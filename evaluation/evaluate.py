#!/usr/bin/env python3
"""
Evaluation harness for SWE-Mutation.

Computes the three metrics defined in the paper for a model's generated test suites:

  Pass@1  — the generated patch applies cleanly and runs without compilation errors.

  VRR     — Verified Reproduction Rate: the test suite fails on the original buggy
             repository (Reproduction) AND passes on the golden fixed code (Validity).

  RDR     — Relative Detection Rate (micro-average across all instances):
                RDR = Σ |M_gen^(i) \ M_base^(i)| / Σ |M^(i) \ M_base^(i)|
             where M^(i) is the full mutant set for instance i,
             M_base^(i) is the set killed by the baseline test suite
             (empty for test-generation, non-empty for test-repair),
             and M_gen^(i) is the set killed by the model-generated test suite.

Bootstrap 95 % CIs and Wilcoxon signed-rank tests are reported in the summary.
"""

from __future__ import annotations

import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
from rich.live import Live

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

from minisweagent.environments.docker import DockerEnvironment
from framework._utils import RunBatchProgressManager

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InstanceResult:
    instance_id: str
    pass_at_1: bool = False
    vrr: bool = False
    m_total: set = field(default_factory=set)   # all mutant IDs for this instance
    m_base:  set = field(default_factory=set)   # mutants killed by baseline suite
    m_gen:   set = field(default_factory=set)   # mutants killed by generated suite
    error: str = ""

    @property
    def m_hidden(self) -> set:
        return self.m_total - self.m_base

    @property
    def m_gen_new(self) -> set:
        return self.m_gen - self.m_base

    @property
    def instance_rdr_numerator(self) -> int:
        return len(self.m_gen_new)

    @property
    def instance_rdr_denominator(self) -> int:
        return len(self.m_hidden)

    def to_dict(self) -> dict:
        return {
            "pass_at_1": self.pass_at_1,
            "vrr": self.vrr,
            "m_total": sorted(self.m_total),
            "m_base": sorted(self.m_base),
            "m_gen": sorted(self.m_gen),
            "m_gen_new": sorted(self.m_gen_new),
            "m_hidden": sorted(self.m_hidden),
            "rdr_numerator": self.instance_rdr_numerator,
            "rdr_denominator": self.instance_rdr_denominator,
            "error": self.error,
        }


@dataclass
class AggregateMetrics:
    n_instances: int = 0
    n_pass_at_1: int = 0
    n_vrr: int = 0
    rdr_numerator: int = 0
    rdr_denominator: int = 0
    # per-instance arrays for statistical tests
    instance_rdr_nums: list = field(default_factory=list)
    instance_rdr_dens: list = field(default_factory=list)

    @property
    def pass_at_1(self) -> float:
        return self.n_pass_at_1 / self.n_instances if self.n_instances else 0.0

    @property
    def vrr(self) -> float:
        return self.n_vrr / self.n_instances if self.n_instances else 0.0

    @property
    def rdr(self) -> float:
        return self.rdr_numerator / self.rdr_denominator if self.rdr_denominator else 0.0

    def bootstrap_ci(self, n_resamples: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
        """Bootstrap 95 % CI for RDR (micro-average)."""
        n = len(self.instance_rdr_nums)
        if n == 0:
            return 0.0, 0.0
        rng = random.Random(42)
        boot_rdrs = []
        for _ in range(n_resamples):
            idxs = [rng.randrange(n) for _ in range(n)]
            num = sum(self.instance_rdr_nums[i] for i in idxs)
            den = sum(self.instance_rdr_dens[i] for i in idxs)
            boot_rdrs.append(num / den if den else 0.0)
        boot_rdrs.sort()
        lo = boot_rdrs[int(alpha / 2 * n_resamples)]
        hi = boot_rdrs[int((1 - alpha / 2) * n_resamples)]
        return lo, hi

    def to_dict(self) -> dict:
        ci_lo, ci_hi = self.bootstrap_ci()
        return {
            "n_instances": self.n_instances,
            "pass_at_1": round(self.pass_at_1 * 100, 2),
            "vrr": round(self.vrr * 100, 2),
            "rdr": round(self.rdr * 100, 2),
            "rdr_ci_95_lower": round(ci_lo * 100, 2),
            "rdr_ci_95_upper": round(ci_hi * 100, 2),
            "rdr_numerator": self.rdr_numerator,
            "rdr_denominator": self.rdr_denominator,
        }


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _start_env(image: str, cwd: str = "/testbed", timeout: int = 600) -> DockerEnvironment:
    env = DockerEnvironment(image=image, cwd=cwd, timeout=timeout, use_sudo=True)
    env.execute("git config --global user.email 'eval@swe-mutation.dev'")
    env.execute("git config --global user.name 'SWE-Mutation Eval'")
    return env


def _git_reset_clean(env: DockerEnvironment) -> None:
    env.execute("git reset --hard && git clean -fd && git checkout .")


def _write_and_apply_patch(env: DockerEnvironment, patch_text: str, label: str) -> bool:
    """Apply patch; returns True on success."""
    if not patch_text or not patch_text.strip():
        return True  # empty patch is a no-op, not a failure
    marker = f"SWE_MUTATION_{label.upper()}_PATCH_EOF"
    env.execute(f"cat > /tmp/{label}.patch << '{marker}'\n{patch_text}\n{marker}")
    result = env.execute(f"git apply -p1 /tmp/{label}.patch 2>&1")
    return result.get("returncode", 1) == 0


def _get_instance_test_cmd(instance: dict) -> str:
    try:
        repo = instance.get("repo", "")
        version = instance.get("version", "")
        if repo and version:
            raw = MAP_REPO_VERSION_TO_SPECS.get(repo, {}).get(version, {}).get("test_cmd", "")
            if isinstance(raw, list):
                return raw[0] if raw else ""
            return raw or ""
    except Exception:
        pass
    return ""


def _get_image_name(instance: dict) -> str:
    if "image_name" in instance:
        return instance["image_name"]
    iid = instance["instance_id"]
    return f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def _parse_test_output(output: str) -> dict:
    passed = failed = errors = skipped = 0
    try:
        for line in reversed(output.split("\n")):
            line = line.strip()
            if not line:
                continue
            # PHPUnit
            if line.startswith("Tests:") and "Assertions:" in line:
                m = re.search(r"Tests:\s*(\d+)", line)
                if m:
                    total = int(m.group(1))
                    fm = re.search(r"Failures:\s*(\d+)", line)
                    em = re.search(r"Errors:\s*(\d+)", line)
                    failed = int(fm.group(1)) if fm else 0
                    errors = int(em.group(1)) if em else 0
                    passed = total - failed - errors
                    break
            # Maven / JUnit
            if "Tests run:" in line:
                m = re.search(r"Tests run:\s*(\d+)", line)
                if m:
                    total = int(m.group(1))
                    fm = re.search(r"Failures:\s*(\d+)", line)
                    em = re.search(r"Errors:\s*(\d+)", line)
                    failed = int(fm.group(1)) if fm else 0
                    errors = int(em.group(1)) if em else 0
                    passed = total - failed - errors
                    break
            # pytest / django
            if any(k in line for k in ("passed", "failed", "error", "skipped")):
                nums = re.findall(r"(\d+)\s+(passed|failed|errors?|skipped)", line)
                if nums:
                    for cnt, kw in nums:
                        c = int(cnt)
                        if kw == "passed":      passed = c
                        elif kw == "failed":    failed = c
                        elif kw in ("error", "errors"): errors = c
                        elif kw == "skipped":   skipped = c
                    break
    except Exception:
        pass
    return {"passed": passed, "failed": failed, "errors": errors, "skipped": skipped,
            "total": passed + failed + errors + skipped}


def _is_compilation_error(output: str, returncode: int) -> bool:
    """Detect catastrophic failures: syntax errors, import failures, test collection errors."""
    if returncode in (125, 126, 127):  # command not found / permission denied
        return True
    signals = [
        "SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError",
        "cannot import name", "error: could not compile",
        "BUILD FAILURE", "COMPILATION ERROR",
        "error[E",           # Rust compile error
        "cannot find symbol", # Java compile error
        "exit status 1\nno test files",
    ]
    out_lower = output.lower()
    for sig in signals:
        if sig.lower() in out_lower:
            return True
    # pytest collection error (exit code 2)
    if returncode == 2 and "ERROR collecting" in output:
        return True
    return False


def _run_test_cmd(env: DockerEnvironment, test_cmd: str, test_files: list[str],
                   instance_id: str = "") -> dict:
    """Run the test command and return {passed, failed, errors, output, returncode}."""
    if not test_cmd:
        return {"passed": 0, "failed": 0, "errors": 0, "output": "", "returncode": 0}

    cmd = test_cmd
    if test_files:
        if "runtests.py" in test_cmd:
            # Django: convert file paths to module names
            modules = []
            for f in test_files:
                if f.endswith(".py") and f.startswith("tests/"):
                    modules.append(f[6:-3].replace("/", "."))
            if modules:
                cmd = f"{test_cmd} {' '.join(modules)}"
        elif "pytest" in test_cmd or "mvn" not in test_cmd:
            cmd = f"{test_cmd} {' '.join(test_files)}"

    if "runtests.py" in cmd or "django" in instance_id.lower():
        cmd = f"export PYTHONIOENCODING=utf-8 && export LC_ALL=C.UTF-8 && {cmd}"
    if "cargo" in cmd:
        cmd = f"source $HOME/.cargo/env 2>/dev/null || true && {cmd}"

    try:
        res = env.execute(f"{cmd} 2>&1 | cat")
        output = res.get("output", "")
        returncode = res.get("returncode", 1)
    except Exception as e:
        return {"passed": 0, "failed": 0, "errors": 1, "output": str(e), "returncode": 1}

    results = _parse_test_output(output)
    results["output"] = output
    results["returncode"] = returncode
    return results


# ---------------------------------------------------------------------------
# Scenario runners (each resets the environment first)
# ---------------------------------------------------------------------------

def _scenario_pass_at_1(env: DockerEnvironment, generated_test_patch: str,
                          test_files: list[str], test_cmd: str,
                          instance_id: str = "") -> bool:
    """
    Pass@1: the generated patch applies cleanly and runs without compilation errors.
    Does NOT apply code_patch — tested on the original (buggy) repo with the model's tests.
    """
    _git_reset_clean(env)
    if not _write_and_apply_patch(env, generated_test_patch, "gen_test"):
        return False
    results = _run_test_cmd(env, test_cmd, test_files, instance_id)
    return not _is_compilation_error(results.get("output", ""), results.get("returncode", 1))


def _scenario_vrr_reproduction(env: DockerEnvironment, generated_test_patch: str,
                                 test_files: list[str], test_cmd: str,
                                 instance_id: str = "") -> bool:
    """
    VRR (Reproduction): generated tests FAIL on original buggy repository.
    """
    _git_reset_clean(env)
    if not _write_and_apply_patch(env, generated_test_patch, "gen_test"):
        return False
    results = _run_test_cmd(env, test_cmd, test_files, instance_id)
    return results.get("failed", 0) > 0 or results.get("errors", 0) > 0


def _scenario_vrr_validity(env: DockerEnvironment, code_patch: str, generated_test_patch: str,
                             test_files: list[str], test_cmd: str,
                             instance_id: str = "") -> bool:
    """
    VRR (Validity): generated tests PASS on golden fixed code.
    """
    _git_reset_clean(env)
    if not _write_and_apply_patch(env, code_patch, "code"):
        return False
    if not _write_and_apply_patch(env, generated_test_patch, "gen_test"):
        return False
    results = _run_test_cmd(env, test_cmd, test_files, instance_id)
    return (results.get("failed", 0) == 0 and results.get("errors", 0) == 0
            and results.get("total", 0) > 0)


def _scenario_mutant_killed_by(env: DockerEnvironment, code_patch: str, test_suite_patch: str,
                                 mutant_diff: str, test_files: list[str], test_cmd: str,
                                 instance_id: str = "") -> bool:
    """
    Returns True if the test suite kills the mutant (tests fail on mutated code).

    Setup:
      original repo → apply code_patch (golden state) → apply mutant_diff (re-introduce bug)
      → apply test_suite_patch → run tests → FAIL means mutant killed.
    """
    _git_reset_clean(env)
    if not _write_and_apply_patch(env, code_patch, "code"):
        return False
    if not _write_and_apply_patch(env, mutant_diff, "mutant"):
        return False
    if test_suite_patch and not _write_and_apply_patch(env, test_suite_patch, "suite"):
        return False
    results = _run_test_cmd(env, test_cmd, test_files, instance_id)
    return results.get("failed", 0) > 0 or results.get("errors", 0) > 0


# ---------------------------------------------------------------------------
# Per-instance evaluator
# ---------------------------------------------------------------------------

class InstanceEvaluator:
    def evaluate(
        self,
        instance: dict,
        mutants: dict[str, str],         # {mutant_id: diff_text}
        code_patch: str,
        golden_test_patch: str,
        generated_test_patch: str,
        task: str,                       # "test_generation" or "test_repair"
        image_name: str,
        timeout: int = 600,
    ) -> InstanceResult:
        instance_id = instance["instance_id"]
        result = InstanceResult(instance_id=instance_id, m_total=set(mutants.keys()))
        test_files = instance.get("test_files", [])
        test_cmd = _get_instance_test_cmd(instance)

        env = _start_env(image_name, timeout=timeout)
        try:
            # --- Pass@1 ---
            result.pass_at_1 = _scenario_pass_at_1(env, generated_test_patch, test_files, test_cmd, instance_id)
            if not result.pass_at_1:
                return result

            # --- VRR ---
            reproduction = _scenario_vrr_reproduction(env, generated_test_patch, test_files, test_cmd, instance_id)
            validity = _scenario_vrr_validity(env, code_patch, generated_test_patch, test_files, test_cmd, instance_id)
            result.vrr = reproduction and validity

            # --- M_base (test repair only; empty for test generation) ---
            if task == "test_repair":
                for mut_id, mut_diff in mutants.items():
                    if _scenario_mutant_killed_by(env, code_patch, golden_test_patch, mut_diff, test_files, test_cmd, instance_id):
                        result.m_base.add(mut_id)

            # --- M_gen ---
            for mut_id, mut_diff in mutants.items():
                if _scenario_mutant_killed_by(env, code_patch, generated_test_patch, mut_diff, test_files, test_cmd, instance_id):
                    result.m_gen.add(mut_id)

        except Exception as e:
            result.error = str(e)
            traceback.print_exc()
        finally:
            try:
                env.cleanup()
            except Exception:
                pass

        return result


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _parse_list_field(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                return [str(x) for x in json.loads(s)]
            return [x.strip() for x in s.split(",") if x.strip()]
        except Exception:
            return [s] if s else []
    return []


def _load_patches(patches_file: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for line in patches_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        iid = obj.get("instance_id") or obj.get("id") or obj.get("name")
        if not iid:
            continue
        mapping[str(iid)] = {
            "repo":         obj.get("repo", ""),
            "version":      obj.get("version", ""),
            "base_commit":  obj.get("base_commit", ""),
            "patch":        obj.get("patch", ""),
            "test_patch":   obj.get("test_patch", ""),
            "test_files":   _parse_list_field(obj.get("test_files")),
            "F2P":          _parse_list_field(obj.get("FAIL_TO_PASS")),
            "P2P":          _parse_list_field(obj.get("PASS_TO_PASS")),
            "problem_statement": obj.get("problem_statement", ""),
        }
    return mapping


def _load_mutants(mutants_file: Path) -> dict[str, dict[str, str]]:
    """
    Load mutants from mutation.py's preds.json.
    Returns {instance_id: {mutant_id: diff_text}}.
    """
    if not mutants_file.exists():
        return {}
    data = json.loads(mutants_file.read_text())
    result: dict[str, dict[str, str]] = {}
    for iid, entry in data.items():
        patch_str = entry.get("model_patch", "") if isinstance(entry, dict) else str(entry)
        try:
            obj = json.loads(patch_str)
            mutations = obj.get("mutations", obj.get("hacks", []))
            mutant_dict: dict[str, str] = {}
            for idx, mut in enumerate(mutations, 1):
                if isinstance(mut, dict) and "diff" in mut:
                    mutant_dict[f"mutant_{idx}"] = mut["diff"]
            result[iid] = mutant_dict
        except Exception:
            result[iid] = {}
    return result


def _load_test_preds(test_preds_file: Path) -> dict[str, str]:
    """
    Load generated test patches from the agent's preds.json.
    Returns {instance_id: test_diff_text}.
    """
    if not test_preds_file.exists():
        return {}
    data = json.loads(test_preds_file.read_text())
    result: dict[str, str] = {}
    for iid, entry in data.items():
        if isinstance(entry, dict):
            result[iid] = entry.get("model_patch", "")
        else:
            result[iid] = str(entry)
    return result


# ---------------------------------------------------------------------------
# Aggregate and save
# ---------------------------------------------------------------------------

def _aggregate(results: list[InstanceResult]) -> AggregateMetrics:
    agg = AggregateMetrics()
    for r in results:
        agg.n_instances += 1
        if r.pass_at_1:
            agg.n_pass_at_1 += 1
        if r.vrr:
            agg.n_vrr += 1
        agg.rdr_numerator += r.instance_rdr_numerator
        agg.rdr_denominator += r.instance_rdr_denominator
        agg.instance_rdr_nums.append(r.instance_rdr_numerator)
        agg.instance_rdr_dens.append(r.instance_rdr_denominator)
    return agg


def _wilcoxon_p(nums_a: list, dens_a: list, nums_b: list, dens_b: list) -> Optional[float]:
    """Instance-level Wilcoxon signed-rank test between two RDR series."""
    try:
        from scipy.stats import wilcoxon  # type: ignore
        scores_a = [n / d if d else 0.0 for n, d in zip(nums_a, dens_a)]
        scores_b = [n / d if d else 0.0 for n, d in zip(nums_b, dens_b)]
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return None
        stat, p = wilcoxon(scores_a, scores_b)
        return float(p)
    except Exception:
        return None


def _save_results(output_dir: Path, instance_results: list[InstanceResult],
                   agg: AggregateMetrics) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_FILE_LOCK:
        per_instance = {r.instance_id: r.to_dict() for r in instance_results}
        (output_dir / "instance_results.json").write_text(
            json.dumps(per_instance, indent=2, ensure_ascii=False)
        )
        (output_dir / "summary.json").write_text(
            json.dumps(agg.to_dict(), indent=2, ensure_ascii=False)
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    patches_file: Path,
    mutants_file: Path,
    test_preds_file: Path,
    task: str,
    output_dir: Path,
    workers: int = 1,
    filter_spec: str = "",
    image_override: Optional[str] = None,
    timeout: int = 600,
) -> AggregateMetrics:
    patches     = _load_patches(patches_file)
    all_mutants = _load_mutants(mutants_file)
    test_preds  = _load_test_preds(test_preds_file)

    instance_ids = [iid for iid in patches if iid in test_preds and iid in all_mutants]
    if filter_spec:
        import re as _re
        instance_ids = [iid for iid in instance_ids if _re.match(filter_spec, iid)]

    if not instance_ids:
        print("No instances to evaluate.")
        return AggregateMetrics()

    print(f"Evaluating {len(instance_ids)} instances (task={task}, workers={workers}) …")

    progress = RunBatchProgressManager(len(instance_ids), output_dir / "progress.yaml")
    results: list[InstanceResult] = []
    lock = threading.Lock()

    def process(iid: str) -> None:
        patch_data = patches[iid]
        instance = {
            "instance_id": iid,
            "repo": patch_data["repo"],
            "version": patch_data["version"],
            "base_commit": patch_data["base_commit"],
            "test_files": patch_data["test_files"],
            "problem_statement": patch_data["problem_statement"],
        }
        mutants = all_mutants.get(iid, {})
        if not mutants:
            progress.on_instance_start(iid)
            progress.on_instance_end(iid, "No mutants — skipped")
            return

        generated_test_patch = test_preds.get(iid, "")
        code_patch = patch_data["patch"]
        golden_test_patch = patch_data["test_patch"]
        image_name = image_override or _get_image_name(instance)

        progress.on_instance_start(iid)
        progress.update_instance_status(iid, "Running …")

        evaluator = InstanceEvaluator()
        r = evaluator.evaluate(
            instance=instance,
            mutants=mutants,
            code_patch=code_patch,
            golden_test_patch=golden_test_patch,
            generated_test_patch=generated_test_patch,
            task=task,
            image_name=image_name,
            timeout=timeout,
        )

        with lock:
            results.append(r)

        status = (f"Pass@1={r.pass_at_1} VRR={r.vrr} "
                  f"M_gen={len(r.m_gen)}/{len(r.m_total)} "
                  f"RDR_i={r.instance_rdr_numerator}/{r.instance_rdr_denominator}")
        progress.on_instance_end(iid, status)

    def drain(futures: dict) -> None:
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                iid = futures[f]
                print(f"Error for {iid}: {e}")
                traceback.print_exc()
                progress.on_uncaught_exception(iid, e)

    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process, iid): iid for iid in instance_ids}
            try:
                drain(futures)
            except KeyboardInterrupt:
                print("Cancelling …")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                drain(futures)

    agg = _aggregate(results)
    _save_results(output_dir, results, agg)

    print("\n=== Evaluation Summary ===")
    for k, v in agg.to_dict().items():
        print(f"  {k}: {v}")

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    patches_file: Path = typer.Option(..., "--patches-file",
        help="JSONL file with instance patch data (same format as mutation.py)"),
    mutants_file: Path = typer.Option(..., "--mutants-file",
        help="preds.json produced by mutation.py"),
    test_preds_file: Path = typer.Option(..., "--test-preds-file",
        help="preds.json produced by the test-generation or test-repair agent"),
    task: str = typer.Option("test_generation", "--task",
        help="'test_generation' or 'test_repair'"),
    output: Path = typer.Option(Path("./eval_results"), "-o", "--output"),
    workers: int = typer.Option(1, "-w", "--workers"),
    filter_spec: str = typer.Option("", "--filter",
        help="Regex to filter instance IDs"),
    image: Optional[str] = typer.Option(None, "--image",
        help="Override Docker image for all instances"),
    timeout: int = typer.Option(600, "--timeout",
        help="Container timeout in seconds per scenario"),
    compare_file: Optional[Path] = typer.Option(None, "--compare",
        help="Another instance_results.json to run Wilcoxon test against"),
) -> None:
    """Evaluate generated test suites with Pass@1, VRR, and RDR."""
    if task not in ("test_generation", "test_repair"):
        print(f"Unknown task '{task}'. Must be 'test_generation' or 'test_repair'.")
        raise typer.Exit(1)

    agg = evaluate(
        patches_file=patches_file,
        mutants_file=mutants_file,
        test_preds_file=test_preds_file,
        task=task,
        output_dir=output,
        workers=workers,
        filter_spec=filter_spec,
        image_override=image,
        timeout=timeout,
    )

    # Optional Wilcoxon comparison
    if compare_file and compare_file.exists():
        try:
            other_data = json.loads(compare_file.read_text())
            other_nums = [v.get("rdr_numerator", 0) for v in other_data.values()]
            other_dens = [v.get("rdr_denominator", 0) for v in other_data.values()]
            p = _wilcoxon_p(agg.instance_rdr_nums, agg.instance_rdr_dens, other_nums, other_dens)
            if p is not None:
                print(f"\nWilcoxon signed-rank vs {compare_file.name}: p = {p:.4f}")
        except Exception as e:
            print(f"Wilcoxon comparison failed: {e}")


if __name__ == "__main__":
    app()
