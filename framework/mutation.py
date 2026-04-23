#!/usr/bin/env python3
"""
Mutation module of the Agentic Mutation Framework (SWE-Mutation).

For each instance, five rounds are executed — one per strategy group.
In each round:
  1. A MutationAgent explores the golden repository and injects a bug.
  2. The Judge step applies the candidate patch to a fresh environment and
     runs Fail-to-Pass (F2P) tests to verify that the injection is effective.
  3. Rejected candidates are retried with corrective feedback up to retry_limit times.
"""

from __future__ import annotations

import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.test_spec import make_test_spec

from minisweagent.agents.default import DefaultAgent, NonTerminatingException, TerminatingException
from minisweagent.config import get_config_path
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from framework._utils import RunBatchProgressManager, save_traj

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}

STRATEGY_GROUPS = [
    ("A", "API Specifications & Contracts",   ["A1", "A2", "A3", "A4"]),
    ("B", "Boundaries & Conditional Logic",   ["B1", "B2", "B3"]),
    ("C", "Type & Data Shape",                ["C1", "C2", "C3"]),
    ("D", "Stateful Logic & Sequences",       ["D1", "D2", "D3", "D4", "D5", "D6"]),
    ("E", "Test-Expectation Alignment",       ["E1", "E2"]),
]

_OUTPUT_FILE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Docker image helpers
# ---------------------------------------------------------------------------

def get_swebench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name")
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def _start_env(image: str, cwd: str = "/testbed", timeout: int = 600) -> DockerEnvironment:
    env = DockerEnvironment(image=image, cwd=cwd, timeout=timeout, use_sudo=True)
    env.execute("git config --global user.email 'agent@swe-mutation.dev'")
    env.execute("git config --global user.name 'SWE-Mutation'")
    return env


# ---------------------------------------------------------------------------
# Output file helpers
# ---------------------------------------------------------------------------

def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str) -> None:
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text()) if output_path.exists() else {}
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str) -> None:
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def _check_instance_exists(output_path: Path, instance_id: str) -> bool:
    if not output_path.exists():
        return False
    try:
        content = output_path.read_text()
        if not content.strip():
            return False
        return instance_id in json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Patch data loader
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
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()]
            if s:
                return [s]
        except Exception:
            return [s]
    return []


def _load_test_patches(patches_file: Path) -> dict[str, dict]:
    if not patches_file.exists():
        return {}
    mapping: dict[str, dict] = {}
    for line in patches_file.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        instance_id = obj.get("instance_id") or obj.get("id") or obj.get("name")
        if not instance_id:
            continue
        mapping[str(instance_id)] = {
            "patch":            obj.get("patch", ""),
            "test_patch":       obj.get("test_patch", ""),
            "test_files":       _parse_list_field(obj.get("test_files")),
            "files":            _parse_list_field(obj.get("files")),
            "F2P":              _parse_list_field(obj.get("FAIL_TO_PASS")),
            "P2P":              _parse_list_field(obj.get("PASS_TO_PASS")),
            "repo_description": obj.get("repo_description", obj.get("problem_statement", "")),
        }
    return mapping


def _load_all_instance_ids(patches_file: Path) -> list[str]:
    if not patches_file.exists():
        return []
    ids: list[str] = []
    for line in patches_file.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        iid = obj.get("instance_id") or obj.get("id") or obj.get("name")
        if iid:
            ids.append(str(iid))
    return ids


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def _git_reset_clean(env: DockerEnvironment) -> None:
    env.execute("git reset --hard && git clean -fd && git checkout .")


def _write_and_apply_patch(env: DockerEnvironment, patch_text: str, label: str) -> str:
    if not patch_text or not patch_text.strip():
        return f"{label} patch: empty (skipped)"
    marker = f"SWE_MUTATION_{label.upper()}_PATCH_EOF"
    env.execute(f"cat > /tmp/{label}.patch << '{marker}'\n{patch_text}\n{marker}")
    result = env.execute(f"git apply -p1 /tmp/{label}.patch 2>&1")
    if result.get("returncode", 1) != 0:
        return f"{label} patch apply failed: {(result.get('stdout') or result.get('output', ''))[:300]}"
    return f"{label} patch applied"


def _apply_candidate(
    env: DockerEnvironment,
    code_patch: str,
    test_patch: str,
    candidate_diff: str,
    instance_id: str = "",
) -> dict:
    """Reset to original, apply golden baseline, commit, then apply the candidate."""
    _git_reset_clean(env)
    code_r = _write_and_apply_patch(env, code_patch, "code")
    test_r = _write_and_apply_patch(env, test_patch, "test")
    if "failed" in code_r.lower() or "failed" in test_r.lower():
        return {"ok": False, "reason": "baseline_apply_failed", "error": f"{code_r}; {test_r}"}

    env.execute("git add -A && git commit --allow-empty -m 'chore: apply baseline patches'")

    if not candidate_diff.strip():
        return {"ok": False, "reason": "empty_candidate"}

    apply_r = _write_and_apply_patch(env, candidate_diff, "candidate")
    if "failed" in apply_r.lower():
        return {"ok": False, "reason": "candidate_apply_failed", "error": apply_r}

    return {"ok": True}


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def _get_instance_test_cmd(instance: dict) -> str | list:
    try:
        repo = instance.get("repo", "")
        version = instance.get("version", "")
        if repo and version:
            return MAP_REPO_VERSION_TO_SPECS.get(repo, {}).get(version, {}).get("test_cmd", "")
    except Exception:
        pass
    return ""


def _convert_django_test_name(test_name: str) -> str:
    m = re.match(r"^(\S+)\s+\(([^)]+)\)$", test_name)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    return test_name


def _match_test_cmd_for_tests(test_cmd_list: list, test_names: list[str]) -> str:
    for test_name in test_names:
        for cmd in test_cmd_list:
            if test_name in cmd:
                return cmd
    return test_cmd_list[0] if test_cmd_list else ""


def _parse_jest_test_status(output: str, test_names: list[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    lines = output.split("\n")
    for test_name in test_names:
        found = False
        for line in lines:
            if test_name.lower() in line.lower():
                if "✓" in line or "PASS" in line:
                    status[test_name] = "passed"; found = True; break
                elif "✕" in line or "FAIL" in line or "×" in line:
                    status[test_name] = "failed"; found = True; break
                elif "○" in line or "skipped" in line.lower():
                    status[test_name] = "skipped"; found = True; break
        if not found:
            status[test_name] = "unknown"
    return status


def _parse_test_output(output: str, test_cmd: str = "") -> dict:
    passed = failed = errors = skipped = 0
    summary = ""
    try:
        lines = output.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            # PHPUnit: "Tests: 6, Assertions: 19, Errors: 1, Failures: 3."
            if line.startswith("Tests:") and "Assertions:" in line:
                m = re.search(r"Tests:\s*(\d+)", line)
                if m:
                    total = int(m.group(1))
                    failed = int(re.search(r"Failures:\s*(\d+)", line).group(1)) if re.search(r"Failures:\s*(\d+)", line) else 0
                    errors = int(re.search(r"Errors:\s*(\d+)", line).group(1)) if re.search(r"Errors:\s*(\d+)", line) else 0
                    passed = total - failed - errors
                    summary = line
                    break

            # TAP: "# tests N"
            if line.startswith("# tests"):
                m = re.search(r"#\s*tests\s+(\d+)", line)
                if m:
                    total = int(m.group(1))
                    for other in lines:
                        pm = re.search(r"#\s*pass\s+(\d+)", other.strip())
                        fm = re.search(r"#\s*fail\s+(\d+)", other.strip())
                        if pm:
                            passed = int(pm.group(1))
                        if fm:
                            failed = int(fm.group(1))
                    summary = f"TAP: {total} tests, {passed} passed, {failed} failed"
                    break

            # Maven / JUnit: "Tests run: X, Failures: Y, Errors: Z"
            if "Tests run:" in line:
                m = re.search(r"Tests run:\s*(\d+)", line)
                if m:
                    total = int(m.group(1))
                    fm = re.search(r"Failures:\s*(\d+)", line)
                    em = re.search(r"Errors:\s*(\d+)", line)
                    failed = int(fm.group(1)) if fm else 0
                    errors = int(em.group(1)) if em else 0
                    passed = total - failed - errors
                    summary = line
                    break

            # pytest-style: "X passed, Y failed, Z error"
            if any(k in line for k in ("passed", "failed", "error", "skipped")):
                nums = re.findall(r"(\d+)\s+(passed|failed|errors?|skipped)", line)
                if nums:
                    for count, status_word in nums:
                        count = int(count)
                        if status_word == "passed":
                            passed = count
                        elif status_word == "failed":
                            failed = count
                        elif status_word in ("error", "errors"):
                            errors = count
                        elif status_word == "skipped":
                            skipped = count
                    summary = line
                    break

            # Django: "Ran X tests in Y.ZZZs"
            if "Ran" in line and "test" in line:
                m = re.search(r"Ran\s+(\d+)\s+test", line)
                if m:
                    total = int(m.group(1))
                    summary = line
                    idx = lines.index(line + "\n") if (line + "\n") in lines else -1
                    for next_line in (lines[idx + 1:idx + 6] if idx >= 0 else []):
                        nl = next_line.strip()
                        if re.match(r"^OK", nl):
                            passed = total
                            break
                        fm2 = re.match(r"^FAILED\s+\((.+)\)$", nl)
                        if fm2:
                            parts = fm2.group(1)
                            fm3 = re.search(r"failures?=(\d+)", parts)
                            em3 = re.search(r"errors?=(\d+)", parts)
                            sm3 = re.search(r"skipped=(\d+)", parts)
                            failed = int(fm3.group(1)) if fm3 else 0
                            errors = int(em3.group(1)) if em3 else 0
                            skipped = int(sm3.group(1)) if sm3 else 0
                            passed = max(0, total - failed - errors - skipped)
                            break
                    break

    except Exception:
        pass

    return {
        "passed": passed, "failed": failed,
        "errors": errors, "skipped": skipped,
        "total": passed + failed + errors + skipped,
        "summary": summary,
    }


def _run_tests_with_cmd(
    env: DockerEnvironment,
    test_cmd: str,
    test_nodeids: list[str] | None = None,
    instance_id: str = "",
) -> tuple[bool, dict]:
    if not test_cmd:
        return True, {"test_results": {}, "output": ""}

    cmd = test_cmd
    if test_nodeids:
        if "pytest" in test_cmd:
            cmd = f"{test_cmd} {' '.join(test_nodeids)}"
        elif "runtests.py" in test_cmd:
            converted = [_convert_django_test_name(t) for t in test_nodeids]
            cmd = f"{test_cmd} {' '.join(converted)}"
        elif "jest" in test_cmd or "yarn jest" in test_cmd:
            pass  # run all; filter results by name below
        elif "mocha" in test_cmd or "npx mocha" in test_cmd:
            cmd = f"{test_cmd} --grep '{' | '.join(test_nodeids)}'"
        elif "cargo test" in test_cmd:
            cmd = f"{test_cmd} {' '.join(test_nodeids)}"
        elif test_cmd.strip().endswith("--"):
            cmd = f"{test_cmd} {' '.join(test_nodeids)}"
        else:
            cmd = f"{test_cmd} {' '.join(test_nodeids)}"

    if "cargo" in cmd:
        cmd = f"source $HOME/.cargo/env 2>/dev/null || true && {cmd}"
    if "runtests.py" in cmd or "django" in instance_id.lower():
        cmd = f"export PYTHONIOENCODING=utf-8 && export LC_ALL=C.UTF-8 && {cmd}"

    try:
        res = env.execute(f"{cmd} 2>&1 | cat")
        output = res.get("output", "")
        returncode = res.get("returncode", 1)
    except Exception as e:
        return False, {
            "test_results": {"passed": 0, "failed": 0, "errors": 1, "skipped": 0, "total": 1, "summary": str(e)},
            "output": str(e),
        }

    if test_nodeids and ("jest" in test_cmd or "yarn jest" in test_cmd):
        jest_status = _parse_jest_test_status(output, test_nodeids)
        failed_tests = [t for t, s in jest_status.items() if s == "failed"]
        results = _parse_test_output(output, test_cmd)
        if not results.get("failed") and not results.get("errors"):
            results["failed"] = len(failed_tests)
        return returncode == 0, {"test_results": results, "output": output}

    return returncode == 0, {"test_results": _parse_test_output(output, test_cmd), "output": output}


# ---------------------------------------------------------------------------
# Judge step (candidate validation)
# ---------------------------------------------------------------------------

def _verify_candidate(
    env: DockerEnvironment,
    instance: dict,
    f2p_tests: list[str],
    test_cmd: str | list | None = None,
) -> dict:
    """
    Judge module: apply the candidate and confirm at least one F2P test fails.
    """
    if not f2p_tests:
        return {"ok": False, "reason": "no_f2p_tests", "f2p_results": {}}

    if test_cmd is None:
        test_cmd = _get_instance_test_cmd(instance)

    if isinstance(test_cmd, list):
        f2p_cmd = _match_test_cmd_for_tests(test_cmd, f2p_tests)
    else:
        f2p_cmd = test_cmd or ""

    _, info = _run_tests_with_cmd(env, f2p_cmd, f2p_tests, instance.get("instance_id", ""))
    results = info.get("test_results", {})
    f2p_failed = results.get("failed", 0) > 0 or results.get("errors", 0) > 0

    return {
        "ok": f2p_failed,
        "reason": "f2p_failed" if f2p_failed else "no_f2p_failure",
        "f2p_results": results,
        "output_summary": info.get("output", "")[-2000:],
    }


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------

def _parse_agent_output(payload: str) -> tuple[str, str]:
    try:
        obj = json.loads(payload)
        return obj.get("diff", ""), obj.get("explanation", "")
    except Exception:
        return payload, ""


class MutationAgent(DefaultAgent):
    def run_once(self, task: str, *, template_vars: dict | None = None) -> tuple[str, str]:
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template, task=task, **(template_vars or {})))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def continue_with_feedback(self, feedback: str) -> tuple[str, str]:
        self.add_message("user", feedback)
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)


class ProgressTrackingMutationAgent(MutationAgent):
    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


# ---------------------------------------------------------------------------
# Core per-instance pipeline
# ---------------------------------------------------------------------------

def process_instance(
    instance: dict,
    output_dir: Path,
    patches_file: Path,
    model_name: str | None,
    config_path: str | Path,
    progress_manager: RunBatchProgressManager,
    base_url: str | None = None,
    api_key: str | None = None,
    retry_limit: int = 3,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    image_name = get_swebench_docker_image_name(instance)
    config = yaml.safe_load(get_config_path(config_path).read_text())
    model_cfg = config.get("model", {})
    if base_url:
        model_cfg["base_url"] = base_url
    if api_key:
        model_cfg.setdefault("model_kwargs", {})["api_key"] = api_key

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Loading patch data")

    entry = _load_test_patches(patches_file).get(instance_id, {})
    code_patch:  str       = entry.get("patch", "")
    test_patch:  str       = entry.get("test_patch", "")
    test_files:  list[str] = entry.get("test_files", [])
    allowed_files: list[str] = entry.get("files", [])
    repo_desc:   str       = entry.get("repo_description", instance.get("problem_statement", ""))
    f2p_tests:   list[str] = entry.get("F2P", [])

    try:
        make_test_spec(instance)
    except Exception as e:
        progress_manager.on_instance_end(instance_id, "Error creating test spec")
        print(f"Error creating test spec for {instance_id}: {e}")
        return

    test_cmd = _get_instance_test_cmd(instance)

    mutation_results: list[dict] = []

    try:
        import shutil

        for round_idx, (group_code, group_name, strategies) in enumerate(STRATEGY_GROUPS, 1):
            progress_manager.update_instance_status(instance_id, f"Round {round_idx}/5: {group_name}")
            round_dir = instance_dir / f"round_{round_idx}_{group_code}"
            round_dir.mkdir(parents=True, exist_ok=True)

            env_mut = _start_env(image_name, timeout=600)
            env_val = _start_env(image_name, timeout=600)

            try:
                # Prepare mutation agent baseline (golden state)
                _git_reset_clean(env_mut)
                code_r = _write_and_apply_patch(env_mut, code_patch, "code")
                test_r = _write_and_apply_patch(env_mut, test_patch, "test")
                if "failed" in code_r.lower() or "failed" in test_r.lower():
                    raise RuntimeError(f"Failed to apply baseline patches: {code_r}; {test_r}")
                env_mut.execute("git add -A && git commit --allow-empty -m 'chore: apply baseline patches'")

                template_vars = {
                    "repo_description": repo_desc,
                    "test_files": test_files,
                    "allowed_files": allowed_files,
                    "strategy_group": group_code,
                    "strategy_group_name": group_name,
                    "allowed_strategies": strategies,
                    "test_cmd": (test_cmd[0] if isinstance(test_cmd, list) and test_cmd else test_cmd) or "",
                }

                agent = ProgressTrackingMutationAgent(
                    get_model(model_name, config=model_cfg),
                    env_mut,
                    progress_manager=progress_manager,
                    instance_id=instance_id,
                    **config.get("agent", {}),
                )

                exit_state, payload = agent.run_once(repo_desc, template_vars=template_vars)
                diff_text, explan = _parse_agent_output(payload)

                attempt = 0
                accepted = False
                veri_details: dict = {}

                while attempt <= retry_limit:
                    attempt += 1
                    progress_manager.update_instance_status(
                        instance_id, f"Round {round_idx}/5: {group_name} – Attempt {attempt}"
                    )

                    applied = _apply_candidate(env_val, code_patch, test_patch, diff_text, instance_id)
                    if not applied.get("ok"):
                        veri_details = {"ok": False, "reason": applied.get("reason", "apply_failed")}
                    else:
                        veri_details = _verify_candidate(env_val, instance, f2p_tests, test_cmd)

                    if veri_details.get("ok"):
                        accepted = True
                        break

                    if attempt < retry_limit:
                        feedback = (
                            f"Validator rejected (reason: {veri_details.get('reason')}). "
                            f"Try a different subtle bug using {group_name} strategies {strategies}. "
                            f"Do not modify test files. Only modify: {allowed_files}. "
                            f"Output strict JSON with fields 'diff' and 'explanation'."
                        )
                        exit_state, payload = agent.continue_with_feedback(feedback)
                        diff_text, explan = _parse_agent_output(payload)

                save_traj(
                    agent,
                    round_dir / f"{instance_id}__round{round_idx}_{group_code}.traj.json",
                    exit_status=exit_state if accepted else f"Rejected_{exit_state}",
                    result=diff_text,
                    extra_info={
                        "round": round_idx,
                        "strategy_group": group_code,
                        "strategy_group_name": group_name,
                        "allowed_strategies": strategies,
                        "accepted": accepted,
                        "validation": veri_details,
                        "explanation": explan,
                        "attempts": attempt,
                    },
                    instance_id=f"{instance_id}:round{round_idx}_{group_code}",
                )

                mutation_results.append({
                    "round": round_idx,
                    "strategy_group": group_code,
                    "strategy_group_name": group_name,
                    "exit_status": exit_state,
                    "accepted": accepted,
                    "diff": diff_text,
                    "explanation": explan,
                    "validation": veri_details,
                    "attempts": attempt,
                })

            finally:
                for env in (env_mut, env_val):
                    try:
                        env.cleanup()
                    except Exception:
                        pass

        n_accepted = sum(1 for r in mutation_results if r.get("accepted"))
        save_traj(
            None,
            instance_dir / f"{instance_id}.traj.json",
            exit_status="Finished",
            result=f"5 rounds processed, {n_accepted} accepted",
            extra_info={"mutations": mutation_results},
            instance_id=instance_id,
        )

    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        traceback.print_exc()
        save_traj(None, instance_dir / f"{instance_id}.traj.json",
                  exit_status="Error", result=str(e), instance_id=instance_id)

    finally:
        accepted = [r for r in mutation_results if r.get("accepted")]
        if accepted:
            update_preds_file(
                output_dir / "preds.json",
                instance_id,
                model_name or "",
                json.dumps({"mutations": accepted}, ensure_ascii=False),
            )
        progress_manager.on_instance_end(instance_id, "Finished")


# ---------------------------------------------------------------------------
# Instance filtering
# ---------------------------------------------------------------------------

def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before = len(instances)
    if filter_spec:
        instances = [i for i in instances if re.match(filter_spec, i["instance_id"])]
    if len(instances) != before:
        print(f"Instance filter: {before} -> {len(instances)}")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
    return instances


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    patches_file: Path = typer.Option(..., "--patches-file", help="JSONL file with instance patch data"),
    subset:       str   = typer.Option("verified", "--subset"),
    split:        str   = typer.Option("test", "--split"),
    slice_spec:   str   = typer.Option("", "--slice"),
    filter_spec:  str   = typer.Option("", "--filter"),
    shuffle:      bool  = typer.Option(False, "--shuffle"),
    output:       str   = typer.Option("", "-o", "--output"),
    workers:      int   = typer.Option(1, "-w", "--workers"),
    model:        str | None = typer.Option(None, "-m", "--model"),
    config:       Path  = typer.Option(Path("agents/configs/mutation.yaml"), "-c", "--config"),
    retry_limit:  int   = typer.Option(2, "--retry-limit"),
    base_url:     str | None = typer.Option(None, "--base-url"),
    api_key:      str | None = typer.Option(None, "--api-key"),
    start_index:  int   = typer.Option(0, "--start-index"),
    num_instances: int | None = typer.Option(None, "--num-instances"),
    suffix:       str   = typer.Option("", "--suffix"),
    skip_existing: bool = typer.Option(False, "--skip-existing"),
) -> None:
    all_ids = _load_all_instance_ids(patches_file)
    if not all_ids:
        print(f"No instances found in {patches_file}.")
        raise typer.Exit(code=1)

    end_index = start_index + num_instances if num_instances is not None else None
    selected_ids = all_ids[start_index:end_index]

    print(f"Loading dataset {DATASET_MAPPING.get(subset, subset)} split={split} ...")
    dataset_instances = list(load_dataset(DATASET_MAPPING.get(subset, subset), split=split))
    dataset_dict = {i["instance_id"]: i for i in dataset_instances}

    instances = [dataset_dict[iid] for iid in selected_ids if iid in dataset_dict]
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    if not output:
        cfg_data = yaml.safe_load(get_config_path(config).read_text())
        mc = cfg_data.get("model", {})
        if base_url:
            mc["base_url"] = base_url
        if api_key:
            mc.setdefault("model_kwargs", {})["api_key"] = api_key
        temp_model = get_model(model, config=mc)
        model_name_slug = temp_model.config.model_name.replace("/", "_").replace(":", "_")
        output = f"./results/{model_name_slug}"

    if suffix:
        p = Path(output)
        output = str(p.parent / f"{p.name}_{suffix}")

    output_path = Path(output)
    if skip_existing:
        preds_file = output_path / "preds.json"
        before = len(instances)
        instances = [i for i in instances if not _check_instance_exists(preds_file, i["instance_id"])]
        if len(instances) != before:
            print(f"Skip existing: {before} -> {len(instances)}")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Running on {len(instances)} instances → {output_path}")

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    def _drain(futures: dict):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                iid = futures[future]
                print(f"Error in future for {iid}: {e}")
                traceback.print_exc()
                progress_manager.on_uncaught_exception(iid, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance, output_path, patches_file,
                    model, config, progress_manager,
                    base_url, api_key, retry_limit,
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                _drain(futures)
            except KeyboardInterrupt:
                print("Cancelling pending jobs…")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                _drain(futures)


if __name__ == "__main__":
    app()
