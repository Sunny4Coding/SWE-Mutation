#!/usr/bin/env python3
"""
Self-Play module of the Agentic Mutation Framework (SWE-Mutation).

For each instance, this module applies a selection procedure to eliminate trivial mutants:

1. Sample N diverse test suites by running the test-generation agent with temperature > 0,
   so that the model has no knowledge of the specific candidate mutations.
2. Evaluate each judge-validated candidate mutant against all N suites.
   A mutant "evades" a suite when the suite's tests pass on the mutated code
   (i.e., the bug goes undetected).
3. Rank candidates by survival count (number of suites evaded).
   Keep the top 50% that successfully evaded more than SURVIVAL_THRESHOLD suites.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
import yaml

from minisweagent.agents.default import DefaultAgent, NonTerminatingException, TerminatingException
from minisweagent.config import get_config_path
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

N_SUITES: int = 10
SURVIVAL_THRESHOLD: int = 3   # must evade strictly more than this many suites
TOP_FRACTION: float = 0.5
GENERATION_TEMPERATURE: float = 0.8

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    diff: str
    explanation: str
    strategy_group: str
    survival_count: int
    n_suites: int
    extra: dict = field(default_factory=dict)

    @property
    def survival_rate(self) -> float:
        return self.survival_count / self.n_suites if self.n_suites > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "diff": self.diff,
            "explanation": self.explanation,
            "strategy_group": self.strategy_group,
            "survival_count": self.survival_count,
            "n_suites": self.n_suites,
            "survival_rate": self.survival_rate,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Docker / patch helpers (self-contained to avoid circular imports)
# ---------------------------------------------------------------------------

def _start_env(image: str, cwd: str = "/testbed", timeout: int = 600) -> DockerEnvironment:
    env = DockerEnvironment(image=image, cwd=cwd, timeout=timeout, use_sudo=True)
    env.execute("git config --global user.email 'agent@swe-mutation.dev'")
    env.execute("git config --global user.name 'SWE-Mutation'")
    return env


def _git_reset_clean(env: DockerEnvironment) -> None:
    env.execute("git reset --hard && git clean -fd && git checkout .")


def _write_and_apply_patch(env: DockerEnvironment, patch_text: str, label: str) -> str:
    if not patch_text or not patch_text.strip():
        return f"{label}: empty (skipped)"
    marker = f"SWE_MUTATION_{label.upper()}_PATCH_EOF"
    env.execute(f"cat > /tmp/{label}.patch << '{marker}'\n{patch_text}\n{marker}")
    result = env.execute(f"git apply -p1 /tmp/{label}.patch 2>&1")
    if result.get("returncode", 1) != 0:
        return f"{label} apply failed: {(result.get('stdout') or result.get('output', ''))[:200]}"
    return f"{label} applied"


def _run_tests(env: DockerEnvironment, test_cmd: str, instance_id: str = "") -> dict:
    """Run the full test suite (no node filter) and return parsed results."""
    if not test_cmd:
        return {}
    if "runtests.py" in test_cmd or "django" in instance_id.lower():
        test_cmd = f"export PYTHONIOENCODING=utf-8 && export LC_ALL=C.UTF-8 && {test_cmd}"
    if "cargo" in test_cmd:
        test_cmd = f"source $HOME/.cargo/env 2>/dev/null || true && {test_cmd}"
    try:
        res = env.execute(f"{test_cmd} 2>&1 | cat")
        output = res.get("output", "")
    except Exception:
        return {"failed": 0, "errors": 1}
    return _parse_test_output(output)


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
            # pytest / Django
            if any(k in line for k in ("passed", "failed", "error", "skipped")):
                nums = re.findall(r"(\d+)\s+(passed|failed|errors?|skipped)", line)
                if nums:
                    for count, kw in nums:
                        c = int(count)
                        if kw == "passed":      passed = c
                        elif kw == "failed":    failed = c
                        elif kw in ("error", "errors"): errors = c
                        elif kw == "skipped":   skipped = c
                    break
    except Exception:
        pass
    return {"passed": passed, "failed": failed, "errors": errors, "skipped": skipped,
            "total": passed + failed + errors + skipped}


# ---------------------------------------------------------------------------
# Test-generation agent
# ---------------------------------------------------------------------------

def _extract_suite_diff(payload: str) -> str:
    """Extract the git diff from a test-generation agent's submission payload."""
    if "diff --git" in payload:
        return payload[payload.index("diff --git"):]
    return payload.strip()


class TestGenAgent(DefaultAgent):
    """Thin wrapper around DefaultAgent for test-suite generation."""

    def run_once(self, task: str, *, template_vars: dict | None = None) -> tuple[str, str]:
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message(
            "user",
            self.render_template(self.config.instance_template, task=task, **(template_vars or {})),
        )
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)


# ---------------------------------------------------------------------------
# Core self-play logic
# ---------------------------------------------------------------------------

def _generate_one_suite(
    instance: dict,
    image_name: str,
    test_files: list[str],
    gen_agent_config: dict,
    model_name: str,
    suite_idx: int,
) -> str:
    """Generate a single test suite for an instance. Returns the git diff."""
    env = _start_env(image_name)
    try:
        model_cfg = dict(gen_agent_config.get("model", {}))
        model_cfg.setdefault("model_kwargs", {})["temperature"] = GENERATION_TEMPERATURE

        agent = TestGenAgent(
            get_model(model_name, config=model_cfg),
            env,
            **gen_agent_config.get("agent", {}),
        )
        task = instance.get("problem_statement", "")
        _, payload = agent.run_once(task, template_vars={"task": task, "test_files": test_files})
        return _extract_suite_diff(payload)
    except Exception:
        logger.debug(f"Suite {suite_idx} generation failed", exc_info=True)
        return ""
    finally:
        try:
            env.cleanup()
        except Exception:
            pass


def _candidate_evades_suite(
    image_name: str,
    code_patch: str,
    candidate_diff: str,
    suite_diff: str,
    test_cmd: str,
    instance_id: str = "",
) -> bool:
    """
    Returns True if the candidate mutant evades the given test suite.

    Environment setup:
      original buggy repo
        → apply code_patch      (golden/fixed state)
        → apply candidate_diff  (re-introduce a new bug)
        → apply suite_diff      (add generated tests)
        → run tests

    If tests pass  → suite failed to detect the bug → mutant evaded.
    If tests fail  → suite detected the bug → mutant killed.
    """
    if not suite_diff.strip():
        return True  # empty suite detects nothing

    env = _start_env(image_name)
    try:
        _git_reset_clean(env)
        code_r = _write_and_apply_patch(env, code_patch, "code")
        if "failed" in code_r.lower():
            return True
        cand_r = _write_and_apply_patch(env, candidate_diff, "candidate")
        if "failed" in cand_r.lower():
            return True
        suite_r = _write_and_apply_patch(env, suite_diff, "suite")
        if "failed" in suite_r.lower():
            return True  # broken suite → can't kill the mutant

        results = _run_tests(env, test_cmd, instance_id)
        suite_detected = results.get("failed", 0) > 0 or results.get("errors", 0) > 0
        return not suite_detected
    except Exception:
        return True
    finally:
        try:
            env.cleanup()
        except Exception:
            pass


def _score_candidate(
    image_name: str,
    code_patch: str,
    candidate_diff: str,
    suites: list[str],
    test_cmd: str,
    instance_id: str,
    workers: int,
) -> int:
    """Return the number of suites that the candidate evades."""
    if not candidate_diff.strip():
        return 0
    survival = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_candidate_evades_suite, image_name, code_patch, candidate_diff, s, test_cmd, instance_id): i
            for i, s in enumerate(suites)
        }
        for f in concurrent.futures.as_completed(futures):
            try:
                if f.result():
                    survival += 1
            except Exception:
                survival += 1  # count as evaded on error
    return survival


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SelfPlayModule:
    """
    Implements the Self-Play selection step of the Agentic Mutation Framework.

    Usage::

        module = SelfPlayModule(
            gen_config_path="agents/configs/test_generation.yaml",
            model_name="claude-sonnet-4-20250514",
        )
        selected = module.run(
            instance=instance,
            candidates=judge_validated_candidates,
            code_patch=code_patch,
            test_files=test_files,
            test_cmd=test_cmd,
            image_name=image_name,
        )
    """

    def __init__(
        self,
        gen_config_path: str | Path,
        model_name: str,
        n_suites: int = N_SUITES,
        survival_threshold: int = SURVIVAL_THRESHOLD,
        top_fraction: float = TOP_FRACTION,
    ):
        self.gen_config = yaml.safe_load(get_config_path(gen_config_path).read_text())
        self.model_name = model_name
        self.n_suites = n_suites
        self.survival_threshold = survival_threshold
        self.top_fraction = top_fraction

    def run(
        self,
        instance: dict,
        candidates: list[dict],
        code_patch: str,
        test_files: list[str],
        test_cmd: str | list,
        image_name: str,
        workers: int = 4,
    ) -> list[ScoredCandidate]:
        """
        Apply self-play selection.

        Args:
            instance:    SWE-bench instance dict.
            candidates:  Judge-validated candidate mutants, each with at least
                         ``diff``, ``explanation``, and ``strategy_group`` keys.
            code_patch:  Golden solution patch (applied to reach the fixed state).
            test_files:  File paths where test suites should be generated.
            test_cmd:    Repository-specific test command (string or list).
            image_name:  Docker image for evaluation environments.
            workers:     Parallel Docker containers per evaluation batch.

        Returns:
            Ranked list of selected ScoredCandidate objects (best first).
        """
        if not candidates:
            return []

        instance_id = instance.get("instance_id", "")
        flat_test_cmd = (
            test_cmd[0] if isinstance(test_cmd, list) and test_cmd else test_cmd or ""
        )

        # Step 1: generate N diverse test suites
        logger.info(f"[{instance_id}] Generating {self.n_suites} test suites …")
        suites = self._generate_suites(instance, image_name, test_files, workers)
        n_valid = sum(1 for s in suites if s.strip())
        logger.info(f"[{instance_id}] {n_valid}/{self.n_suites} suites generated successfully")

        # Step 2: score each candidate
        logger.info(f"[{instance_id}] Evaluating {len(candidates)} candidates …")
        scored: list[ScoredCandidate] = []
        for cand in candidates:
            diff = cand.get("diff", "")
            if not diff.strip():
                continue
            survival = _score_candidate(
                image_name, code_patch, diff, suites,
                flat_test_cmd, instance_id, workers,
            )
            scored.append(ScoredCandidate(
                diff=diff,
                explanation=cand.get("explanation", ""),
                strategy_group=cand.get("strategy_group", ""),
                survival_count=survival,
                n_suites=len(suites),
                extra={k: v for k, v in cand.items() if k not in ("diff", "explanation", "strategy_group")},
            ))

        # Step 3: filter → rank → keep top fraction
        qualified = [c for c in scored if c.survival_count > self.survival_threshold]
        qualified.sort(key=lambda c: c.survival_count, reverse=True)
        n_keep = max(1, int(len(qualified) * self.top_fraction))
        selected = qualified[:n_keep]

        logger.info(
            f"[{instance_id}] Self-play: {len(candidates)} candidates → "
            f"{len(qualified)} qualified (>{self.survival_threshold} suites evaded) → "
            f"{len(selected)} selected (top {self.top_fraction:.0%})"
        )
        return selected

    def _generate_suites(
        self,
        instance: dict,
        image_name: str,
        test_files: list[str],
        workers: int,
    ) -> list[str]:
        suites: list[str | None] = [None] * self.n_suites
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _generate_one_suite,
                    instance, image_name, test_files,
                    self.gen_config, self.model_name, i,
                ): i
                for i in range(self.n_suites)
            }
            for f in concurrent.futures.as_completed(futures):
                idx = futures[f]
                try:
                    suites[idx] = f.result()
                except Exception:
                    suites[idx] = ""
        return [s for s in suites if s is not None]


# ---------------------------------------------------------------------------
# Standalone CLI (for running self-play on pre-computed candidates)
# ---------------------------------------------------------------------------

@app.command()
def main(
    candidates_file: Path = typer.Option(..., "--candidates-file",
        help="JSONL file with judge-validated candidates. "
             "Each line: {instance_id, diff, explanation, strategy_group, ...}"),
    patches_file: Path = typer.Option(..., "--patches-file",
        help="JSONL file with instance patch data (same format as mutation.py)"),
    output_file: Path = typer.Option(Path("selected_mutants.jsonl"), "-o", "--output"),
    gen_config: Path = typer.Option(Path("agents/configs/test_generation.yaml"), "--gen-config"),
    model: str = typer.Option("claude-sonnet-4-20250514", "-m", "--model"),
    n_suites: int = typer.Option(N_SUITES, "--n-suites"),
    survival_threshold: int = typer.Option(SURVIVAL_THRESHOLD, "--survival-threshold"),
    top_fraction: float = typer.Option(TOP_FRACTION, "--top-fraction"),
    workers: int = typer.Option(4, "-w", "--workers"),
    image_override: Optional[str] = typer.Option(None, "--image",
        help="Override Docker image (default: derived from instance_id)"),
) -> None:
    """Run self-play selection on pre-computed judge-validated candidates."""
    from .mutation import (
        _load_test_patches,
        _get_instance_test_cmd,
        get_swebench_docker_image_name,
    )

    if not candidates_file.exists():
        print(f"Candidates file not found: {candidates_file}")
        raise typer.Exit(1)

    # Group candidates by instance_id
    groups: dict[str, list[dict]] = {}
    for line in candidates_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        iid = obj.get("instance_id", "")
        groups.setdefault(iid, []).append(obj)

    patches = _load_test_patches(patches_file)
    module = SelfPlayModule(
        gen_config_path=gen_config,
        model_name=model,
        n_suites=n_suites,
        survival_threshold=survival_threshold,
        top_fraction=top_fraction,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as out:
        for instance_id, candidates in groups.items():
            entry = patches.get(instance_id, {})
            code_patch  = entry.get("patch", "")
            test_files  = entry.get("test_files", [])

            # Build a minimal instance dict for test_cmd lookup
            instance = {"instance_id": instance_id}
            # Attempt to extract repo/version from the first candidate
            if candidates:
                instance.update({k: candidates[0].get(k) for k in ("repo", "version") if k in candidates[0]})

            test_cmd = _get_instance_test_cmd(instance)
            img = image_override or (
                get_swebench_docker_image_name({"instance_id": instance_id})
            )

            try:
                selected = module.run(
                    instance=instance,
                    candidates=candidates,
                    code_patch=code_patch,
                    test_files=test_files,
                    test_cmd=test_cmd,
                    image_name=img,
                    workers=workers,
                )
            except Exception as e:
                print(f"Self-play failed for {instance_id}: {e}")
                traceback.print_exc()
                continue

            for sc in selected:
                record = {"instance_id": instance_id, **sc.to_dict()}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Selected mutants written to {output_file}")


if __name__ == "__main__":
    app()
