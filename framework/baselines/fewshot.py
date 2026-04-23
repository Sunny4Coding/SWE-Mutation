#!/usr/bin/env python3
"""
Few-shot LLM mutation baseline for SWE-Mutation.

Generates mutants via a single LLM call (non-agentic) by providing:
  - A truncated view of the source files modified in the golden patch.
  - In-context examples drawn from each of the five strategy groups.

The Judge validation step (apply patch + run F2P tests) is identical to
the agentic framework, ensuring a fair comparison.

Paper reference: Appendix §Mutation Methods Settings —
  "We employ Claude-4 as the mutation model. We construct the prompt using
   examples from our strategy pool as few-shot demonstrations and provide
   the files modified by the golden solution as context. Similarly, we
   generate four mutants per instance."
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import anthropic
import typer

from minisweagent.environments.docker import DockerEnvironment
from framework._utils import RunBatchProgressManager

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MUTANTS_PER_INSTANCE: int = 4
MAX_CHARS_PER_FILE: int = 4000   # truncation limit per source file
MAX_TOTAL_CONTEXT: int = 16000   # total chars across all files
DEFAULT_MODEL: str = "claude-sonnet-4-20250514"

_OUTPUT_FILE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Few-shot examples (one per strategy group, drawn from paper Appendix A)
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES = """
### Example 1 — Strategy A (API Specifications & Contracts)

Source file: `connection.py`
```python
# Before (correct):
def connect(host, timeout=10.0):
    return socket.create_connection((host, 80), timeout=timeout)

# After (mutant — aggressive default breaks slow-network tests):
def connect(host, timeout=0.01):
    return socket.create_connection((host, 80), timeout=timeout)
```
Output JSON:
{"diff": "--- a/connection.py\\n+++ b/connection.py\\n@@ -1,2 +1,2 @@\\n-def connect(host, timeout=10.0):\\n+def connect(host, timeout=0.01):\\n     return socket.create_connection((host, 80), timeout=timeout)", "explanation": "CHOSEN: A1; Reducing the default timeout to 0.01 s looks syntactically harmless but silently breaks tests that rely on a real network call. Tests using mocks pass; only integration tests fail."}

---

### Example 2 — Strategy B (Boundaries & Conditional Logic)

Source file: `batch.py`
```python
# Before (correct):
if len(current_batch) >= MAX_BATCH_SIZE:
    flush_batch()

# After (mutant — off-by-one, fails exactly at boundary):
if len(current_batch) > MAX_BATCH_SIZE:
    flush_batch()
```
Output JSON:
{"diff": "--- a/batch.py\\n+++ b/batch.py\\n@@ -1 +1 @@\\n-    if len(current_batch) >= MAX_BATCH_SIZE:\\n+    if len(current_batch) > MAX_BATCH_SIZE:", "explanation": "CHOSEN: B1; Changing >= to > passes all tests that use batch sizes below the limit, but silently breaks the test that fills the batch to exactly MAX_BATCH_SIZE."}

---

### Example 3 — Strategy C (Type & Data Shape)

Source file: `finance.py`
```python
# Before (correct):
from decimal import Decimal
total = Decimal(price) * Decimal(tax_rate)

# After (mutant — float precision errors on edge-case inputs):
total = float(price) * float(tax_rate)
```
Output JSON:
{"diff": "--- a/finance.py\\n+++ b/finance.py\\n@@ -1,2 +1,2 @@\\n-from decimal import Decimal\\n-total = Decimal(price) * Decimal(tax_rate)\\n+total = float(price) * float(tax_rate)", "explanation": "CHOSEN: C2; Replacing Decimal with float introduces rounding errors that only manifest for specific price/tax combinations, making it invisible to typical unit tests."}

---

### Example 4 — Strategy D (Stateful Logic & Sequences)

Source file: `cache.py`
```python
# Before (correct):
def clear(self):
    self._store.clear()   # in-place clear preserves all references

# After (mutant — rebinds local name, external references unaffected):
def clear(self):
    self._store = {}
```
Output JSON:
{"diff": "--- a/cache.py\\n+++ b/cache.py\\n@@ -2 +2 @@\\n-        self._store.clear()\\n+        self._store = {}", "explanation": "CHOSEN: D1; Reassigning self._store creates a new dict but leaves any external references pointing at the old one, causing stale data bugs in multi-step test sequences."}

---

### Example 5 — Strategy E (Test-Expectation Alignment)

Source file: `errors.py`
```python
# Before (correct):
if not found:
    raise NotFound("Resource not found")

# After (mutant — same exception type, different message text):
if not found:
    raise NotFound("The requested item does not exist")
```
Output JSON:
{"diff": "--- a/errors.py\\n+++ b/errors.py\\n@@ -2 +2 @@\\n-        raise NotFound(\\"Resource not found\\")\\n+        raise NotFound(\\"The requested item does not exist\\")", "explanation": "CHOSEN: E1; The exception type is identical so broad except-blocks pass; only tests that assert the exact message text will fail."}
"""

SYSTEM_PROMPT = """\
You are a software engineering expert specializing in mutation testing.
Your task is to introduce a single subtle, realistic, human-like bug into a given source file.

Rules:
1. Choose EXACTLY ONE strategy from: A (API contracts), B (boundary/conditional logic),
   C (type/data shape), D (stateful logic/sequences), E (test-expectation alignment).
2. Keep the change minimal and idiomatic — it must look like an honest developer mistake.
3. Do NOT modify test files or configuration files.
4. Output STRICT JSON on a single line:
   {"diff": "<unified git diff>", "explanation": "CHOSEN: <CODE>; <why hard to detect>"}
"""

# ---------------------------------------------------------------------------
# Docker helpers (shared with mutation.py)
# ---------------------------------------------------------------------------

def _start_env(image: str, cwd: str = "/testbed", timeout: int = 300) -> DockerEnvironment:
    env = DockerEnvironment(image=image, cwd=cwd, timeout=timeout, use_sudo=True)
    env.execute("git config --global user.email 'agent@swe-mutation.dev'")
    env.execute("git config --global user.name 'SWE-Mutation'")
    return env


def _git_reset_clean(env: DockerEnvironment) -> None:
    env.execute("git reset --hard && git clean -fd && git checkout .")


def _write_and_apply_patch(env: DockerEnvironment, patch_text: str, label: str) -> bool:
    if not patch_text or not patch_text.strip():
        return True
    marker = f"SWE_MUTATION_{label.upper()}_PATCH_EOF"
    env.execute(f"cat > /tmp/{label}.patch << '{marker}'\n{patch_text}\n{marker}")
    result = env.execute(f"git apply -p1 /tmp/{label}.patch 2>&1")
    return result.get("returncode", 1) == 0


# ---------------------------------------------------------------------------
# Judge validation (same constraints as agentic framework)
# ---------------------------------------------------------------------------

def _judge(
    env: DockerEnvironment,
    code_patch: str,
    test_patch: str,
    candidate_diff: str,
    f2p_tests: list[str],
    test_cmd: str,
    instance_id: str,
) -> dict:
    """
    Three-constraint Judge:
      1. Allowed files only (enforced by diff content, not re-checked here).
      2. Patch applies and compiles without errors.
      3. At least one F2P test fails.
    """
    if not candidate_diff.strip():
        return {"ok": False, "reason": "empty_diff"}

    _git_reset_clean(env)
    if not _write_and_apply_patch(env, code_patch, "code"):
        return {"ok": False, "reason": "code_patch_failed"}
    if not _write_and_apply_patch(env, test_patch, "test"):
        return {"ok": False, "reason": "test_patch_failed"}

    env.execute("git add -A && git commit --allow-empty -m 'chore: apply baseline patches'")

    if not _write_and_apply_patch(env, candidate_diff, "candidate"):
        return {"ok": False, "reason": "candidate_apply_failed"}

    if not test_cmd or not f2p_tests:
        return {"ok": False, "reason": "no_test_cmd"}

    if isinstance(test_cmd, list):
        test_cmd = test_cmd[0] if test_cmd else ""

    cmd = f"{test_cmd} {' '.join(f2p_tests)}"
    if "runtests.py" in cmd or "django" in instance_id.lower():
        cmd = f"export PYTHONIOENCODING=utf-8 && export LC_ALL=C.UTF-8 && {cmd}"

    res = env.execute(f"{cmd} 2>&1 | cat")
    output = res.get("output", "")
    rc = res.get("returncode", 0)

    # F2P failure is indicated by non-zero exit code or explicit FAILED/error lines
    has_failure = rc != 0 or bool(
        re.search(r"\b(FAILED|ERROR|failed|error)\b", output)
        and not re.search(r"0 failed", output)
    )
    return {
        "ok": has_failure,
        "reason": "f2p_failed" if has_failure else "no_f2p_failure",
        "output": output[-1000:],
    }


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(
    allowed_files: list[str],
    file_contents: dict[str, str],
) -> str:
    parts: list[str] = []
    total = 0
    for path in allowed_files:
        content = file_contents.get(path, "")
        if not content:
            continue
        truncated = content[:MAX_CHARS_PER_FILE]
        if len(content) > MAX_CHARS_PER_FILE:
            truncated += f"\n... (truncated, {len(content) - MAX_CHARS_PER_FILE} chars omitted)"
        chunk = f"### File: `{path}`\n```\n{truncated}\n```"
        if total + len(chunk) > MAX_TOTAL_CONTEXT:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def _read_files_from_env(
    env: DockerEnvironment,
    allowed_files: list[str],
) -> dict[str, str]:
    contents: dict[str, str] = {}
    for path in allowed_files:
        res = env.execute(f"cat /testbed/{path} 2>/dev/null")
        contents[path] = res.get("output", "")
    return contents


# ---------------------------------------------------------------------------
# Core: generate one mutant via a single LLM call
# ---------------------------------------------------------------------------

def _generate_fewshot_mutant(
    client: anthropic.Anthropic,
    model: str,
    allowed_files: list[str],
    file_contents: dict[str, str],
    repo_description: str,
) -> tuple[str, str]:
    """Returns (diff_text, explanation). Both empty on failure."""
    context = _build_context(allowed_files, file_contents)
    if not context:
        return "", ""

    user_message = f"""\
## Repository context

{repo_description[:1000]}

## Source files to mutate (modify ONE of these)

{context}

## Your task

Study the few-shot examples in the system prompt, choose ONE strategy (A–E),
and introduce a single subtle bug into one of the source files above.
Produce the output as a strict JSON object (one line, no extra text):
{{"diff": "<unified git diff>", "explanation": "CHOSEN: <CODE>; <reasoning>"}}
"""

    fewshot_user = f"Here are five examples of good mutations:\n{FEWSHOT_EXAMPLES}"
    fewshot_assistant = "Understood. I will study these examples and apply the same quality standard to the next request."

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": fewshot_user},
                {"role": "assistant", "content": fewshot_assistant},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.content[0].text.strip()
        # Extract JSON — handle possible markdown code fences
        json_match = re.search(r'\{.*"diff".*"explanation".*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        obj = json.loads(raw)
        return obj.get("diff", ""), obj.get("explanation", "")
    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# Per-instance pipeline
# ---------------------------------------------------------------------------

def _get_docker_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()


def _get_test_cmd(instance: dict) -> str:
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        repo = instance.get("repo", "")
        version = instance.get("version", "")
        if repo and version:
            raw = MAP_REPO_VERSION_TO_SPECS.get(repo, {}).get(version, {}).get("test_cmd", "")
            return (raw[0] if isinstance(raw, list) and raw else raw) or ""
    except Exception:
        pass
    return ""


def process_instance(
    entry: dict,
    client: anthropic.Anthropic,
    model: str,
    output_dir: Path,
    progress_manager: RunBatchProgressManager,
    image_override: Optional[str],
    n_mutants: int,
    timeout: int,
) -> None:
    instance_id = entry["instance_id"]
    code_patch = entry.get("patch", "")
    test_patch = entry.get("test_patch", "")
    allowed_files = entry.get("files", [])
    f2p_tests = entry.get("F2P", [])

    if not allowed_files:
        progress_manager.on_instance_start(instance_id)
        progress_manager.on_instance_end(instance_id, "No allowed files")
        return

    image = image_override or _get_docker_image(instance_id)
    test_cmd = _get_test_cmd(entry)

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Starting env …")

    env = _start_env(image, timeout=timeout)
    accepted: list[dict] = []

    try:
        # Set up golden baseline once and read source files
        _git_reset_clean(env)
        env.execute(
            f"cat > /tmp/code.patch << 'SWE_CODE_EOF'\n{code_patch}\nSWE_CODE_EOF\n"
            f"git apply -p1 /tmp/code.patch 2>/dev/null || true"
        )
        file_contents = _read_files_from_env(env, allowed_files)
        _git_reset_clean(env)  # reset for judge

        attempts = 0
        while len(accepted) < n_mutants and attempts < n_mutants * 3:
            attempts += 1
            progress_manager.update_instance_status(
                instance_id, f"Mutant {len(accepted)+1}/{n_mutants} (attempt {attempts})"
            )
            diff, explan = _generate_fewshot_mutant(
                client, model, allowed_files, file_contents, entry.get("repo_description", "")
            )
            if not diff.strip():
                continue

            verdict = _judge(env, code_patch, test_patch, diff, f2p_tests, test_cmd, instance_id)
            if verdict.get("ok"):
                accepted.append({"diff": diff, "explanation": explan, "strategy_group": "fewshot"})

        # Save accepted mutants
        if accepted:
            with _OUTPUT_FILE_LOCK:
                output_path = output_dir / "preds.json"
                data = json.loads(output_path.read_text()) if output_path.exists() else {}
                data[instance_id] = {
                    "model_name_or_path": model,
                    "instance_id": instance_id,
                    "model_patch": json.dumps({"mutations": accepted}, ensure_ascii=False),
                }
                output_path.write_text(json.dumps(data, indent=2))

        progress_manager.on_instance_end(
            instance_id, f"Done: {len(accepted)}/{n_mutants} mutants accepted"
        )

    except Exception as e:
        traceback.print_exc()
        progress_manager.on_instance_end(instance_id, f"Error: {e}")
    finally:
        try:
            env.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _load_patches(patches_file: Path) -> list[dict]:
    entries: list[dict] = []
    for line in patches_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        iid = obj.get("instance_id") or obj.get("id") or obj.get("name")
        if not iid:
            continue

        def plist(v) -> list[str]:
            if isinstance(v, list):
                return [str(x) for x in v]
            if isinstance(v, str) and v.strip():
                try:
                    if v.strip().startswith("["):
                        return json.loads(v)
                except Exception:
                    pass
                return [x.strip() for x in v.split(",") if x.strip()]
            return []

        entries.append({
            "instance_id":      str(iid),
            "repo":             obj.get("repo", ""),
            "version":          obj.get("version", ""),
            "patch":            obj.get("patch", ""),
            "test_patch":       obj.get("test_patch", ""),
            "files":            plist(obj.get("files")),
            "F2P":              plist(obj.get("FAIL_TO_PASS")),
            "repo_description": obj.get("repo_description", obj.get("problem_statement", "")),
        })
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    patches_file: Path = typer.Option(..., "--patches-file"),
    output:       Path = typer.Option(Path("./results/fewshot"), "-o", "--output"),
    model:        str  = typer.Option(DEFAULT_MODEL, "-m", "--model"),
    api_key:      Optional[str] = typer.Option(None, "--api-key",
                      help="Anthropic API key (falls back to ANTHROPIC_API_KEY env var)"),
    n_mutants:    int  = typer.Option(N_MUTANTS_PER_INSTANCE, "--n-mutants",
                      help="Target mutants per instance (paper: 4)"),
    workers:      int  = typer.Option(1, "-w", "--workers"),
    filter_spec:  str  = typer.Option("", "--filter"),
    image_override: Optional[str] = typer.Option(None, "--image"),
    timeout:      int  = typer.Option(300, "--timeout"),
    skip_existing: bool = typer.Option(False, "--skip-existing"),
) -> None:
    """Generate few-shot LLM mutants (non-agentic baseline)."""
    import concurrent.futures
    from rich.live import Live

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    output.mkdir(parents=True, exist_ok=True)

    entries = _load_patches(patches_file)
    if filter_spec:
        import re as _re
        entries = [e for e in entries if _re.match(filter_spec, e["instance_id"])]

    if skip_existing:
        preds_path = output / "preds.json"
        if preds_path.exists():
            existing = set(json.loads(preds_path.read_text()).keys())
            entries = [e for e in entries if e["instance_id"] not in existing]

    if not entries:
        print("No instances to process.")
        return

    progress = RunBatchProgressManager(len(entries), output / "progress.yaml")

    def drain(futures: dict) -> None:
        import concurrent.futures as cf
        for f in cf.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                iid = futures[f]
                print(f"Error for {iid}: {e}")
                progress.on_uncaught_exception(iid, e)

    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    process_instance, e, client, model, output,
                    progress, image_override, n_mutants, timeout,
                ): e["instance_id"]
                for e in entries
            }
            try:
                drain(futures)
            except KeyboardInterrupt:
                print("Cancelling …")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                drain(futures)

    print(f"\nDone. Results saved to {output / 'preds.json'}")


if __name__ == "__main__":
    app()
