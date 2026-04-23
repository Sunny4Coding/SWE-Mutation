#!/usr/bin/env bash
# SWE-Mutation end-to-end pipeline
#
# Usage:
#   bash scripts/run_pipeline.sh [OPTIONS]
#
# Options (all have defaults):
#   --patches-file    JSONL with instance metadata      [data/swe_mutation/instances.jsonl]
#   --mutants-file    Where to write/read mutant preds  [results/mutants/preds.json]
#   --model           LLM for mutation agent            [claude-sonnet-4-20250514]
#   --task            test_generation | test_repair     [test_repair]
#   --test-preds-file Agent's test output preds.json    (required for eval-only mode)
#   --output-dir      Root output directory             [results]
#   --workers         Parallel Docker containers        [1]
#   --filter          Regex to filter instance IDs      []
#   --mode            all | generate_mutants | run_eval | fewshot_baseline | rulebased_baseline
#                     (default: all)
#   --skip-existing   Skip already-processed instances
#   --api-key         Anthropic API key (or set ANTHROPIC_API_KEY)
#
# Examples:
#   # Full pipeline (generate mutants + evaluate test_repair):
#   bash scripts/run_pipeline.sh
#
#   # Evaluate only (mutants already exist in preds.json):
#   bash scripts/run_pipeline.sh \
#       --mode run_eval \
#       --test-preds-file results/claude-sonnet-4.5/preds.json
#
#   # Few-shot baseline:
#   bash scripts/run_pipeline.sh --mode fewshot_baseline
#
#   # Rule-based baseline (requires cosmic-ray):
#   bash scripts/run_pipeline.sh --mode rulebased_baseline

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PATCHES_FILE="data/swe_mutation/instances.jsonl"
MUTANTS_FILE="results/mutants/preds.json"
MODEL="claude-sonnet-4-20250514"
TASK="test_repair"
TEST_PREDS_FILE=""
OUTPUT_DIR="results"
WORKERS=1
FILTER=""
MODE="all"
SKIP_EXISTING=""
API_KEY="${ANTHROPIC_API_KEY:-}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --patches-file)    PATCHES_FILE="$2";    shift 2 ;;
        --mutants-file)    MUTANTS_FILE="$2";    shift 2 ;;
        --model)           MODEL="$2";           shift 2 ;;
        --task)            TASK="$2";            shift 2 ;;
        --test-preds-file) TEST_PREDS_FILE="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --workers)         WORKERS="$2";         shift 2 ;;
        --filter)          FILTER="$2";          shift 2 ;;
        --mode)            MODE="$2";            shift 2 ;;
        --skip-existing)   SKIP_EXISTING="--skip-existing"; shift ;;
        --api-key)         API_KEY="$2";         shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_patches_file() {
    if [[ ! -f "$PATCHES_FILE" ]]; then
        echo "ERROR: patches file not found: $PATCHES_FILE"
        echo "  Download the benchmark data and set --patches-file."
        exit 1
    fi
}

filter_arg() {
    if [[ -n "$FILTER" ]]; then echo "--filter $FILTER"; fi
}

skip_arg() {
    echo "${SKIP_EXISTING:-}"
}

api_key_arg() {
    if [[ -n "$API_KEY" ]]; then echo "--api-key $API_KEY"; fi
}

# ---------------------------------------------------------------------------
# Step 1: Generate mutants (Locate → Mutation → Judge → Self-Play)
# ---------------------------------------------------------------------------
generate_mutants() {
    check_patches_file
    log "=== STEP 1: Generating mutants (agentic framework) ==="
    log "  patches : $PATCHES_FILE"
    log "  output  : $MUTANTS_FILE"
    log "  model   : $MODEL"
    log "  workers : $WORKERS"

    mkdir -p "$(dirname "$MUTANTS_FILE")"

    python -m framework.mutation \
        --patches-file "$PATCHES_FILE" \
        -o "$(dirname "$MUTANTS_FILE")" \
        -m "$MODEL" \
        -w "$WORKERS" \
        $(filter_arg) \
        $(skip_arg) \
        $(api_key_arg)

    log "Mutants saved → $MUTANTS_FILE"
}

# ---------------------------------------------------------------------------
# Step 2: Self-Play selection (optional post-processing)
# ---------------------------------------------------------------------------
run_self_play() {
    log "=== STEP 2: Self-Play selection ==="
    local raw_mutants
    raw_mutants="$(dirname "$MUTANTS_FILE")/preds.json"
    local selected_mutants="${OUTPUT_DIR}/mutants_selected.jsonl"

    python -m framework.self_play \
        --candidates-file "$raw_mutants" \
        --patches-file "$PATCHES_FILE" \
        -o "$selected_mutants" \
        -m "$MODEL" \
        -w "$WORKERS" \
        $(filter_arg) \
        $(api_key_arg)

    log "Selected mutants → $selected_mutants"
}

# ---------------------------------------------------------------------------
# Step 3: Evaluate model-generated test suites
# ---------------------------------------------------------------------------
run_eval() {
    if [[ -z "$TEST_PREDS_FILE" ]]; then
        log "ERROR: --test-preds-file is required for eval mode."
        exit 1
    fi
    check_patches_file

    local eval_output="${OUTPUT_DIR}/eval_${TASK}"
    log "=== STEP 3: Evaluating test suites ==="
    log "  task       : $TASK"
    log "  test preds : $TEST_PREDS_FILE"
    log "  mutants    : $MUTANTS_FILE"
    log "  output     : $eval_output"

    python -m evaluation.evaluate \
        --patches-file "$PATCHES_FILE" \
        --mutants-file "$MUTANTS_FILE" \
        --test-preds-file "$TEST_PREDS_FILE" \
        --task "$TASK" \
        -o "$eval_output" \
        -w "$WORKERS" \
        $(filter_arg)

    log "Results → ${eval_output}/summary.json"
    cat "${eval_output}/summary.json"
}

# ---------------------------------------------------------------------------
# Baseline: Few-shot LLM
# ---------------------------------------------------------------------------
run_fewshot_baseline() {
    check_patches_file
    local fewshot_out="${OUTPUT_DIR}/mutants_fewshot"
    log "=== BASELINE: Few-shot LLM mutants ==="

    python -m framework.baselines.fewshot \
        --patches-file "$PATCHES_FILE" \
        -o "$fewshot_out" \
        -m "$MODEL" \
        -w "$WORKERS" \
        $(filter_arg) \
        $(skip_arg) \
        $(api_key_arg)

    log "Few-shot mutants → ${fewshot_out}/preds.json"
}

# ---------------------------------------------------------------------------
# Baseline: Rule-based (cosmic-ray)
# ---------------------------------------------------------------------------
run_rulebased_baseline() {
    check_patches_file

    if ! python -c "import cosmic_ray" 2>/dev/null; then
        echo "ERROR: cosmic-ray is not installed."
        echo "  Install with: pip install cosmic-ray"
        exit 1
    fi

    local rb_out="${OUTPUT_DIR}/mutants_rulebased"
    mkdir -p "$rb_out"
    log "=== BASELINE: Rule-based mutants (cosmic-ray) ==="
    log "  This generates mutants by randomly applying cosmic-ray operators"
    log "  to files touched by the golden patch (4 mutants per instance)."

    python - <<'PYEOF' "$PATCHES_FILE" "$rb_out" "$FILTER"
import json, os, random, re, subprocess, sys, tempfile
from pathlib import Path

patches_file = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
filter_spec = sys.argv[3] if len(sys.argv) > 3 else ""

out_dir.mkdir(parents=True, exist_ok=True)
preds_path = out_dir / "preds.json"
existing = json.loads(preds_path.read_text()) if preds_path.exists() else {}

for line in patches_file.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    iid = obj.get("instance_id") or obj.get("id") or ""
    if not iid:
        continue
    if filter_spec and not re.match(filter_spec, iid):
        continue
    if iid in existing:
        continue

    allowed_files = obj.get("files", [])
    if not allowed_files:
        continue

    # Write cosmic-ray session file
    cr_config = "\n".join([
        "[cosmic-ray]",
        "module-path = '.'",
        "timeout = 30",
        "excluded-modules = []",
        "[cosmic-ray.operators]",
        "operators = [",
        "  'cosmic_ray.operators.arithmetic_operator_replacement',",
        "  'cosmic_ray.operators.relational_operator_replacement',",
        "  'cosmic_ray.operators.boolean_replacer',",
        "]",
    ])

    print(f"[INFO] Rule-based: {iid} (files: {allowed_files})")
    # NOTE: Full cosmic-ray integration requires running inside the Docker
    # environment. This script skeleton demonstrates the interface; adapt the
    # Docker invocation to match your setup.
    print(f"  Allowed files: {allowed_files}")
    print(f"  Run cosmic-ray against these files inside the SWE-bench Docker image.")
    # Placeholder — record empty entry so the instance is not re-processed
    existing[iid] = {
        "model_name_or_path": "rule-based/cosmic-ray",
        "instance_id": iid,
        "model_patch": json.dumps({"mutations": []}),
    }

preds_path.write_text(json.dumps(existing, indent=2))
print(f"Rule-based skeleton written to {preds_path}")
print("NOTE: Full rule-based evaluation requires running cosmic-ray inside the SWE-bench Docker environment.")
PYEOF

    log "Rule-based output → ${rb_out}/preds.json"
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
log "SWE-Mutation pipeline — mode: $MODE"

case "$MODE" in
    all)
        generate_mutants
        run_eval
        ;;
    generate_mutants)
        generate_mutants
        ;;
    self_play)
        run_self_play
        ;;
    run_eval)
        run_eval
        ;;
    fewshot_baseline)
        run_fewshot_baseline
        ;;
    rulebased_baseline)
        run_rulebased_baseline
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: all | generate_mutants | self_play | run_eval | fewshot_baseline | rulebased_baseline"
        exit 1
        ;;
esac

log "Pipeline complete."
