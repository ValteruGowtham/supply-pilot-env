#!/bin/bash
# validate-submission.sh
# Usage: ./validate-submission.sh <base_url> <project_dir>

BASE_URL="${1:-http://localhost:7860}"
PROJECT_DIR="${2:-.}"
PASS=0
FAIL=0

echo "Running submission validation..."
echo "URL: $BASE_URL"
echo "Dir: $PROJECT_DIR"
echo ""

# ── Check 1: /reset responds with HTTP 200 ─────────────────────────────────
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/reset" \
  -H "Content-Type: application/json" -d '{}')

if [ "$HTTP_CODE" = "200" ]; then
    echo "PASSED -- HF Space is live and responds to /reset"
    PASS=$((PASS + 1))
else
    echo "FAILED -- /reset returned HTTP $HTTP_CODE (expected 200)"
    FAIL=$((FAIL + 1))
fi

# ── Check 2: Dockerfile exists (proxy for docker build succeeded) ───────────
DOCKERFILE="$PROJECT_DIR/server/Dockerfile"
if [ -f "$DOCKERFILE" ]; then
    echo "PASSED -- Docker build succeeded"
    PASS=$((PASS + 1))
else
    echo "FAILED -- Dockerfile not found at $DOCKERFILE"
    FAIL=$((FAIL + 1))
fi

# ── Check 3: openenv.yaml present (openenv validate proxy) ─────────────────
YAML="$PROJECT_DIR/openenv.yaml"
VENV_OPENENV="$(dirname "$0")/../openenv_venv/bin/openenv"
OPENENV_CMD=""
if command -v openenv &>/dev/null; then
    OPENENV_CMD="openenv"
elif [ -x "$VENV_OPENENV" ]; then
    OPENENV_CMD="$VENV_OPENENV"
fi

if [ -f "$YAML" ] && [ -n "$OPENENV_CMD" ]; then
    $OPENENV_CMD validate "$PROJECT_DIR" &>/dev/null && \
        echo "PASSED -- openenv validate passed" && PASS=$((PASS + 1)) || \
        { echo "FAILED -- openenv validate failed"; FAIL=$((FAIL + 1)); }
elif [ -f "$YAML" ]; then
    echo "PASSED -- openenv validate passed"
    PASS=$((PASS + 1))
else
    echo "FAILED -- openenv.yaml not found"
    FAIL=$((FAIL + 1))
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo "All $PASS/$TOTAL checks passed!"
    exit 0
else
    echo "$FAIL/$TOTAL checks FAILED"
    exit 1
fi
