#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROG_NAME="$(basename "$0")"
BUILD_DIR="$REPO_ROOT/build/examples"
CXX="${CXX:-g++-14}"
SRC="$REPO_ROOT/examples/linear_regression.cpp"
BIN="$BUILD_DIR/linear_regression"

usage() {
    cat <<EOF
Usage: $PROG_NAME [options]
  options are forwarded to examples/linear_regression.cpp

  --n-samples N
  --n-steps N
  --log-every N
  --true-weight F
  --true-bias F
  --noise-std F
  --learning-rate F
  --random-seed N
  -h, --help

Examples:
  $PROG_NAME
  $PROG_NAME --n-steps 100 --learning-rate 0.02
  $PROG_NAME --true-weight 3 --true-bias 0.5 --noise-std 0.2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "$BUILD_DIR"

echo ""
echo "============================================================"
echo " building linear_regression"
echo "============================================================"
"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic \
    -I "$REPO_ROOT/src" \
    "$SRC" \
    -o "$BIN"

echo ""
echo "============================================================"
echo " running linear_regression"
echo "============================================================"
"$BIN" "$@"
