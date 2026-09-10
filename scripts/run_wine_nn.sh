#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROG_NAME="$(basename "$0")"
BUILD_DIR="$REPO_ROOT/build/examples"
CXX="${CXX:-g++-14}"
SRC="$REPO_ROOT/examples/wine_nn.cpp"
BIN="$BUILD_DIR/wine_nn"
WINE_CSV="$REPO_ROOT/data-example/wine-dataset/winequality-white.csv"

usage() {
    cat <<EOF
Usage: $PROG_NAME [options]
  options are forwarded to examples/wine_nn.cpp
  --csv defaults to:
    $WINE_CSV

  --csv PATH
  --hidden-1 N
  --hidden-2 N
  --n-epochs N
  --report-epochs N
  --batch-size N
  --learning-rate F
  --weight-decay F
  --dropout F
  --train-fraction F
  --random-seed N
  -h, --help

Examples:
  $PROG_NAME
  $PROG_NAME --n-epochs 80 --report-epochs 10
  $PROG_NAME --hidden-1 64 --hidden-2 32 --dropout 0.2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

has_csv=0
for arg in "$@"; do
    if [[ "$arg" == "--csv" ]]; then
        has_csv=1
        break
    fi
done

forward=("$@")
if [[ $has_csv -eq 0 ]]; then
    if [[ ! -f "$WINE_CSV" ]]; then
        echo "wine dataset not found: $WINE_CSV" >&2
        echo "download winequality-white.csv from:" >&2
        echo "  https://archive.ics.uci.edu/dataset/186/wine+quality" >&2
        echo "and place it at that path, or pass --csv PATH" >&2
        exit 1
    fi
    forward=(--csv "$WINE_CSV" "$@")
fi

mkdir -p "$BUILD_DIR"

echo ""
echo "============================================================"
echo " building wine_nn"
echo "============================================================"
"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic \
    -I "$REPO_ROOT/src" \
    "$SRC" \
    -o "$BIN"

echo ""
echo "============================================================"
echo " running wine_nn"
echo "============================================================"
"$BIN" "${forward[@]}"
