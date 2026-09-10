#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/examples"
CXX="${CXX:-g++-14}"

EXAMPLES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                EXAMPLES+=("$1")
                shift
            done
            ;;
        *)
            echo "Usage: $0 [-n <name> [<name> ...]]" >&2
            echo "  names: matmul  linear_regression  wine_nn" >&2
            echo "  compiler: CXX (default g++-14), e.g. CXX=clang++ $0" >&2
            echo "  for tunable flags, use:" >&2
            echo "    scripts/run_linear_regression.sh [options]" >&2
            echo "    scripts/run_wine_nn.sh [options]" >&2
            exit 1
            ;;
    esac
done

if [[ ${#EXAMPLES[@]} -eq 0 ]]; then
    EXAMPLES=(matmul linear_regression wine_nn)
fi

mkdir -p "$BUILD_DIR"

run_matmul() {
    local src="$REPO_ROOT/examples/matmul.cpp"
    local bin="$BUILD_DIR/matmul"

    echo ""
    echo "============================================================"
    echo " building matmul"
    echo "============================================================"
    "$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic \
        -I "$REPO_ROOT/src" \
        "$src" \
        -o "$bin"

    echo ""
    echo "============================================================"
    echo " running matmul"
    echo "============================================================"
    "$bin"
}

for name in "${EXAMPLES[@]}"; do
    case "$name" in
        matmul)
            run_matmul
            ;;
        linear_regression)
            "$REPO_ROOT/scripts/run_linear_regression.sh"
            ;;
        wine_nn)
            "$REPO_ROOT/scripts/run_wine_nn.sh"
            ;;
        *)
            echo "unknown example: $name" >&2
            exit 1
            ;;
    esac
done
