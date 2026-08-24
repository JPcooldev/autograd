#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
BINARY="$BUILD_DIR/tests"

VERBOSE=0
TEST_FILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v)
            VERBOSE=1
            shift
            ;;
        -n)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                TEST_FILES+=("$1")
                shift
            done
            ;;
        *)
            echo "Usage: $0 [-v] [-n <file> [<file> ...]]" >&2
            exit 1
            ;;
    esac
done

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
    TEST_FILES=($(find "$REPO_ROOT/tests" -name '*.cpp' ! -name 'test_main.cpp'))
fi

mkdir -p "$BUILD_DIR"

g++-14 -std=c++17 -Wall -Wextra -pedantic \
    -I "$REPO_ROOT/src" \
    "$REPO_ROOT/tests/test_main.cpp" \
    "${TEST_FILES[@]}" \
    -o "$BINARY"

echo ""
if [[ $VERBOSE -eq 1 ]]; then
    "$BINARY" --success=1
else
    "$BINARY"
fi
