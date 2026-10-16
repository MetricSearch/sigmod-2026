#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root
readonly build_dir="$repo_root/build/bench-rabit-avx512"

cmake -S "$repo_root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
    -DRABITQ_BUILD_EXPERIMENTS=OFF \
    -DRABITQ_BUILD_SAMPLES=OFF \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_PYTHON_BINDINGS=OFF

cmake --build "$build_dir" --target bench_rabit
exec "$build_dir/bench_rabit" "$@"