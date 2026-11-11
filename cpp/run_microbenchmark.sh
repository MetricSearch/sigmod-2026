#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root
readonly build_dir="$repo_root/build/microbenchmark-native"


cmake -S "$repo_root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -mavx2 -mfma" \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=ON \
    -DRABITQ_BUILD_EXPERIMENTS=ON \
    -DRABITQ_BUILD_SAMPLES=OFF \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_PYTHON_BINDINGS=OFF

cmake --build "$build_dir" --target microbenchmark --parallel "${BUILD_JOBS:-2}"
exec "$build_dir/sigmod_experiments/microbenchmark" "$@"
