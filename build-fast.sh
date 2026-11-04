#!/usr/bin/env bash
# Compila variantes de llama.cpp en carpetas separadas, sin tocar la compilación "build".
# Uso: ./build-fast.sh lto | prefetch [bloques] | pgo [ruta-del-modelo]
# Cada compilación tarda unos 10 minutos y usa toda la CPU. PGO compila dos veces.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA="$ROOT/vendor/llama.cpp"
MODEL_DEFAULT="$ROOT/models/SmolLM3-Q4_K_M.gguf"
BASE=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)

compile() {
    local dir="$1" flags="$2"; shift 2
    cmake -S "$LLAMA" -B "$LLAMA/$dir" "${BASE[@]}" \
        "-DCMAKE_C_FLAGS=$flags" "-DCMAKE_CXX_FLAGS=$flags" "$@" >/dev/null
    cmake --build "$LLAMA/$dir" -j "$(nproc)" --target llama-server llama-cli
    echo "Listo. Pruébalo con: loboll --engine $dir"
}

case "${1:-}" in
    lto)
        compile build-lto "" -DGGML_LTO=ON
        ;;
    prefetch)
        blocks="${2:-8}"
        compile "build-prefetch$blocks" "-DLOBO_PREFETCH_BLOCKS=$blocks"
        ;;
    pgo)
        model="${2:-$MODEL_DEFAULT}"
        generate="-fprofile-generate -fprofile-update=prefer-atomic"
        compile build-pgo "$generate"
        # Entrena el perfil con la misma configuración que usa loboll.
        "$LLAMA/build-pgo/bin/llama-cli" -m "$model" -t "$(nproc)" -tb "$(nproc)" \
            -ngl 0 --no-repack -c 2048 -n 128 --temp 0 --single-turn \
            -p "Write a short technical explanation of how CPU caches work." >/dev/null
        compile build-pgo "-fprofile-use -fprofile-correction -Wno-missing-profile"
        ;;
    *)
        sed -n '2,4p' "$0"
        exit 1
        ;;
esac
