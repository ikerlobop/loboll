#!/usr/bin/env bash
# Mide la configuración "lobo" con la memoria limitada, sin swap.
# El modelo se expulsa de la caché antes de cada límite: las páginas ya cacheadas
# se cargan a otro cgroup y harían que el límite pareciera no afectar.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/models/SmolLM3-Q4_K_M.gguf"

for cap in 8G 4G 2500M 1500M; do
    python3 -I -c "
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
" "$MODEL"
    echo "=== límite $cap"
    timeout 900 systemd-run --user --scope --quiet \
        -p "MemoryMax=$cap" -p MemorySwapMax=0 \
        python3 "$ROOT/bench/tune.py" --rounds 2 --only lobo \
        --out "$ROOT/bench/cap-$cap.json" 2>&1 | grep -E '^r[0-9]|Error|Killed'
    echo "resultado: ${PIPESTATUS[0]}"
done
