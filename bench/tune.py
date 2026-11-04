#!/usr/bin/env python3
"""Mide ajustes de llama-server frente a una configuración base.

Usa prompts fijos y muestreo voraz, de modo que cualquier configuración debería
producir el mismo texto. Una diferencia frente a la base se marca en el resumen,
porque un ajuste de velocidad no debe cambiar la respuesta.
"""
import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models/SmolLM3-Q4_K_M.gguf"
SERVER = ROOT / "vendor/llama.cpp/build/bin/llama-server"
PORT = 18081
CPU = Path("/sys/devices/system/cpu")

FREE_TEXT = (
    "The following is a detailed technical essay on the history of computer "
    "processors. It explains how pipelining, caches, branch prediction and "
    "vector instructions changed the way software is written, and why modern "
    "chips devote so much of their area to cache.\n\n"
    "Chapter 1: The vacuum tube era\n\n"
)

CODE = '''class Inventory:
    def __init__(self):
        self.items = {}

    def add_item(self, name, quantity, price):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if name in self.items:
            self.items[name]["quantity"] += quantity
            self.items[name]["price"] = price
        else:
            self.items[name] = {"quantity": quantity, "price": price}

    def remove_item(self, name, quantity):
        if name not in self.items:
            raise KeyError(name)
        if self.items[name]["quantity"] < quantity:
            raise ValueError("not enough stock")
        self.items[name]["quantity"] -= quantity
        if self.items[name]["quantity"] == 0:
            del self.items[name]

    def total_value(self):
        total = 0
        for item in self.items.values():
            total += item["quantity"] * item["price"]
        return total

    def report(self):
        lines = []
        for name, item in sorted(self.items.items()):
            lines.append(f"{name}: {item['quantity']} at {item['price']}")
        return "\\n".join(lines)
'''

CODE_EDIT = (
    "Original Python file:\n\n```python\n" + CODE + "```\n\n"
    "The same file, unchanged except that a docstring was added to the class:"
    "\n\n```python\nclass Inventory:\n"
    '    """Tracks items in stock."""\n\n'
)

WORKLOADS = {"texto": (FREE_TEXT, 128), "codigo": (CODE_EDIT, 200)}


def physical_core_mask():
    """Devuelve una máscara con un solo hilo por núcleo físico."""
    seen, mask = set(), 0
    for path in sorted(CPU.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        core = (path / "topology/core_id").read_text().strip()
        if core not in seen:
            seen.add(core)
            mask |= 1 << int(path.name[3:])
    return f"{mask:x}"


def configs():
    pin = ["-t", "2", "-tb", "2", "-C", physical_core_mask(), "--cpu-strict", "1"]
    thp = {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"}
    return {
        "base-t4": (["-t", "4", "-tb", "4"], {}),
        "base-t2": (["-t", "2", "-tb", "2"], {}),
        "pin-t2": (pin, {}),
        "thp-t4": (["-t", "4", "-tb", "4", "--load-mode", "none"], thp),
        "ngram-simple": (["-t", "4", "-tb", "4", "--spec-type", "ngram-simple",
                          "--spec-ngram-simple-size-n", "6",
                          "--spec-ngram-simple-size-m", "24",
                          "--spec-draft-n-max", "24"], {}),
        "lobo": (["-t", "4", "-tb", "4", "--no-repack",
                  "--spec-type", "ngram-simple",
                  "--spec-ngram-simple-size-n", "6",
                  "--spec-ngram-simple-size-m", "24",
                  "--spec-draft-n-max", "24"], {}),
        "ngram-map-k": (["-t", "4", "-tb", "4", "--spec-type", "ngram-map-k",
                         "--spec-ngram-map-k-size-n", "6",
                         "--spec-ngram-map-k-size-m", "24",
                         "--spec-draft-n-max", "24"], {}),
        "ngram-mod": (["-t", "4", "-tb", "4", "--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "6",
                       "--spec-ngram-mod-n-min", "4",
                       "--spec-ngram-mod-n-max", "24"], {}),
    }


def post(url, payload):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)


def huge_kb(pid):
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        if line.startswith("AnonHugePages:"):
            return int(line.split()[1])
    return 0


def run_config(name, args, env, log_path):
    url = f"http://127.0.0.1:{PORT}"
    with open(log_path, "a") as log:
        log.write(f"\n==== {name}\n")
        log.flush()
        proc = subprocess.Popen(
            [str(SERVER), "-m", str(GGUF), "--port", str(PORT), "-c", "2048",
             "-b", "512", "-ngl", "0", "--parallel", "1", *args],
            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env},
        )
    rows = []
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise RuntimeError(f"{name}: el servidor terminó con {proc.returncode}")
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError(f"{name}: el servidor no arrancó")

        def complete(prompt, n):
            return post(f"{url}/completion", {
                "prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1,
                "repeat_penalty": 1.0, "seed": 42, "cache_prompt": False,
            })

        complete("Warm up.", 4)
        for workload, (prompt, n_predict) in WORKLOADS.items():
            data = complete(prompt, n_predict)
            timings = data["timings"]
            rows.append({
                "config": name,
                "workload": workload,
                "gen_n": timings["predicted_n"],
                "gen_tps": timings["predicted_per_second"],
                "prompt_tps": timings["prompt_per_second"],
                "draft_n": timings.get("draft_n", 0),
                "draft_ok": timings.get("draft_n_accepted", 0),
                "huge_mb": huge_kb(proc.pid) / 1024,
                "text": data["content"],
            })
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", default=str(ROOT / "bench/tune.json"))
    args = parser.parse_args()

    log_path = ROOT / "bench/tune-server.log"
    log_path.write_text("")
    selected = {
        name: cfg for name, cfg in configs().items()
        if not args.only or name in args.only
    }
    results = []
    for rnd in range(args.rounds):
        # Rota el orden en cada ronda para repartir la deriva térmica.
        names = list(selected)
        offset = rnd % len(names)
        for name in names[offset:] + names[:offset]:
            cli, env = selected[name]
            for row in run_config(name, cli, env, log_path):
                row["round"] = rnd
                results.append(row)
                print(
                    f"r{rnd} {name:13} {row['workload']:7} "
                    f"gen {row['gen_n']}@{row['gen_tps']:.2f}/s "
                    f"prompt {row['prompt_tps']:.2f}/s "
                    f"draft {row['draft_ok']}/{row['draft_n']} "
                    f"huge {row['huge_mb']:.0f}MB",
                    flush=True,
                )
            Path(args.out).write_text(json.dumps(results, indent=1))

    print("\nconfig         carga    gen tok/s (mediana, min-max)  prompt  mismo texto")
    for workload in WORKLOADS:
        reference = next(
            (r["text"] for r in results
             if r["config"] == "base-t4" and r["workload"] == workload), None
        )
        for name in selected:
            rows = [r for r in results
                    if r["config"] == name and r["workload"] == workload]
            gen = [r["gen_tps"] for r in rows]
            same = all(r["text"] == reference for r in rows)
            print(
                f"{name:14} {workload:7} {statistics.median(gen):7.2f} "
                f"({min(gen):.2f}-{max(gen):.2f})"
                f"{statistics.median(r['prompt_tps'] for r in rows):12.2f}"
                f"  {'sí' if same else 'NO'}"
            )


if __name__ == "__main__":
    main()
