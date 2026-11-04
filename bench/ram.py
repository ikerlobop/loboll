#!/usr/bin/env python3
"""Mide qué memoria usa cada configuración y cuánta velocidad cuesta ahorrarla.

Distingue RssAnon (memoria propia, que solo puede salir a swap) de RssFile
(modelo mapeado desde disco, que el sistema puede liberar y volver a leer).
"""
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from compare import ollama_generate, post
from tune import FREE_TEXT, GGUF, ROOT, SERVER

PORT = 18082
N_PREDICT = 128
ROUNDS = 3
CONFIGS = {
    "repack-c2048": ["-c", "2048"],
    "norepack-c2048": ["-c", "2048", "--no-repack"],
    "repack-c512": ["-c", "512"],
    "norepack-c512": ["-c", "512", "--no-repack"],
}


def memory_gb(pid):
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key = line.split(":")[0]
        if key in ("VmHWM", "RssAnon", "RssFile"):
            fields[key] = int(line.split()[1]) / 1024**2
    return fields


def run_llama(name, args, log):
    url = f"http://127.0.0.1:{PORT}"
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(GGUF), "--port", str(PORT), "-t", "4", "-tb", "4",
         "-b", "512", "-ngl", "0", "--parallel", "1", *args],
        stdout=log, stderr=subprocess.STDOUT,
    )
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

        def complete(prompt, n):
            return post(f"{url}/completion", {
                "prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1,
                "repeat_penalty": 1.0, "seed": 42, "cache_prompt": False,
            })

        complete("Warm up.", 4)
        timings = complete(FREE_TEXT, N_PREDICT)["timings"]
        return {
            "gen_tps": timings["predicted_per_second"],
            "prompt_tps": timings["prompt_per_second"],
            **memory_gb(proc.pid),
        }
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def run_ollama():
    ollama_generate("Warm up.", 4, 4, "5m")
    data = ollama_generate(FREE_TEXT, 4, N_PREDICT, "5m")
    pid = subprocess.run(
        ["pgrep", "-f", "ollama runner"], capture_output=True, text=True
    ).stdout.split()[0]
    memory = memory_gb(pid)
    ollama_generate("", 4, 0, 0)
    return {
        "gen_tps": data["eval_count"] / data["eval_duration"] * 1e9,
        "prompt_tps": data["prompt_eval_count"] / data["prompt_eval_duration"] * 1e9,
        **memory,
    }


def main():
    results = []
    names = [*CONFIGS, "ollama-c2048"]
    with open(ROOT / "bench/ram-server.log", "w") as log:
        for rnd in range(ROUNDS):
            offset = rnd % len(names)
            for name in names[offset:] + names[:offset]:
                row = run_ollama() if name.startswith("ollama") else run_llama(
                    name, CONFIGS[name], log
                )
                row.update(config=name, round=rnd)
                results.append(row)
                print(
                    f"r{rnd} {name:15} gen {row['gen_tps']:.2f}/s "
                    f"prompt {row['prompt_tps']:.2f}/s anon {row['RssAnon']:.2f}GB "
                    f"file {row['RssFile']:.2f}GB pico {row['VmHWM']:.2f}GB",
                    flush=True,
                )
                (ROOT / "bench/ram.json").write_text(json.dumps(results, indent=1))

    print("\nconfig           gen tok/s  prompt   anon GB  file GB  pico GB")
    for name in names:
        rows = [r for r in results if r["config"] == name]
        median = lambda key: statistics.median(r[key] for r in rows)
        print(
            f"{name:15} {median('gen_tps'):9.2f} {median('prompt_tps'):7.2f} "
            f"{median('RssAnon'):9.2f} {median('RssFile'):8.2f} {median('VmHWM'):8.2f}"
        )


if __name__ == "__main__":
    main()
