#!/usr/bin/env python3
"""Compara la velocidad en CPU de Ollama y llama-server con el mismo modelo.

Ambos motores reciben el mismo prompt en crudo y los mismos parámetros de
muestreo. Las ejecuciones se intercalan para repartir el ruido del sistema, y
cada prompt empieza por un número distinto para que la caché de prefijos no
pueda reutilizar el procesamiento anterior.
"""
import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models/SmolLM3-Q4_K_M.gguf"
SERVER = ROOT / "vendor/llama.cpp/build/bin/llama-server"
OLLAMA_MODEL = "lobo-smollm3-q4"
OLLAMA_URL = "http://localhost:11434/api/generate"
PORT = 18080
CTX = 2048
BATCH = 512

BODY = (
    "The following is a detailed technical essay on the history of computer "
    "processors. It begins with the first vacuum tube machines, continues with "
    "the invention of the transistor and the integrated circuit, and then "
    "explains how pipelining, caches, branch prediction, out-of-order execution "
    "and vector instructions changed the way software is written. Each section "
    "describes the main engineering problems of the period, the solutions that "
    "were adopted, and the consequences for programmers and for the industry as "
    "a whole. The essay pays special attention to the growing gap between "
    "processor speed and memory speed, and to why modern chips devote so much "
    "of their area to cache.\n\nChapter 1: The vacuum tube era\n\n"
)


def post(url, payload, timeout=900):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def peak_rss_gb(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024**2
    return None


def ollama_generate(prompt, threads, n_predict, keep_alive):
    return post(OLLAMA_URL, {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_thread": threads, "num_ctx": CTX, "num_batch": BATCH,
            "num_gpu": 0, "num_predict": n_predict, "temperature": 0,
            "top_k": 1, "repeat_penalty": 1.0, "seed": 42,
        },
    })


def run_ollama(prompt, threads, n_predict):
    ollama_generate("Warm up.", threads, 4, "5m")
    data = ollama_generate(prompt, threads, n_predict, "5m")
    runner = subprocess.run(
        ["pgrep", "-f", "ollama runner"], capture_output=True, text=True
    ).stdout.split()
    rss = peak_rss_gb(runner[0]) if runner else None
    ollama_generate("", threads, 0, 0)  # descarga el modelo
    return {
        "prompt_n": data["prompt_eval_count"],
        "prompt_tps": data["prompt_eval_count"] / data["prompt_eval_duration"] * 1e9,
        "gen_n": data["eval_count"],
        "gen_tps": data["eval_count"] / data["eval_duration"] * 1e9,
        "rss_gb": rss,
        "text": data["response"],
    }


def run_llama(prompt, threads, n_predict, log_path):
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [str(SERVER), "-m", str(GGUF), "--port", str(PORT),
             "-c", str(CTX), "-b", str(BATCH), "-t", str(threads),
             "-tb", str(threads), "-ngl", "0", "--parallel", "1"],
            stdout=log, stderr=subprocess.STDOUT,
        )
    try:
        url = f"http://127.0.0.1:{PORT}"
        for _ in range(300):
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server no arrancó")

        def complete(text, n):
            return post(f"{url}/completion", {
                "prompt": text, "n_predict": n, "temperature": 0, "top_k": 1,
                "repeat_penalty": 1.0, "seed": 42, "cache_prompt": False,
            })

        complete("Warm up.", 4)
        data = complete(prompt, n_predict)
        timings = data["timings"]
        return {
            "prompt_n": timings["prompt_n"],
            "prompt_tps": timings["prompt_per_second"],
            "gen_n": timings["predicted_n"],
            "gen_tps": timings["predicted_per_second"],
            "rss_gb": peak_rss_gb(proc.pid),
            "text": data["content"],
        }
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--out", default=str(ROOT / "bench/results.json"))
    args = parser.parse_args()

    log_path = ROOT / "bench/llama-server.log"
    results = []
    for rnd in range(args.rounds):
        # El orden se alterna en cada ronda para no favorecer a ningún motor.
        engines = ["ollama", "llama"] if rnd % 2 == 0 else ["llama", "ollama"]
        for threads in args.threads:
            for engine in engines:
                prompt = f"Document {rnd}-{threads}-{engine}.\n\n{BODY}"
                if engine == "ollama":
                    row = run_ollama(prompt, threads, args.n_predict)
                else:
                    row = run_llama(prompt, threads, args.n_predict, log_path)
                row.update(engine=engine, threads=threads, round=rnd)
                results.append(row)
                print(
                    f"r{rnd} {engine:6} t={threads} "
                    f"prompt {row['prompt_n']}@{row['prompt_tps']:.2f}/s "
                    f"gen {row['gen_n']}@{row['gen_tps']:.2f}/s "
                    f"rss {row['rss_gb'] or 0:.2f}GB",
                    flush=True,
                )
                Path(args.out).write_text(json.dumps(results, indent=1))

    print("\nmotor   hilos  gen tok/s (mediana, min-max)   prompt tok/s")
    for engine in ["ollama", "llama"]:
        for threads in args.threads:
            rows = [r for r in results
                    if r["engine"] == engine and r["threads"] == threads]
            gen = [r["gen_tps"] for r in rows]
            pp = [r["prompt_tps"] for r in rows]
            print(
                f"{engine:7} {threads:5}  {statistics.median(gen):6.2f} "
                f"({min(gen):.2f}-{max(gen):.2f})"
                f"{statistics.median(pp):18.2f}"
            )


if __name__ == "__main__":
    main()
