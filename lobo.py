#!/usr/bin/env python3
"""Arranca LoboLL: llama-server con la configuración medida y la interfaz web."""
import argparse
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import which

import console
import focus
import models

ROOT = Path(__file__).resolve().parent
DEFAULT_ENGINE = "build-prefetch8"
DEFAULT_MODEL = ROOT / "models/SmolLM3-Q4_K_M.gguf"
HOST = "127.0.0.1"
LOG_PATH = ROOT / "loboll-server.log"


def read_stats(pid):
    """Devuelve la memoria del servidor en bytes, tal como la ve el sistema."""
    wanted = {"RssAnon": "fixed", "RssFile": "mapped", "VmHWM": "peak"}
    stats = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key = line.split(":")[0]
        if key in wanted:
            stats[wanted[key]] = int(line.split()[1]) * 1024
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            stats["available"] = int(line.split()[1]) * 1024
    return stats


def start_stats_server(pid, ui_port, stats_port):
    origin = f"http://{HOST}:{ui_port}"
    focus_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/stats":
                self.send_error(404)
                return
            self.reply({**read_stats(pid), "frozen": len(focus.load_state())})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self):
            # Solo la propia interfaz puede congelar programas, no otra web abierta.
            if self.path != "/focus" or self.headers.get("Origin") != origin:
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            with focus_lock:
                if request.get("action") == "on":
                    also = ["code"] if request.get("vscode") else []
                    counts = focus.freeze(also, request.get("exclude", []))
                    self.reply({"counts": counts, "freed": focus.release_memory()})
                else:
                    self.reply({"resumed": focus.resume()})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((HOST, stats_port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def wait_until_ready(proc, port):
    url = f"http://{HOST}:{port}/health"
    while proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def servers_on_port(port):
    """Devuelve (pid, padre) de los llama-server de LoboLL que usan ese puerto."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if (
                Path(args[0].decode()).name == "llama-server"
                and str(ROOT).encode() in args[0]
                and str(port).encode() == args[args.index(b"--port") + 1]
            ):
                stat = (entry / "stat").read_text()
                found.append((int(entry.name), int(stat[stat.rindex(")") + 2:].split()[1])))
        except (OSError, ValueError, IndexError):
            continue
    return found


def parent_is_loboll(ppid):
    try:
        args = Path(f"/proc/{ppid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return any(Path(arg.decode(errors="ignore")).name in ("loboll", "lobo.py") for arg in args[:2])


def ensure_port_free(port):
    """Cierra servidores huérfanos y se niega a compartir el puerto con otro LoboLL."""
    for pid, ppid in servers_on_port(port):
        if parent_is_loboll(ppid):
            sys.exit(f"Ya hay otro LoboLL abierto en el puerto {port} (proceso {pid}). Ciérralo antes.")
        print(f"Cerrando un servidor anterior que quedó abierto (proceso {pid}).", flush=True)
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((HOST, port))
                return
            except OSError:
                time.sleep(0.3)
    sys.exit(f"El puerto {port} está ocupado por otro programa.")


def die_with_parent():
    """Hace que el servidor reciba SIGTERM si LoboLL muere de cualquier forma."""
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def choose_model(installed, web):
    """Pregunta qué modelo abrir si hay varios y se usa una terminal real."""
    if not installed:
        sys.exit(f"No hay modelos en {models.MODELS_DIR}")
    default = DEFAULT_MODEL if DEFAULT_MODEL in installed else installed[0]
    if is_interactive(web) and len(installed) > 1:
        chosen = models.pick(installed, default)
        return str(chosen) if chosen else None
    return str(default)


def is_interactive(web):
    return not web and sys.stdin.isatty() and sys.stdout.isatty()


def physical_cores():
    cores = {
        path.read_text().strip()
        for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/core_id")
    }
    return len(cores) or os.cpu_count()


def choose_threads(web):
    """Pregunta cuántos hilos usar. Devuelve None si se cancela."""
    total = os.cpu_count()
    if not is_interactive(web):
        return total
    physical = physical_cores()
    notes = {physical: "un hilo por núcleo físico; deja más CPU libre", total: "todos los hilos"}
    if physical == total:
        notes[total] = "todos los núcleos"
    labels = [
        f"{count} ({notes[count]})" if count in notes else str(count)
        for count in range(1, total + 1)
    ]
    index = models.menu("Elige los hilos", labels, total - 1)
    return None if index is None else index + 1


def engine_path(name):
    return ROOT / "vendor/llama.cpp" / name / "bin/llama-server"


def available_engines():
    """Devuelve las compilaciones disponibles, con la preferida en primer lugar."""
    names = [path.name for path in (ROOT / "vendor/llama.cpp").glob("build*") if engine_path(path.name).exists()]
    return sorted(names, key=lambda name: (name != DEFAULT_ENGINE, name != "build", name))


def engine_label(name):
    if name == "build":
        return "Normal"
    if name.startswith("build-prefetch"):
        return f"Prefetch {name.removeprefix('build-prefetch')}"
    return name.removeprefix("build-").upper()


def choose_engine(web):
    """Pregunta qué compilación usar si hay varias. Devuelve None si se cancela."""
    engines = available_engines()
    if not engines:
        sys.exit("No encuentro ninguna compilación de llama-server.")
    if not is_interactive(web) or len(engines) == 1:
        return engines[0]
    index = models.menu("Elige el motor", [engine_label(name) for name in engines])
    return None if index is None else engines[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="ruta del modelo; sin ella se puede elegir de un menú")
    parser.add_argument("--list", action="store_true", help="lista los modelos instalados")
    parser.add_argument("--ctx", type=int, default=4096, help="tokens de contexto")
    parser.add_argument("--threads", type=int, help="hilos; sin ella se puede elegir de un menú")
    parser.add_argument(
        "--engine",
        help="compilación de llama.cpp, por ejemplo build-prefetch8; sin ella se puede elegir de un menú",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stats-port", type=int, default=8081)
    parser.add_argument("--web", action="store_true", help="usa la interfaz web en lugar de la consola")
    parser.add_argument("--no-browser", action="store_true", help="con --web, no abre el navegador")
    parser.add_argument(
        "--focus", action="store_true",
        help="congela los programas pesados mientras LoboLL está en marcha",
    )
    parser.add_argument(
        "--also", nargs="*", default=[],
        help="otros programas a congelar con --focus, por ejemplo: code",
    )
    args = parser.parse_args()

    installed = models.find_models()
    if args.list:
        models.print_list(installed)
        return
    if args.engine and not engine_path(args.engine).exists():
        sys.exit(f"No encuentro llama-server en {engine_path(args.engine)}")
    if args.model is None:
        args.model = choose_model(installed, args.web)
        if args.model is None:
            return
    if not Path(args.model).exists():
        sys.exit(f"No encuentro el modelo en {args.model}")
    if args.threads is None:
        args.threads = choose_threads(args.web)
        if args.threads is None:
            return
    if args.engine is None:
        args.engine = choose_engine(args.web)
        if args.engine is None:
            return
    server = engine_path(args.engine)
    # Por si una ejecución anterior terminó de golpe con programas congelados.
    focus.resume()
    ensure_port_free(args.port)

    log = open(LOG_PATH, "w")
    proc = subprocess.Popen([
        str(server), "-m", args.model, "--host", HOST, "--port", str(args.port),
        "--path", str(ROOT / "ui"), "--cors-origins", "localhost",
        "-c", str(args.ctx), "--parallel", "1", "--cache-ram", "0",
        "-t", str(args.threads), "-tb", str(args.threads), "-ngl", "0",
        "--no-repack", "--reasoning", "off",
        "--spec-type", "ngram-simple",
        "--spec-ngram-simple-size-n", "6",
        "--spec-ngram-simple-size-m", "24",
        "--spec-draft-n-max", "24",
    # Sesión propia: Ctrl+C debe detener la respuesta, no apagar el servidor.
    ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=die_with_parent)
    log.close()

    # Cerrar la terminal envía SIGHUP: hay que apagar el servidor y reanudar programas.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))
    try:
        if not args.web:
            print(
                f"Cargando {Path(args.model).stem} con {args.threads} hilos "
                f"(motor {engine_label(args.engine)})…",
                flush=True,
            )
        if not wait_until_ready(proc, args.port):
            sys.exit(f"llama-server terminó al arrancar. Revisa {LOG_PATH}")
        url = f"http://{HOST}:{args.port}"
        if args.focus:
            print(f"Modo concentración: {focus.freeze(args.also)}", flush=True)
        if args.web:
            start_stats_server(proc.pid, args.port, args.stats_port)
            print(f"LoboLL listo en {url}/", flush=True)
            if not args.no_browser:
                # Con el modo concentración, Chrome está congelado y no podría abrirla.
                opener = "firefox" if args.focus and which("firefox") else "xdg-open"
                subprocess.Popen(
                    [opener, f"{url}/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            proc.wait()
        else:
            console.run(url, lambda: read_stats(proc.pid), args.model)
    except KeyboardInterrupt:
        pass
    finally:
        # Siempre, para que el botón de la interfaz no deje nada congelado.
        print(f"Procesos reanudados: {focus.resume()}", flush=True)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
