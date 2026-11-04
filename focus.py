#!/usr/bin/env python3
"""Congela programas pesados mientras corre el modelo, sin cerrarlos.

Los procesos se pausan con SIGSTOP y se reanudan con SIGCONT, por lo que conservan
su estado y el trabajo sin guardar. Su memoria puede pasarse a swap para liberar
RAM; al reanudarlos vuelve poco a poco, así que tardan algo en responder.

Uso: focus.py on [--also code] | off | status
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path

# El escritorio y el navegador con la interfaz no se tocan. VS Code solo se congela
# si se pide con --also code, porque también aloja a Claude Code.
PROGRAMS = {
    "chrome", "chromium", "brave", "opera", "slack", "discord", "spotify",
    "telegram-deskto", "teams", "thunderbird", "steam", "zoom", "obsidian",
    "dropbox",
}
STATE = Path.home() / ".local/state/lobo/frozen.json"
CGROUP_ROOT = Path("/sys/fs/cgroup")
RECLAIM_STEP = "128M"
RECLAIM_SECONDS = 20


def process_info(pid):
    """Devuelve (nombre, instante de arranque) o None si el proceso ya no existe."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return None
    except OSError:
        return None
    # El nombre va entre paréntesis y puede contener espacios.
    name = stat[stat.index("(") + 1:stat.rindex(")")]
    return name, stat[stat.rindex(")") + 2:].split()[19]


def matching_processes(names):
    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != me:
            info = process_info(int(entry.name))
            if info and info[0] in names:
                yield int(entry.name), info


def load_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return []


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def release_memory():
    """Pasa a swap la memoria de los programas congelados.

    Solo actúa sobre grupos de memoria en los que todo está congelado, para no
    tocar nunca un proceso activo como el propio modelo. Devuelve los bytes de RAM
    disponibles que se han ganado.
    """
    frozen_pids = {row["pid"] for row in load_state()}
    groups = set()
    for pid in frozen_pids:
        try:
            line = Path(f"/proc/{pid}/cgroup").read_text().strip()
        except OSError:
            continue
        groups.add(CGROUP_ROOT / line.split("::", 1)[-1].lstrip("/"))

    before = available_memory()
    deadline = time.monotonic() + RECLAIM_SECONDS
    for group in groups:
        try:
            procs = {int(p) for p in (group / "cgroup.procs").read_text().split()}
            if not procs <= frozen_pids:
                continue
            control = group / "memory.reclaim"
            # Por tramos: el kernel da error en cuanto no puede liberar más.
            while time.monotonic() < deadline:
                control.write_text(RECLAIM_STEP)
        except OSError:
            continue
    return max(0, available_memory() - before)


def freeze(also=(), exclude=()):
    """Pausa los programas y devuelve cuántos procesos hay de cada uno."""
    names = (PROGRAMS | set(also)) - set(exclude)
    frozen = {row["pid"]: row for row in load_state()}
    # Dos pasadas: un proceso puede haber creado hijos justo antes de pausarlo.
    for _ in range(2):
        found = {
            pid: {"pid": pid, "name": info[0], "start": info[1]}
            for pid, info in matching_processes(names)
        }
        frozen.update(found)
        # El estado se guarda antes de pausar para poder reanudar tras un fallo.
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(list(frozen.values())))
        for pid in found:
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass
    counts = {}
    for row in frozen.values():
        counts[row["name"]] = counts.get(row["name"], 0) + 1
    return counts


def resume():
    """Reanuda lo congelado y devuelve cuántos procesos se han reanudado."""
    resumed = 0
    for row in load_state():
        info = process_info(row["pid"])
        # Comprueba el arranque para no tocar un proceso nuevo con el mismo PID.
        if info and info == (row["name"], row["start"]):
            try:
                os.kill(row["pid"], signal.SIGCONT)
                resumed += 1
            except OSError:
                pass
    STATE.unlink(missing_ok=True)
    return resumed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["on", "off", "status"])
    parser.add_argument("--also", nargs="*", default=[], help="otros programas a congelar")
    args = parser.parse_args()

    if args.action == "on":
        counts = freeze(args.also)
        print(f"Congelados: {counts}" if counts else "No había programas que congelar.")
    elif args.action == "off":
        print(f"Procesos reanudados: {resume()}")
    else:
        rows = load_state()
        print(f"Procesos congelados: {len(rows)}")


if __name__ == "__main__":
    main()
