"""Busca los modelos GGUF instalados y muestra menús que se manejan con los cursores."""
import os
import shutil
import sys
import termios
import tty
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
UP = {b"\x1b[A", b"\x1bOA", b"k"}
DOWN = {b"\x1b[B", b"\x1bOB", b"j"}
ENTER = {b"\r", b"\n"}
CANCEL = {b"\x1b", b"q", b"\x03"}


def find_models():
    """Devuelve los modelos ordenados por tamaño, sin archivos auxiliares de visión."""
    found = [
        path for path in MODELS_DIR.glob("*.gguf")
        if not path.name.lower().startswith("mmproj")
    ]
    return sorted(found, key=lambda path: path.stat().st_size)


def describe(path):
    return f"{path.stem}  ({path.stat().st_size / 1e9:.2f} GB)"


def print_list(models, current=None):
    if not models:
        print(f"No hay modelos en {MODELS_DIR}")
    for path in models:
        marker = "*" if current and path == Path(current) else " "
        print(f"{marker} {describe(path)}")


def _render(labels, selected, width):
    for index, label in enumerate(labels):
        line = label[:width - 3]
        if index == selected:
            print(f"\033[2K\033[7m> {line}\033[0m", end="\r\n")
        else:
            print(f"\033[2K  {line}", end="\r\n")
    sys.stdout.flush()


def menu(title, labels, selected=0):
    """Muestra un menú y devuelve el índice elegido, o None si se cancela."""
    width = shutil.get_terminal_size().columns
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    print(f"{title} (↑/↓ para moverte, Enter para aceptar, q para salir):")
    print("\033[?25l", end="")  # oculta el cursor
    try:
        tty.setraw(fd)
        _render(labels, selected, width)
        while True:
            key = os.read(fd, 8)
            if key in ENTER:
                return selected
            if key in CANCEL:
                return None
            if key in UP:
                selected = (selected - 1) % len(labels)
            elif key in DOWN:
                selected = (selected + 1) % len(labels)
            else:
                continue
            print(f"\033[{len(labels)}A", end="")
            _render(labels, selected, width)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print("\033[?25h", end="", flush=True)  # recupera el cursor


def pick(models, default=None):
    """Devuelve el modelo elegido en el menú, o None si se cancela."""
    selected = models.index(default) if default in models else 0
    index = menu("Elige un modelo", [describe(path) for path in models], selected)
    return None if index is None else models[index]
