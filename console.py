"""Chat de consola para LoboLL, solo con la biblioteca estándar."""
import json
import urllib.error
import urllib.request

import focus
import models

try:
    import readline  # noqa: F401  (activa la edición de línea en input)
except ImportError:
    pass

HELP = """Comandos:
  /nueva          empieza otra conversación
  /liberar        congela programas pesados y libera su RAM
  /liberar code   lo mismo, incluido VS Code (Claude Code dejará de responder)
  /reanudar       reanuda los programas congelados
  /ram            muestra la memoria actual
  /modelos        lista los modelos instalados (* es el abierto)
  /ayuda          muestra esta ayuda
  /salir          termina (también Ctrl+D)
Ctrl+C detiene una respuesta en curso."""


def gb(value):
    return f"{value / 1024 ** 3:.2f} GB"


def print_memory(read_stats):
    stats = read_stats()
    print(
        f"RAM fija {gb(stats['fixed'])} · modelo mapeado {gb(stats['mapped'])} · "
        f"pico {gb(stats['peak'])} · libre en el sistema {gb(stats['available'])}"
    )


def print_timings(timings):
    line = (
        f"{timings['predicted_per_second']:.1f} tok/s · "
        f"{timings['predicted_n']} tokens · "
        f"prompt {timings['prompt_n']} leídos, {timings.get('cache_n', 0)} de caché"
    )
    if timings.get("draft_n"):
        line += f" · especulación {timings['draft_n_accepted']}/{timings['draft_n']}"
    print(f"\033[2m[{line}]\033[0m")


def stream_reply(url, messages):
    """Imprime la respuesta según llega. Devuelve el texto aunque se interrumpa."""
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        json.dumps({
            "messages": messages, "stream": True, "cache_prompt": True,
            "temperature": 0.7, "max_tokens": 1024,
        }).encode(),
        {"Content-Type": "application/json"},
    )
    text = ""
    timings = None
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw in response:
                line = raw.decode()
                if not line.startswith("data: ") or line.startswith("data: [DONE]"):
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    raise RuntimeError(event["error"].get("message", "error desconocido"))
                choices = event.get("choices") or [{}]
                piece = choices[0].get("delta", {}).get("content")
                if piece:
                    text += piece
                    print(piece, end="", flush=True)
                timings = event.get("timings", timings)
        print()
        if timings:
            print_timings(timings)
    except KeyboardInterrupt:
        # Al cerrar la conexión, el servidor deja de generar.
        print("\n[detenido]")
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read() or b"{}").get("error", {}).get("message", error)
        print(f"\n[error] {detail}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"\n[error] {error}")
    return text


def handle_command(command, messages, read_stats, model):
    """Devuelve False si hay que salir."""
    name, *rest = command.split()
    if name == "/salir":
        return False
    if name == "/modelos":
        models.print_list(models.find_models(), model)
        print("Para cambiar de modelo, sal con /salir y vuelve a abrir loboll.")
        return True
    if name == "/nueva":
        messages.clear()
        print("Conversación nueva.")
    elif name == "/liberar":
        counts = focus.freeze(rest)
        if counts:
            print(f"Congelados: {', '.join(counts)} · liberando memoria…", flush=True)
            print(f"RAM liberada: {gb(focus.release_memory())}")
        else:
            print("No había programas pesados que congelar.")
    elif name == "/reanudar":
        print(f"Procesos reanudados: {focus.resume()}")
    elif name == "/ram":
        print_memory(read_stats)
    else:
        print(HELP)
    return True


def run(url, read_stats, model):
    print("LoboLL listo. Escribe /ayuda para ver los comandos.")
    messages = []
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.startswith("/"):
            if not handle_command(text, messages, read_stats, model):
                return
            continue
        messages.append({"role": "user", "content": text})
        reply = stream_reply(url, messages)
        if reply:
            messages.append({"role": "assistant", "content": reply})
        else:
            messages.pop()
