"""Ollama HTTP client. Run on the Labs VM or inside your own image."""
import os
import sys
import time

import requests

URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("MODEL", "smollm2:135m-instruct-q2_K")


def ask(prompt):
    started = time.perf_counter()
    response = requests.post(
        URL + "/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 1024, "num_predict": 64},
        },
        timeout=(10, 180),
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(data["error"])
    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("No non-empty response field in API result")
    if data.get("done") is not True:
        raise ValueError("Generation did not finish")
    print(answer.strip())
    print(f"[elapsed={time.perf_counter() - started:.2f}s]", file=sys.stderr)


def safe_ask(prompt):
    try:
        ask(prompt)
        return True
    except requests.exceptions.Timeout:
        print("Timeout: check CPU load and retry with a shorter prompt.", file=sys.stderr)
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to {URL}. Check container, port, and network.", file=sys.stderr)
    except requests.exceptions.HTTPError as exc:
        print(f"HTTP error: {exc.response.status_code} {exc.response.text[:240]}", file=sys.stderr)
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
    return False


def main():
    if len(sys.argv) > 1:
        return 0 if safe_ask(" ".join(sys.argv[1:])) else 1
    print(f"Ollama client: {URL} / {MODEL}. Type exit to quit.")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.lower() in {"exit", "quit"}:
            return 0
        if prompt:
            safe_ask(prompt)


if __name__ == "__main__":
    raise SystemExit(main())