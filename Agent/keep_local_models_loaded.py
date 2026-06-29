from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
from threading import Event
from typing import Any

from langchain_core.messages import HumanMessage

from memeagent.config import MemeAgentConfig, load_project_env
from memeagent.llm import create_controller_llm, create_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load MemeAgent local models into memory and keep the process alive."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved model configuration without loading model weights.",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only keep the primary multimodal model loaded.",
    )
    parser.add_argument(
        "--controller-only",
        action="store_true",
        help="Only keep the controller model loaded.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="After loading, run a tiny text warmup generation for each loaded model.",
    )
    return parser.parse_args()


def _load_client(name: str, client: Any, *, warmup: bool) -> None:
    ensure_loaded = getattr(client, "_ensure_loaded", None)
    if ensure_loaded is None:
        raise TypeError(f"{name} client does not expose a local _ensure_loaded method.")

    model_path = getattr(client, "model_path", "<unknown>")
    print(f"[{name}] loading {model_path}", flush=True)
    ensure_loaded()

    model = getattr(client, "_model", None)
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        print(f"[{name}] loaded; hf_device_map={hf_device_map}", flush=True)
    else:
        device = getattr(model, "device", "<unknown>")
        print(f"[{name}] loaded; device={device}", flush=True)

    if warmup:
        print(f"[{name}] running warmup generation", flush=True)
        response = client.invoke([HumanMessage(content="Reply with OK.")])
        print(f"[{name}] warmup output: {response.content[:120]}", flush=True)


def main() -> int:
    args = parse_args()
    if args.main_only and args.controller_only:
        raise SystemExit("--main-only and --controller-only cannot be used together.")

    project_root = Path(__file__).resolve().parent
    load_project_env(project_root)
    config = MemeAgentConfig.from_env()

    print("Resolved local model configuration:", flush=True)
    print(f"  provider: {config.provider}", flush=True)
    print(f"  model: {config.model}", flush=True)
    print(f"  controller_provider: {config.controller_provider}", flush=True)
    print(f"  controller_model: {config.controller_model}", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)

    if args.dry_run:
        return 0

    loaded_clients: list[Any] = []
    if not args.controller_only:
        main_llm = create_llm(config)
        _load_client("main", main_llm, warmup=args.warmup)
        loaded_clients.append(main_llm)

    if not args.main_only:
        controller_llm = create_controller_llm(config)
        if controller_llm is None:
            print("[controller] skipped; MEMEAGENT_CONTROLLER_PROVIDER is empty", flush=True)
        else:
            _load_client("controller", controller_llm, warmup=args.warmup)
            loaded_clients.append(controller_llm)

    if not loaded_clients:
        raise RuntimeError("No model clients were loaded.")

    stop_event = Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}; exiting.", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("Models are loaded and held in memory. Press Ctrl+C to release them.", flush=True)
    stop_event.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
