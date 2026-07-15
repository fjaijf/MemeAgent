from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import batch_agent_evaluator
from memeagent.config import MemeAgentConfig, load_project_env


def _option_value(argv: list[str], option: str) -> str:
    for index, argument in enumerate(argv):
        if argument == option and index + 1 < len(argv):
            return argv[index + 1]
        prefix = option + "="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return ""


def _without_option(argv: list[str], option: str) -> list[str]:
    result: list[str] = []
    skip_next = False
    for argument in argv:
        if skip_next:
            skip_next = False
            continue
        if argument == option:
            skip_next = True
            continue
        if argument.startswith(option + "="):
            continue
        result.append(argument)
    return result


def _ablation_argv(argv: list[str], *, main_model: str, project_root: Path) -> list[str]:
    forwarded = _without_option(argv, "--controller-model")
    forwarded.extend(["--controller-model", main_model])

    if not _option_value(forwarded, "--output-dir"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "runs" / f"batch_agent_main_as_controller_{timestamp}"
        forwarded.extend(["--output-dir", str(output_dir)])
    return forwarded


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_project_env(project_root)
    config = MemeAgentConfig.from_env()
    main_model = (
        _option_value(sys.argv[1:], "--main-model").strip()
        or config.model
        or batch_agent_evaluator.DEFAULT_BATCH_MAIN_MODEL
    )

    sys.argv[1:] = _ablation_argv(
        sys.argv[1:],
        main_model=main_model,
        project_root=project_root,
    )
    print(
        f"Ablation: controller and final stages use main model {main_model}",
        flush=True,
    )
    return batch_agent_evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
