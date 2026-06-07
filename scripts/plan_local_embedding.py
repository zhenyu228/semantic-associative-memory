from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.embedding import inspect_embedding_provider_config  # noqa: E402
from sam.env import load_env_file  # noqa: E402


MODEL_MARKER_FILES = [
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "model.safetensors",
    "pytorch_model.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查本地 sentence-transformers embedding 运行准备状态")
    parser.add_argument("--env-file", default=None, help="可选：加载本地 .env.local")
    parser.add_argument("--model-path", default=None, help="本地模型目录；默认读取 SAM_SENTENCE_TRANSFORMER_MODEL")
    parser.add_argument("--output-dir", default="outputs/plans/local_embedding_plan", help="输出目录")
    parser.add_argument("--json", action="store_true", help="同时在终端打印 JSON")
    return parser.parse_args()


def build_local_embedding_plan(model_path: str | None = None) -> dict[str, object]:
    status = inspect_embedding_provider_config("sentence_transformers")
    resolved_model = model_path or _env_model_path()
    model_info = _inspect_model_path(resolved_model)
    ready = bool(status.get("ready")) and bool(model_info["ready"])
    return {
        "provider": "sentence_transformers",
        "ready": ready,
        "provider_status": status,
        "model": model_info,
        "install_command": 'conda run -n sam python -m pip install -e ".[local-embedding]"',
        "probe_command": _probe_command(resolved_model),
        "run_command": _run_command(resolved_model),
        "notes": _notes(status, model_info),
    }


def write_local_embedding_plan(plan: dict[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "local_embedding_plan.json"
    markdown_path = target / "local_embedding_plan.md"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(plan), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_env_file(_resolve_path(args.env_file))
    output_dir = _resolve_path(args.output_dir)
    plan = build_local_embedding_plan(args.model_path)
    json_path, markdown_path = write_local_embedding_plan(plan, output_dir)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"本地 embedding 准备计划完成：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"Ready：{plan['ready']}")


def _env_model_path() -> str:
    import os

    return os.environ.get("SAM_SENTENCE_TRANSFORMER_MODEL", "Qwen/Qwen3-Embedding-0.6B")


def _inspect_model_path(model_path: str) -> dict[str, object]:
    path = Path(model_path).expanduser()
    is_local_path = path.is_absolute() or "/" in model_path or model_path.startswith(".")
    if not is_local_path:
        return {
            "model_path": model_path,
            "is_local_path": False,
            "exists": False,
            "ready": True,
            "found_marker_files": [],
            "missing_marker_files": [],
            "message": "模型将按 HuggingFace model id 加载，首次运行可能需要联网下载。",
        }
    exists = path.exists()
    found = [
        marker
        for marker in MODEL_MARKER_FILES
        if (path / marker).exists()
    ]
    ready = exists and any(marker in found for marker in ["modules.json", "config.json"])
    return {
        "model_path": str(path),
        "is_local_path": True,
        "exists": exists,
        "ready": ready,
        "found_marker_files": found,
        "missing_marker_files": [
            marker for marker in MODEL_MARKER_FILES if marker not in found
        ],
        "message": (
            "本地模型目录存在且包含基础配置文件。"
            if ready
            else "本地模型目录不存在或缺少 config/modules 配置文件。"
        ),
    }


def _notes(status: dict[str, object], model_info: dict[str, object]) -> list[str]:
    notes: list[str] = []
    if status.get("missing_packages"):
        notes.append("当前环境缺少 sentence-transformers/torch/transformers 相关依赖，需要先安装可选依赖。")
    if model_info.get("is_local_path") and not model_info.get("ready"):
        notes.append("本地模型目录尚不可用，需要先下载 Qwen3-Embedding-0.6B 或改用 HuggingFace model id。")
    if not notes:
        notes.append("本地 embedding provider 准备就绪，可以先 probe，再运行 HotpotQA smoke。")
    return notes


def _probe_command(model_path: str) -> str:
    return (
        f'SAM_SENTENCE_TRANSFORMER_MODEL="{model_path}" '
        "conda run -n sam python scripts/check_embedding_provider.py "
        "--provider sentence_transformers --probe \"SAM local embedding probe.\""
    )


def _run_command(model_path: str) -> str:
    return (
        f'SAM_SENTENCE_TRANSFORMER_MODEL="{model_path}" '
        "conda run -n sam python scripts/run_demo.py "
        "--reset --dataset hotpotqa --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json "
        "--embedding-provider sentence_transformers --embedding-cache --query-limit 30"
    )


def _markdown(plan: dict[str, object]) -> str:
    provider_status = plan.get("provider_status", {})
    model = plan.get("model", {})
    if not isinstance(provider_status, dict):
        provider_status = {}
    if not isinstance(model, dict):
        model = {}
    lines = [
        "# 本地 Embedding 准备计划",
        "",
        f"- Provider：{plan.get('provider')}",
        f"- Ready：{plan.get('ready')}",
        f"- 缺少依赖：{', '.join(str(item) for item in provider_status.get('missing_packages', [])) or '无'}",
        f"- 模型路径：{model.get('model_path')}",
        f"- 模型目录可用：{model.get('ready')}",
        f"- 模型检查说明：{model.get('message')}",
        "",
        "## 建议命令",
        "",
        "```bash",
        str(plan.get("install_command", "")),
        "",
        str(plan.get("probe_command", "")),
        "",
        str(plan.get("run_command", "")),
        "```",
        "",
        "## 备注",
        "",
    ]
    for note in plan.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
