from __future__ import annotations

import os
from pathlib import Path


PROVIDER_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    # GPT-5.4 / 官方 baseline 常用变量名 -> SAM 聊天模型变量名。
    "SAM_AZURE_CHAT_API_KEY": ("GPT54_API_KEY", "OPENAI_API_KEY"),
    "SAM_AZURE_CHAT_ENDPOINT": ("GPT54_BASE_URL", "RAPTOR_AZURE_ENDPOINT"),
    "SAM_AZURE_CHAT_API_VERSION": ("GPT54_API_VERSION", "RAPTOR_API_VERSION"),
    "SAM_AZURE_CHAT_MODEL": ("GPT54_MODEL", "RAPTOR_QA_MODEL", "GRAPHRAG_CHAT_MODEL"),
    # Embedding provider 允许使用更通用的本地变量名，避免维护多套 env。
    "SAM_AZURE_EMBEDDING_API_KEY": (
        "EMBEDDING_API_KEY",
        "AZURE_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    ),
    "SAM_AZURE_EMBEDDING_ENDPOINT": (
        "EMBEDDING_BASE_URL",
        "EMBEDDING_ENDPOINT",
        "AZURE_EMBEDDING_ENDPOINT",
    ),
    "SAM_AZURE_EMBEDDING_API_VERSION": (
        "EMBEDDING_API_VERSION",
        "AZURE_EMBEDDING_API_VERSION",
    ),
    "SAM_AZURE_EMBEDDING_MODEL": (
        "EMBEDDING_MODEL",
        "AZURE_EMBEDDING_MODEL",
        "RAPTOR_EMBEDDING_MODEL",
        "GRAPHRAG_EMBEDDING_MODEL",
    ),
    "SAM_AZURE_EMBEDDING_DIMENSIONS": (
        "EMBEDDING_DIMENSIONS",
        "AZURE_EMBEDDING_DIMENSIONS",
    ),
}


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, bool]:
    """加载本地 env 文件，返回每个变量是否写入当前进程环境。"""

    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"env 文件不存在：{env_path}")
    loaded: dict[str, bool] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        clean_value = _strip_quotes(value.strip())
        if key in os.environ and not override:
            loaded[key] = False
            continue
        os.environ[key] = clean_value
        loaded[key] = True
    return loaded


def apply_provider_env_aliases(
    *,
    override: bool = False,
    target_prefix: str | None = None,
) -> dict[str, str]:
    """把常见 provider 变量名归一化为 SAM 使用的变量名。

    返回值只包含目标变量名与来源变量名，不包含任何变量值。
    """

    applied: dict[str, str] = {}
    for target, sources in PROVIDER_ENV_ALIASES.items():
        if target_prefix is not None and not target.startswith(target_prefix):
            continue
        if os.environ.get(target) and not override:
            continue
        for source in sources:
            value = os.environ.get(source)
            if _is_missing_env_value(value):
                continue
            os.environ[target] = value.strip()
            applied[target] = source
            break
    return applied


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _is_missing_env_value(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.startswith("replace-with-") or stripped.startswith("your-")
