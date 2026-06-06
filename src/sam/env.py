from __future__ import annotations

import os
from pathlib import Path


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


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
