from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "evaluation/official_baselines"
EXTERNAL_DIR = ROOT / "evaluation/external"
VENV_DIR = ROOT / "evaluation/.venvs"
DEFAULT_OUTPUT_DIR = ROOT / "docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计官方 baseline 的本地就绪状态")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="可重复传入本地 env 文件，只统计变量名和是否配置，不输出值",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="审计报告输出目录")
    parser.add_argument("--timeout", type=float, default=30.0, help="导入检查子进程超时时间")
    parser.add_argument("--json", action="store_true", help="同时打印 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    env_sources = {key: "process" for key in env}
    for raw_env_file in args.env_file:
        env_path = _resolve_path(raw_env_file)
        file_env = _read_env_file(env_path)
        env.update(file_env)
        env_sources.update({key: str(env_path) for key in file_env})
    _apply_official_baseline_aliases(env, env_sources)

    audit = build_official_baseline_audit(env=env, env_sources=env_sources, timeout=args.timeout)
    json_path, markdown_path = write_official_baseline_audit(audit, _resolve_path(args.output_dir))
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"官方 baseline 审计完成：{json_path}")
    print(f"Markdown：{markdown_path}")


def build_official_baseline_audit(
    *,
    env: dict[str, str] | None = None,
    env_sources: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """构建官方 baseline 就绪状态审计，不输出任何密钥或 endpoint 明文。"""

    effective_env = dict(os.environ)
    if env:
        effective_env.update(env)
    source_by_key = env_sources or {}
    baselines = json.loads((BASELINE_DIR / "baselines.json").read_text(encoding="utf-8"))
    prepared_datasets = _prepared_dataset_summary(ROOT / "evaluation/runs")

    methods = {
        "raptor": _audit_raptor(effective_env, source_by_key, timeout),
        "graphrag": _audit_graphrag(effective_env, source_by_key, timeout),
        "hipporag": _audit_hipporag(effective_env, source_by_key, timeout),
    }
    for method_id, method in methods.items():
        method["official_repo"] = baselines.get(method_id, {}).get("official_repo")
        method["display_name"] = baselines.get(method_id, {}).get("display_name", method_id)

    runnable_count = sum(1 for method in methods.values() if method["status"] == "ready")
    partial_count = sum(1 for method in methods.values() if method["status"] == "partial")
    blocked_count = sum(1 for method in methods.values() if method["status"] == "blocked")
    return {
        "title": "官方 baseline 就绪状态审计",
        "summary": {
            "method_count": len(methods),
            "ready_count": runnable_count,
            "partial_count": partial_count,
            "blocked_count": blocked_count,
            "prepared_dataset_count": len(prepared_datasets),
        },
        "methods": methods,
        "prepared_datasets": prepared_datasets,
        "next_actions": _next_actions(methods, prepared_datasets),
    }


def write_official_baseline_audit(audit: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "official_baseline_audit.json"
    markdown_path = target / "official_baseline_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def _audit_raptor(env: dict[str, str], env_sources: dict[str, str], timeout: float) -> dict[str, Any]:
    repo = EXTERNAL_DIR / "raptor"
    python = VENV_DIR / "raptor/bin/python"
    import_check = _run_check(
        [str(python), "-c", "from raptor import RetrievalAugmentation; print('ok')"],
        timeout=timeout,
        env={**env, "PYTHONPATH": str(repo)},
        cwd=ROOT,
    )
    config = _config_status(
        env,
        env_sources,
        required_any=[
            ["OPENAI_API_KEY", "GPT54_API_KEY"],
            ["RAPTOR_EMBEDDING_API_KEY", "EMBEDDING_API_KEY", "SAM_AZURE_EMBEDDING_API_KEY"],
        ],
        required=[
            "RAPTOR_QA_MODEL",
            "RAPTOR_SUMMARY_MODEL",
            "RAPTOR_EMBEDDING_MODEL",
        ],
        required_for_azure=[
            "RAPTOR_AZURE_ENDPOINT",
            "RAPTOR_API_VERSION",
            "RAPTOR_EMBEDDING_AZURE_ENDPOINT",
            "RAPTOR_EMBEDDING_API_VERSION",
        ] if (env.get("RAPTOR_CLIENT_TYPE") == "azure") else [],
    )
    checks = {
        "official_repo_present": repo.exists(),
        "venv_python_present": python.exists(),
        "official_import_ok": import_check["ok"],
        "model_config_ready": config["ready"],
    }
    return {
        "status": _status_from_checks(checks),
        "checks": checks,
        "config": config,
        "import_check": import_check,
        "runner": "evaluation/official_baselines/run_raptor_official.py",
        "notes": [
            "RAPTOR 官方高层 API 主要返回答案文本，当前 runner 的 evidence 字段只作诊断。",
        ],
    }


def _audit_graphrag(env: dict[str, str], env_sources: dict[str, str], timeout: float) -> dict[str, Any]:
    repo = EXTERNAL_DIR / "graphrag"
    cli = VENV_DIR / "graphrag/bin/graphrag"
    cli_check = _run_check([str(cli), "--help"], timeout=timeout, env=env, cwd=ROOT)
    config = _config_status(
        env,
        env_sources,
        required_any=[
            ["GRAPHRAG_API_KEY", "OPENAI_API_KEY", "GPT54_API_KEY"],
            ["GRAPHRAG_EMBEDDING_API_KEY", "EMBEDDING_API_KEY", "SAM_AZURE_EMBEDDING_API_KEY"],
        ],
        required=[
            "GRAPHRAG_MODEL_PROVIDER",
            "GRAPHRAG_CHAT_MODEL",
            "GRAPHRAG_EMBEDDING_MODEL",
        ],
        required_for_azure=[
            "GRAPHRAG_API_BASE",
            "GRAPHRAG_API_VERSION",
            "GRAPHRAG_CHAT_DEPLOYMENT",
            "GRAPHRAG_EMBEDDING_API_BASE",
            "GRAPHRAG_EMBEDDING_API_VERSION",
            "GRAPHRAG_EMBEDDING_DEPLOYMENT",
        ] if (env.get("GRAPHRAG_MODEL_PROVIDER") == "azure") else [],
    )
    checks = {
        "official_repo_present": repo.exists(),
        "cli_present": cli.exists(),
        "cli_runs": cli_check["ok"],
        "model_config_ready": config["ready"],
    }
    return {
        "status": _status_from_checks(checks),
        "checks": checks,
        "config": config,
        "cli_check": cli_check,
        "runner": "evaluation/official_baselines/run_graphrag_official.py",
        "notes": [
            "GraphRAG 官方 CLI 需要先完成 index，再运行 local/global/drift query。",
        ],
    }


def _audit_hipporag(env: dict[str, str], env_sources: dict[str, str], timeout: float) -> dict[str, Any]:
    repo = EXTERNAL_DIR / "hipporag"
    source = repo / "src"
    python = VENV_DIR / "hipporag/bin/python"
    import_check = _run_check(
        [str(python), "-c", "from hipporag import HippoRAG; print('ok')"],
        timeout=timeout,
        env={**env, "PYTHONPATH": str(source)},
        cwd=ROOT,
    )
    config = _config_status(
        env,
        env_sources,
        required_any=[["OPENAI_API_KEY", "GPT54_API_KEY"]],
        required=[],
        required_for_azure=[],
    )
    checks = {
        "official_repo_present": repo.exists(),
        "venv_python_present": python.exists(),
        "official_import_ok": import_check["ok"],
        "model_config_ready": config["ready"],
    }
    notes = [
        "当前 macOS arm64 本机未完整安装 HippoRAG 官方依赖；官方 requirements 包含更适合 Linux/CUDA 的重依赖。",
    ]
    return {
        "status": _status_from_checks(checks),
        "checks": checks,
        "config": config,
        "import_check": import_check,
        "runner": "evaluation/official_baselines/run_hipporag_official.py",
        "notes": notes,
    }


def _config_status(
    env: dict[str, str],
    env_sources: dict[str, str],
    *,
    required_any: list[list[str]],
    required: list[str],
    required_for_azure: list[str],
) -> dict[str, Any]:
    missing = [key for key in [*required, *required_for_azure] if _missing(env, key)]
    missing_any = [
        group
        for group in required_any
        if not any(not _missing(env, key) for key in group)
    ]
    configured = [
        key
        for key in sorted(set([*required, *required_for_azure, *[item for group in required_any for item in group]]))
        if not _missing(env, key)
    ]
    return {
        "ready": not missing and not missing_any,
        "configured_variables": configured,
        "configured_sources": {
            key: _source_label(env_sources.get(key, "process"))
            for key in configured
        },
        "missing_variables": missing,
        "missing_any_groups": missing_any,
    }


def _apply_official_baseline_aliases(env: dict[str, str], env_sources: dict[str, str]) -> None:
    """把 SAM 本地模型配置映射为官方 baseline 变量名。"""

    alias_pairs = [
        ("EMBEDDING_API_KEY", "SAM_AZURE_EMBEDDING_API_KEY"),
        ("EMBEDDING_BASE_URL", "SAM_AZURE_EMBEDDING_ENDPOINT"),
        ("EMBEDDING_API_VERSION", "SAM_AZURE_EMBEDDING_API_VERSION"),
        ("EMBEDDING_MODEL", "SAM_AZURE_EMBEDDING_MODEL"),
        ("EMBEDDING_DIMENSIONS", "SAM_AZURE_EMBEDDING_DIMENSIONS"),
        ("RAPTOR_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
        ("RAPTOR_EMBEDDING_API_KEY", "EMBEDDING_API_KEY"),
        ("RAPTOR_EMBEDDING_AZURE_ENDPOINT", "EMBEDDING_BASE_URL"),
        ("RAPTOR_EMBEDDING_API_VERSION", "EMBEDDING_API_VERSION"),
        ("RAPTOR_EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSIONS"),
        ("GRAPHRAG_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
        ("GRAPHRAG_EMBEDDING_DEPLOYMENT", "EMBEDDING_MODEL"),
        ("GRAPHRAG_EMBEDDING_API_KEY", "EMBEDDING_API_KEY"),
        ("GRAPHRAG_EMBEDDING_API_BASE", "EMBEDDING_BASE_URL"),
        ("GRAPHRAG_EMBEDDING_API_VERSION", "EMBEDDING_API_VERSION"),
    ]
    for target, source in alias_pairs:
        if not _missing(env, target) or _missing(env, source):
            continue
        env[target] = env[source]
        env_sources[target] = env_sources.get(source, f"alias:{source}")


def _prepared_dataset_summary(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/prepared/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            summaries.append({"path": str(manifest_path), "error": type(exc).__name__})
            continue
        summaries.append(
            {
                "dataset_name": manifest.get("dataset_name"),
                "document_count": manifest.get("document_count"),
                "query_count": manifest.get("query_count"),
                "prepared_dir": str(manifest_path.parent.relative_to(ROOT)),
            }
        )
    return summaries


def _run_check(command: list[str], *, timeout: float, env: dict[str, str], cwd: Path) -> dict[str, Any]:
    if not Path(command[0]).exists():
        return {"ok": False, "error_type": "missing_executable", "message": command[0]}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "timeout", "message": f">{timeout}s"}
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "message": _safe_message(output),
    }


def _status_from_checks(checks: dict[str, bool]) -> str:
    if all(checks.values()):
        return "ready"
    if checks.get("official_repo_present") and any(checks.values()):
        return "partial"
    return "blocked"


def _next_actions(methods: dict[str, dict[str, Any]], prepared_datasets: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    if not prepared_datasets:
        actions.append("先运行 export_sam_for_official.py 导出至少一个 prepared 数据集。")
    for method_id, method in methods.items():
        if method["status"] == "ready":
            actions.append(f"{method['display_name']} 已具备本地运行条件，可选择小样本 limit=1 做 smoke。")
        else:
            missing = method.get("config", {}).get("missing_variables", [])
            missing_any = method.get("config", {}).get("missing_any_groups", [])
            if missing or missing_any:
                actions.append(f"{method['display_name']} 需要补齐模型配置变量后再运行官方 runner。")
            if not method.get("checks", {}).get("official_import_ok", method.get("checks", {}).get("cli_runs", False)):
                actions.append(f"{method['display_name']} 需要先修复官方依赖导入或 CLI 可用性。")
    return actions


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# 官方 baseline 就绪状态审计",
        "",
        f"- 方法数量：{audit['summary']['method_count']}",
        f"- Ready：{audit['summary']['ready_count']}",
        f"- Partial：{audit['summary']['partial_count']}",
        f"- Blocked：{audit['summary']['blocked_count']}",
        f"- 已导出 prepared 数据集：{audit['summary']['prepared_dataset_count']}",
        "",
        "## 方法状态",
        "",
        "| 方法 | 状态 | 官方代码 | 运行入口 | 配置状态 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in audit["methods"].values():
        checks = method["checks"]
        repo_status = "已存在" if checks.get("official_repo_present") else "缺失"
        config_status = "完整" if method["config"]["ready"] else "不完整"
        lines.append(
            f"| {method['display_name']} | {method['status']} | {repo_status} | `{method['runner']}` | {config_status} |"
        )
    lines.extend(["", "## 配置缺口", ""])
    for method in audit["methods"].values():
        config = method["config"]
        missing_parts = []
        if config["missing_variables"]:
            missing_parts.append("缺少变量：" + ", ".join(config["missing_variables"]))
        if config["missing_any_groups"]:
            groups = [" 或 ".join(group) for group in config["missing_any_groups"]]
            missing_parts.append("至少配置其一：" + "; ".join(groups))
        if not missing_parts:
            missing_parts.append("配置变量完整")
        lines.append(f"- {method['display_name']}：{'；'.join(missing_parts)}。")
    lines.extend(["", "## 已导出数据集", ""])
    if audit["prepared_datasets"]:
        for dataset in audit["prepared_datasets"]:
            lines.append(
                f"- {dataset.get('dataset_name')}：documents={dataset.get('document_count')}，queries={dataset.get('query_count')}，目录 `{dataset.get('prepared_dir')}`"
            )
    else:
        lines.append("- 暂无 prepared 数据集。")
    lines.extend(["", "## 下一步", ""])
    lines.extend(f"- {action}" for action in audit["next_actions"])
    lines.append("")
    return "\n".join(lines)


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = _expand_env_value(value, {**os.environ, **values})
    return values


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, character in enumerate(value):
        if character == "'" and not in_double:
            in_single = not in_single
        elif character == '"' and not in_single:
            in_double = not in_double
        elif character == "#" and not in_single and not in_double:
            return value[:index].strip()
    return value


def _expand_env_value(value: str, env: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        return env.get(match.group(1), "")

    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replace, value)


def _missing(env: dict[str, str], key: str) -> bool:
    value = env.get(key)
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.startswith("your-") or stripped.startswith("replace-with-")


def _source_label(source: str) -> str:
    if source == "process":
        return source
    if source.startswith("alias:"):
        return source
    try:
        return str(Path(source).relative_to(ROOT))
    except ValueError:
        return str(Path(source).name)


def _safe_message(message: str) -> str:
    message = re.sub(r"https?://[^\s)]+", "<redacted-url>", message)
    message = re.sub(r"api[_-]?key[=:]\s*[^,\s]+", "api_key=<redacted>", message, flags=re.IGNORECASE)
    return message[:500]


def _resolve_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


if __name__ == "__main__":
    main()
