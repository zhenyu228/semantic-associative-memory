from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINES_PATH = ROOT / "evaluation/official_baselines/baselines.json"
DEFAULT_EXTERNAL_DIR = ROOT / "evaluation/external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载官方 baseline 仓库到 evaluation/external")
    parser.add_argument("--external-dir", default=str(DEFAULT_EXTERNAL_DIR), help="官方仓库保存目录")
    parser.add_argument(
        "--methods",
        default="raptor,graphrag,hipporag",
        help="逗号分隔的方法名：raptor,graphrag,hipporag",
    )
    parser.add_argument("--update", action="store_true", help="如果仓库已存在，则执行 git pull")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_dir = Path(args.external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    baselines = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    for method in methods:
        if method not in baselines:
            raise ValueError(f"未知官方 baseline：{method}")
        repo_url = baselines[method]["official_repo"]
        target = external_dir / method
        if target.exists():
            if args.update:
                print(f"更新 {method}: {target}")
                subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
            else:
                print(f"跳过 {method}: {target} 已存在")
            continue
        print(f"下载 {method}: {repo_url}")
        subprocess.run(["git", "clone", repo_url, str(target)], check=True)


if __name__ == "__main__":
    main()
