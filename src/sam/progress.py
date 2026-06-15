from __future__ import annotations


def progress_iter(iterable, *, total: int | None = None, desc: str = "", enabled: bool = True, progress_factory=None):
    """统一实验进度条入口；未安装 tqdm 时自动退化为普通迭代。"""

    if not enabled:
        return iterable
    factory = progress_factory
    if factory is None:
        try:
            from tqdm import tqdm as factory
        except ImportError:
            return iterable
    return factory(iterable, total=total, desc=desc)
