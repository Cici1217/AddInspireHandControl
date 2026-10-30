"""Read-only directory navigation for the local model picker."""

from pathlib import Path
from .config import ROOT


def list_directories(path, *, project_root=ROOT):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("请输入目录路径")
    if not isinstance(project_root, (str, Path)):
        raise ValueError('项目目录无效')
    directory = (ROOT / project_root / Path(path).expanduser())
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError("目录不存在，请检查路径或选择其他目录")
    children = []
    for child in directory.iterdir():
        if not child.name.startswith('.') and child.is_dir():
            children.append({"name": child.name, "path": str(child)})
    children.sort(key=lambda row: row['name'].casefold())
    return {
        "path": str(directory),
        "parent": str(directory.parent) if directory.parent != directory else None,
        "directories": children,
    }
