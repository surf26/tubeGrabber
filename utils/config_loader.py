from pathlib import Path

import yaml

# tubeGrabber 项目根目录（本文件在 utils/ 下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: str | Path) -> dict:
    """加载 YAML 文件 相对路径相对于项目根目录"""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config() -> dict:
    """加载主配置 config/default.yaml"""
    return load_yaml("config/default.yaml")
