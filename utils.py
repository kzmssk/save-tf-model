from pathlib import Path


def get_export_dir():
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=False, exist_ok=True)
    export_dir = str(tmp_dir / "my_model")
    return export_dir
