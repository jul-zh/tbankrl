"""Архив исходников для Colab: без данных, результатов и весов."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


root = Path(__file__).resolve().parents[1]
files = [root / "pyproject.toml", root / "constraints_colab.txt"]
for folder in ["src", "tests", "configs", "scripts", "notebooks"]:
    files.extend(p for p in (root / folder).rglob("*")
                 if p.is_file() and p.suffix in {".py", ".json", ".ipynb"} and "__pycache__" not in p.parts)
destination = root / "dist/colab_bundle.zip"
destination.parent.mkdir(exist_ok=True)
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, path.relative_to(root))
print(f"{destination}: {destination.stat().st_size:,} байт")
