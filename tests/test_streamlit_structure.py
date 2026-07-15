from __future__ import annotations

import py_compile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STREAMLIT_FILES = [
    PROJECT_ROOT / "streamlit_app" / "Home.py",
    PROJECT_ROOT / "streamlit_app" / "pages" / "1_Prediction.py",
    PROJECT_ROOT / "streamlit_app" / "pages" / "2_Feedback.py",
    PROJECT_ROOT / "streamlit_app" / "pages" / "3_Monitoring.py",
    PROJECT_ROOT / "streamlit_app" / "services" / "api_client.py",
]


def test_required_streamlit_files_exist() -> None:
    missing_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in STREAMLIT_FILES
        if not path.is_file()
    ]

    assert not missing_files, (
        "Fichiers Streamlit manquants : "
        + ", ".join(missing_files)
    )


@pytest.mark.parametrize("source_file", STREAMLIT_FILES)
def test_streamlit_python_files_compile(
    source_file: Path,
    tmp_path: Path,
) -> None:
    compiled_file = tmp_path / f"{source_file.stem}.pyc"

    py_compile.compile(
        str(source_file),
        cfile=str(compiled_file),
        doraise=True,
    )


def test_streamlit_dockerfile_exists() -> None:
    dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.streamlit"

    assert dockerfile.is_file()


def test_streamlit_compose_service_exists() -> None:
    compose_file = PROJECT_ROOT / "docker-compose.streamlit.yml"
    content = compose_file.read_text(encoding="utf-8")

    assert "streamlit:" in content
    assert "8501:8501" in content
    assert "API_BASE_URL" in content
