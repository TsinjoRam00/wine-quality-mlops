from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import connection


def connection_kwargs() -> dict:
    uri = (
        os.getenv("MONITORING_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("MLFLOW_BACKEND_STORE_URI")
    )

    if uri and uri.startswith(("postgresql://", "postgres://")):
        parsed = urlparse(uri)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "dbname": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
        }

    return {
        "host": os.getenv("POSTGRES_HOST", "postgres"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "mlflow"),
        "user": os.getenv("POSTGRES_USER", "mlflow"),
        "password": os.getenv("POSTGRES_PASSWORD", "mlflowpassword"),
    }


@contextmanager
def connect() -> Iterator[connection]:
    database = psycopg2.connect(**connection_kwargs())
    try:
        yield database
        database.commit()
    except Exception:
        database.rollback()
        raise
    finally:
        database.close()
