"""Database connectivity and initialization. CineMatch | Roll Number: 2025AG05135."""
from .config import PROJECT_ROOT, database_url


def connect():
    """Open a PostgreSQL connection using the validated configured database URL."""
    import psycopg
    return psycopg.connect(database_url())


def init_db() -> None:
    """Create Stage 1 PostgreSQL staging and ingestion audit tables."""
    sql = (PROJECT_ROOT / "sql" / "001_stage1.sql").read_text(encoding="utf-8")
    with connect() as conn:
        conn.execute(sql)
