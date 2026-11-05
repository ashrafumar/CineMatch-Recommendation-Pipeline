"""Assignment stage: Ingestion.

Public stage interface delegates to the tested ``cinematch`` implementation.
The CLI and shared data contracts stay in ``src/cinematch`` to avoid duplicated
business logic and preserve compatibility with earlier experiments.
"""
from cinematch.ingest import ingest_one,reconcile

__all__ = ['ingest_one', 'reconcile']
