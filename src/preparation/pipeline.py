"""Assignment stage: Preparation.

Public stage interface delegates to the tested ``cinematch`` implementation.
The CLI and shared data contracts stay in ``src/cinematch`` to avoid duplicated
business logic and preserve compatibility with earlier experiments.
"""
from cinematch.stage2 import prepare

__all__ = ['prepare']
