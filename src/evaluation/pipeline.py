"""Assignment stage: Evaluation.

Public stage interface delegates to the tested ``cinematch`` implementation.
The CLI and shared data contracts stay in ``src/cinematch`` to avoid duplicated
business logic and preserve compatibility with earlier experiments.
"""
from cinematch.stage5 import run_evaluation

__all__ = ['run_evaluation']
