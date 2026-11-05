"""Assignment stage: Features.

Public stage interface delegates to the tested ``cinematch`` implementation.
The CLI and shared data contracts stay in ``src/cinematch`` to avoid duplicated
business logic and preserve compatibility with earlier experiments.
"""
from cinematch.stage3 import build_features,feature_status

__all__ = ['build_features', 'feature_status']
