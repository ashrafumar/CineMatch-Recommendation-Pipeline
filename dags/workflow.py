"""Synchronous dependency DAG for the DM4ML recommendation assignment.

This is a structured Python workflow (not an Apache Airflow deployment).
Run with ``cinematch run-all`` after installing the project. All training
is performed from scratch when the PostgreSQL database is empty.
"""
from cinematch.orchestrate import run_pipeline

if __name__ == "__main__":
    print(run_pipeline())
