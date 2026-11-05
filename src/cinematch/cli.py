"""CineMatch command line. Roll Number: 2025AG05135."""
import argparse
import json
import logging
import sys
from .config import source_dir
from .db import init_db
from .ingest import SOURCES, ingest_one, reconcile, status
from .verify import verify_dataset
from .stage2 import init_stage2, run_profile, prepare, stage2_status
from .stage3 import init_stage3, build_features, feature_status, write_feature_report
from .stage4 import init_stage4, train_models, recommend, model_status, write_stage4_report
from .ranker import train_ranker, recommend_ranker
from .stage5 import init_stage5, run_evaluation, stage5_status
from .orchestrate import run_pipeline


def make_parser() -> argparse.ArgumentParser:
    """Build all stage-specific command-line parsers and typed arguments."""
    parser = argparse.ArgumentParser(prog="cinematch", description="MovieLens 32M Stage 1 — Roll Number 2025AG05135")
    parser.add_argument("--source-dir", help="Directory containing extracted ml-32m CSVs")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="Create Stage 1 tables in an existing PostgreSQL database")
    sub.add_parser("verify", help="Verify official pinned MD5 checksums and headers for all four CSVs")
    ingestion = sub.add_parser("ingest", help="Load movies, links, and ratings with PostgreSQL COPY")
    ingestion.add_argument("--kind", choices=[*SOURCES, "all"], default="all")
    ingestion.add_argument("--force", action="store_true", help="Reload even if source already completed")
    sub.add_parser("status", help="Display ingestion ledger")
    sub.add_parser("reconcile", help="Compare database counts with the source baseline")
    sub.add_parser("init-stage2", help="Create quality and canonical Stage 2 schemas")
    sub.add_parser("profile", help="Run complete read-only SQL profiling and save JSON report")
    prep = sub.add_parser("prepare", help="Validate and build canonical tables transactionally")
    prep.add_argument("--force", action="store_true", help="Rebuild canonical tables even if reconciled")
    sub.add_parser("stage2-status", help="Show last profile and preparation runs")
    sub.add_parser("init-stage3", help="Create versioned feature snapshot tables")
    sub.add_parser("build-features", help="Compute chronological split and leakage-safe feature snapshot")
    sub.add_parser("stage3-status", help="Inspect latest completed feature snapshot")
    sub.add_parser("init-stage4", help="Create Stage 4 model registry and recommendation tables")
    training=sub.add_parser("train-models", help="Train sparse collaborative SVD from the full training partition")
    training.add_argument("--rank",type=int,default=32)
    training.add_argument("--feature-run-id",type=int)
    rec=sub.add_parser("recommend", help="Generate popularity, collaborative, genre, and hybrid Top-K rankings")
    rec.add_argument("--user-id",type=int,required=True)
    rec.add_argument("--model-run-id",type=int)
    rec.add_argument("--top-k",type=int,default=10)
    newtrain=sub.add_parser("train-ranker",help="Train second model on early validation ratings only")
    newtrain.add_argument("--svd-model-run-id",type=int,default=1)
    newtrain.add_argument("--chunk-size",type=int,default=100000)
    learned=sub.add_parser("recommend-ranker",help="Generate Top-K from trained learned hybrid ranker")
    learned.add_argument("--user-id",type=int,required=True)
    learned.add_argument("--model-run-id",type=int)
    learned.add_argument("--top-k",type=int,default=10)
    sub.add_parser("stage4-status",help="Display model registry")
    sub.add_parser("init-stage5",help="Create evaluation and selection audit tables")
    ev=sub.add_parser("evaluate",help="Evaluate all strategies on later validation (no tuning)")
    ev.add_argument("--split",choices=["validation"],default="validation")
    ev.add_argument("--max-users",type=int,help="Optional partial smoke evaluation")
    ev.add_argument("--top-k",type=int,default=10)
    tune=sub.add_parser("tune-hybrid",help="Compare all models, select on later validation only")
    tune.add_argument("--max-users",type=int,help="Optional partial exploratory run")
    tune.add_argument("--allow-partial-selection",action="store_true",
                      help="Explicitly allow partial selection (never eligible for final test)")
    tune.add_argument("--top-k",type=int,default=10)
    final=sub.add_parser("final-test",help="Run once on untouched test using full validation selection")
    final.add_argument("--selection-id",type=int,required=True)
    final.add_argument("--top-k",type=int,default=10)
    sub.add_parser("stage5-status",help="View evaluation runs and selections")
    allcmd=sub.add_parser("run-all",help="Run complete pipeline with resumable stages")
    allcmd.add_argument("--rank",type=int,default=32)
    allcmd.add_argument("--max-users",type=int)
    allcmd.add_argument("--allow-partial-selection",action="store_true")
    allcmd.add_argument("--final-test",action="store_true",
                        help="Only after full validation tuning; not default")
    return parser


def main(argv=None) -> int:
    """Parse and dispatch CLI commands, returning a shell-compatible exit code."""
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    directory = source_dir(args.source_dir)
    try:
        if args.command == "verify":
            for filename, checksum in verify_dataset(directory).items():
                print(f"PASS {filename} MD5={checksum}")
        elif args.command == "init-db":
            init_db()
            print("Stage 1 CineMatch schema initialized")
        elif args.command == "ingest":
            kinds = list(SOURCES) if args.kind == "all" else [args.kind]
            for kind in kinds:
                ingest_one(kind, directory, force=args.force)
        elif args.command == "status":
            rows = status()
            if not rows:
                print("No ingestion batches yet")
            for row in rows:
                print(row)
        elif args.command == "reconcile":
            results = reconcile()
            for file, (expected, actual, passed) in results.items():
                print(f"{'PASS' if passed else 'FAIL'} {file}: expected={expected:,} loaded={actual:,}")
            if not all(result[2] for result in results.values()):
                return 2
        elif args.command == "init-stage2":
            init_stage2()
            print("Stage 2 CineMatch schema initialized")
        elif args.command == "profile":
            report = run_profile()
            print(f"Profile #{report['run_id']}: {'PASS' if report['passed'] else 'FAIL'}")
            for check in report['metrics']:
                print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['value']:,}")
            if not report['passed']:
                return 2
        elif args.command == "prepare":
            counts = prepare(force=args.force)
            for name, count in counts.items():
                print(f"PASS core.{name}: {count:,} rows")
        elif args.command == "stage2-status":
            results = stage2_status()
            for category, rows in results.items():
                print(category)
                for row in rows:
                    print("  ", row)
        elif args.command == "init-stage3":
            init_stage3()
            print("Stage 3 feature schema initialized")
        elif args.command == "build-features":
            result = build_features()
            report_path = write_feature_report(result)
            print(f"PASS feature snapshot #{result['run_id']}")
            for key, value in result.items():
                print(f"  {key}: {value}")
            print(f"Report saved: {report_path}")
        elif args.command == "stage3-status":
            result = feature_status()
            if not result:
                print("No completed feature snapshots")
            else:
                for key, value in result.items():
                    print(f"{key}: {value}")
        elif args.command == "init-stage4":
            init_stage4()
            print("Stage 4 model registry initialized")
        elif args.command == "train-models":
            report=train_models(feature_run_id=args.feature_run_id,rank=args.rank)
            print(json.dumps(report,indent=2))
            print("Saved:",write_stage4_report(report,f"training_run_{report['model_run_id']}"))
        elif args.command == "recommend":
            report=recommend(args.user_id,model_run_id=args.model_run_id,k=args.top_k)
            print(json.dumps(report,indent=2))
            print("Saved:",write_stage4_report(report,f"recommend_user_{args.user_id}"))
        elif args.command == "train-ranker":
            report=train_ranker(svd_model_run_id=args.svd_model_run_id,chunk_size=args.chunk_size)
            print(json.dumps(report,indent=2))
            print("Saved:",write_stage4_report(report,f"ranker_training_run_{report['model_run_id']}"))
        elif args.command == "recommend-ranker":
            report=recommend_ranker(args.user_id,model_run_id=args.model_run_id,top_n=args.top_k)
            print(json.dumps(report,indent=2))
            print("Saved:",write_stage4_report(report,f"ranker_recommend_user_{args.user_id}"))
        elif args.command == "stage4-status":
            for row in model_status():print(row)
        elif args.command == "init-stage5":
            init_stage5()
            print("Stage 5 evaluation schema initialized")
        elif args.command == "evaluate":
            report=run_evaluation("validation",max_users=args.max_users,k=args.top_k)
            print(json.dumps(report,indent=2))
        elif args.command == "tune-hybrid":
            report=run_evaluation("validation",max_users=args.max_users,k=args.top_k,
                tune=True,allow_partial_selection=args.allow_partial_selection)
            print(json.dumps(report,indent=2))
        elif args.command == "final-test":
            report=run_evaluation("test",selection_id=args.selection_id,k=args.top_k)
            print(json.dumps(report,indent=2))
        elif args.command == "stage5-status":
            for group,rows in stage5_status().items():
                print(group)
                for row in rows:print(" ",row)
        elif args.command == "run-all":
            path=run_pipeline(rank=args.rank,max_users=args.max_users,
                include_final_test=args.final_test,
                allow_partial_selection=args.allow_partial_selection)
            print("Pipeline manifest:",path)
        return 0
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
