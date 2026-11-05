"""CineMatch end-to-end orchestrator | Roll Number watermark: 2025AG05135.

Runs dependencies synchronously and records a reproducible manifest. Already
completed source ingestions, features, and model training are reused by the
existing stage implementations. The final test is explicit, not automatic.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime,timezone
from pathlib import Path
from .config import PROJECT_ROOT, source_dir
from .verify import verify_dataset
from .db import init_db
from .ingest import SOURCES, ingest_one, reconcile
from .stage2 import init_stage2, run_profile, prepare
from .stage3 import init_stage3, build_features, feature_status
from .stage4 import init_stage4, train_models
from .ranker import train_ranker
from .stage5 import init_stage5, run_evaluation

LOG=logging.getLogger(__name__)
ROLL='2025AG05135'


def run_pipeline(rank=32, max_users=None, include_final_test=False, allow_partial_selection=False):
    """Run the dependency-ordered recommendation workflow and persist a manifest.

    Args:
        rank: Latent dimension for collaborative SVD on a fresh training run.
        max_users: Optional validation smoke-test limit. Omit for full validation.
        include_final_test: Evaluate the held-out test only after full validation.
        allow_partial_selection: Permit *exploratory* partial selection; never final test.

    Returns:
        Absolute path to the JSON run manifest. The manifest is written after
        every completed step and upon failure for postmortem inspection.

    Notes:
        Existing Stage 1-4 implementations handle reuse. By design validation
        evaluation currently runs again; it is not memoized. A fresh PostgreSQL
        database and empty model-artifact directory result in from-scratch fit.
    """
    manifest={'roll_number':ROLL,'started_at':datetime.now(timezone.utc).isoformat(),
              'steps':[],'status':'RUNNING','rank':rank,'max_users':max_users}
    out=PROJECT_ROOT/'artifacts'/'stage5'
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/f'pipeline_manifest_{ROLL}.json'
    def write():
        manifest_path.write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    def step(name,func):
        LOG.info('Pipeline step: %s',name)
        result=func()
        manifest['steps'].append({'name':name,'result':result,'status':'COMPLETED'})
        write()
        return result
    try:
        step('verify',lambda: verify_dataset(source_dir()))
        step('init_stage1',init_db)
        for name in SOURCES:
            step(f'ingest_{name}',lambda n=name:ingest_one(n,source_dir()))
        reconciliation=step('reconcile',reconcile)
        if not all(v[2] for v in reconciliation.values()):
            raise ValueError('Source-to-database reconciliation failed')
        step('init_stage2',init_stage2)
        profile=step('profile',run_profile)
        if not profile['passed']:
            raise ValueError('Blocking profiling check failed')
        step('prepare',prepare)
        step('init_stage3',init_stage3)
        step('build_features',build_features)
        step('init_stage4',init_stage4)
        svd=step('train_svd',lambda:train_models(rank=rank))
        svd_id=svd['model_run_id']
        ranker=step('train_ranker',lambda:train_ranker(svd_model_run_id=svd_id))
        step('init_stage5',init_stage5)
        validation=step('validate_and_select',lambda:run_evaluation(
            'validation',svd_model_run_id=svd_id,
            ranker_model_run_id=ranker['model_run_id'],max_users=max_users,
            tune=True,allow_partial_selection=allow_partial_selection))
        if include_final_test:
            if validation['partial']:
                raise ValueError('Cannot finalize test after partial validation')
            step('final_test',lambda:run_evaluation('test',svd_model_run_id=svd_id,
                 ranker_model_run_id=ranker['model_run_id'],selection_id=validation['selection_id']))
        manifest['status']='COMPLETED'
    except Exception as exc:
        manifest['status']='FAILED'
        manifest['error']=f'{type(exc).__name__}: {exc}'
        LOG.exception('Pipeline failed')
        raise
    finally:
        manifest['completed_at']=datetime.now(timezone.utc).isoformat()
        write()
    return manifest_path
