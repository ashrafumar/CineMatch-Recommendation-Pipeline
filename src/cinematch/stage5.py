"""CineMatch Stage 5: leakage-controlled full-catalog evaluation and selection.

DMML Assignment 2 | Roll number / watermark: 2025AG05135.

Training-period features (Stage 3), SVD (Stage 4), and early-validation
ranker fit (Stage 4B) remain frozen. Selection sees LATE validation labels
only; final test requires a stored selection and never tunes on test data.

IMPORTANT: MovieLens has observed ratings, not impressions/exposure logs.
Unrated candidates are UNKNOWN, not verified negatives. Recall/NDCG are
observed-positive retrieval metrics with this important limitation.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit

from .config import PROJECT_ROOT
from .db import connect
from .ranker import feature_vector, load_context, _ranker_meta
from .stage4 import normalized_scores, blend_scores, top_k
from .reporting import export_evaluation_report

ROLL = '2025AG05135'
VERSION = 'full_catalog_late_validation_v1'
LOG = logging.getLogger(__name__)
DEFAULT_WEIGHTS = (0.15, 0.60, 0.25)
# Small, prespecified set; no test-set tuning. Scores tied -> lexical strategy name.
CANDIDATE_WEIGHTS = (
    (0.15, 0.60, 0.25),
    (0.10, 0.75, 0.15),
    (0.25, 0.50, 0.25),
    (0.10, 0.45, 0.45),
    (0.30, 0.40, 0.30),
    (0.05, 0.90, 0.05),
)


def init_stage5():
    """Create evaluation and model-selection tables without modifying earlier schemas."""
    with connect() as conn:
        conn.execute((PROJECT_ROOT/'sql'/'005_stage5.sql').read_text(encoding='utf-8'))


def gain(rank: int) -> float:
    """Compute binary relevance discounted gain at a one-based rank."""
    return 1.0 / np.log2(rank + 1)


def user_metrics(predicted: list[int], relevant: set[int], k: int) -> tuple[float,float,float]:
    """Recall/NDCG/hit among observed positive held-out items; no negatives assumed."""
    if not relevant:
        raise ValueError('User must have >=1 observed positive held-out rating')
    picks = predicted[:k]
    hits = [int(mid in relevant) for mid in picks]
    recall = sum(hits) / len(relevant)
    dcg = sum(h * gain(i + 1) for i,h in enumerate(hits))
    idcg = sum(gain(i + 1) for i in range(min(k,len(relevant))))
    return float(recall), float(dcg / idcg), float(any(hits))


def precision_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    """Fraction of K recommendation slots occupied by observed positive items.

    Uses the fixed K denominator, including when fewer than K results exist.
    Unobserved items are *not* interpreted as verified dislikes; this is a
    partial-label retrieval metric rather than an exposure-aware precision.
    """
    if k < 1:
        raise ValueError("k must be positive")
    return sum(movie_id in relevant for movie_id in predicted[:k]) / k


def intra_list_diversity(movie_ids: list[int], movie_lookup: dict[int, int],
                         genre_matrix: np.ndarray) -> float:
    """Average pairwise Jaccard distance between recommended movie genre sets.

    A missing genre set is excluded from diversity pairs (not interpreted as
    maximally dissimilar). Returns 0.0 for lists with <2 informative movies.
    Movie metadata is a fixed snapshot and is not historically versioned.
    """
    rows = [movie_lookup[movie_id] for movie_id in movie_ids
            if movie_id in movie_lookup]
    valid = [genre_matrix[idx].astype(bool) for idx in rows
             if np.any(genre_matrix[idx])]
    if len(valid) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            intersection = np.count_nonzero(valid[i] & valid[j])
            union = np.count_nonzero(valid[i] | valid[j])
            total += 1.0 - intersection / union
            pairs += 1
    return total / pairs


def effective_weights(weights, known, has_genre):
    """Remove unavailable cold-start signals and renormalize blending weights."""
    w=np.asarray(weights,dtype=np.float32).copy()
    if not known: w[1]=0
    if not has_genre: w[2]=0
    if not np.all(np.isfinite(w)) or np.any(w<0) or w.sum() <= 0:
        raise ValueError('Invalid effective blend weights')
    return w/w.sum()


def candidate_scores(ctx, user_id: int, coef: np.ndarray, intercept: float):
    """Score every catalog movie without loading any held-out interactions."""
    movie_ids=ctx['movie_ids']
    pos=np.searchsorted(ctx['user_ids'], user_id)
    known=bool(pos<len(ctx['user_ids']) and ctx['user_ids'][pos]==user_id)
    has_genre=bool(known and ctx['pref_nonempty'][pos])
    pop=ctx['popularity']
    if known:
        cf=ctx['movie_factors']@ctx['user_factors'][pos]
    else:
        cf=np.zeros(len(movie_ids),dtype=np.float32)
    if has_genre:
        numerator=ctx['movie_genres']@ctx['preferences'][pos]
        denominator=ctx['movie_genre_count']
        genre=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    else:
        genre=np.zeros(len(movie_ids),dtype=np.float32)
    cf_n=normalized_scores(cf)
    genre_n=normalized_scores(genre)
    x=feature_vector(pop,cf,genre,known,has_genre)
    learned=expit(x@coef+intercept).astype(np.float32)
    return known,has_genre,{'popularity':pop,'collaborative':cf_n,
                             'genre':genre_n,'learned_hybrid':learned},(pop,cf,genre)


def rank_scores(movie_ids, scores, seen, k):
    """Return movie identifiers in deterministic top-K order excluding train history."""
    return [mid for mid,_ in top_k(movie_ids,scores,seen,k)]


@dataclass
class Stat:
    users: int=0
    relevant: int=0
    recall: float=0.0
    precision: float=0.0
    diversity: float=0.0
    ndcg: float=0.0
    hit: float=0.0
    recommended: set[int]=field(default_factory=set)

    def add(self, predictions, truth, k, diversity=0.0):
        r,n,h=user_metrics(predictions,truth,k)
        self.users+=1;self.relevant+=len(truth)
        self.recall+=r;self.ndcg+=n;self.hit+=h
        self.precision+=precision_at_k(predictions,truth,k)
        self.diversity+=diversity
        self.recommended.update(predictions[:k])

    def output(self,catalog_size):
        if not self.users:
            return {'evaluated_users':0,'relevant_interactions':0,
                    'recall_at_k':None,'precision_at_k':None,'diversity_at_k':None,'ndcg_at_k':None,'hit_rate_at_k':None,
                    'distinct_recommended_movies':0,'catalog_coverage':None}
        return {'evaluated_users':self.users,'relevant_interactions':self.relevant,
                'recall_at_k':self.recall/self.users,
                'precision_at_k':self.precision/self.users,
                'diversity_at_k':self.diversity/self.users,
                'ndcg_at_k':self.ndcg/self.users,
                'hit_rate_at_k':self.hit/self.users,
                'distinct_recommended_movies':len(self.recommended),
                'catalog_coverage':len(self.recommended)/catalog_size}


def _read_model_metadata(conn,svd_model_run_id,ranker_model_run_id):
    """Validate SVD/ranker lineage and recover chronological partition boundaries."""
    rid,feature_id,path,params=_ranker_meta(conn,ranker_model_run_id)
    if int(params['svd_model_run_id'])!=svd_model_run_id:
        raise ValueError('Ranker was not trained against the requested SVD run')
    with np.load(path,allow_pickle=False) as z:
        if int(z['feature_run_id'])!=feature_id or int(z['svd_model_run_id'])!=svd_model_run_id:
            raise ValueError('Ranker artifact linkage mismatch')
        coef=z['coef'][0].copy();intercept=float(z['intercept'][0]);fit_end=int(z['fit_end'])
    train_end,val_end=conn.execute('''SELECT train_end,validation_end FROM features.feature_runs
       WHERE run_id=%s AND status='COMPLETED' ''',(feature_id,)).fetchone()
    if not (int(train_end)<fit_end<int(val_end)):
        raise ValueError('Ranker fit cutoff outside validation period')
    return feature_id,coef,intercept,int(train_end),int(val_end),fit_end


def _truth_sql(split):
    """Build the positive-label query for a strictly held-out time window."""
    if split=='validation':
        # Crucial: exclude early-validation examples used to fit the ranker.
        return '''SELECT user_id,array_agg(movie_id ORDER BY movie_id)
             FROM core.ratings WHERE rating_timestamp>%s AND rating_timestamp<=%s AND rating>=4
             GROUP BY user_id ORDER BY user_id'''
    if split=='test':
        return '''SELECT user_id,array_agg(movie_id ORDER BY movie_id)
             FROM core.ratings WHERE rating_timestamp>%s AND rating_timestamp<=%s AND rating>=4
             GROUP BY user_id ORDER BY user_id'''
    raise ValueError('Split must be validation or test')


def _validate_weights(weights):
    """Check that three hybrid weights are finite, nonnegative and sum to one."""
    w=np.asarray(weights,dtype=float)
    if w.shape!=(3,) or not np.all(np.isfinite(w)) or np.any(w<0) or not np.isclose(w.sum(),1):
        raise ValueError('Weights must be three nonnegative numbers summing to one')
    return tuple(float(x) for x in w)


def _record_run(feature_id,svd_id,ranker_id,split,config,selection_id=None):
    """Start an auditable PostgreSQL evaluation run and return its identifier."""
    with connect() as conn:
        return conn.execute('''INSERT INTO ml.evaluation_runs(
            feature_run_id,svd_model_run_id,ranker_model_run_id,selection_id,
            evaluation_split,specification_version,status,configuration)
            VALUES (%s,%s,%s,%s,%s,%s,'RUNNING',%s::jsonb)
            RETURNING evaluation_run_id''',(feature_id,svd_id,ranker_id,selection_id,
                       split,VERSION,json.dumps(config))).fetchone()[0]


def _finish_run(run_id,metrics,all_users,all_relevant,selected=None,split='validation', partial=False):
    """Persist legacy evaluation metrics and optional selection transactionally."""
    with connect() as conn:
        for (strategy,segment),stat in metrics.items():
            m=stat
            conn.execute('''INSERT INTO ml.evaluation_metrics
             (evaluation_run_id,strategy,segment,evaluated_users,relevant_interactions,
              recall_at_k,ndcg_at_k,hit_rate_at_k,catalog_coverage,distinct_recommended_movies)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
              (run_id,strategy,segment,m['evaluated_users'],m['relevant_interactions'],
               m['recall_at_k'],m['ndcg_at_k'],m['hit_rate_at_k'],m['catalog_coverage'],
               m['distinct_recommended_movies']))
        conn.execute('''UPDATE ml.evaluation_runs SET status='COMPLETED',
            completed_at=now(),users_evaluated=%s,relevant_interactions=%s
            WHERE evaluation_run_id=%s''',(all_users,all_relevant,run_id))
        if selected and split=='validation':
            strategy,weights,ndcg=selected
            return conn.execute('''INSERT INTO ml.model_selections
                (feature_run_id,validation_evaluation_run_id,selected_strategy,
                 selected_weights,validation_ndcg_at_k,specification_version,selection_basis)
                SELECT feature_run_id,%s,%s,%s::jsonb,%s,%s,%s FROM ml.evaluation_runs
                WHERE evaluation_run_id=%s RETURNING selection_id''',
                (run_id,strategy,json.dumps(list(weights) if weights is not None else None),
                 ndcg,VERSION,'partial_validation' if partial else 'full_validation',run_id)).fetchone()[0]
    return None


def run_evaluation(split: str,svd_model_run_id=1,ranker_model_run_id=2,k=10,
                   max_users=None,selection_id=None,tune=False, allow_partial_selection=False,
                   progress_every=1000):
    """Evaluate all catalog candidates, same eligible users and training-history exclusions.

    Default evaluates ALL users with >=1 held-out positive rating; optional
    max_users enables explicitly labeled smoke runs only. No user sampling by
    default. Evaluation on full catalog requires no negative sampling.
    """
    if split not in {'validation','test'} or k<1 or (max_users is not None and max_users<1):
        raise ValueError('Invalid evaluation configuration')
    if tune and split!='validation':
        raise ValueError('Model selection must use validation, never test')
    if split=='test' and selection_id is None:
        raise ValueError('Final test requires --selection-id from validation tuning')
    if tune and max_users is not None and not allow_partial_selection:
        raise ValueError('Partial tuning requires explicit --allow-partial-selection')
    init_stage5()
    with connect() as conn:
        feature_id,coef,intercept,train_end,val_end,fit_end=_read_model_metadata(
            conn,svd_model_run_id,ranker_model_run_id)
        if split=='test':
            row=conn.execute('''SELECT s.feature_run_id,s.selected_strategy,s.selected_weights,
                      s.selection_basis,e.svd_model_run_id,e.ranker_model_run_id
              FROM ml.model_selections s JOIN ml.evaluation_runs e
                 ON e.evaluation_run_id=s.validation_evaluation_run_id
              WHERE s.selection_id=%s AND e.status='COMPLETED' AND e.evaluation_split='validation'
              ''',(selection_id,)).fetchone()
            if not row or row[0]!=feature_id:
                raise ValueError('Selection absent or does not match feature snapshot')
            if row[3]!='full_validation':
                raise ValueError('Final test requires a full-validation model selection')
            if row[4]!=svd_model_run_id or row[5]!=ranker_model_run_id:
                raise ValueError('Test models must match the validation selection models')
            selected_strategy=row[1]
            selected_weights=_validate_weights(row[2]) if row[2] is not None else None
        else:
            selected_strategy=None
            selected_weights=None
        ctx=load_context(conn,svd_model_run_id)
        if ctx['feature_run_id']!=feature_id:
            raise ValueError('SVD and learned ranker feature snapshots differ')
    base=['popularity','collaborative','genre','fixed_hybrid','learned_hybrid']
    strategies=list(base)
    grid={f'blend_{i}':_validate_weights(w) for i,w in enumerate(CANDIDATE_WEIGHTS)} if tune else {}
    strategies+=list(grid)
    if split=='test' and selected_strategy not in strategies:
        strategies.append(selected_strategy)
    genre_lookup = {int(movie_id): idx for idx, movie_id in enumerate(ctx['movie_ids'])}
    stats={(s,seg):Stat() for s in strategies for seg in ('all','known_user','cold_user')}
    config={'roll_number':ROLL,'k':k,'max_users':max_users,'partial':max_users is not None,
            'candidate_universe':'full_movie_catalog','positive_threshold':4,
            'exclusion':'train_period_ratings_only','ranker_fit_end':fit_end,
            'validation_start_exclusive':fit_end,'test_start_exclusive':val_end,
            'fixed_weights':list(DEFAULT_WEIGHTS),'grid':{s:list(w) for s,w in grid.items()},
            'catalog_size':len(ctx['movie_ids'])}
    run_id=_record_run(feature_id,svd_model_run_id,ranker_model_run_id,split,config,selection_id)
    evaluated=0; relevant_total=0
    try:
        with connect() as conn:
            lower,upper=(fit_end,val_end) if split=='validation' else (val_end,2**63-1)
            with conn.cursor(name=f'cm_eval_{run_id}') as cur:
                cur.execute(_truth_sql(split),(lower,upper))
                while batch:=cur.fetchmany(200):
                    for user_id,movies in batch:
                        if max_users is not None and evaluated>=max_users:break
                        truth=set(map(int,movies))
                        known,has_genre,base_scores,raw=candidate_scores(ctx,int(user_id),coef,intercept)
                        with conn.cursor() as seen_cur:
                            seen_cur.execute('''SELECT movie_id FROM core.ratings
                              WHERE user_id=%s AND rating_timestamp<=%s''',(user_id,train_end))
                            seen={int(r[0]) for r in seen_cur.fetchall()}
                        # Test/validation truth never participates in exclusion or scoring.
                        base_scores['fixed_hybrid']=blend_scores(
                            *raw,effective_weights(DEFAULT_WEIGHTS,known,has_genre))
                        for name,w in grid.items():
                            base_scores[name]=blend_scores(*raw,effective_weights(w,known,has_genre))
                        if split=='test' and selected_strategy not in base_scores:
                            if selected_weights is None:
                                raise ValueError('Selected strategy unavailable in test')
                            base_scores[selected_strategy]=blend_scores(
                                *raw,effective_weights(selected_weights,known,has_genre))
                        segment='known_user' if known else 'cold_user'
                        for name in strategies:
                            ranked=rank_scores(ctx['movie_ids'],base_scores[name],seen,k)
                            diversity = intra_list_diversity(
                                ranked, genre_lookup, ctx['movie_genres'])
                            stats[(name,'all')].add(ranked,truth,k,diversity)
                            stats[(name,segment)].add(ranked,truth,k,diversity)
                        evaluated+=1;relevant_total+=len(truth)
                        if evaluated%progress_every==0:
                            LOG.info('Stage 5 %s: %s users evaluated',split,f'{evaluated:,}')
                    if max_users is not None and evaluated>=max_users:break
        if not evaluated:raise ValueError('No eligible users with observed positive held-out ratings')
        metrics={key:value.output(len(ctx['movie_ids'])) for key,value in stats.items()}
        selected=None
        if tune:
            candidates=['fixed_hybrid','learned_hybrid',*grid]
            best=max(candidates,key=lambda s:(metrics[(s,'all')]['ndcg_at_k'],-candidates.index(s)))
            w=DEFAULT_WEIGHTS if best=='fixed_hybrid' else grid.get(best)
            selected=(best,w,metrics[(best,'all')]['ndcg_at_k'])
        selected_id=_finish_run(run_id,metrics,evaluated,relevant_total,selected,split,
                                partial=max_users is not None)
    except Exception as exc:
        with connect() as conn:
            conn.execute('''UPDATE ml.evaluation_runs SET status='FAILED',completed_at=now(),
                     error_message=%s WHERE evaluation_run_id=%s''',(str(exc)[:3000],run_id))
        raise
    output={'roll_number':ROLL,'evaluation_run_id':run_id,'split':split,
            'users_evaluated':evaluated,'relevant_interactions':relevant_total,
            'partial':max_users is not None,'k':k,
            'metrics':{s:{seg:metrics[(s,seg)] for seg in ('all','known_user','cold_user')}
                       for s in strategies},'selection_id':selected_id if tune else selection_id,
            'selected_strategy':selected[0] if selected else selected_strategy,
            'selected_weights':list(selected[1]) if selected and selected[1] is not None else
                  list(selected_weights) if selected_weights else None,
            'limitations':['Unrated movies are not known negatives',
                           'Ratings do not represent logged impressions',
                           'Movie metadata is a single snapshot, not historically versioned',
                           'Precision@K treats unobserved items as nonhits, not proven negatives',
                           'Diversity@K excludes movies with missing genre labels from pairwise comparisons']}
    directory=PROJECT_ROOT/'artifacts'/'stage5'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'evaluation_{split}_run_{run_id}_{ROLL}.json'
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    output['report_path']=str(path)
    output['exports']=export_evaluation_report(output,directory)
    return output


def stage5_status():
    """List recent evaluation and selection runs from the PostgreSQL registry."""
    init_stage5()
    with connect() as conn:
        evaluations=conn.execute('''SELECT evaluation_run_id,evaluation_split,status,
           users_evaluated,relevant_interactions,selection_id FROM ml.evaluation_runs
           ORDER BY evaluation_run_id DESC LIMIT 10''').fetchall()
        selections=conn.execute('''SELECT selection_id,validation_evaluation_run_id,
              selected_strategy,validation_ndcg_at_k,selection_basis
              FROM ml.model_selections ORDER BY selection_id DESC LIMIT 10''').fetchall()
    return {'evaluations':evaluations,'selections':selections}
