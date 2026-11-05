"""CineMatch Stage 4 models; Roll Number / watermark 2025AG05135.

Training-only data: never read held-out interaction labels when fitting models.
Pure structured genre and rating features. No NLP or movie-media processing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .config import PROJECT_ROOT
from .db import connect

LOG = logging.getLogger(__name__)
VERSION = 'positive_sparse_svd_v1'
ROLL = '2025AG05135'
SEED = 2025


def init_stage4():
    """Create model registry and scored recommendation tables."""
    with connect() as conn:
        conn.execute((PROJECT_ROOT / 'sql' / '004_stage4.sql').read_text(encoding='utf-8'))


def latest_features(conn, feature_run_id=None):
    """Resolve an existing completed versioned feature snapshot."""
    if feature_run_id is None:
        row = conn.execute("SELECT run_id,train_end FROM features.feature_runs WHERE status='COMPLETED' ORDER BY run_id DESC LIMIT 1").fetchone()
    else:
        row = conn.execute("SELECT run_id,train_end FROM features.feature_runs WHERE run_id=%s AND status='COMPLETED'", (feature_run_id,)).fetchone()
    if not row:
        raise ValueError('No completed Stage 3 feature snapshot found')
    return int(row[0]), int(row[1])


def positive_only_score(rating: float, threshold: float = 4.0) -> float:
    """Positive implicit preference signal. Ratings below threshold are not positives."""
    return float(rating >= threshold)


def make_factors(matrix: sparse.csr_matrix, rank: int, seed: int=SEED):
    """Truncated sparse SVD with deterministic initialization and decreasing singular values."""
    max_rank = min(matrix.shape) - 1
    if max_rank < 1 or matrix.nnz == 0:
        raise ValueError('Insufficient nonzero interactions for collaborative training')
    k = min(rank, max_rank)
    rng = np.random.default_rng(seed)
    # Sparse linear algebra; never densify the full user-item matrix.
    u, s, vt = svds(matrix.astype(np.float32), k=k, which='LM',
                    v0=rng.standard_normal(min(matrix.shape)).astype(np.float32),
                    tol=1e-3, solver='arpack')
    order = np.argsort(s)[::-1]
    s = s[order].astype(np.float32)
    return (u[:,order] * np.sqrt(s)).astype(np.float32), (vt[order,:].T * np.sqrt(s)).astype(np.float32)


def genre_scores(genre_matrix: sparse.csr_matrix, user_weights: np.ndarray) -> np.ndarray:
    """Normalized weighted genre similarity. Missing genres yield zero score."""
    numerator = genre_matrix @ user_weights
    denom = np.asarray(genre_matrix.sum(axis=1)).reshape(-1)
    return np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=np.float32), where=denom>0)


def normalized_scores(values: np.ndarray) -> np.ndarray:
    """Apply per-candidate min–max normalization, handling constant vectors."""
    v = np.asarray(values, dtype=np.float32)
    mn, mx = float(np.min(v)), float(np.max(v))
    if mx-mn < 1e-8:
        return np.zeros_like(v)
    return (v-mn)/(mx-mn)


def blend_scores(pop, collaborative, genre, weights=(0.15, 0.60, 0.25)):
    """Combine popularity, collaborative, and genre signals with validated weights."""
    if len(weights)!=3 or any(w<0 for w in weights) or not np.isclose(sum(weights),1):
        raise ValueError('Hybrid weights must be nonnegative and sum to one')
    return sum(w*normalized_scores(x) for w,x in zip(weights,(pop,collaborative,genre)))


def top_k(movie_ids, scores, seen_movie_ids=(), k=10):
    """Rank eligible movies with stable movie-ID tie-breaking and history exclusion."""
    scores = np.asarray(scores,dtype=np.float32).copy()
    movie_ids = np.asarray(movie_ids,dtype=np.int32)
    if len(movie_ids)!=len(scores):
        raise ValueError('movie_ids and scores have different lengths')
    excluded = np.isin(movie_ids, list(seen_movie_ids))
    scores[excluded] = -np.inf
    eligible = np.flatnonzero(np.isfinite(scores))
    if not len(eligible):
        return []
    # Deterministic ties: ascending movie_id.
    ranked = np.lexsort((movie_ids[eligible], -scores[eligible]))[:k]
    return [(int(movie_ids[i]), float(scores[i])) for i in eligible[ranked]]


def _register(conn, feature_id, algorithm, parameters):
    """Create or resume a versioned model run, checking parameter consistency."""
    row = conn.execute('''SELECT model_run_id,status,artifact_path,parameters FROM ml.model_runs
        WHERE feature_run_id=%s AND specification_version=%s AND algorithm=%s''',
        (feature_id, VERSION, algorithm)).fetchone()
    if row and row[3] != parameters:
        raise ValueError('Existing model version uses different parameters; change VERSION to create a new model specification')
    if row and row[1]=='COMPLETED' and (not row[2] or Path(row[2]).exists()):
        return row[0], True
    if row:
        conn.execute('''UPDATE ml.model_runs SET status='RUNNING',created_at=now(),
            completed_at=NULL,error_message=NULL,parameters=%s::jsonb
            WHERE model_run_id=%s''', (json.dumps(parameters),row[0]))
        return row[0], False
    new_id = conn.execute('''INSERT INTO ml.model_runs(feature_run_id,specification_version,
        status,algorithm,parameters) VALUES(%s,%s,'RUNNING',%s,%s::jsonb) RETURNING model_run_id''',
        (feature_id, VERSION, algorithm,json.dumps(parameters))).fetchone()[0]
    return new_id, False


def train_models(feature_run_id=None, rank=32, chunk_size=100000):
    """Train SVD on all eligible training positive interactions; store frozen artifacts.

    Uses one server-side cursor and sparse COO chunks. No arbitrary dataset sampling.
    """
    init_stage4()
    if rank < 1 or chunk_size < 1:
        raise ValueError('rank and chunk_size must be positive')
    with connect() as conn:
        feature_id, train_end = latest_features(conn, feature_run_id)
        movie_rows = conn.execute('''SELECT movie_id,COALESCE(bayesian_rating,0),rating_count
                FROM features.movie_features WHERE run_id=%s ORDER BY movie_id''',(feature_id,)).fetchall()
        user_rows = conn.execute('''SELECT user_id FROM features.user_features
                WHERE run_id=%s ORDER BY user_id''',(feature_id,)).fetchall()
        movie_ids = np.array([r[0] for r in movie_rows], dtype=np.int32)
        user_ids = np.array([r[0] for r in user_rows], dtype=np.int32)
        if not len(movie_ids) or not len(user_ids):
            raise ValueError('Feature snapshot has no users or movies')
        params = {'rank':rank,'seed':SEED,'threshold':4.0,'all_training_positives':True,
                  'feature_run_id':feature_id,'train_end':train_end}
        model_run_id, reuse = _register(conn,feature_id,'sparse_svd',params)
        artifact = PROJECT_ROOT / 'artifacts' / 'stage4' / f'svd_run_{model_run_id}_{ROLL}.npz'
        if reuse:
            LOG.info('Reusing completed model run %s',model_run_id)
            return {'model_run_id':model_run_id,'feature_run_id':feature_id,'artifact_path':str(artifact),'reused':True}
        LOG.info('Streaming positive training interactions for feature run %s',feature_id)
        row_chunks, col_chunks = [], []
        total_positive = 0
        # Named cursor prevents loading the result set in PostgreSQL client memory.
        with conn.cursor(name=f'cm_svd_{model_run_id}') as cursor:
            cursor.execute('''SELECT user_id,movie_id FROM core.ratings
                WHERE rating_timestamp<=%s AND rating>=4''',(train_end,))
            while True:
                batch = cursor.fetchmany(chunk_size)
                if not batch: break
                u = np.fromiter((r[0] for r in batch),dtype=np.int32,count=len(batch))
                m = np.fromiter((r[1] for r in batch),dtype=np.int32,count=len(batch))
                ur = np.searchsorted(user_ids,u)
                mc = np.searchsorted(movie_ids,m)
                if np.any(ur>=len(user_ids)) or np.any(mc>=len(movie_ids)):
                    raise ValueError('Training interaction missing matching feature entity')
                if not (np.array_equal(user_ids[ur],u) and np.array_equal(movie_ids[mc],m)):
                    raise ValueError('Training interaction references unknown user/movie')
                row_chunks.append(ur.astype(np.int32))
                col_chunks.append(mc.astype(np.int32))
                total_positive += len(batch)
                if total_positive % 1000000 < chunk_size:
                    LOG.info('Read %s positive interactions',f'{total_positive:,}')
        rr = np.concatenate(row_chunks) if row_chunks else np.array([],dtype=np.int32)
        cc = np.concatenate(col_chunks) if col_chunks else np.array([],dtype=np.int32)
        del row_chunks,col_chunks
        matrix = sparse.coo_matrix((np.ones(len(rr),dtype=np.float32),(rr,cc)),
                                   shape=(len(user_ids),len(movie_ids))).tocsr()
        del rr,cc
        if matrix.nnz != total_positive:
            raise ValueError(f'Nonunique positive pairs detected: {matrix.nnz} != {total_positive}')
        LOG.info('Fitting sparse SVD: matrix=%s nonzero=%s rank=%s',matrix.shape,matrix.nnz,rank)
        user_factors, movie_factors = make_factors(matrix,rank)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        tmp = artifact.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, user_ids=user_ids,movie_ids=movie_ids,
                            user_factors=user_factors,movie_factors=movie_factors,
                            feature_run_id=np.array(feature_id,dtype=np.int64),
                            train_end=np.array(train_end,dtype=np.int64))
        tmp.replace(artifact)
        train_count=conn.execute('SELECT train_rows FROM features.feature_runs WHERE run_id=%s',(feature_id,)).fetchone()[0]
        conn.execute('''UPDATE ml.model_runs SET status='COMPLETED',completed_at=now(),
            artifact_path=%s,train_interactions=%s,train_positive_interactions=%s
            WHERE model_run_id=%s''',(str(artifact),train_count,total_positive,model_run_id))
        return {'model_run_id':model_run_id,'feature_run_id':feature_id,'artifact_path':str(artifact),
                'train_positive_interactions':total_positive,'train_interactions':train_count,'reused':False}


def _load_model(conn, model_run_id=None):
    """Load a completed SVD model registration and validate artifact availability."""
    if model_run_id is None:
        row=conn.execute('''SELECT model_run_id,artifact_path,feature_run_id FROM ml.model_runs
             WHERE status='COMPLETED' AND algorithm='sparse_svd'
             ORDER BY model_run_id DESC LIMIT 1''').fetchone()
    else:
        row=conn.execute('''SELECT model_run_id,artifact_path,feature_run_id FROM ml.model_runs
             WHERE status='COMPLETED' AND algorithm='sparse_svd' AND model_run_id=%s''',(model_run_id,)).fetchone()
    if row is None: raise ValueError('No completed SVD model; run cinematch train-models')
    if not Path(row[1]).exists(): raise FileNotFoundError(f'Model artifact is missing: {row[1]}')
    return int(row[0]),str(row[1]),int(row[2])


def recommend(user_id: int, model_run_id=None, k=10, weights=(0.15,0.60,0.25), persist=True):
    """Rank full movie catalog, excluding only movies rated *in training*.

    New users have no collaborative profile and fall back to popularity.
    Do not use validation/test ratings, even for seen-item exclusions.
    """
    init_stage4()
    with connect() as conn:
        run_id,path,feature_id=_load_model(conn,model_run_id)
        stored=np.load(path,allow_pickle=False)
        if int(stored['feature_run_id'])!=feature_id:
            raise ValueError('Artifact feature run mismatch')
        movie_ids=stored['movie_ids']
        user_ids=stored['user_ids']
        rows=conn.execute('''SELECT movie_id,COALESCE(bayesian_rating,0),rating_count,has_genres
            FROM features.movie_features WHERE run_id=%s ORDER BY movie_id''',(feature_id,)).fetchall()
        if not np.array_equal(movie_ids,np.array([r[0] for r in rows],dtype=np.int32)):
            raise ValueError('Feature catalog and model artifact do not match')
        pop=np.array([max(0,r[1])*np.log1p(r[2]) if r[2] else 0 for r in rows],dtype=np.float32)
        uidx=np.searchsorted(user_ids,user_id)
        known=uidx<len(user_ids) and user_ids[uidx]==user_id
        cf=(stored['movie_factors']@stored['user_factors'][uidx]) if known else np.zeros(len(movie_ids),dtype=np.float32)
        # Structured genres only, no titles/tags/video content.
        genre_rows=conn.execute('''SELECT movie_id,genre FROM features.movie_genres WHERE run_id=%s''',(feature_id,)).fetchall()
        vocab=sorted({genre for _,genre in genre_rows})
        gmap={g:i for i,g in enumerate(vocab)}
        movie_index={int(m):i for i,m in enumerate(movie_ids)}
        gm=sparse.csr_matrix((np.ones(len(genre_rows),dtype=np.float32),
                  ([movie_index[m] for m,_ in genre_rows],[gmap[g] for _,g in genre_rows])),
                  shape=(len(movie_ids),len(vocab)))
        prefs=conn.execute('''SELECT genre,positive_weight FROM features.user_genre_preferences
                       WHERE run_id=%s AND user_id=%s''',(feature_id,user_id)).fetchall()
        uw=np.zeros(len(vocab),dtype=np.float32)
        for genre,value in prefs:
            if genre in gmap: uw[gmap[genre]]=float(value)
        if uw.sum()>0: uw/=uw.sum()
        genre=genre_scores(gm,uw)
        # Fallback weights for new user / no genre history: do not dilute popularity.
        if not known:
            cf[:]=0
        if not prefs:
            genre[:]=0
        effective=np.array(weights,dtype=np.float32)
        if not known:effective[1]=0
        if not prefs:effective[2]=0
        effective/=effective.sum()
        score_sets={'popularity':normalized_scores(pop),'collaborative':normalized_scores(cf),
                    'genre':normalized_scores(genre)}
        score_sets['hybrid']=blend_scores(pop,cf,genre,effective)
        train_end=conn.execute('SELECT train_end FROM features.feature_runs WHERE run_id=%s',(feature_id,)).fetchone()[0]
        seen={r[0] for r in conn.execute('''SELECT movie_id FROM core.ratings
                          WHERE user_id=%s AND rating_timestamp<=%s''',(user_id,train_end)).fetchall()}
        results={name:top_k(movie_ids,s,seen,k) for name,s in score_sets.items()}
        if persist:
            # Hybrid is the final deployed ranking; component rankings are independently reproducible.
            conn.execute('DELETE FROM ml.recommendations WHERE model_run_id=%s AND user_id=%s',(run_id,user_id))
            with conn.cursor() as cur:
                cur.executemany('''INSERT INTO ml.recommendations
                       (model_run_id,user_id,movie_id,rank,score,strategy)
                       VALUES(%s,%s,%s,%s,%s,'hybrid')''',
                    [(run_id,user_id,mid,i+1,score) for i,(mid,score) in enumerate(results['hybrid'])])
        return {'model_run_id':run_id,'feature_run_id':feature_id,'user_id':user_id,
                'known_training_user':bool(known),'weights':effective.tolist(),'top_k':results}


def model_status():
    """List registered training runs and artifact references."""
    init_stage4()
    with connect() as conn:
        return conn.execute('''SELECT model_run_id,feature_run_id,algorithm,status,
            train_interactions,train_positive_interactions,artifact_path
            FROM ml.model_runs ORDER BY model_run_id DESC''').fetchall()


def write_stage4_report(data, name):
    """Export an auditable summary of the trained recommendation model."""
    directory=PROJECT_ROOT/'artifacts'/'stage4'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{name}_{ROLL}.json'
    path.write_text(json.dumps({'roll_number':ROLL,**data},indent=2,default=str),encoding='utf-8')
    return path
