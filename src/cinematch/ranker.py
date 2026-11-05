'''Second trained CineMatch model: leakage-controlled learned hybrid ranker.
Roll number / watermark 2025AG05135.

The SVD and all Stage 3 behavioral features come from pre-2018 train interactions.
Ranker is fit on the EARLY validation time segment only. The LATE validation
segment and all test interactions are never read during fitting.
No text, tags, images, or video processing.
'''
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from .config import PROJECT_ROOT
from .db import connect
from .stage4 import (_load_model, latest_features, top_k, normalized_scores)

ROLL = '2025AG05135'
VERSION = 'learned_hybrid_early_validation_v1'
LOG = logging.getLogger(__name__)


def fit_cutoff(train_end: int, validation_end: int, share: float = 0.7) -> int:
    '''70% of the validation TIME WINDOW, not 70% of validation rows.'''
    if not train_end < validation_end or not 0 < share < 1:
        raise ValueError('Invalid time boundaries/share')
    return train_end + int((validation_end - train_end) * share)


def feature_vector(pop, cf, genre, has_cf, has_genre):
    '''Same deterministic feature mapping used for fit and inference.'''
    p, c, g = np.broadcast_arrays(np.asarray(pop,dtype=np.float32),
                                   np.asarray(cf,dtype=np.float32),
                                   np.asarray(genre,dtype=np.float32))
    shape = p.shape
    return np.stack((p, np.where(has_cf, expit(np.clip(c, -20, 20)), 0.),
                     np.clip(g,0.,1.),
                     np.broadcast_to(np.asarray(has_cf,dtype=np.float32),shape),
                     np.broadcast_to(np.asarray(has_genre,dtype=np.float32),shape)),axis=-1).astype(np.float32)


def load_context(conn, svd_model_run_id=None):
    """Load SVD factors and training-only feature arrays for hybrid inference."""
    run_id,path,feature_id = _load_model(conn,svd_model_run_id)
    with np.load(path,allow_pickle=False) as z:
        uid=z['user_ids'].copy(); mid=z['movie_ids'].copy()
        uf=z['user_factors'].copy(); mf=z['movie_factors'].copy()
        if int(z['feature_run_id']) != feature_id:
            raise ValueError('SVD feature snapshot does not match registered run')
    movie_rows=conn.execute('''SELECT movie_id, COALESCE(bayesian_rating,0), rating_count
          FROM features.movie_features WHERE run_id=%s ORDER BY movie_id''',(feature_id,)).fetchall()
    if not np.array_equal(mid,np.array([r[0] for r in movie_rows],dtype=np.int32)):
        raise ValueError('Movie feature ordering does not match SVD artifact')
    popularity=np.asarray([max(0,float(r[1]))*np.log1p(int(r[2])) if r[2] else 0 for r in movie_rows],dtype=np.float32)
    popularity=normalized_scores(popularity)
    genres=conn.execute('SELECT genre FROM features.genre_catalog WHERE run_id=%s ORDER BY genre',(feature_id,)).fetchall()
    lookup={g[0]:i for i,g in enumerate(genres)}
    movie_genres=np.zeros((len(mid),len(lookup)),dtype=np.float32)
    with conn.cursor(name='cm_ranker_mgenres') as cur:
        cur.execute('SELECT movie_id,genre FROM features.movie_genres WHERE run_id=%s',(feature_id,))
        while batch:=cur.fetchmany(50000):
            for movie_id,genre in batch:
                i=np.searchsorted(mid,movie_id)
                if i<len(mid) and mid[i]==movie_id:movie_genres[i,lookup[genre]]=1
    movie_genre_count=movie_genres.sum(axis=1)
    preferences=np.zeros((len(uid),len(lookup)),dtype=np.float32)
    with conn.cursor(name='cm_ranker_uprefs') as cur:
        cur.execute('SELECT user_id,genre,positive_weight FROM features.user_genre_preferences WHERE run_id=%s',(feature_id,))
        while batch:=cur.fetchmany(100000):
            for user_id,genre,weight in batch:
                i=np.searchsorted(uid,user_id)
                if i<len(uid) and uid[i]==user_id: preferences[i,lookup[genre]]=float(weight)
    sums=preferences.sum(axis=1)
    np.divide(preferences,sums[:,None],out=preferences,where=sums[:,None]>0)
    return {'svd_model_run_id':run_id,'feature_run_id':feature_id,'user_ids':uid,'movie_ids':mid,
            'user_factors':uf,'movie_factors':mf,'popularity':popularity,
            'movie_genres':movie_genres,'movie_genre_count':movie_genre_count,
            'preferences':preferences,'pref_nonempty':sums>0}


def vectorize_pairs(ctx, user_ids, movie_ids):
    """Build consistent numeric ranker features for user–movie pairs."""
    users=np.asarray(user_ids,dtype=np.int32)
    movies=np.asarray(movie_ids,dtype=np.int32)
    ui=np.searchsorted(ctx['user_ids'],users)
    mi=np.searchsorted(ctx['movie_ids'],movies)
    valid_m=(mi<len(ctx['movie_ids']))
    if not np.all(valid_m):raise ValueError('Validation/test movie missing from canonical catalog')
    if not np.array_equal(ctx['movie_ids'][mi],movies):raise ValueError('Bad movie id')
    known=(ui<len(ctx['user_ids']))
    known[known] = ctx['user_ids'][ui[known]]==users[known]
    safe_ui=np.where(known,ui,0)
    cf=np.einsum('ij,ij->i',ctx['user_factors'][safe_ui],ctx['movie_factors'][mi])
    cf[~known]=0
    good_genre=known & ctx['pref_nonempty'][safe_ui]
    genre=np.sum(ctx['preferences'][safe_ui]*ctx['movie_genres'][mi],axis=1)
    den=ctx['movie_genre_count'][mi]
    genre=np.divide(genre,den,out=np.zeros_like(genre),where=den>0)
    genre[~good_genre]=0
    return feature_vector(ctx['popularity'][mi],cf,genre,known,good_genre)


def _ranker_meta(conn, model_run_id):
    """Fetch a completed learned-ranker model and its run metadata."""
    row=conn.execute('''SELECT model_run_id,feature_run_id,artifact_path,parameters
      FROM ml.model_runs WHERE model_run_id=%s AND algorithm='learned_hybrid_sgd'
      AND status='COMPLETED' ''',(model_run_id,)).fetchone()
    if not row:raise ValueError('Completed learned hybrid run not found')
    return row


def train_ranker(svd_model_run_id=None, chunk_size=100000, force=False):
    '''Fit on ALL observed ratings in early-validation time range; no arbitrary sampling.
    Label 1 when explicit rating >=4; label 0 for ratings <4 (not unobserved items).
    '''
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    with connect() as conn:
        ctx=load_context(conn,svd_model_run_id)
        feature_id=ctx['feature_run_id']
        train_end,val_end=conn.execute('SELECT train_end,validation_end FROM features.feature_runs WHERE run_id=%s',(feature_id,)).fetchone()
        cutoff=fit_cutoff(int(train_end),int(val_end))
        params={'svd_model_run_id':ctx['svd_model_run_id'],'fit_start_exclusive':int(train_end),
                'fit_end_inclusive':cutoff,'validation_end':int(val_end),
                'algorithm':'SGDClassifier(log_loss)','label':'rating>=4.0',
                'feature_order':['popularity','sigmoid_svd','genre','known_user','genre_profile_present'],
                'random_seed':2025,'epochs':1,'training_rows':'all_early_validation_ratings'}
        row=conn.execute('''SELECT model_run_id,status,parameters,artifact_path FROM ml.model_runs
              WHERE feature_run_id=%s AND specification_version=%s AND algorithm='learned_hybrid_sgd' ''',(feature_id,VERSION)).fetchone()
        if row:
            rid,status,old_params,old_path=row
            if old_params!=params:raise ValueError('Existing model run has different parameters; update VERSION')
            if status=='COMPLETED' and Path(old_path).exists() and not force:
                return {'model_run_id':rid,'reused':True,'artifact_path':old_path,'fit_end':cutoff}
            conn.execute("UPDATE ml.model_runs SET status='RUNNING',error_message=NULL,completed_at=NULL WHERE model_run_id=%s",(rid,))
        else:
            rid=conn.execute('''INSERT INTO ml.model_runs (feature_run_id,specification_version,status,algorithm,parameters)
                    VALUES(%s,%s,'RUNNING','learned_hybrid_sgd',%s::jsonb) RETURNING model_run_id''',
                    (feature_id,VERSION,json.dumps(params))).fetchone()[0]
        # Commit RUNNING registration before a possibly long fit.
        conn.commit()
    fitted=SGDClassifier(loss='log_loss',alpha=0.0001,random_state=2025,
                         learning_rate='optimal',average=True)
    count=0; positives=0
    with connect() as conn:
        ctx=load_context(conn,svd_model_run_id)
        with conn.cursor(name='cm_ranker_training') as cur:
            cur.execute('''SELECT user_id,movie_id,rating FROM core.ratings
                WHERE rating_timestamp > %s AND rating_timestamp <= %s
                ORDER BY rating_timestamp,user_id,movie_id''',(train_end,cutoff))
            while batch:=cur.fetchmany(chunk_size):
                x=vectorize_pairs(ctx,[r[0] for r in batch],[r[1] for r in batch])
                y=np.asarray([float(r[2])>=4.0 for r in batch],dtype=np.int32)
                fitted.partial_fit(x,y,classes=np.array([0,1],dtype=np.int32))
                count+=len(batch);positives+=int(y.sum())
                if count%1000000<chunk_size:LOG.info('Hybrid ranker fitted %s ratings',f'{count:,}')
    if count==0:raise ValueError('Early validation segment has no ratings')
    artifact=PROJECT_ROOT/'artifacts'/'stage4'/f'learned_hybrid_run_{rid}_{ROLL}.npz'
    artifact.parent.mkdir(parents=True,exist_ok=True)
    tmp=artifact.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,coef=fitted.coef_,intercept=fitted.intercept_,
                        feature_run_id=np.asarray(feature_id),svd_model_run_id=np.asarray(ctx['svd_model_run_id']),
                        fit_end=np.asarray(cutoff))
    tmp.replace(artifact)
    with connect() as conn:
        conn.execute('''UPDATE ml.model_runs SET status='COMPLETED',completed_at=now(),
          artifact_path=%s,train_interactions=%s,train_positive_interactions=%s WHERE model_run_id=%s''',
          (str(artifact),count,positives,rid))
    return {'model_run_id':rid,'svd_model_run_id':ctx['svd_model_run_id'],
            'feature_run_id':feature_id,'fit_end':cutoff,'fit_rows':count,
            'positive_rows':positives,'artifact_path':str(artifact),'reused':False}


def recommend_ranker(user_id, model_run_id=None, top_n=10):
    """Rank unseen catalog movies using saved logistic hybrid coefficients."""
    if top_n<1:raise ValueError('top_n must be positive')
    with connect() as conn:
        if model_run_id is None:
            row=conn.execute('''SELECT model_run_id FROM ml.model_runs
               WHERE status='COMPLETED' AND algorithm='learned_hybrid_sgd'
               ORDER BY model_run_id DESC LIMIT 1''').fetchone()
            if not row:raise ValueError('No completed learned hybrid ranker')
            model_run_id=int(row[0])
        rid,feature_id,path,params=_ranker_meta(conn,model_run_id)
        ctx=load_context(conn,params['svd_model_run_id'])
        if ctx['feature_run_id']!=feature_id:raise ValueError('Ranker/SVD feature snapshot mismatch')
        with np.load(path,allow_pickle=False) as z:
            if int(z['feature_run_id'])!=feature_id:raise ValueError('Ranker artifact feature mismatch')
            coef=z['coef'][0].copy();intercept=float(z['intercept'][0])
        movie_ids=ctx['movie_ids']
        movie_user_ids=np.full(len(movie_ids),user_id,dtype=np.int32)
        # Chunk vectorization avoids enormous temporary matrices.
        probs=np.empty(len(movie_ids),dtype=np.float32)
        for start in range(0,len(movie_ids),10000):
            stop=min(start+10000,len(movie_ids))
            x=vectorize_pairs(ctx,movie_user_ids[start:stop],movie_ids[start:stop])
            probs[start:stop]=expit(x @ coef + intercept)
        train_end=conn.execute('SELECT train_end FROM features.feature_runs WHERE run_id=%s',(feature_id,)).fetchone()[0]
        seen={r[0] for r in conn.execute('''SELECT movie_id FROM core.ratings
                      WHERE user_id=%s AND rating_timestamp<=%s''',(user_id,train_end)).fetchall()}
        results=top_k(movie_ids,probs,seen,top_n)
        return {'roll_number':ROLL,'model_run_id':rid,'svd_model_run_id':ctx['svd_model_run_id'],
                'feature_run_id':feature_id,'user_id':user_id,'top_k':results,
                'score_description':'P(rating >= 4 | observed rating, candidate features); not click/conversion probability'}
