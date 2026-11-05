"""CineMatch Stage 5 report tests | Roll Number: 2025AG05135."""
from cinematch.reporting import export_evaluation_report

def test_report(tmp_path):
    report={'split':'validation','evaluation_run_id':7,'roll_number':'2025AG05135',
            'users_evaluated':2,'relevant_interactions':3,'k':10,'partial':True,
            'selection_id':None,'selected_strategy':None,'selected_weights':None,
            'limitations':['unobserved != negative'],
            'metrics':{'popularity':{'all':{'evaluated_users':2,'relevant_interactions':3,
            'recall_at_k':.2,'ndcg_at_k':.1,'hit_rate_at_k':.3,
            'catalog_coverage':.04,'distinct_recommended_movies':4}}}}
    result=export_evaluation_report(report,tmp_path)
    assert '2025AG05135' in (tmp_path/result['markdown'].split('/')[-1]).read_text()
    assert 'popularity' in (tmp_path/result['csv'].split('/')[-1]).read_text()
