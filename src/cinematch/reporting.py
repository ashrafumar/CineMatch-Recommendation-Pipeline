"""CineMatch Stage 5 reports | Roll Number watermark: 2025AG05135."""
from __future__ import annotations
import csv
from pathlib import Path


def export_evaluation_report(report: dict, output_dir: Path) -> dict:
    """Write a readable markdown and flat CSV, plus comparison chart if available."""
    output_dir.mkdir(parents=True,exist_ok=True)
    split=report['split']; rid=report['evaluation_run_id']; roll=report['roll_number']
    prefix=f'evaluation_{split}_run_{rid}_{roll}'
    csv_path=output_dir/f'{prefix}.csv'
    fields=['roll_number','strategy','segment','evaluated_users','relevant_interactions','recall_at_k','precision_at_k','diversity_at_k',
            'ndcg_at_k','hit_rate_at_k','catalog_coverage','distinct_recommended_movies']
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for strat,segments in report['metrics'].items():
            for segment,metrics in segments.items():
                writer.writerow({'roll_number':roll,'strategy':strat,'segment':segment,**metrics})
    md_path=output_dir/f'{prefix}.md'
    lines=[f'# CineMatch evaluation — {split}',f'**Roll Number / watermark:** {roll}',
           f'**Evaluation run:** {rid}',
           f'**Eligible users:** {report["users_evaluated"]:,}',
           f'**Observed positive ratings:** {report["relevant_interactions"]:,}',
           f'**Top-K:** {report["k"]}',
           f'**Partial smoke run:** {report["partial"]}',
           '', '## Comparison (all eligible users)',
           '| Strategy | Precision@K | Recall@K | NDCG@K | Diversity@K | Hit rate@K | Catalog coverage |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,segments in report['metrics'].items():
        m=segments['all']
        lines.append('| '+ ' | '.join([name,*[f'{m.get(x):.6f}' if m.get(x) is not None else 'N/A'
               for x in ('precision_at_k','recall_at_k','ndcg_at_k','diversity_at_k','hit_rate_at_k','catalog_coverage')]])+' |')
    lines+=['','## Selection',
            f'- Model selection ID: {report.get("selection_id")}',
            f'- Selected strategy: {report.get("selected_strategy")}',
            f'- Selected weights: {report.get("selected_weights")}',
            '', '## Limitations']
    lines.extend(f'- {item}' for item in report['limitations'])
    lines+=['','## Interpretation',
            'These are offline retrieval metrics against held-out positively rated movies. '
            'Unrated movies were not necessarily shown or disliked. '
            'No causal uplift or engagement gain is established.','']
    md_path.write_text('\n'.join(lines),encoding='utf-8')
    outputs={'csv':str(csv_path),'markdown':str(md_path)}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=list(report['metrics'])
        values=[report['metrics'][n]['all']['ndcg_at_k'] for n in names]
        fig,ax=plt.subplots(figsize=(max(8,len(names)*1.1),4.8))
        ax.bar(range(len(names)),values)
        ax.set_xticks(range(len(names)),names,rotation=35,ha='right')
        ax.set_ylabel(f'NDCG@{report["k"]}')
        ax.set_title(f'CineMatch {split.capitalize()} (run {rid}) — {roll}')
        ax.set_ylim(0,max(0.01,max(values)*1.15))
        fig.tight_layout()
        chart=output_dir/f'{prefix}_ndcg.png'
        fig.savefig(chart,dpi=160)
        plt.close(fig)
        outputs['chart']=str(chart)
    except ImportError:
        pass
    return outputs
