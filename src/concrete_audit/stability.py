"""固定折外预测的配对来源重采样；描述评估对来源构成的敏感性，不重新训练。"""
import numpy as np


def paired_source_stability(model_rows,baseline_rows,resamples=2000,seed=42):
    """以来源连通组为重采样单位，保留组内全部样本及模型/均值配对关系。"""
    model={r['sample_id']:r for r in model_rows};baseline={r['sample_id']:r for r in baseline_rows}
    if len(model)!=len(model_rows) or len(baseline)!=len(baseline_rows):raise ValueError('预测ID重复')
    if not model or set(model)!=set(baseline):raise ValueError('模型与参照预测不对应')
    if resamples<1:raise ValueError('重采样次数必须为正')
    bygroup={}
    for sample_id,r in model.items():
        b=baseline[sample_id]
        if r['actual']!=b['actual'] or r['group']!=b['group']:raise ValueError('目标或来源组不一致')
        values=[r['actual'],r['predicted'],b['predicted']]
        if not np.isfinite(values).all():raise ValueError('非有限预测')
        bygroup.setdefault(r['group'],[]).append((r,b))
    source=[]
    for group,rows in sorted(bygroup.items()):
        em=np.array([abs(r['actual']-r['predicted']) for r,_ in rows])
        eb=np.array([abs(b['actual']-b['predicted']) for _,b in rows])
        source.append({'group':group,'n':len(rows),'papers':sorted({r['paper'] for r,_ in rows}),
            'model_MAE_MPa':float(em.mean()),'mean_reference_MAE_MPa':float(eb.mean()),
            'delta_MAE_MPa':float(em.mean()-eb.mean())})
    n=np.array([s['n'] for s in source]);em=np.array([s['model_MAE_MPa'] for s in source]);eb=np.array([s['mean_reference_MAE_MPa'] for s in source])
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(source),size=(resamples,len(source)))
    model_macro=em[draw].mean(axis=1);base_macro=eb[draw].mean(axis=1)
    model_pooled=(em[draw]*n[draw]).sum(axis=1)/n[draw].sum(axis=1)
    base_pooled=(eb[draw]*n[draw]).sum(axis=1)/n[draw].sum(axis=1)
    def summary(x):
        q=np.quantile(x,[.025,.5,.975]);return {'p2_5':float(q[0]),'median':float(q[1]),'p97_5':float(q[2])}
    return {'n':len(model),'source_groups':len(source),'resamples':resamples,'seed':seed,
        'sources_improved':int((em<eb).sum()),'sources_worsened':int((em>eb).sum()),
        'model_sample_MAE_MPa':float(np.average(em,weights=n)),'reference_sample_MAE_MPa':float(np.average(eb,weights=n)),
        'model_macro_MAE_MPa':float(em.mean()),'reference_macro_MAE_MPa':float(eb.mean()),
        'source_resampling_percentiles':{'model_sample_MAE_MPa':summary(model_pooled),
            'model_macro_MAE_MPa':summary(model_macro),'paired_sample_MAE_delta_MPa':summary(model_pooled-base_pooled),
            'paired_macro_MAE_delta_MPa':summary(model_macro-base_macro)},
        'sources':source,'interpretation':'固定训练折外预测的来源构成稳定性描述；不重训练，非独立测试置信区间，不能当p值或泛化保证。'}
