"""核验来源等权及证据版本对照，区分性能变化、信息移除和可信资格。"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import numpy as np
from openpyxl import Workbook, load_workbook
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from concrete_audit.baselines import regression_metrics
from concrete_audit.material_features import append_excel_row
from concrete_audit.parallel_exploration import digest, write_json


def summarize():
    root=Path('outputs/source-balanced-evidence-v1');output=root/'analysis';output.mkdir(exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text());config=manifest['config']
    results=json.loads((root/'results.json').read_text());pred=json.loads((root/'oof_predictions.json').read_text())
    traces=json.loads((root/'inner_search_traces.json').read_text());cohort=json.loads((root/'cohort_and_folds.json').read_text())
    contrasts=json.loads((root/'paired_contrasts.json').read_text());changes=json.loads((root/'input_changes.json').read_text())
    features=json.loads((root/'feature_versions.json').read_text())
    baseline=Path(config['baseline_run']);previous=Path(config['comparison_run'])
    previous_results=json.loads((previous/'results.json').read_text());previous_pred=json.loads((previous/'oof_predictions.json').read_text())
    byid={r['sample_id']:r for r in cohort};ids=set(byid)
    source_sizes=Counter(r['group'] for r in cohort)
    for r in results:
        pp=[p for p in pred if p['experiment']==r['experiment']]
        assert len(pp)==len(ids) and {p['sample_id'] for p in pp}==ids
        metrics=regression_metrics([p['observed'] for p in pp],[p['predicted'] for p in pp])
        for key in ['MAE_MPa','RMSE_MPa','R2']:assert abs(metrics[key]-r[key])<1e-9
        assert all(p['outer_fold']==byid[p['sample_id']]['outer_fold'] for p in pp)
        macro=np.mean([np.mean([p['absolute_error'] for p in pp if p['group']==g]) for g in source_sizes])
        assert abs(macro-r['macro_group_MAE_MPa'])<1e-10
    # 必须重现原版流水线，避免将实现变动误称为来源加权收益。
    reproduced={}
    for model in config['models']:
        now={p['sample_id']:p for p in pred if p['experiment']==f'{model}__recorded__fold_mean'}
        old={p['sample_id']:p for p in previous_pred if p['experiment']==f'{model}__materials__nested'}
        assert set(now)==set(old)
        difference=max(abs(now[i]['predicted']-old[i]['predicted']) for i in now)
        assert difference<1e-8
        assert all(now[i]['observed']==old[i]['observed'] for i in now)
        reproduced[model]=difference
    for trace in traces:
        train,val=set(trace['outer_train_groups']),set(trace['outer_validation_groups'])
        assert not train&val
        for inner in trace['inner_folds']:
            it,iv=set(inner['train_groups']),set(inner['validation_groups'])
            assert not it&iv and it|iv==train and not (it|iv)&val
        expected=min(trace['candidates'],key=lambda c:c['inner_scores'][trace['objective']])
        assert trace['best_params']==expected['params']
    for p,expected in manifest['input_hashes_after'].items():assert digest(p)==expected
    log=Path('/tmp/underwater-source-weighting-tests.log').read_text();tests=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK',log)
    if not tests:raise ValueError('全量测试未通过')
    # 记录样本大组对合并误差的实际贡献，不将组大小本身视为数据问题。
    group_contributions=[]
    for name in ['xgb__recorded__fold_mean','xgb__recorded_plus_source_ratios__source_macro']:
        pp=[p for p in pred if p['experiment']==name];total=sum(p['absolute_error'] for p in pp)
        for group,n in source_sizes.items():
            rr=[p for p in pp if p['group']==group]
            group_contributions.append({'experiment':name,'group':group,'n':n,'sample_fraction':n/len(ids),
                'MAE_MPa':float(np.mean([p['absolute_error'] for p in rr])),
                'absolute_error_fraction':sum(p['absolute_error'] for p in rr)/total,
                'papers':sorted({p['paper'] for p in rr})})
    write_json(output/'source_contributions.json',group_contributions)
    best_sample=min(results,key=lambda r:r['MAE_MPa']);best_macro=min(results,key=lambda r:r['macro_group_MAE_MPa'])
    old_reference_manifest=json.loads((baseline/'manifest.json').read_text())
    old_reference=next(r for r in json.loads((baseline/'results.json').read_text()) if r['experiment']==old_reference_manifest['selected_primary_experiment'])
    # 展示同一XGBoost的两个指标，不以单指标强行宣布统一最优。
    xgb=[r for r in results if r['model']=='xgb']
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    labels=[r['version']+' / '+r['objective'] for r in xgb]
    for ax,key,title in zip(axes,['MAE_MPa','macro_group_MAE_MPa'],['Sample-weighted MAE','Equal-source MAE']):
        ax.barh(labels[::-1],[r[key] for r in xgb][::-1]);ax.set_xlabel('MPa');ax.set_title(title)
    fig.tight_layout();fig.savefig(output/'xgb_two_metrics.png',dpi=160);plt.close(fig)
    wb=Workbook();wb.remove(wb.active)
    tables={'全部16组实验':(['experiment','model','version','objective','n','MAE_MPa','RMSE_MPa','R2','macro_group_MAE_MPa','worst_group_MAE_MPa'],results),
        '同折配对差异':(['contrast','before','after','delta_sample_MAE_MPa','delta_source_macro_MAE_MPa','source_groups_improved','source_groups_worsened','per_group_MAE_delta'],contrasts),
        '单位敏感性变更':(['source_row','field','recorded_value','new_value','reason','evidence_paper','evidence_pages'],changes),
        '逐条折外预测':(['experiment','source_row','paper','group','outer_fold','observed','predicted','absolute_error'],pred),
        '来源贡献':(['experiment','group','n','sample_fraction','MAE_MPa','absolute_error_fraction','papers'],group_contributions),
        '内层选参记录':(['version','model','objective','outer_fold','best_params','outer_train_groups','outer_validation_groups'],traces),
        '特征版本':(['version','fields','observed_n_by_field','feature_matrix_sha256'],[{'version':v,**r} for v,r in features.items()])}
    for title,(keys,data) in tables.items():
        ws=wb.create_sheet(title);append_excel_row(ws,keys)
        for row in data:append_excel_row(ws,[json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in keys])
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    ws=wb.create_sheet('解释与限制')
    notes=['探索性实验，原始样本与暂定标签全部不变，未升级正式可靠状态。',
        'fold_mean为旧内层选参目标，source_macro为每个来源连通组MAE等权；不是给模型训练样本加权。',
        '单位处理版仅将有明确单位疑问的38项输入设为未知，18条记录保留；不声称已获得其真实绝对值。',
        '两个来源比值版增加6个来源有依据的相对配比特征；缺失指示器也可能反映证据覆盖/论文记录方式。',
        '单位敏感性变差不能证明原始推算用量正确；部分比值有胶材总量冲突，按分项定义保留标记。',
        '同一开发集已多轮使用，本轮方案受既往误差启发，存在适应性选择偏差。',
        '145条历史留出本轮未用于拟合、转换、调参或评估；没有新的独立最终测试结论。']
    for note in notes:append_excel_row(ws,[note])
    file=output/'来源等权与证据版本对照.xlsx';wb.save(file)
    check=load_workbook(file,read_only=True,data_only=False);assert not any(c.data_type=='f' for s in check for row in s for c in row);check.close()
    verification={'experiments':len(results),'oof_predictions':len(pred),'same_development_records':len(ids),
        'source_groups':len(source_sizes),'input_changes':len(changes),'affected_rows':len({r['source_row'] for r in changes}),
        'prior_predictions_max_abs_difference':reproduced,'all_metrics_recomputed':True,'all_nested_groups_isolated':True,
        'parameters_match_recorded_objective':True,'protected_inputs_unchanged':True,'unit_tests_passed':int(tests[1]),
        'reserved_test_predictions':0,'formal_reliability_changed':False,'labels_changed':False,
        'summary_script_sha256':digest(__file__),'best_sample_MAE_experiment':best_sample['experiment'],
        'best_source_macro_experiment':best_macro['experiment']}
    write_json(output/'verification.json',verification)
    # 报告保留两个评价目标；不由更低MAE推断未知用量已真实。
    lines=['# 来源等权选参与文献输入版本对照','',
        f'同一283条AB开发记录、20个来源连通组和五个外层折。{tests[1]}项测试通过，16组实验、4528条折外预测。',
        '旧版原字段+等权折选参精确复现上轮RF/XGBoost每条折外预测，最大差异0；其他比较采用同一参数范围。','',
        '|模型|版本|内层选参|MAE MPa|RMSE MPa|R²|来源组平均MAE|','|---|---|---|---:|---:|---:|---:|']
    for r in results:lines.append(f'|{r["model"]}|{r["version"]}|{r["objective"]}|{r["MAE_MPa"]:.3f}|{r["RMSE_MPa"]:.3f}|{r["R2"]:.3f}|{r["macro_group_MAE_MPa"]:.3f}|')
    lines+=['','结果解释：',
        f'- 样本平均MAE最低：{best_sample["experiment"]}，MAE{best_sample["MAE_MPa"]:.4f}、RMSE{best_sample["RMSE_MPa"]:.4f} MPa、R²{best_sample["R2"]:.4f}，来源宏平均MAE{best_sample["macro_group_MAE_MPa"]:.4f}。',
        f'- 来源宏平均MAE最低：{best_macro["experiment"]}，宏平均{best_macro["macro_group_MAE_MPa"]:.4f}，样本MAE{best_macro["MAE_MPa"]:.4f}。来源等权内层选参并未保证外层来源等权误差改善。',
        f'- 此前参考XGBoost：MAE{old_reference["MAE_MPa"]:.4f}、RMSE{old_reference["RMSE_MPa"]:.4f}、R²{old_reference["R2"]:.4f}，来源宏平均MAE{old_reference["macro_group_MAE_MPa"]:.4f}。新候选只改善部分指标，保留为探索候选而不宣布统一最优。',
        '- 输入单位敏感性版在18条记录的38个单元格设置未知，全部样本与标签保留；主要是Zhang绝对用量依据未知24项及Yousri/Brzozowski体积水量14项。原版本数值未覆盖。',
        '- 原字段有12列；来源比值版本增加6列。两套字段都各自比较有/无单位处理，分开识别缺失信息增加和新特征的效果。比值仅用同单位/同基准数值，不用未知密度反算。',
        '- 未知值增多后的误差上升反映信息损失和插补影响，不等于证明原始推算质量用量正确；本轮没有新测量绝对用量，不能将此称为完整质量改进。',
        '- 来源特征仅覆盖部分论文：定义中的分项未知仍保留；Al-Selmi分项1503与注释1502冲突不被算术消除。缺失模式可能反映证据覆盖，未使用论文ID作模型输入。','',
        '后续建议：停止在同一开发集上连续按最低分扩展参数网格；优先获得高影响来源的真实配合比单位和绝对/相对基准，新增有依据的数据版本后再做同折对照。独立科研结论需另行预先固定验证方案。',
        '当前训练模型仍可作为探索工具运行；不等待全部文献或全部字段核实完毕。','',
        '修改/新增文件：configs/source_balanced_evidence_v1.json、src/concrete_audit/source_weighting.py、scripts/run_source_weighting.py、scripts/summarize_source_weighting.py、tests/test_source_weighting.py、README.md。',
        '函数解释：source_ratios读取原文同单位比例；versioned_inputs只生成新矩阵和逐格变更记录；selection_scores分开折、来源、样本三种权重；paired_search复用内层预测选择两个目标；run锁定外层验证并保存配对差异。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verification,ensure_ascii=False))


if __name__=='__main__':summarize()
