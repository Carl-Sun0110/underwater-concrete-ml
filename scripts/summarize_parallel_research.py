"""汇总真实输出并核对折外覆盖、来源隔离、原始输入哈希和指标。"""
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from openpyxl import load_workbook
from concrete_audit.baselines import regression_metrics
from concrete_audit.parallel_exploration import digest, write_json


def summarize():
    root=Path('outputs/parallel-exploration-v1');explain=Path('outputs/parallel-explanation-v1')
    manifest=json.loads((root/'manifest.json').read_text())
    results=json.loads((root/'results.json').read_text());pred=json.loads((root/'oof_predictions.json').read_text())
    snapshot=json.loads((root/'quality_snapshot.json').read_text());meta={r['sample_id']:r for r in snapshot}
    reserved={r['sample_id'] for r in snapshot if r['split']=='reserved_historical_test'}
    assert not reserved.intersection(p['sample_id'] for p in pred)
    for result in results:
        pp=[r for r in pred if r['experiment']==result['experiment']]
        assert len(pp)==result['n']==len({r['sample_id'] for r in pp})
        expected=regression_metrics([r['observed'] for r in pp],[r['predicted'] for r in pp])
        for key in ['MAE_MPa','RMSE_MPa','R2']:assert abs(expected[key]-result[key])<1e-10
        for fold in result['folds']:assert not set(fold['train_groups'])&set(fold['validation_groups'])
        for r in pp:assert meta[r['sample_id']]['grade'] in result['cohort']
    for path,expected in manifest['input_hashes_before'].items():assert digest(path)==expected
    sv=json.loads((explain/'oof_shap.json').read_text())
    best=manifest['selected_primary_experiment'];by_id={r['sample_id']:r for r in pred if r['experiment']==best}
    assert len(sv['rows'])==len(by_id)
    assert max(abs(r['prediction']-by_id[r['sample_id']]['predicted']) for r in sv['rows'])<1e-5
    wb=load_workbook(explain/'探索实验汇总.xlsx',read_only=True,data_only=False)
    assert all(cell.data_type!='f' for sheet in wb for row in sheet for cell in row)
    workbook_rows={s.title:s.max_row for s in wb};wb.close()
    # 文件名匹配只报告可获得性，不把“存在同名PDF”当成已完成证据核查。
    def normalized(name):
        return re.sub(r'[\W_]+','',re.sub(r'\(\d+\)(?=\.pdf$)','',name).casefold())
    available={normalized(p.name):str(p) for p in Path('/workspace/attachments').rglob('*.pdf')}
    sources=sorted({r['paper'] for r in snapshot})
    inventory=[{'source_name':s,'available_path':available.get(normalized(s)),
                'status':'文件名匹配，非内容核实' if normalized(s) in available else '未匹配本地文件名，可能缺失或另有命名'} for s in sources]
    write_json(root/'paper_availability.json',inventory)
    tests=Path('/tmp/underwater-parallel-tests.log').read_text()
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK',tests)
    if not match:raise ValueError('未找到本次全量测试通过记录')
    verified={'experiments':len(results),'oof_predictions':len(pred),'reserved_test_predictions':0,
              'metric_recalculation_passed':True,'fold_group_isolation_passed':True,
              'protected_input_hashes_unchanged':True,'SHAP_matches_oof_predictions':True,
              'excel_formula_cells':0,'workbook_rows':workbook_rows,
              'unit_tests_passed':int(match[1]),'unit_test_seconds':float(match[2]),
              'all_empty_column_warning':'部分训练折有完全未知特征，StandardScaler产生RuntimeWarning；插补保留列，所有预测与指标有限。该列没有训练信息，不代表真实用量为零。'}
    write_json(root/'verification.json',verified)
    selected=next(r for r in results if r['experiment']==best)
    ab_mat=next(r for r in results if r['experiment']=='AB__materials__median__xgb__unknown')
    abc_mat=next(r for r in results if r['experiment']=='ABC__materials__median__xgb__unknown')
    facts=[f"- 候选{manifest['candidate_n']}条：{manifest['candidate_grades']}；开发{manifest['development_n']}条：{manifest['development_grades']}；历史留出{manifest['reserved_n']}条。",
           f"- 候选涉及{len(sources)}个论文文件名，匹配到当前本地PDF {sum(bool(r['available_path']) for r in inventory)}个；其余逐项见paper_availability.json，不因未匹配就阻断全部建模。",
           f"- {len(results)}组实验，{len(pred)}条折外预测记录；98项测试通过，原始Excel和输入版本哈希全部不变。",
           f"- 主要模型：{best}；MAE {selected['MAE_MPa']:.4f}、RMSE {selected['RMSE_MPa']:.4f} MPa、R² {selected['R2']:.4f}。",
           f"- 各来源组等权平均MAE {selected['macro_group_MAE_MPa']:.4f} MPa，最差来源组MAE {max(g['MAE_MPa'] for g in selected['groups']):.4f} MPa。不能仅看合并样本分数。",
           f"- 同样B级验证样本：材料+中位数XGBoost加入C级训练样本后MAE从{ab_mat['shared_AB_metrics']['MAE_MPa']:.4f}变成{abc_mat['shared_AB_metrics']['MAE_MPa']:.4f} MPa，暂未显示收益。该比较是一次固定划分结果，不作显著性结论。",
           '- 训练折全未知列产生数值警告，插补后预测均为有限值，验证通过。下一版可显式处理全未知列以减少日志噪声，无需删除样本。',
           '- 已有折外置换重要性、283条折外SHAP、逐论文误差及Excel汇总。SHAP中缺失标记也有贡献，可能反映论文记录习惯，需关注迁移风险。',
           '- 原始测量强度并非全部逐项核实，换算仍为0.95/0.8/0.76暂定规则；本次B级不是正式可信资格。未给未入选的其他624条记录强行分级或伪造标签。',
           '- 部分材料原表量纲和绝对用量依据仍待核查；外加剂14字段只作敏感性。旧严格128条实验与本次283条不同，不能直接把分数差异解释为方法提升。',
           '- 下一轮先做内层分组调参、无量纲配合比特征、CatBoost/SVR/MLP对照；文献轨道同步优先核查高误差且多样本来源，保留可信高强度样本。']
    files=['configs/parallel_exploration_v1.json','src/concrete_audit/parallel_exploration.py',
           'scripts/run_parallel_exploration.py','scripts/explain_parallel_exploration.py',
           'scripts/summarize_parallel_research.py','tests/test_parallel_exploration.py',
           'requirements-exploration.txt','docs/parallel_research_plan.md','README.md']
    (root/'implementation_summary.md').write_text('# 本轮实施与运行结果\n\n'+'\n'.join(facts)
        +'\n\n修改/新增代码与说明文件：\n\n'+'\n'.join('- '+f for f in files)
        +'\n\n关键函数：feature_matrix解析输入和零值策略；quality保留正式状态并生成研究分层；allocate隔离论文与重复配合比；make_pipeline确保训练折内预处理；run运行统一外层分组对照；explain解释折外预测；summarize重新核对结果。\n')
    print(json.dumps(verified,ensure_ascii=False))


if __name__=='__main__':summarize()
