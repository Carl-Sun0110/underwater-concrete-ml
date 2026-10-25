"""审计固定范围训练的同折结果，导出指标、来源误差与完整折外预测。"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import tempfile
import platform
from importlib.metadata import version
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.baselines import digest
from concrete_audit.scope_results import validate_fold_layout, aggregate_predictions
from concrete_audit.material_features import append_excel_row


def run(output):
    output = Path(output)
    scope_folder = ROOT / 'outputs/casting-scope-gate-v1'
    views = json.loads((scope_folder / 'scope_views.json').read_text())
    planned = json.loads((scope_folder / 'training_plan.json').read_text())['plans']
    split_path = ROOT / 'outputs/exploratory-baselines-v2/split_manifest.json'
    split = json.loads(split_path.read_text())['rows']
    group_by_id = {r['sample_id']: r['leakage_group'] for r in split}
    test_ids = {r['sample_id'] for r in split if r['split'] == 'test'}
    labels_path = ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json'
    labels = {r['sample_id']: r for r in json.loads(labels_path.read_text()) if r['sample_id'] not in test_ids}
    model_names = ['mean', 'ridge', 'svr', 'rf', 'xgb', 'mlp']
    combined, source_rows, fold_rows, predictions, cohort_notes = [], [], [], [], []
    protected = {str(p): digest(p) for p in [split_path, labels_path, scope_folder / 'scope_views.json', scope_folder / 'training_plan.json']}
    for mode, folder in [('strict_evidence_scope', ROOT / 'outputs/scope-strict-train-only-v1'),
                         ('isolate_only_exploratory', ROOT / 'outputs/scope-isolation-sensitivity-v1')]:
        paths = [folder / name for name in ['summary.json', 'nested_cv_results.json', 'training_oof_predictions.json',
                                            'training_scope.json', 'config_snapshot.json', 'manifest.json']]
        protected.update({str(p): digest(p) for p in paths})
        summary, reports, pred, scope, config, manifest = [json.loads(p.read_text()) for p in paths]
        assert summary['scope_mode'] == mode and not summary['test_set_evaluated'] and not summary['input_or_reliability_changed']
        expected = set(views[mode]['train_sample_ids'])
        assert set(scope['train_sample_ids']) == expected and set(scope['protected_test_ids']) == test_ids
        assert not any(r['source_row'] in {x['source_row'] for x in scope['excluded_by_scope_gate']} for r in pred)
        assert len(reports) == len(config['feature_sets']) * len(config['models']) == 12
        parent = Path(config['parent_plan_path'])
        assert digest(parent) == config['parent_plan_sha256']
        old = json.loads(parent.read_text())
        changes = {'training_execution_authorized', 'scope', 'parent_plan_path', 'parent_plan_sha256'}
        assert {k: v for k, v in config.items() if k not in changes} == {k: v for k, v in old.items() if k not in changes}
        assert config['training_execution_authorized'] is True and old['training_execution_authorized'] is False
        validate_fold_layout(reports, expected, planned[mode]['cv'])
        for path, expected_hash in manifest['inputs_sha256'].items():
            assert digest(path) == expected_hash, path
            protected[path] = expected_hash
        protected[str(parent)] = digest(parent)
        for r in pred:
            assert r['sample_id'] in expected and r['sample_id'] not in test_ids
            assert abs(r['actual_provisional_MPa'] - labels[r['sample_id']]['Provisional_Converted_CS_28d']) < 1e-10
        pairs = {(f, m) for f in config['feature_sets'] for m in config['models']}
        scores, groups = aggregate_predictions(pred, group_by_id, expected, test_ids, pairs)
        for r in scores:
            original = next(x['pooled_nested_oof'] for x in reports if (x['feature_set'], x['model']) == (r['feature_set'], r['model']))
            assert all(original[k] is None and r[k] is None or abs(original[k] - r[k]) < 1e-8 for k in original)
            combined.append({'scope': mode, **r})
        source_rows += [{'scope': mode, **r} for r in groups]
        for report in reports:
            fold_rows += [{'scope': mode, 'feature_set': report['feature_set'], 'model': report['model'],
                           'fold': f['fold'], 'train_n': f['train_n'], 'validation_n': f['validation_n'],
                           'train_groups': f['train_groups'], 'validation_groups': f['validation_groups'],
                           'best_params': f['best_params'], **f['metrics']} for f in report['folds']]
        predictions += [{'scope': mode, 'leakage_group': group_by_id[r['sample_id']], **r} for r in pred]
        cohort_notes.append({'scope': mode, 'train_records': len(expected), 'source_groups': summary['training_groups'],
                             'models': config['models'], 'feature_sets': config['feature_sets'],
                             'historical_test_ids_preserved': len(test_ids), 'scope_excluded': len(scope['excluded_by_scope_gate']),
                             'actual_folds_match_predeclared_plan': True})
    assert all(digest(path) == expected_hash for path, expected_hash in protected.items())
    output.mkdir(parents=True, exist_ok=False)
    summary = {'cohorts': cohort_notes, 'experiments': len(combined), 'oof_predictions': len(predictions),
               'test_set_evaluated': False, 'new_trusted_samples': 0, 'formal_model_claimed': False,
               'source_and_sample_weighted_errors_reported': True, 'inputs_unchanged': True}
    best = {mode: min((r for r in combined if r['scope'] == mode), key=lambda r: r['MAE_MPa'])
            for mode in ['strict_evidence_scope', 'isolate_only_exploratory']}
    domain_checks = []
    for cohort in cohort_notes:
        mode = cohort['scope']
        for feature_set in cohort['feature_sets']:
            for model in model_names:
                selected = [r for r in predictions if (r['scope'], r['feature_set'], r['model']) == (mode, feature_set, model)]
                largest = sorted(selected, key=lambda r: abs(r['actual_provisional_MPa'] - r['oof_prediction_MPa']), reverse=True)[:5]
                domain_checks.append({'scope': mode, 'feature_set': feature_set, 'model': model,
                                      'negative_prediction_count': sum(r['oof_prediction_MPa'] < 0 for r in selected),
                                      'top_absolute_errors': largest,
                                      'action': '原样保留，负预测为物理不可用警示；不截断后重新报分、不据此删样本。'})
    summary['best_by_training_oof_mae_exploratory_only'] = best
    versions = {'python': platform.python_version(), **{name: version(name) for name in ['numpy', 'scikit-learn', 'torch', 'xgboost-cpu']}}
    for name, data in [('combined_metrics.json', combined), ('source_group_metrics.json', source_rows),
                       ('prediction_domain_checks.json', domain_checks),
                       ('outer_fold_metrics.json', fold_rows), ('all_training_oof_predictions.json', predictions),
                       ('summary.json', summary), ('runtime_versions.json', versions),
                       ('manifest.json', {'inputs_sha256': protected, 'script_sha256': digest(__file__)})]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (output / '模型指标.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    book = openpyxl.Workbook()
    book.remove(book.active)
    for title, data in [('模型指标', combined), ('来源组误差', source_rows), ('外层验证与超参', fold_rows), ('全部训练折外预测', predictions)]:
        sheet = book.create_sheet(title)
        keys = list(data[0])
        append_excel_row(sheet, keys)
        for r in data:
            append_excel_row(sheet, [json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], (dict, list)) else r[k] for k in keys])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
    book.save(output / '范围审查后模型对比.xlsx')
    lines = ['# 范围审查后的训练内模型对比', '',
             '按已固定预案实际完成两种范围×两种输入表示×6模型，共24项实验、120个外层验证折。所有数值为训练内嵌套分组CV折外成绩，不是测试成绩或最终可信模型。', '',
             '严格范围128条/5来源组，仅隔离30条的敏感性范围269条/19来源组；后者仍含范围未知记录，不能称严格主分析。原145条测试ID保持封存，本轮没有预测或评价测试。', '',
             '输入维持Water、Cem以及由二者计算的比值；未把目标派生信息或未知材料零值加入输入。比例是现有两项输入的计算表示，不增加独立测量信息。']
    for mode, title in [('strict_evidence_scope', '严格方法证据范围（128条、5组）'),
                        ('isolate_only_exploratory', '仅隔离30条的探索敏感性范围（269条、19组）')]:
        for feature_set, representation in [('core_plus_computed_ratio', 'Water+Cem+计算比值'), ('core2', 'Water+Cem')]:
            lines += ['', f'## {title}：{representation}', '',
                      '|模型|R²|MSE MPa²|RMSE MPa|MAE MPa|MAPE %|来源组等权MAE|最差来源组MAE|',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for name in model_names:
                r = next(r for r in combined if (r['scope'], r['feature_set'], r['model']) == (mode, feature_set, name))
                lines.append(f'|{name}|{r["R2"]:.4f}|{r["MSE_MPa2"]:.3f}|{r["RMSE_MPa"]:.3f}|{r["MAE_MPa"]:.3f}|{r["MAPE_percent"]:.2f}|{r["source_macro_MAE_MPa"]:.3f}|{r["worst_source_MAE_MPa"]:.3f}|')
    lines += ['', '## 解释与限制', '',
              '- 两种范围样本不同、重建的外折不同；不能比较两个总分后宣称隔离提高或降低了模型质量，也不能与历史299条成绩直接排名。',
              '- 严格范围中Nasr两篇作为同一组，占104/128=81.25%；因此同时报告按样本和按来源组等权误差，最差来源误差完整保留。',
              '- 每个范围12项配置使用相同事前外折；网格选择只用内层4折。输入缩放、MLP目标缩放均只在当前训练折拟合。',
              '- MLP为32/16隐层及实际Dropout、固定200轮，无外层/测试早停；随机种子42、CPU单线程。负R²、负预测或大误差均保留，不临时重选种子或调网格追分。',
              '- 从多个配置中挑出的最好CV值存在选择乐观性。来源仅5组的严格范围不能支持稳定、广泛泛化的最终结论。',
              '- 目标仍用用户允许的暂定换算系数；方法证据通过不表示换算、绝对用量、密度假设或整条可靠状态已核实。新增可信样本0。',
              '- 历史测试集曾作基线评价，本轮没有重评；最终模型需独立外部来源或前瞻验证。', '',
              '## 运行及输出', '',
              '`outputs/scope-strict-train-only-v1`和`outputs/scope-isolation-sensitivity-v1`分别保存完整外折、参数、逐条预测、配置和输入指纹；本目录Excel与JSON仅汇总，不再次训练。', '',
              '关键函数：validate_fold_layout检查相同外折、组隔离及预案一致；aggregate_predictions检查训练ID覆盖，计算样本加权与来源等权误差；本脚本复核暂定目标、父预案及上游输入指纹。', '',
              '本轮仅保存训练内折外预测和参数记录，没有选定/部署最终模型。下一步应检查逐来源误差及共同可用的物理特征，预先定义范围与材料信息的敏感性方案，继续保留测试封存。']
    strict_best = best['strict_evidence_scope']
    sensitivity_best = best['isolate_only_exploratory']
    lines += ['', '## 本轮主要发现', '',
              f'严格范围按训练折外MAE最小的是{strict_best["model"]}/{strict_best["feature_set"]}：MAE={strict_best["MAE_MPa"]:.3f}MPa，R²={strict_best["R2"]:.4f}；来源组等权MAE={strict_best["source_macro_MAE_MPa"]:.3f}MPa，最差来源组MAE={strict_best["worst_source_MAE_MPa"]:.3f}MPa。总体表现仍弱，不足以建立可靠主模型。',
              f'仅隔离范围内按MAE最小的是{sensitivity_best["model"]}/{sensitivity_best["feature_set"]}：MAE={sensitivity_best["MAE_MPa"]:.3f}MPa，R²={sensitivity_best["R2"]:.4f}。这是另一人群的探索结果，不能据此宣布优于严格模型。',
              '严格范围MLP+计算比值在Al-Selmi6条记录上输出约−133.19MPa，该组MAE约201.91MPa，整体R²约−10.21；这些负预测物理上不可用，已完整保存，没有截断或为改善得分临时改网络。',
              '严格范围XGBoost核心两输入在Zhang4条上的MAE约59.75MPa，而Nasr104条约7.51MPa，来源异质性不能由大组的总体表现掩盖。',
              '新增负预测和大误差清单见prediction_domain_checks.json。当前主要未解决问题是只有两个独立输入、材料体系差异、少数来源组及标签/剂量可靠性尚未定稿。']
    cache = Path(tempfile.gettempdir()) / 'underwater-concrete-plot-cache'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, mode, title in zip(axes, ['strict_evidence_scope', 'isolate_only_exploratory'], ['Strict scope: n=128, groups=5', 'Isolation sensitivity: n=269, groups=19']):
        data = [next(r for r in combined if (r['scope'], r['feature_set'], r['model']) == (mode, 'core_plus_computed_ratio', name)) for name in model_names]
        x = np.arange(len(data))
        ax.bar(x - .18, [r['MAE_MPa'] for r in data], .36, label='Sample-weighted MAE')
        ax.bar(x + .18, [r['source_macro_MAE_MPa'] for r in data], .36, label='Group-macro MAE')
        ax.set_xticks(x, model_names, rotation=25)
        ax.set_ylabel('Nested grouped OOF MAE (MPa)')
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.savefig(output / 'scope_training_mae.png', dpi=180)
    plt.close(fig)
    lines += ['', '![各范围内的样本与来源等权误差](scope_training_mae.png)']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    assert all(digest(path) == expected_hash for path, expected_hash in protected.items())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
