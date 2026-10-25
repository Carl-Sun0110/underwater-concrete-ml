"""合并同一训练样本、相同外折的四算法结果，不执行训练或测试预测。"""
from pathlib import Path
import argparse,csv,json,hashlib,os,tempfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(earlier,advanced,output):
    paths=[earlier/'nested_cv_results.json',advanced/'nested_cv_results.json',
           earlier/'training_scope.json',advanced/'training_scope.json',advanced/'training_oof_predictions.json']
    fingerprints={str(p):digest(p) for p in paths}
    scope=json.loads(paths[2].read_text());scope2=json.loads(paths[3].read_text())
    if scope!=scope2:raise ValueError('两轮训练范围不一致，不能合并')
    reports=json.loads(paths[0].read_text())+json.loads(paths[1].read_text())
    reference=reports[0]['folds']
    for report in reports:
        if [f['validation_sample_ids'] for f in report['folds']]!=[f['validation_sample_ids'] for f in reference]:
            raise ValueError('外层验证划分不同，不能直接比较')
    rows=[]
    for report in reports:
        rows.append({'feature_set':report['feature_set'],'model':report['model'],**report['pooled_nested_oof']})
    if len({(r['feature_set'],r['model']) for r in rows})!=len(rows):raise ValueError('试验重复')
    predictions=json.loads(paths[4].read_text())
    large_errors=[]
    for name in ['xgb','mlp']:
        selected=[r for r in predictions if r['model']==name and r['feature_set']=='core_plus_computed_ratio']
        large_errors.extend(sorted(selected,key=lambda r:abs(r['actual_provisional_MPa']-r['oof_prediction_MPa']),reverse=True)[:5])
    output.mkdir(parents=True,exist_ok=False)
    for name,obj in [('combined_metrics.json',rows),('large_training_oof_errors.json',large_errors),('training_scope.json',scope),
                     ('manifest.json',{'inputs_sha256':fingerprints,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                      'script_sha256':digest(Path(__file__)),'test_set_evaluated':False})]:
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'model_comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    versions=json.loads((ROOT/'outputs/advanced-runtime-versions.json').read_text())
    lines=['# 四算法探索性训练内对比','',
      '已经实际运行SVR、随机森林、XGBoost和带Dropout的MLP；仍是Water/Cem两项原始输入下的探索流程，不是完整配合比可信模型。',
      '相同299条训练样本、20个来源隔离组、5外折×4内折分组验证。145条测试记录本轮未重新评估。','',
      '## 结果：Water+Cem+计算比值Water/Cem','',
      '以下全部为训练部分嵌套CV折外预测，不是独立测试成绩。','',
      '|模型|R²|MSE（MPa²）|RMSE（MPa）|MAE（MPa）|MAPE|','|---|---:|---:|---:|---:|---:|']
    for name in ['mean','ridge','svr','rf','xgb','mlp']:
        r=next(r for r in rows if r['model']==name and r['feature_set']=='core_plus_computed_ratio')
        lines.append(f'|{name}|{r["R2"]:.4f}|{r["MSE_MPa2"]:.3f}|{r["RMSE_MPa"]:.3f}|{r["MAE_MPa"]:.3f}|{r["MAPE_percent"]:.2f}%|')
    lines+=['', '## 结果：Water+Cem，不加计算比值','',
            '|模型|R²|RMSE（MPa）|MAE（MPa）|','|---|---:|---:|---:|']
    for name in ['mean','ridge','svr','rf','xgb','mlp']:
        r=next(r for r in rows if r['model']==name and r['feature_set']=='core2')
        lines.append(f'|{name}|{r["R2"]:.4f}|{r["RMSE_MPa"]:.3f}|{r["MAE_MPa"]:.3f}|')
    lines+=['', 'XGBoost在本轮比较中MAE和R²较好，但R²仍只有约0.216，不足以声称预测可靠。',
      'MLP虽有较低MAE，平方误差仍大、R²为负，不能仅看MAE宣布其表现良好。没有为改进该结果临时换种子或网络。',
      '比较多个配置后选出的最佳CV值具有选择乐观性，尚不等于最终独立泛化能力。','',
      '## 两个新增模型的实现','',
      '- XGBoost：CPU histogram，learning_rate=0.05，max_depth∈{2,4}，n_estimators∈{100,200}，subsample=0.8，L2=1，min_child_weight=3；每内折4个候选。',
      '- MLP：PyTorch CPU，隐层32→16，ReLU，两层均含Dropout；dropout∈{0.1,0.25}；AdamW，学习率0.003、weight_decay=0.001，batch64，固定200轮。每内折2个候选。',
      '- 神经网络输入和目标的StandardScaler均只在当前训练折拟合；目标逆变换后按MPa评价。没有外层验证/测试早停。',
      '- CPU单线程、固定seed42、确定性训练。未使用GPU、没有删除难预测样本。',
      '- 参数网格在本轮结果产生前固定；不同模型搜索预算不同，本轮属于有限预算探索对比，不是穷尽调优。',
      '- 计算比值仅由Water/Cem构造，没有额外测量信息，不改原表w/c或w/b；未知材料列继续不使用。','',
      '## 已识别的大误差','',
      '- 原表942行的暂定强度为102.296 MPa；该记录所在外折中，XGBoost约预测30.09 MPa、MLP约预测28.04 MPa。',
      '- 同来源943、944等高强度记录也存在大误差，详见large_training_oof_errors.json。不能因模型预测差就将其判为错误或删除。',
      '- 这些现象说明仅有Water/Cem的当前模型难以覆盖部分来源，不能单凭本次误差判定原因一定是标签、输入或来源中的哪一项。','',
      '## 依赖、验证和复现','',
      '- 独立环境：/workspace/.venv-underwater-ml，使用系统已有基础依赖，仅新增CPU XGBoost/PyTorch及所需包；原审计解释器未改。',
      '- 包版本：'+json.dumps(versions,ensure_ascii=False),
      '- 使用官方PyPI与PyTorch CPU索引安装，TLS和索引哈希验证保持启用，安装报告位于outputs下。',
      '- 新增5项测试：真实Dropout、训练损失下降、固定seed复现、折内目标缩放、CPU XGBoost及输入边界；完整43项测试通过，无跳过。',
      '- 新增4个配置×5外折完成；1196条折外预测只涉及299条训练样本；各折样本ID与此前完全一致。',
      '- 原始Excel/PDF及可靠状态未改；新增可信样本0；没有生成最终部署模型。','',
      '```bash','bash scripts/setup_advanced_env.sh',
      'PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v',
      '/workspace/.venv-underwater-ml/bin/python scripts/run_train_comparison.py --config configs/advanced_train_only.json --output outputs/new-advanced-run',
      '```','',
      '## 产物与下一步','',
      '- 本目录：四算法及两个参考基线的完整指标CSV/JSON、大误差索引、对比图。',
      '- ../advanced-train-only-v1：所有外折结果、最佳参数、仅训练折外预测、配置和输入/代码指纹。',
      '- 实现：advanced_models.py、扩展的train_compare.py、advanced_train_only.json、test_advanced_models.py、安装脚本及依赖版本文件。',
      '- 下一步应先根据训练折外误差定位哪些来源/材料条件缺失，再优先补可追溯特征；不反复查看测试集来挑方案。',
      '- 暂定强度系数仍按用户约定；不因此认定标签或材料来源已核实。']
    cache=Path(tempfile.gettempdir())/'underwater-concrete-plot-cache';cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    selected=[next(r for r in rows if r['model']==n and r['feature_set']=='core_plus_computed_ratio') for n in ['mean','ridge','svr','rf','xgb','mlp']]
    for ax,key,title in [(axes[0],'MAE_MPa','Nested grouped OOF MAE (MPa)'),(axes[1],'R2','Nested grouped OOF R2')]:
        ax.bar([r['model'] for r in selected],[r[key] for r in selected]);ax.set_title(title);ax.axhline(0,color='black',linewidth=.7)
    fig.savefig(output/'four_model_training_cv.png',dpi=180);plt.close(fig)
    lines+=['','![四算法训练内验证](four_model_training_cv.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(digest(Path(k))!=v for k,v in fingerprints.items()):raise RuntimeError('源结果被改写')
    print('四算法汇总完成；全部指标来自同一训练样本和相同验证折，未评估测试。')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--earlier',type=Path,default=ROOT/'outputs/train-only-comparison-v1')
 p.add_argument('--advanced',type=Path,default=ROOT/'outputs/advanced-train-only-v1')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();run(a.earlier,a.advanced,a.output)
