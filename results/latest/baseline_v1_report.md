# Baseline V1 第一轮实际建模总结（2026-10-24）

本报告为本次明确协议的独立Baseline V1运行，不沿用历史分数。数据及部分来源此前已参加探索，不能称全新盲测；本轮测试评价后不再用于反复选参并声称独立。

## 文件与实际数据

未找到《水下混凝土_人工核查版.xlsx》《水下混凝土_筛选结果_20261012.xlsx》。实际读取原版Total；Selected为编码版，批注版与筛选方案为独立版本，不冒充所缺文件。工作簿结构、原始附件哈希和所有项目Excel清单见excel_inventory.json。
Total 1082条，水下标记710条；其中原始28天强度正值656条，Converted正值582条。
497条满足本轮暂定换算/一致性候选，其中30条已知非直接成型单独隔离。最终主集212条、扩展探索集467条（包含主集及额外范围待核查255条）。
原文辅助证据支持主集成型范围，整条标签/换算仍非正式人工核实。使用原表存储Converted，不重算标签、不补目标；0.95/0.8/0.76为用户既有暂定许可，不因算术一致就认定可靠。未强行纳入其他系数对应记录，也未判为伪造。

## 特征、划分与固定参数

11项输入：Water, Cem, SF, FA, GGBS, MK, NP, RS, SS, CA, SCA。材料正值保留，0含义未明按未知；中位数插补加缺失指示器仅训练折拟合，整列未知的0是数值占位。未使用Density、强度衍生变量、其他实测性能；SP/AWA/纤维及w/c、w/b的单位、掺量基准或公式来源待统一，未在本轮直接混用。
三模型固定150棵/轮；XGBoost/CatBoost深度4、学习率.04，RF叶节点最少3；无参数搜索。增加均值参照，完整参数见config.json、代码与模型文件。
随机种子42。随机对照先按近重复配比簇划分，允许同论文不同配比跨集；未做有近重复泄漏的逐行随机。组别验证按论文/研究组及跨论文近重复连通组隔离。近重复为全配比字段已知模式相同且逐项1%以内，不看目标，不据此删除记录。

|数据集|方案|训练数|验证/测试数|
|---|---|---:|---:|
|primary_direct|random_recipe_holdout|170|42|
|primary_direct|paper_group_holdout|117|95|
|exploratory_underwater|random_recipe_holdout|374|93|
|exploratory_underwater|paper_group_holdout|320|147|

## 真实评价

|数据集|方案|模型|n|MAE MPa|RMSE MPa|R²|
|---|---|---|---:|---:|---:|---:|
|primary_direct|random_recipe_holdout|rf|42|4.387|6.438|0.784|
|primary_direct|random_recipe_holdout|xgb|42|4.185|5.405|0.848|
|primary_direct|random_recipe_holdout|catboost|42|3.974|5.291|0.854|
|primary_direct|random_recipe_holdout|mean|42|7.912|13.863|-0.000|
|primary_direct|paper_group_holdout|rf|95|12.206|15.171|-1.098|
|primary_direct|paper_group_holdout|xgb|95|8.942|11.297|-0.163|
|primary_direct|paper_group_holdout|catboost|95|9.507|11.799|-0.269|
|primary_direct|paper_group_holdout|mean|95|8.753|10.526|-0.010|
|primary_direct|paper_group_cv|rf|212|7.171|11.822|0.160|
|primary_direct|paper_group_cv|xgb|212|7.060|12.423|0.072|
|primary_direct|paper_group_cv|catboost|212|8.225|13.489|-0.094|
|primary_direct|paper_group_cv|mean|212|10.127|13.983|-0.175|
|exploratory_underwater|random_recipe_holdout|rf|93|3.190|4.700|0.818|
|exploratory_underwater|random_recipe_holdout|xgb|93|3.429|4.487|0.834|
|exploratory_underwater|random_recipe_holdout|catboost|93|4.089|5.353|0.764|
|exploratory_underwater|random_recipe_holdout|mean|93|7.340|11.021|-0.002|
|exploratory_underwater|paper_group_holdout|rf|147|7.899|9.135|-1.251|
|exploratory_underwater|paper_group_holdout|xgb|147|6.090|8.169|-0.800|
|exploratory_underwater|paper_group_holdout|catboost|147|4.908|7.380|-0.469|
|exploratory_underwater|paper_group_holdout|mean|147|6.207|7.421|-0.485|
|exploratory_underwater|paper_group_cv|rf|467|7.651|10.913|0.065|
|exploratory_underwater|paper_group_cv|xgb|467|7.117|10.309|0.165|
|exploratory_underwater|paper_group_cv|catboost|467|7.410|10.813|0.082|
|exploratory_underwater|paper_group_cv|mean|467|8.368|11.445|-0.028|

## 解释、限制与下一轮

- 主集随机配比簇划分表现明显好于论文留出。随机对照虽隔离了近重复配比，仍可学习同论文体系特征；按论文留出存在材料体系、强度范围及实验条件迁移。差距不能全部归因于泄漏或训练过拟合，也不能把随机R²当作跨文献能力。
- 主集仅6个来源连通组、7个论文文件名，五折中验证数不均；扩展集29组。组测试比例按组抽取，样本比例不一定20%，不得为改善指标重新挑测试论文。各折、训练分数、来源宏平均误差均保留。
- 论文分组固定留出三模型R²均为负，部分模型不及均值参照；样本强度方差较小也影响R²。主集五折的三指标排名不一致，暂不宣布最终最优模型。
- 来源覆盖、单位、密度假设和未知零值语义仍待核；Converted使用暂定系数，需按原试件尺寸、形状、龄期、养护条件和换算文献逐条复核，不用预测值倒推标签。
- 高误差记录及论文按分组折外预测产生；高强度值只是复核线索，不自动认定错误。优先核对高影响来源的配合比绝对/相对基准、砂/粉煤灰字段和标签图形读取。
- 特征严重缺失与重要性见feature_review_priorities.json。测试置换仅用于本轮描述，相关列可相互替代，不能根据这次测试重要性删列后继续称独立测试。
- 下一轮：固定本轮协议，在训练来源内部比较KNN/中位数及单位一致的配比比值；扩充独立来源、进行标签换算敏感性、按预先确定方案开展新验证。SHAP本轮未强制追加。
- 118项全量测试通过；原始输入哈希未变，全部指标重算验证通过；12张预测/残差图已生成。

## 文件与复现

结果目录含Excel/metrics.csv、selection_records.json、modeling_records.json、features_and_missingness.json、preprocessing.json、split_manifest.json、duplicate_analysis.json、predictions.json、paper_error_analysis.json、high_error_records.json、plots/、模型joblib、code/、training.log、console.log、tests.log和manifest.json。
仓库新增configs/baseline_v1.json、src/concrete_audit/baseline_v1.py、scripts/run_baseline_v1.py、scripts/summarize_baseline_v1.py、tests/test_baseline_v1.py；README更新本轮入口。
```bash
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/run_baseline_v1.py --output outputs/baseline_v1_rerun
```

输出目录必须不存在。关键函数：read_source映射参数字典及身份合并；select_records保留逐条理由和存储目标；clusters在划分前识别近重复；model_pipeline封装训练折插补；run执行固定参数评估；summarize重算指标并导出图表。
