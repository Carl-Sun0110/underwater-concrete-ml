# Baseline V1.1：训练集内缺失处理对照及CSV预测入口

本轮锁定Baseline V1的论文留出划分；未用留出目标重新评价、选方案或拟合最终模型。分组身份仅从既有输出提取，目标仅取训练ID对应记录。
已用同一训练CV挑选方案和报告，因此仍有选择偏差。历史留出之前已评价过，本轮没有新独立最终测试结果。

|队列|训练数|训练来源组|CV折数|锁定留出数|
|---|---:|---:|---:|---:|
|primary_direct|117|4|4|95|
|exploratory_underwater|320|23|5|147|

固定RF/XGBoost/CatBoost参数不变，比较Water/Cem两特征和完整11配合比特征；中位数或标准化距离KNN插补附缺失标记，全部只在训练折拟合。KNN用训练观察值估计距离尺度，填补后还原单位，全未知列保留数值占位。

|按来源组平均MAE选出的候选|MAE MPa|RMSE MPa|R²|来源组平均MAE|
|---|---:|---:|---:|---:|
|primary_direct__materials__scaled_knn__catboost|19.826|21.141|-1.109|21.178|
|exploratory_underwater__core__median__xgb|9.118|12.227|0.084|10.668|

结果解释：
- 主集只剩4个训练来源组。Nasr组104条占117条的绝大部分；留出该组做训练CV时，仅剩其他体系13条，造成严重跨域外推。不能把本轮MAE与前轮212条全体CV直接比较，也不能把变化解释为模型代码退化。
- 扩展训练集按预先指定的来源宏平均误差选择两特征/中位数XGBoost。KNN改善部分全特征组合，但没有形成稳定高精度的跨来源模型。
- 两个模型都保留为探索演示；主集R²为负，不能作为有效泛化结论。扩展模型也只略高于均值参照，不能宣称可以准确设计配合比。
- 121项测试通过，两个模型已保存并各生成3条CSV演示预测。演示来自训练样本，不计为新实验数据或独立性能。
- 优先扩充独立来源与真实单位证据；继续保留原样本和强度标签，不以删难样本或重复看测试集来制造提升。

## 直接预测CSV

CSV使用模型白名单中的原始字段名。扩展模型此次只需Water、Cem；名义单位遵循原库，单位疑问仍保留。留空或非正值会标记未知并使用训练时拟合的填补器；多余列（包括强度标签）不作为输入。
```bash
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/predict_strength.py --model outputs/baseline_v1_1/exploratory_underwater__selected.joblib --input your_recipes.csv --output your_predictions.csv
```

输入示例见exploratory_underwater__demo_inputs.csv，格式含case_id,Water,Cem。预测输出含MPa、缺失字段、超出训练单变量范围字段和探索状态。所有输入缺失仍可能有数值预测，但状态会明确标为all_inputs_missing，不应据此作科研判断。
范围内不代表组合受训练支持或预测可靠；没有自动单位换算、概率置信区间或工程保证。模型仅加载本项目生成的本地joblib。

主要文件：results.json、training_oof_predictions.json、paired_imputation_comparison.json、cohort_and_training_folds.json、selected_models.json、*.joblib、*demo_inputs.csv、*demo_predictions.csv、verification.json、tests.log、training.log、console.log。
新增代码与配置：configs/baseline_v1_1.json、src/concrete_audit/baseline_refinement.py、src/concrete_audit/predict.py、scripts/run_baseline_v1_1.py、scripts/predict_strength.py、scripts/summarize_baseline_v1_1.py、tests/test_baseline_refinement.py；README更新入口。
函数说明：load_training只取锁定训练ID；ScaledKNNImputer按训练观察值定义距离尺度；pipeline封装插补和模型；run生成CV证据并拟合仅训练集模型；predict_csv白名单读入、范围标记及独立输出；summarize复核无留出样本参与。
