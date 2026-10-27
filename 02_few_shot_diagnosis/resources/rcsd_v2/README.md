# 固定 RCSD 半合成诊断资源

39份资源：3个K值 ×（9个神经单模型＋3个神经集成＋1个RF）。每份对应支持抽样11；集成成员为训练种子42/43/44，使用同一批目标标注。模型没有新增现场实测验证。

权威索引 [resource_manifest.json](resource_manifest.json) 包含路径、算子文件名和SHA256；[input_contract.json](input_contract.json)给出25通道顺序、单位、物理量、聚合、900秒间隔及20点窗口。调用者必须声明实际输入契约，不能为绕过检查而复制到不相符的现场表。

| method参数 | 模型 | 算子 |
|---|---|---|
| random_forest | .joblib | process_random_forest.py |
| cnn_scratch_seed42 / cnn_scratch_seed43 / cnn_scratch_seed44 | 单CNN .pt | process_cnn_diagnosis.py |
| transfer_seed42 / transfer_seed43 / transfer_seed44 | 单迁移 .pt | process_transfer_learning.py |
| meta_adapted_prototype_seed42 / meta_adapted_prototype_seed43 / meta_adapted_prototype_seed44 | 单元学习适配原型 .pt | process_meta_learning.py |
| cnn_scratch_seed_ensemble | 三CNN集成 .pt | process_seed_ensemble.py |
| transfer_seed_ensemble | 三迁移集成 .pt | process_seed_ensemble.py |
| meta_adapted_prototype_seed_ensemble | 三元学习适配原型集成 .pt | process_seed_ensemble.py |

```text
python 02_few_shot_diagnosis/scripts/run_rcsd_demo.py --method transfer_seed_ensemble --k 5
```

仅加载可信模型，PyTorch使用 `weights_only=True`；joblib资源须可信。推荐精确依赖版本。评分未经概率校准，不可称作真实故障概率。

`expected_resource_predictions.csv` 为39资源×352查询的冻结输出，不含传感器原值。不同构造版本复用22背景，不是新增故障事件。取得工作簿后按[复现说明](../../../docs/凌晨/数据与复现.md)恢复输入并比较。
