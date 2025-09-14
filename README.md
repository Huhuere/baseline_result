# SpeechFormer_dev_test_result.csv 说明

## 1. 文件来源
训练过程按 epoch 在 dev 上评估，用 dev 集macro f1选出最佳 epoch，保存为 `best.pt`。随后仅用该最佳模型在测试集 (test) 上跑 **一次**，生成 `dev_test_result.csv`。
## 2. 列含义
dev_xxx列记录了训练过程中在验证集表现最好的模型评估结果，在epoch=8时达到最优，test_xxx记录了使用最佳模型的测试结果。
| 列名 | 含义 | 备注 |
|------|------|------|
| dev_accuracy | dev集准确率 | |
| dev_recall | dev集宏平均召回 | macro recall |
| dev_macro_f1 | dev集 macro f1| 用于选择 best.pt |
| dev_precision | dev集宏平均精度 | macro precision |
| dev_auc | dev集 AUC | 二分类：正类概率|
| dev_sensitivity | dev集灵敏度 | 二分类=正类召回|
| dev_specificity | dev集特异度 | 二分类=TN/(TN+FP)|
| test_accuracy | test集准确率 | |
| test_recall | test集宏平均召回 | 同上 |
| test_macro_f1 | test集 macro f1 ||
| test_precision | test集宏平均精度 |  |
| test_auc | test集 AUC |  |
| test_sensitivity | test集灵敏度 |  |
| test_specificity | test集特异度 |  |
## 3. epoch=60时验证集模型评估结果
|dev_accuracy | dev_recall |dev_macro_f1 |dev_precision |dev_auc| dev_sensitivity| dev_specificity |
|------|------|------|------|------|------|------|
|0.58857| 0.50960| 0.50474|0.51167|0.51877|0.25833|0.76087|
## 4. epoch=8时验证集最优模型评估结果
|dev_accuracy | dev_recall |dev_macro_f1 |dev_precision |dev_auc| dev_sensitivity| dev_specificity |
|------|------|------|------|------|------|------|
|0.63571| 0.58333| 0.58486|0.58823|0.56506|0.41667|0.75000|
## 5. 验证集最优模型在测试集评估结果
|train_accuracy | train_recall |train_macro_f1 |train_precision |train_auc| train_sensitivity| train_specificity |
|------|------|------|------|------|------|------|
|0.65957| 0.55195| 0.55238|0.56486|0.53680|0.28571|0.81818|
