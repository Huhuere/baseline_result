# dev_test_result.csv 说明

## 1. 文件来源
训练过程按 epoch 在 dev 上评估，用 dev 集macro f1选出最佳 epoch，保存为 `best.pt`。随后仅用该最佳模型在测试集 (test) 上跑 **一次**，生成 `dev_test_result.csv`。
## 3. 列含义
dev_xxx列记录了训练过程中在验证集表现最好的模型评估结果，test_xxx记录了使用最佳模型的测试结果。
| 列名 | 含义 | 备注 |
|------|------|------|
| dev_accuracy | dev集准确率 | |
| dev_recall | dev集宏平均召回 (UA) | macro recall |
| dev_f1 | dev集 macro f1| 用于选择 best.pt |
| dev_precision | dev集宏平均精度 | macro precision |
| dev_auc | dev集 AUC | 二分类：正类概率|
| dev_sensitivity | dev集灵敏度 | 二分类=正类召回|
| dev_specificity | dev集特异度 | 二分类=TN/(TN+FP)|
| test_accuracy | test集准确率 | |
| test_recall | test集宏平均召回 | 同上 |
| test_f1 | test集 macro f1 ||
| test_precision | test集宏平均精度 |  |
| test_auc | test集 AUC |  |
| test_sensitivity | test集灵敏度 |  |
| test_specificity | test集特异度 |  |
