import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np

# 读取CSV文件
file_path = '/root/autodl-tmp/v1/3d/code/3d_code/results_post/best_results.csv'
data = pd.read_csv(file_path)

# 验证数据列（可选）
print("数据列名:", data.columns.tolist())
print("前5行数据:\n", data.head())

# 提取Label 1的分数作为预测概率（正例分数）
scores = data['Label 1 Score']
# 提取真实标签
true_labels = data['True Label']

# 1. 计算AUC
try:
    auc = roc_auc_score(true_labels, scores)
except Exception as e:
    print(f"计算AUC时出错: {str(e)}")
    auc = None

# 2. 生成预测类别（阈值为0.5）
pred_labels = np.where(scores >= 0.5, 1, 0)

# 3. 计算其他指标
try:
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    
    # 计算各项指标
    acc = accuracy_score(true_labels, pred_labels)
    sens = tp / (tp + fn)         # 灵敏度 = TP/(TP+FN)
    spec = tn / (tn + fp)         # 特异度 = TN/(TN+FP)
    ppv = precision_score(true_labels, pred_labels)  # 阳性预测值 = TP/(TP+FP)
    npv = tn / (tn + fn)          # 阴性预测值 = TN/(TN+FN)

except Exception as e:
    print(f"计算指标时出错: {str(e)}")
    acc, sens, spec, ppv, npv = [None]*5
    tn, fp, fn, tp = [None]*4

# 输出结果
print("\n===== 性能指标报告 =====")
print(f"AUC:      {auc:.4f}" if auc is not None else "AUC:      计算错误")
print(f"Accuracy: {acc:.4f}" if acc is not None else "Accuracy: 计算错误")
print(f"Sens:     {sens:.4f}" if sens is not None else "Sens:     计算错误")
print(f"Spec:     {spec:.4f}" if spec is not None else "Spec:     计算错误")
print(f"PPV:      {ppv:.4f}" if ppv is not None else "PPV:      计算错误")
print(f"NPV:      {npv:.4f}" if npv is not None else "NPV:      计算错误")
print("\n混淆矩阵统计:")
print(f"真阳性(TP): {tp} | 假阳性(FP): {fp}")
print(f"假阴性(FN): {fn} | 真阴性(TN): {tn}")
