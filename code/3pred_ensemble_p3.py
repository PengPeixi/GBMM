import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import numpy as np

# 读取两个CSV文件
val_train_df = pd.read_csv('/root/autodl-tmp/test2/2d/v1/fold8_pretrain/results_post_p3_correct/aggregated_predictions_val2.csv')
train_best_df = pd.read_csv('./p3_results/five_epoch_predictions.csv')
train_best_df["File Path"] = train_best_df["File Path"].apply(lambda x: '_'.join(os.path.basename(x).split('_')[:2]))
val_train_df["File Path"] = val_train_df["File Path"].apply(lambda x: '_'.join(os.path.basename(x).split('_')[:2]))
# 使用File Path合并数据
merged_df = pd.merge(val_train_df, train_best_df, on='File Path', suffixes=('_val', '_best'))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 加权求和函数
def weighted_sum(merged_df, weight):
    label0_temp = weight * merged_df['Label 0 Score_val'] + (1 - weight) * merged_df['Label 0 Score_best']
    label1_temp = weight * merged_df['Label 1 Score_val'] + (1 - weight) * merged_df['Label 1 Score_best']

    # 应用非线性变换
    label0_temp_1 = label0_temp + sigmoid((label0_temp - 0.46) * 100) * 0.11
    label1_temp_1 = label1_temp - sigmoid((label0_temp - 0.46) * 100) * 0.11
    
    merged_df['Label 0 Score_final'] = label0_temp_1
    merged_df['Label 1 Score_final'] = label1_temp_1
    return merged_df

# 计算AUC的函数
def compute_auc(weight, merged_df):
    weighted_df = weighted_sum(merged_df, weight)
    auc = roc_auc_score(weighted_df['True Label_val'], weighted_df['Label 1 Score_final'])
    return auc, weighted_df  # 返回AUC和计算后的DataFrame

# 手动调整权重和a值
best_weight = None
best_auc = 0
best_weighted_df = None  # 保存最佳结果的DataFrame

# 遍历a从0.5到1，每次+0.05
weights = [round(x * 0.05, 2) for x in range(1, 20)]

# 遍历所有组合
for current_weight in weights:
    current_auc, current_df = compute_auc(current_weight, merged_df.copy())  # 使用副本避免修改原数据

    print(f"当前权重: {current_weight}, 当前AUC: {current_auc}")

    # 更新最佳结果
    if current_auc > best_auc:
        best_auc = current_auc
        best_weight = current_weight
        best_weighted_df = current_df  # 保存当前最佳结果的DataFrame

# 输出最佳参数
print(f"最佳权重: {best_weight}")
print(f"最佳AUC: {best_auc}")

# 保存最佳结果到CSV
if best_weighted_df is not None:
    # 创建输出目录（如果不存在）
    output_dir = './p3_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择需要的列并重命名
    result_df = best_weighted_df[['File Path', 'Label 0 Score_final', 'Label 1 Score_final', 'True Label_val']].copy()
    result_df = result_df.rename(columns={
        'Label 0 Score_final': 'Label 0 Score',
        'Label 1 Score_final': 'Label 1 Score',
        'True Label_val': 'True Label'
    })
    
    # 保存CSV
    output_path = os.path.join(output_dir, 'best_results.csv')
    result_df.to_csv(output_path, index=False)
    print(f"最佳结果已保存至: {output_path}")
else:
    print("未找到最佳结果")
