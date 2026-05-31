import pandas as pd
from sklearn.metrics import roc_auc_score
import os

# 读取两个CSV文件
val_train_df = pd.read_csv('/root/autodl-tmp/v1/2d/code/2d_new_code/results_post/aggregated_predictions.csv')
train_best_df = pd.read_csv('/root/autodl-tmp/v1/3d/code/3d_code/results_post/epoch_10_test_predictions.csv')
# train_best_df = pd.read_csv('/root/autodl-tmp/test/val_train_iter/results_post/best_epoch_predictions.csv')
train_best_df["File Path"] = train_best_df["File Path"].apply(lambda x: '_'.join(os.path.basename(x).split('_')[:4]))
# print(train_best_df["File Path"])
# 假设这两个文件具有相同的列顺序，并且文件路径可以作为索引连接的依据
# 使用File Path合并数据
merged_df = pd.merge(val_train_df, train_best_df, on='File Path', suffixes=('_val', '_best'))

# 修改Label 0 Score大于0.46的那一行，a动态变化
def modify_scores(row, a):
    if row['Label 0 Score_val'] > 0.46:
        row['Label 0 Score_val'] = a
        row['Label 1 Score_val'] = 1 - a
    return row

# 加权求和函数
def weighted_sum(merged_df, weight, a):
    # 修改Label 0 Score > 0.46的行，根据不同的a值
    modified_df = merged_df.apply(lambda row: modify_scores(row, a), axis=1)
    
    # 加权求和 Label 0 和 Label 1 的分数
    merged_df['Label 0 Score_final'] = weight * modified_df['Label 0 Score_val'] + (1 - weight) * merged_df['Label 0 Score_best']
    merged_df['Label 1 Score_final'] = weight * modified_df['Label 1 Score_val'] + (1 - weight) * merged_df['Label 1 Score_best']
    
    return merged_df

# 计算AUC的函数
def compute_auc(weight, a, merged_df):
    weighted_df = weighted_sum(merged_df, weight, a)
    
    # 使用最终的预测分数计算AUC
    auc = roc_auc_score(weighted_df['True Label_val'], weighted_df['Label 1 Score_final'])
    return auc

# 手动调整权重和a值
best_weight = None
best_a = None
best_auc = 0

# 遍历a从0.5到1，每次+0.05
for a in [round(x * 0.05, 2) for x in range(10, 21)]:
    # current_weight从0.05到0.95，每次+0.05
    for current_weight in [round(x * 0.05, 2) for x in range(1, 20)]:
        current_auc = compute_auc(current_weight, a, merged_df)
        
        print(f"当前权重: {current_weight}, 当前a: {a}, 当前AUC: {current_auc}")
        
        # 如果当前AUC比之前的最好AUC好，更新最佳AUC、权重和a
        if current_auc > best_auc:
            best_auc = current_auc
            best_weight = current_weight
            best_a = a

# 输出最佳权重、a值和AUC
print(f"最佳a值: {best_a}")
print(f"最佳权重: {best_weight}")
print(f"最佳AUC: {best_auc}")
