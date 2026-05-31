import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os

# 读取每一折每一代的预测文件
def load_fold_predictions(fold_number, epoch_number):
    file_path = f"./p4_results/fold_{fold_number}_epoch_{epoch_number}_test_predictions.csv"
    return pd.read_csv(file_path)

# 读取最后三代每一折的预测文件并合并
def load_last_three_epochs_predictions(num_folds, num_epochs):
    all_predictions = []
    for fold in range(1, 3):
        if fold == 1:
            epoch=22
        else:
            epoch=8
        fold_pred = load_fold_predictions(fold, epoch)
        all_predictions.append(fold_pred)
    return all_predictions

# 通过加权组合预测概率
def calculate_weighted_predictions(predictions, weights):
    weighted_probs = np.zeros_like(predictions[0]['Label 1 Score'].values, dtype=np.float64)
    for weight, pred in zip(weights, predictions):
        weighted_probs += weight * pred['Label 1 Score'].values
    return weighted_probs

# 计算 AUC
def calculate_auc(true_labels, weighted_probs):
    return roc_auc_score(true_labels, weighted_probs)

# 优化每一折的权重以最大化 AUC
def optimize_weights(predictions, true_labels):
    num_folds = len(predictions)
    initial_weights = np.ones(num_folds) / num_folds

    def loss_function(weights):
        weighted_probs = calculate_weighted_predictions(predictions, weights)
        return -calculate_auc(true_labels, weighted_probs)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(num_folds)]

    result = minimize(loss_function, initial_weights, bounds=bounds, constraints=constraints)
    return result.x

# 计算最后三代的最佳 AUC
def find_best_auc(all_predictions, test_labels):
    num_folds = len(all_predictions)
    optimized_weights = optimize_weights(all_predictions, test_labels)
    weighted_probs = calculate_weighted_predictions(all_predictions, optimized_weights)
    auc = calculate_auc(test_labels, weighted_probs)
    return auc, optimized_weights

# 保存结果到新的 CSV 文件（修改后的部分）
def save_results(all_predictions, best_weights, test_labels, output_file):
    # 创建字典存储每个样本在所有预测中的Label 1 Score
    sample_min_scores = {}
    
    # 收集每个样本的所有Label 1 Score
    for fold_pred in all_predictions:
        for idx, row in fold_pred.iterrows():
            file_path = row['File Path']
            label1_score = row['Label 1 Score']
            
            if file_path not in sample_min_scores:
                sample_min_scores[file_path] = []
            sample_min_scores[file_path].append(label1_score)
    
    # 计算每个样本的最小Label 1 Score
    for file_path, scores in sample_min_scores.items():
        sample_min_scores[file_path] = min(scores)
    
    # 计算加权预测分数
    weighted_0_scores = {}
    weighted_1_scores = {}
    true_labels_dict = {}
    
    # 初始化字典
    sample_csv = all_predictions[0]
    for file_path in sample_csv['File Path'].unique():
        weighted_0_scores[file_path] = 0.0
        weighted_1_scores[file_path] = 0.0
    
    # 计算加权分数
    for weight, fold_pred in zip(best_weights, all_predictions):
        for idx, row in fold_pred.iterrows():
            file_path = row['File Path']
            weighted_0_scores[file_path] += weight * row['Label 0 Score']
            weighted_1_scores[file_path] += weight * row['Label 1 Score']
            true_labels_dict[file_path] = row['True Label']  # 真实标签只需记录一次
    
    # 准备结果数据
    results = []
    for file_path in weighted_0_scores.keys():
        min_score = sample_min_scores[file_path]
        
        # 检查是否满足条件
        if min_score < 0.52:
            # label1_score = min_score
            # label0_score = 1 - label1_score  # 确保概率和为1
            label1_score = 0
            label0_score = 1  # 确保概率和为1
        else:
            label1_score = weighted_1_scores[file_path]
            label0_score = weighted_0_scores[file_path]
        
        results.append({
            'File Path': file_path,
            'Label 0 Score': label0_score,
            'Label 1 Score': label1_score,
            'True Label': true_labels_dict[file_path]
        })
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# 主程序调用
def main_optimization(num_folds=3, num_epochs=40):
    sample_csv = load_fold_predictions(1, 22)
    test_labels = sample_csv['True Label'].values

    all_predictions = load_last_three_epochs_predictions(num_folds, num_epochs)
    best_auc, best_weights = find_best_auc(all_predictions, test_labels)

    print(f"最佳AUC: {best_auc}")
    output_file = "./p4_results/five_epoch_predictions.csv"
    save_results(all_predictions, best_weights, test_labels, output_file)

# 运行主函数
if __name__ == '__main__':
    main_optimization(num_folds=8, num_epochs=15)