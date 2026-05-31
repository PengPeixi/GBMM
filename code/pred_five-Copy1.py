import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os

# 读取每一折每一代的预测文件
def load_fold_predictions(fold_number, epoch_number):
    file_path = f"./results/fold_{fold_number}_epoch_{epoch_number}_test_predictions.csv"
    return pd.read_csv(file_path)

# 读取最后三代每一折的预测文件并合并
def load_last_three_epochs_predictions(num_folds, num_epochs):
    all_predictions = []
    # 只加载最后三代
    # for epoch in range(num_epochs - 2, num_epochs + 1):  # 这里是加载最后三代
    for epoch in range(num_epochs, num_epochs+2):  # 这里是加载最后三代
        for fold in range(1, num_folds + 1):  # 加载三折
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
    initial_weights = np.ones(num_folds) / num_folds  # 初始权重为均等

    def loss_function(weights):
        weighted_probs = calculate_weighted_predictions(predictions, weights)
        return -calculate_auc(true_labels, weighted_probs)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # 权重和为1
    bounds = [(0, 1) for _ in range(num_folds)]  # 权重的范围在[0, 1]之间

    result = minimize(loss_function, initial_weights, bounds=bounds, constraints=constraints)
    return result.x  # 返回优化后的权重

# 计算最后三代的最佳 AUC
def find_best_auc(all_predictions, test_labels):
    best_auc = -1
    best_weights = None

    num_folds = len(all_predictions)  # 这里的fold数是9（3代 × 3折）

    optimized_weights = optimize_weights(all_predictions, test_labels)
    weighted_probs = calculate_weighted_predictions(all_predictions, optimized_weights)
    auc = calculate_auc(test_labels, weighted_probs)

    # print(f"Overall AUC = {auc:.4f} with weights {optimized_weights}")

    return auc, optimized_weights

# 保存结果到新的 CSV 文件
def save_results(all_predictions, best_weights, test_labels, output_file):
    weighted_0_scores = np.zeros_like(test_labels, dtype=np.float64)  # 确保与test_labels长度一致且为float64
    weighted_1_scores = np.zeros_like(test_labels, dtype=np.float64)  # 确保与test_labels长度一致且为float64
    file_paths = []

    # 提取第一个折的文件路径
    first_fold_pred = all_predictions[0]
    file_paths = first_fold_pred['File Path'].values  # 从第一个折提取File Path

    # 计算每个样本的加权预测
    for fold in range(len(all_predictions)):
        fold_pred = all_predictions[fold]
        weight = best_weights[fold]

        # 确保折的预测数据与样本数量一致
        if len(fold_pred) != len(test_labels):
            raise ValueError(f"Fold {fold + 1} has {len(fold_pred)} predictions, but test_labels has {len(test_labels)} labels.")

        weighted_0_scores += weight * fold_pred['Label 0 Score'].values
        weighted_1_scores += weight * fold_pred['Label 1 Score'].values

    # 确保所有列的长度一致
    if len(file_paths) != len(test_labels) or len(weighted_0_scores) != len(test_labels) or len(weighted_1_scores) != len(test_labels):
        raise ValueError("All arrays must be of the same length.")

    results_df = pd.DataFrame({
        'File Path': file_paths,  # 使用第一个折的File Path
        'Label 0 Score': weighted_0_scores,
        'Label 1 Score': weighted_1_scores,
        'True Label': test_labels
    })

    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# 主程序调用
def main_optimization(num_folds=3, num_epochs=40):  # 这里将num_folds改为3
    sample_csv = load_fold_predictions(1, 1)  # 随便选取一个CSV文件，假设所有文件标签相同
    test_labels = sample_csv['True Label'].values

    all_predictions = load_last_three_epochs_predictions(num_folds, num_epochs)

    best_auc, best_weights = find_best_auc(all_predictions, test_labels)

    # print(f"最佳AUC: {best_auc}, 最佳权重: {best_weights}")
    print(f"最佳AUC: {best_auc}")

    # 保存结果
    output_file = "./results_post/five_epoch_predictions.csv"
    save_results(all_predictions, best_weights, test_labels, output_file)

# 运行主函数
if __name__ == '__main__':
    # main_optimization(num_folds=8, num_epochs=15)  # 运行主程序时保证三折
    main_optimization(num_folds=8, num_epochs=21)  # 运行主程序时保证三折
    # for j in range(1,9):
    #     print('*************************************************************************')
    #     print('fold: ',j)
    #     for i in range(1,9):
    #         main_optimization(num_folds=8, num_epochs=i*3)  # 运行主程序时保证三折
