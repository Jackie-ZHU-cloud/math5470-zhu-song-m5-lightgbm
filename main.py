import pickle

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from lightgbm.callback import early_stopping, record_evaluation
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import prepare_data_and_features
from visual import plot_feature_importance, plot_loss_curves, plot_prediction_performance, run_advanced_visualizations

INPUT_DIR = './input/'
WORKING_DIR = './working/'

# 确保工作目录存在
os.makedirs(WORKING_DIR, exist_ok=True)

# 训练周期定义
END_TRAIN = 1941  # 训练集中的最后一天
HORIZON = 28  # 预测未来28天

RUN_TRAINING = True
RUN_VISUALIZATION = True

# LightGBM 模型超参数
LGB_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'learning_rate': 0.03,
    'num_leaves': 2 ** 11 - 1,
    'min_data_in_leaf': 2 ** 12 - 1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    # 'n_estimators': 1400,
    'n_estimators': 600,
    'boost_from_average': False,
    'verbosity': -1,
    'n_jobs': -1,  # 使用所有可用核心
}


# 模型训练与预测
def train_and_predict(grouping_cols, output_filename):
    """
    根据给定的分组列，为每个数据子集训练一个LightGBM模型并进行预测。
    同时，收集每个模型的特征重要性和损失历史。

    参数:
        grouping_cols (list): 用于划分数据子集的列名。
        output_filename (str): 保存预测结果的文件名。

    返回:
        tuple: (特征重要性DataFrame列表, 损失历史列表)
    """
    print(f"\n===== 开始按 {grouping_cols} 分组进行训练和预测... =====")
    base_grid = pd.read_pickle(f'{WORKING_DIR}grid_merged.pkl')
    lag_features = pd.read_pickle(f'{WORKING_DIR}features_lags.pkl')
    grid = base_grid.merge(lag_features, on=['id', 'd'], how='left')
    del base_grid, lag_features
    cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in cat_features:
        if grid[col].dtype == 'object':
            grid[col] = grid[col].astype('category').cat.codes.astype("int16")
            grid[col] -= grid[col].min()
    remove_features = ['id', 'sales', 'date', 'wm_yr_wk', 'weekday', 'd']
    features = [col for col in grid.columns if col not in remove_features]

    all_predictions = pd.DataFrame()
    all_valid_predictions = pd.DataFrame()
    feature_importances_list = []
    loss_histories = []
    unique_groups = grid[grouping_cols].drop_duplicates().values

    for group_values in unique_groups:
        group_filter = (grid[grouping_cols[0]] == group_values[0])
        if len(grouping_cols) > 1: group_filter &= (grid[grouping_cols[1]] == group_values[1])
        print(f"\n--- 正在处理分组: {group_values} ---")
        sub_grid = grid[group_filter]
        train_mask = (sub_grid['d'] <= END_TRAIN - HORIZON)
        valid_mask = (sub_grid['d'] > END_TRAIN - HORIZON) & (sub_grid['d'] <= END_TRAIN)
        preds_mask = (sub_grid['d'] > END_TRAIN)
        X_train, y_train = sub_grid[train_mask][features], sub_grid[train_mask]['sales']
        X_valid, y_valid = sub_grid[valid_mask][features], sub_grid[valid_mask]['sales']
        X_test = sub_grid[preds_mask][features]
        evals_result = {}

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse',
                  callbacks=[early_stopping(50, verbose=False), record_evaluation(evals_result)])

        loss_histories.append(evals_result['valid_0']['rmse'])
        fi_df = pd.DataFrame({'feature': model.feature_name_, 'importance': model.feature_importances_})
        feature_importances_list.append(fi_df)

        # 1. 预测测试集 (用于最终提交)
        test_predictions = model.predict(X_test, num_iteration=model.best_iteration_)
        temp_preds = sub_grid[preds_mask][['id', 'd']]
        temp_preds['sales'] = test_predictions
        all_predictions = pd.concat([all_predictions, temp_preds])

        # 2. 预测验证集 (用于性能分析)
        valid_predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
        temp_valid_preds = sub_grid[valid_mask][['id', 'd']]
        temp_valid_preds['sales'] = valid_predictions
        all_valid_predictions = pd.concat([all_valid_predictions, temp_valid_preds])

    all_predictions.to_pickle(f'{WORKING_DIR}{output_filename}')
    print(f"===== 按 {grouping_cols} 分组的预测已保存至 {output_filename} =====")

    return feature_importances_list, loss_histories, all_valid_predictions


# 生成提交文件
def generate_submission_file():
    print("\n===== 开始生成最终提交文件... =====")
    pred1 = pd.read_pickle(f'{WORKING_DIR}preds_by_store_dept.pkl')
    pred2 = pd.read_pickle(f'{WORKING_DIR}preds_by_store_cat.pkl')
    df_avg = pred1.copy()
    df_avg['sales'] = (pred1['sales'] + pred2['sales']) / 2
    df_pivot = df_avg.pivot(index='id', columns='d', values='sales').reset_index()
    day_cols = sorted([col for col in df_pivot.columns if isinstance(col, (int, np.int16))])
    df_pivot = df_pivot[['id'] + day_cols]
    forecast_cols = [f'F{i}' for i in range(1, HORIZON + 1)]
    df_pivot.columns = ['id'] + forecast_cols
    df_validation = df_pivot.copy()
    df_validation['id'] = df_validation['id'].str.replace('_evaluation', '_validation')
    df_evaluation = df_pivot
    full_submission = pd.concat([df_validation, df_evaluation], ignore_index=True)
    sample_sub = pd.read_csv(f'{INPUT_DIR}sample_submission.csv')
    final_submission = sample_sub[['id']].merge(full_submission, on='id', how='left')
    final_submission.fillna(0, inplace=True)
    submission_path = f'{WORKING_DIR}submission.csv'
    final_submission.to_csv(submission_path, index=False)
    print(f"提交文件已成功生成: {submission_path}")
    print("\n提交文件预览:")
    print(final_submission.head())


# 主函数入口
if __name__ == '__main__':
    if RUN_TRAINING:
        print("=" * 60 + "\n===== [阶段 1] 开始执行: 模型训练与数据生成 =====\n" + "=" * 60)

        prepare_data_and_features(WORKING_DIR, INPUT_DIR, HORIZON, END_TRAIN)

        all_feature_importances = []
        all_loss_histories = []
        all_valid_preds_list = []

        # 策略一: 按商店和部门分组
        fi_list_1, loss_list_1, valid_preds_1 = train_and_predict(
            grouping_cols=['store_id', 'dept_id'],
            output_filename='preds_by_store_dept.pkl'
        )
        all_feature_importances.extend(fi_list_1)
        all_loss_histories.extend(loss_list_1)
        all_valid_preds_list.append(valid_preds_1)

        # 策略二: 按商店和品类分组
        fi_list_2, loss_list_2, valid_preds_2 = train_and_predict(
            grouping_cols=['store_id', 'cat_id'],
            output_filename='preds_by_store_cat.pkl'
        )
        all_feature_importances.extend(fi_list_2)
        all_loss_histories.extend(loss_list_2)
        all_valid_preds_list.append(valid_preds_2)

        print("\n===== 正在保存用于可视化的中间数据... =====")

        # 保存特征重要性 和 损失历史
        fi_path = f'{WORKING_DIR}viz_data_feature_importances.pkl'
        pd.concat(all_feature_importances).to_pickle(fi_path)
        print(f"特征重要性数据已保存至: {fi_path}")
        loss_path = f'{WORKING_DIR}viz_data_loss_histories.pkl'
        with open(loss_path, 'wb') as f: pickle.dump(all_loss_histories, f)
        print(f"损失历史数据已保存至: {loss_path}")

        valid_preds_path = f'{WORKING_DIR}viz_data_validation_predictions.pkl'
        with open(valid_preds_path, 'wb') as f:
            pickle.dump(all_valid_preds_list, f)
        print(f"验证集预测数据已保存至: {valid_preds_path}")

        generate_submission_file()
        print("\n===== [阶段 1] 执行完毕 =====")

    # 可视化
    if RUN_VISUALIZATION:
        print("\n" + "=" * 60)
        print("===== [阶段 2] 开始执行: 数据可视化 =====")
        print("=" * 60)

        print("\n===== 正在加载用于可视化的数据... =====")

        # 加载特征重要性
        fi_path = f'{WORKING_DIR}viz_data_feature_importances.pkl'
        fi_df = pd.read_pickle(fi_path)
        print(f"成功加载: {fi_path}")

        # 加载损失历史
        loss_path = f'{WORKING_DIR}viz_data_loss_histories.pkl'
        with open(loss_path, 'rb') as f:
            loss_histories = pickle.load(f)
        print(f"成功加载: {loss_path}")

        # 1. 特征重要性图
        avg_fi_df = fi_df.groupby('feature')['importance'].mean().reset_index()
        plot_feature_importance(avg_fi_df)

        # 2. 损失曲线图
        plot_loss_curves(loss_histories)

        # 3. 所有高级性能分析图 (函数现在自己加载数据)
        run_advanced_visualizations(WORKING_DIR, END_TRAIN, HORIZON)

        print("\n===== [阶段 2] 执行完毕 =====")
