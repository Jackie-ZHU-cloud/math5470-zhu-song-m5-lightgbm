import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from lightgbm.callback import early_stopping, record_evaluation
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import prepare_data_and_features
from visual import plot_feature_importance, plot_loss_curves

INPUT_DIR = './input/'
WORKING_DIR = './working/'

# 确保工作目录存在
os.makedirs(WORKING_DIR, exist_ok=True)

# 训练周期定义
END_TRAIN = 1941  # 训练集中的最后一天
HORIZON = 28  # 预测未来28天

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
    'n_estimators': 1400,
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
            grid[col] = grid[col].astype('category')
            grid[col] = grid[col].cat.codes.astype("int16")
            grid[col] -= grid[col].min()

    remove_features = ['id', 'sales', 'date', 'wm_yr_wk', 'weekday', 'd']
    features = [col for col in grid.columns if col not in remove_features]

    all_predictions = pd.DataFrame()
    feature_importances_list = []
    loss_histories = []  # 用于存储每个模型的损失历史

    unique_groups = grid[grouping_cols].drop_duplicates().values

    for group_values in unique_groups:
        group_filter = (grid[grouping_cols[0]] == group_values[0])
        if len(grouping_cols) > 1:
            group_filter &= (grid[grouping_cols[1]] == group_values[1])

        print(f"\n--- 正在处理分组: {group_values} ---")
        sub_grid = grid[group_filter]

        train_mask = (sub_grid['d'] <= END_TRAIN - HORIZON)
        valid_mask = (sub_grid['d'] > END_TRAIN - HORIZON) & (sub_grid['d'] <= END_TRAIN)
        preds_mask = (sub_grid['d'] > END_TRAIN)

        X_train = sub_grid[train_mask][features]
        y_train = sub_grid[train_mask]['sales']
        X_valid = sub_grid[valid_mask][features]
        y_valid = sub_grid[valid_mask]['sales']
        X_test = sub_grid[preds_mask][features]

        # 用于记录评估结果的字典
        evals_result = {}

        # 训练模型
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='rmse',
                  callbacks=[
                      early_stopping(50, verbose=False),
                      record_evaluation(evals_result)  # 添加记录回调
                  ])

        # 收集损失历史
        loss_histories.append(evals_result['valid_0']['rmse'])

        # 收集特征重要性
        fi_df = pd.DataFrame({'feature': model.feature_name_, 'importance': model.feature_importances_})
        feature_importances_list.append(fi_df)

        # 预测
        predictions = model.predict(X_test, num_iteration=model.best_iteration_)
        temp_preds = sub_grid[preds_mask][['id', 'd']]
        temp_preds['sales'] = predictions
        all_predictions = pd.concat([all_predictions, temp_preds])

    all_predictions.to_pickle(f'{WORKING_DIR}{output_filename}')
    print(f"===== 按 {grouping_cols} 分组的预测已保存至 {output_filename} =====")

    return feature_importances_list, loss_histories


# 生成提交文件
def generate_submission_file():
    """整合所有模型的预测结果，计算平均值，并生成最终的提交文件。"""
    print("\n===== 开始生成最终提交文件... =====")
    pred1 = pd.read_pickle(f'{WORKING_DIR}preds_by_store_dept.pkl')
    pred2 = pd.read_pickle(f'{WORKING_DIR}preds_by_store_cat.pkl')
    pred1.set_index(['id', 'd'], inplace=True)
    pred2.set_index(['id', 'd'], inplace=True)
    df_avg = (pred1 + pred2) / 2
    df_avg.reset_index(inplace=True)
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
    prepare_data_and_features(WORKING_DIR, INPUT_DIR, HORIZON, END_TRAIN)

    all_feature_importances = []
    all_loss_histories = []

    # 策略一: 按商店和部门分组
    fi_list_1, loss_list_1 = train_and_predict(
        grouping_cols=['store_id', 'dept_id'],
        output_filename='preds_by_store_dept.pkl'
    )
    all_feature_importances.extend(fi_list_1)
    all_loss_histories.extend(loss_list_1)

    # 策略二: 按商店和品类分组
    fi_list_2, loss_list_2 = train_and_predict(
        grouping_cols=['store_id', 'cat_id'],
        output_filename='preds_by_store_cat.pkl'
    )
    all_feature_importances.extend(fi_list_2)
    all_loss_histories.extend(loss_list_2)

    # 绘制并保存特征重要性图表
    plot_feature_importance(all_feature_importances)

    # 绘制并保存平均损失曲线图表
    plot_loss_curves(all_loss_histories)

    generate_submission_file()
