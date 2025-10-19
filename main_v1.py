import pickle
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from lightgbm.callback import early_stopping, record_evaluation
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any

from etl_pipeline import process_and_feature_engineer
# 假设visual.py文件存在，包含可视化函数
from visual import plot_feature_importance, plot_loss_curves, plot_prediction_performance, run_advanced_visualizations

# --- 常量定义 ---
INPUT_DIR = './input/'
WORKING_DIR = './working/'
END_TRAIN = 1941  # 训练集中的最后一天
HORIZON = 28  # 预测未来28天
RUN_TRAINING = True
RUN_VISUALIZATION = True

# --- LightGBM 模型超参数 ---
LGB_PARAMS = {
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
    'n_estimators': 600,
    'boost_from_average': False,
    'verbosity': -1,
    'n_jobs': -1,  # 使用所有可用核心
}


# --- 辅助函数 ---
def load_processed_data(working_dir: str) -> pd.DataFrame:
    """加载并合并预处理后的数据。"""
    print("加载预处理数据...")
    base_grid_path = f'{working_dir}grid_merged.pkl'
    lags_path = f'{working_dir}features_lags.pkl'
    base_grid = pd.read_pickle(base_grid_path)
    lag_features = pd.read_pickle(lags_path)
    grid = pd.merge(base_grid, lag_features, on=['id', 'd'], how='left')
    del base_grid, lag_features
    return grid


def preprocess_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """自动检测并转换对象类型的分类特征为数值编码。"""
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        df[col] = df[col].astype('category').cat.codes.astype("int16")
        # 确保编码从0开始
        if df[col].min() < 0:
            df[col] -= df[col].min()
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """自动获取特征列列表，排除已知的非特征列。"""
    remove_cols = ['id', 'sales', 'date', 'wm_yr_wk', 'weekday', 'd']
    return [col for col in df.columns if col not in remove_cols]


def create_time_based_masks(df: pd.DataFrame, end_train: int, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    根据时间创建训练集、验证集和测试集的布尔掩码。
    这是对原split_data函数的重构，以避免复杂的元组解包。
    """
    train_mask = df['d'] <= end_train - horizon
    valid_mask = (df['d'] > end_train - horizon) & (df['d'] <= end_train)
    preds_mask = df['d'] > end_train
    return train_mask, valid_mask, preds_mask


def train_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
                     lgbm_params: Dict[str, Any]) -> Tuple[lgb.LGBMRegressor, Dict]:
    """训练LightGBM模型并返回模型和评估结果。"""
    model = lgb.LGBMRegressor(**lgbm_params)
    evals_result = {}
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              callbacks=[early_stopping(50, verbose=False), record_evaluation(evals_result)])
    return model, evals_result


def generate_predictions(model: lgb.LGBMRegressor, X_test: pd.DataFrame, X_valid: pd.DataFrame,
                         sub_grid: pd.DataFrame, valid_mask: pd.Series, preds_mask: pd.Series,
                         lgbm_params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """为测试集和验证集生成预测。"""
    best_iteration = model.best_iteration_ if model.best_iteration_ else lgbm_params['n_estimators']

    test_predictions = model.predict(X_test, num_iteration=best_iteration)
    valid_predictions = model.predict(X_valid, num_iteration=best_iteration)

    preds_df = sub_grid.loc[preds_mask, ['id', 'd']].copy()
    preds_df['sales'] = test_predictions

    valid_preds_df = sub_grid.loc[valid_mask, ['id', 'd']].copy()
    valid_preds_df['sales'] = valid_predictions

    return preds_df, valid_preds_df


def log_training_results(model: lgb.LGBMRegressor, evals_result: Dict, feature_importances_list: List,
                         loss_histories: List, group_name: Tuple) -> None:
    """记录模型训练结果（特征重要性、损失历史）。"""
    fi_df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_,
        'group': '_'.join(map(str, group_name))
    })
    feature_importances_list.append(fi_df)

    if 'valid_0' in evals_result and 'rmse' in evals_result['valid_0']:
        loss_histories.append(evals_result['valid_0']['rmse'])
    else:
        print(f"警告: 验证集RMSE未在evals_result中找到。Group: {group_name}")


# --- 核心流程 ---
def train_and_predict(grouping_cols: List[str], output_filename: str, working_dir: str, end_train: int, horizon: int,
                      lgbm_params: Dict) -> Tuple[pd.DataFrame, List, pd.DataFrame]:
    """
    根据给定的分组列，为每个数据子集训练一个LightGBM模型并进行预测。
    """
    print(f"\n===== 开始按 {grouping_cols} 分组进行训练和预测... =====")
    grid = load_processed_data(working_dir)
    grid = preprocess_categorical_features(grid)
    features = get_feature_columns(grid)

    all_predictions_list, all_valid_predictions_list, feature_importances_list, loss_histories = [], [], [], []

    unique_groups = grid[grouping_cols].drop_duplicates().values

    for group_values in unique_groups:
        group_filter = np.logical_and.reduce([grid[col] == val for col, val in zip(grouping_cols, group_values)])

        print(f"\n--- 正在处理分组: {dict(zip(grouping_cols, group_values))} ---")
        sub_grid = grid[group_filter]

        if sub_grid.empty:
            print("警告: 分组数据为空，已跳过。")
            continue

        train_mask, valid_mask, preds_mask = create_time_based_masks(sub_grid, end_train, horizon)

        # 检查是否有足够的训练和验证数据
        if not train_mask.any() or not valid_mask.any():
            print("警告: 缺乏足够的训练或验证数据，已跳过。")
            continue

        X_train, y_train = sub_grid.loc[train_mask, features], sub_grid.loc[train_mask, 'sales']
        X_valid, y_valid = sub_grid.loc[valid_mask, features], sub_grid.loc[valid_mask, 'sales']
        X_test = sub_grid.loc[preds_mask, features]

        model, evals_result = train_lgbm_model(X_train, y_train, X_valid, y_valid, lgbm_params)

        log_training_results(model, evals_result, feature_importances_list, loss_histories, tuple(group_values))

        preds_df, valid_preds_df = generate_predictions(model, X_test, X_valid, sub_grid, valid_mask, preds_mask, lgbm_params)
        all_predictions_list.append(preds_df)
        all_valid_predictions_list.append(valid_preds_df)

    # 整合结果
    all_predictions = pd.concat(all_predictions_list, ignore_index=True)
    all_valid_predictions = pd.concat(all_valid_predictions_list, ignore_index=True)
    all_feature_importances = pd.concat(feature_importances_list,
                                        ignore_index=True) if feature_importances_list else pd.DataFrame()

    all_predictions.to_pickle(f'{working_dir}{output_filename}')
    print(f"===== 按 {grouping_cols} 分组的预测已保存至 {output_filename} =====")

    return all_feature_importances, loss_histories, all_valid_predictions


def generate_submission_file(working_dir: str, horizon: int) -> None:
    """生成最终的提交文件。"""
    print("\n===== 开始生成最终提交文件... =====")
    try:
        pred1 = pd.read_pickle(f'{working_dir}preds_by_store_dept.pkl')
        pred2 = pd.read_pickle(f'{working_dir}preds_by_store_cat.pkl')
    except FileNotFoundError as e:
        print(f"错误: 预测文件缺失 - {e}。请确保训练已成功运行。")
        return

    df_avg = pred1.copy()
    df_avg['sales'] = 0.5 * pred1['sales'] + 0.5 * pred2['sales']

    df_pivot = df_avg.pivot(index='id', columns='d', values='sales').reset_index()

    forecast_cols = [f'F{i}' for i in range(1, horizon + 1)]
    day_cols = sorted([col for col in df_pivot.columns if isinstance(col, (int, np.int16))])
    df_pivot = df_pivot[['id'] + day_cols]
    df_pivot.columns = ['id'] + forecast_cols

    df_validation = df_pivot.copy()
    df_validation['id'] = df_validation['id'].str.replace('_evaluation', '_validation')

    full_submission = pd.concat([df_validation, df_pivot], ignore_index=True)

    try:
        sample_sub = pd.read_csv(f'{INPUT_DIR}sample_submission.csv')
        final_submission = pd.merge(sample_sub[['id']], full_submission, on='id', how='left').fillna(0)
    except FileNotFoundError:
        print(f"警告: {INPUT_DIR}sample_submission.csv 未找到。将直接使用生成的提交数据。")
        final_submission = full_submission.fillna(0)

    submission_path = f'{working_dir}submission.csv'
    final_submission.to_csv(submission_path, index=False)
    print(f"提交文件已成功生成: {submission_path}")
    print("\n提交文件预览:\n", final_submission.head())


def save_visualization_data(fi_list: List, loss_list: List, valid_preds_list: List, working_dir: str) -> None:
    """保存用于可视化的中间数据。"""
    print("\n===== 正在保存用于可视化的中间数据... =====")
    if fi_list:
        combined_fi_df = pd.concat(fi_list, ignore_index=True)
        fi_path = f'{working_dir}viz_data_feature_importances.pkl'
        combined_fi_df.to_pickle(fi_path)
        print(f"特征重要性数据已保存至: {fi_path}")

    if loss_list:
        loss_path = f'{working_dir}viz_data_loss_histories.pkl'
        with open(loss_path, 'wb') as f: pickle.dump(loss_list, f)
        print(f"损失历史数据已保存至: {loss_path}")

    if valid_preds_list:
        valid_preds_path = f'{working_dir}viz_data_validation_predictions.pkl'
        with open(valid_preds_path, 'wb') as f: pickle.dump(valid_preds_list, f)
        print(f"验证集预测数据已保存至: {valid_preds_path}")


def run_visualization_stage(working_dir: str, end_train: int, horizon: int) -> None:
    """执行数据可视化阶段。"""
    print("\n" + "=" * 60 + "\n===== [阶段 2] 开始执行: 数据可视化 =====\n" + "=" * 60)

    fi_path = f'{working_dir}viz_data_feature_importances.pkl'
    if os.path.exists(fi_path):
        fi_df = pd.read_pickle(fi_path)
        print(f"成功加载: {fi_path}")
        avg_fi_df = fi_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
        plot_feature_importance(avg_fi_df)
    else:
        print(f"未找到特征重要性文件: {fi_path}")

    loss_path = f'{working_dir}viz_data_loss_histories.pkl'
    if os.path.exists(loss_path):
        with open(loss_path, 'rb') as f:
            loss_histories = pickle.load(f)
        print(f"成功加载: {loss_path}")
        plot_loss_curves(loss_histories)
    else:
        print(f"未找到损失历史文件: {loss_path}")

    try:
        run_advanced_visualizations(working_dir, end_train, horizon)
    except Exception as e:
        print(f"高级可视化执行时出错: {e}")

    print("\n===== [阶段 2] 执行完毕 =====")


# --- 主程序入口 ---
if __name__ == '__main__':
    os.makedirs(WORKING_DIR, exist_ok=True)

    if RUN_TRAINING:
        print("=" * 60 + "\n===== [阶段 1] 开始执行: 模型训练与数据生成 =====\n" + "=" * 60)

        process_and_feature_engineer(INPUT_DIR, WORKING_DIR, END_TRAIN, HORIZON)

        training_strategies = {
            'store_dept': ['store_id', 'dept_id'],
            'store_cat': ['store_id', 'cat_id'],
        }

        all_fi, all_losses, all_valid_preds = [], [], []

        for name, cols in training_strategies.items():
            try:
                fi_df, loss_list, valid_preds = train_and_predict(
                    grouping_cols=cols,
                    output_filename=f'preds_by_{name}.pkl',
                    working_dir=WORKING_DIR,
                    end_train=END_TRAIN,
                    horizon=HORIZON,
                    lgbm_params=LGB_PARAMS
                )
                if not fi_df.empty: all_fi.append(fi_df)
                all_losses.extend(loss_list)
                all_valid_preds.append(valid_preds)
            except Exception as e:
                print(f"对分组 {cols} 的训练策略执行失败: {e}")

        save_visualization_data(all_fi, all_losses, all_valid_preds, WORKING_DIR)

        generate_submission_file(WORKING_DIR, HORIZON)
        print("\n===== [阶段 1] 执行完毕 =====")

    if RUN_VISUALIZATION:
        run_visualization_stage(WORKING_DIR, END_TRAIN, HORIZON)
