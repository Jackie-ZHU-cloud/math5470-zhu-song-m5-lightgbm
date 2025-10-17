import pickle

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns


INPUT_DIR = './input/'
WORKING_DIR = './working/'

# 确保工作目录存在
os.makedirs(WORKING_DIR, exist_ok=True)

# 训练周期定义
END_TRAIN = 1941  # 训练集中的最后一天
HORIZON = 28  # 预测未来28天


def visual_sanity_check(num_items_to_plot=5):
    """
    随机抽取几个商品，将其历史销量、模型原始预测以及后处理（取整）后的预测
    进行可视化对比，以进行健全性检查。

    参数:
        num_items_to_plot (int): 希望抽样检查的商品数量。
    """
    print("\n===== 开始执行代表性抽样与网格化可视化检查... =====")

    # --- 1. 加载数据 ---
    try:
        history = pd.read_csv(f'{INPUT_DIR}sales_train_evaluation.csv')
        submission = pd.read_csv(f'{WORKING_DIR}submission.csv')
        calendar = pd.read_csv(f'{INPUT_DIR}calendar.csv')
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。请确保所有必需文件都在正确路径下。")
        return

    # --- 2. 代表性抽样逻辑 ---
    print("正在进行代表性抽样...")

    all_ids = history['id'].unique()
    eval_ids = [i for i in all_ids if 'evaluation' in i]
    history_eval = history[history['id'].isin(eval_ids)]

    history_days = [f'd_{i}' for i in range(END_TRAIN - 180, END_TRAIN + 1)]

    # ******** 关键修正：使用 .copy() 来避免 SettingWithCopyWarning ********
    sales_summary = history_eval[['id'] + history_days].copy()

    sales_summary['mean_sales'] = sales_summary[history_days].mean(axis=1)

    sales_summary = sales_summary[['id', 'mean_sales']].sort_values('mean_sales').reset_index(drop=True)

    n = len(sales_summary)
    try:
        low_ids = sales_summary.iloc[:int(n * 0.2)].sample(2, random_state=38)['id'].tolist()
        mid_ids = sales_summary.iloc[int(n * 0.4):int(n * 0.6)].sample(2, random_state=38)['id'].tolist()
        high_ids = sales_summary.iloc[int(n * 0.8):].sample(2, random_state=38)['id'].tolist()
    except ValueError:
        print("警告：数据量不足以进行分层抽样，将采用随机抽样。")
        sample_ids = random.sample(eval_ids, 6)
    else:
        sample_ids = low_ids + mid_ids + high_ids
        print(f"抽样完成。低销量ID: {low_ids}, 中销量ID: {mid_ids}, 高销量ID: {high_ids}")

    # --- 3. 创建2x3的子图网格 ---
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 11))

    calendar['date'] = pd.to_datetime(calendar['date'])
    date_map = calendar[['d', 'date']]

    for ax, item_id in zip(axes.flatten(), sample_ids):
        item_history = history[history['id'] == item_id][history_days].T
        item_history.columns = ['sales']
        item_history.index.name = 'd'
        item_history = item_history.reset_index().merge(date_map, on='d')

        forecast_cols = [f'F{i}' for i in range(1, HORIZON + 1)]
        item_forecast = submission[submission['id'] == item_id][forecast_cols].T
        item_forecast.columns = ['sales']

        pred_start_day_num = END_TRAIN + 1
        pred_start_date = calendar[calendar['d'] == f'd_{pred_start_day_num}']['date'].iloc[0]
        pred_dates = pd.to_datetime(pd.date_range(start=pred_start_date, periods=HORIZON))
        item_forecast['date'] = pred_dates
        item_forecast['sales_postprocessed'] = np.round(item_forecast['sales'])

        ax.step(item_history['date'], item_history['sales'], where='mid',
                label='Historical', color='blue', alpha=0.8)
        ax.plot(item_forecast['date'], item_forecast['sales'], label='Raw Forecast',
                color='orange', marker='o', linestyle='--', markersize=4, alpha=0.8)
        ax.step(item_forecast['date'], item_forecast['sales_postprocessed'], where='mid',
                label='Rounded Forecast', color='green', linestyle='-', linewidth=2.5)
        ax.axvline(x=item_history['date'].iloc[--1], color='red', linestyle='--', label='Forecast Start')

        mean_val = sales_summary[sales_summary['id'] == item_id]['mean_sales'].iloc[0]
        ax.set_title(f'ID: ...{item_id[-25:]}\n(Avg Sales: {mean_val:.2f})', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize='small')

    fig.suptitle('Representative Sales Forecasts Check (Low, Medium, and High Volume Items)', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_feature_importance(avg_fi_df, num_features=30):
    """
    根据已聚合的平均特征重要性数据，绘制图表。

    参数:
        avg_fi_df (pd.DataFrame): 包含 'feature' 和 'importance' 列的、已聚合和排序的DataFrame。
        num_features (int): 图表中展示的最重要特征的数量。
    """
    print("\n===== 开始绘制特征重要性图表... =====")

    # 修改后的检查方式：使用 .empty 来判断DataFrame是否为空
    if avg_fi_df.empty:
        print("未提供任何特征重要性数据。")
        return

    # --- 特征名称映射字典 ---
    feature_name_mapping = {
        # --- 您已有的特征 ---
        'item_id': 'Item ID',
        'dept_id': 'Department ID',
        'cat_id': 'Category ID',
        'store_id': 'Store ID',
        'state_id': 'State ID',
        'month': 'Month',
        'year': 'Year',
        'wday': 'Weekday (WDay)',
        'event_name_1': 'Event Name 1',
        'event_type_1': 'Event Type 1',
        'snap_CA': 'SNAP Day (California)',
        'snap_TX': 'SNAP Day (Texas)',
        'snap_WI': 'SNAP Day (Wisconsin)',

        # --- 新增/补全的特征 ---

        # 基础特征 (Base Features)
        'sell_price': 'Sell Price',

        # 简单滞后特征 (Simple Lag Features)
        'sales_lag_28': 'Sales 28 Days Ago',
        'sales_lag_29': 'Sales 29 Days Ago',
        'sales_lag_30': 'Sales 30 Days Ago',
        'sales_lag_31': 'Sales 31 Days Ago',
        'sales_lag_32': 'Sales 32 Days Ago',
        'sales_lag_33': 'Sales 33 Days Ago',
        'sales_lag_34': 'Sales 34 Days Ago',
        'sales_lag_35': 'Sales 35 Days Ago',
        'sales_lag_36': 'Sales 36 Days Ago',  # 额外添加一些以防万一

        # 简单滚动统计 (Simple Rolling Statistics) - 窗口期 X
        'rolling_mean_7': '7-Day Rolling Mean Sales',
        'rolling_std_7': '7-Day Rolling Std Dev Sales',
        'rolling_mean_14': '14-Day Rolling Mean Sales',
        'rolling_std_14': '14-Day Rolling Std Dev Sales',
        'rolling_mean_28': '28-Day Rolling Mean Sales',
        'rolling_std_28': '28-Day Rolling Std Dev Sales',
        'rolling_mean_56': '56-Day Rolling Mean Sales',
        'rolling_std_56': '56-Day Rolling Std Dev Sales',

        # 带滞后的滚动统计 (Lagged Rolling Statistics) - 窗口期 X, 滞后期 Y
        'rmean_7_28': '7-Day Rolling Avg Sales (Lag 28)',
        'rmean_7_7': '7-Day Rolling Avg Sales (Lag 7)',
        'rmean_7_1': '7-Day Rolling Avg Sales (Lag 1)',
        'rmean_14_28': '14-Day Rolling Avg Sales (Lag 28)',

        # 时间趋势特征 (Time Trend Features)
        'tm_d': 'Trend (Day of Year)',
        'tm_w': 'Trend (Week of Year)',
        'tm_dw': 'Trend (Day of Week)',
        'tm_m': 'Trend (Month)',
        'tm_y': 'Trend (Year)',

        # 价格相关特征 (Price Features)
        'price_change': 'Price Change Rate',
        'price_momentum': 'Price Momentum (30 Days)',
    }
    # -------------------------------------------------------------

    # 函数现在接收的是已经聚合和排序好的DataFrame，所以不再需要以下步骤：
    # full_fi_df = pd.concat(feature_importances_list, axis=0)
    # avg_fi_df = full_fi_df.groupby('feature')['importance'].mean().reset_index()

    # 确保数据是按重要性降序排列的
    avg_fi_df = avg_fi_df.sort_values(by='importance', ascending=False)

    # 应用特征名称映射
    # 使用 .copy() 来避免 SettingWithCopyWarning
    top_features = avg_fi_df.head(num_features).copy()
    top_features['feature'] = top_features['feature'].apply(
        lambda x: feature_name_mapping.get(x, x)
    )

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, num_features * 0.35))

    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis', ax=ax)

    ax.set_title(f'Top {num_features} Feature Importances (Averaged Across All Models)', fontsize=16)
    ax.set_xlabel('Average Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()

    plot_path = f'{WORKING_DIR}feature_importance.png'
    plt.savefig(plot_path)
    print(f"特征重要性图表已保存至: {plot_path}")
    plt.show()


# 绘制平均损失曲线 (保持不变)
def plot_loss_curves(loss_histories_list):
    """
    计算并绘制所有模型在验证集上的平均RMSE损失曲线。
    """
    print("\n===== 开始计算并绘制平均损失曲线... =====")
    if not loss_histories_list:
        print("未收集到任何损失历史数据。")
        return

    loss_df = pd.DataFrame(loss_histories_list).transpose()
    mean_loss = loss_df.mean(axis=1)
    std_loss = loss_df.std(axis=1)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(mean_loss.index, mean_loss, label='Average Validation RMSE', color='royalblue', lw=2)

    ax.fill_between(mean_loss.index,
                    mean_loss - std_loss,
                    mean_loss + std_loss,
                    color='lightsteelblue',
                    alpha=0.4,
                    label='Std. Dev. of RMSE')

    # # 标注您观察到的最优迭代点
    # best_iter = mean_loss.argmin()
    # ax.axvline(x=best_iter, color='darkred', linestyle='--', lw=1, label=f'Lowest Avg. RMSE at Iter. {best_iter}')

    ax.set_title('Average Model Training Curve (Validation RMSE)', fontsize=16)
    ax.set_xlabel('Boosting Iteration', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()

    plot_path = f'{WORKING_DIR}loss_curve.png'
    plt.savefig(plot_path)

    print(f"平均损失曲线图表已保存至: {plot_path}")
    plt.show()


# 绘制预测表现对比图
def plot_prediction_performance(pred1, pred2, avg_pred, working_dir, period_start, horizon):
    """
    绘制两个独立模型和集成模型的聚合预测与真实值的对比图。
    (版本已增强，会自动缩放Y轴并为线条添加标记以提高清晰度)
    """
    print("\n===== 开始绘制预测表现对比图... =====")

    # 加载并准备数据 (与之前相同)
    base_grid = pd.read_pickle(f'{working_dir}grid_merged.pkl')
    actuals_df = base_grid[(base_grid['d'] > period_start) & (base_grid['d'] <= period_start + horizon)]
    actuals_by_day = actuals_df.groupby('d')['sales'].sum().reset_index().rename(columns={'sales': 'Actual Sales'})

    preds1_by_day = pred1.groupby('d')['sales'].sum().reset_index().rename(columns={'sales': 'Model 1 (Store-Dept)'})
    preds2_by_day = pred2.groupby('d')['sales'].sum().reset_index().rename(columns={'sales': 'Model 2 (Store-Cat)'})
    avg_pred_by_day = avg_pred.groupby('d')['prediction'].sum().reset_index().rename(
        columns={'prediction': 'Ensemble (Average)'})

    plot_df = actuals_by_day.merge(preds1_by_day, on='d').merge(preds2_by_day, on='d').merge(avg_pred_by_day, on='d')

    # 绘制图表
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))  # 稍微调整尺寸以适应更详细的视图

    ax.plot(plot_df['d'], plot_df['Actual Sales'], label='Actual Sales', color='black', lw=2.5, marker='o',
            markersize=6, zorder=5)
    ax.plot(plot_df['d'], plot_df['Model 1 (Store-Dept)'], label='Model 1 (Store-Dept)', color='dodgerblue',
            linestyle='--', lw=2, marker='^', markersize=5, alpha=0.7)
    ax.plot(plot_df['d'], plot_df['Model 2 (Store-Cat)'], label='Model 2 (Store-Cat)', color='orangered', linestyle=':',
            lw=2, marker='s', markersize=5, alpha=0.7)
    ax.plot(plot_df['d'], plot_df['Ensemble (Average)'], label='Ensemble (Average)', color='forestgreen', lw=2.5,
            alpha=0.9)

    value_cols = ['Actual Sales', 'Model 1 (Store-Dept)', 'Model 2 (Store-Cat)', 'Ensemble (Average)']
    min_val = plot_df[value_cols].min().min()
    max_val = plot_df[value_cols].max().max()
    padding = 0  # (max_val - min_val) * 0.05  # 计算5%的边距
    ax.set_ylim(33000, max_val + padding)  # 应用新的Y轴范围

    ax.set_title('Aggregated Sales: Actuals vs. Predictions (Zoomed View)', fontsize=18)
    ax.set_xlabel('Forecast Day (d)', fontsize=14)
    ax.set_ylabel('Total Sales Volume', fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 为了更好的可读性，让x轴的刻度更密集
    ax.set_xticks(plot_df['d'])
    fig.autofmt_xdate()  # 自动旋转x轴标签以防重叠

    plt.tight_layout()

    # 保存图表
    plot_path = f'{working_dir}prediction_performance.png'
    plt.savefig(plot_path)

    print(f"预测表现对比图已保存至: {plot_path}")
    plt.show()


def plot_error_vs_horizon(analysis_df, working_dir, period_start):
    """
    绘制预测误差随预测时域（1-28天）的变化。

    参数:
        analysis_df (pd.DataFrame): 包含 'd', 'sales' (真实值), 和 'prediction' 的DataFrame。
        working_dir (str): 保存图表的工作目录。
    """
    print("\n--- 正在生成: 误差 vs. 预测时域图 ---")

    # 计算预测的第几天 (1 to 28)
    analysis_df['forecast_day'] = analysis_df['d'] - period_start

    # 计算绝对误差
    analysis_df['error'] = (analysis_df['prediction'] - analysis_df['sales']).abs()

    # 按预测天数聚合，计算平均绝对误差 (MAE)
    mae_by_day = analysis_df.groupby('forecast_day')['error'].mean().reset_index()

    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(mae_by_day['forecast_day'], mae_by_day['error'], marker='o', linestyle='-', color='teal')

    ax.set_title('Mean Absolute Error (MAE) vs. Forecast Horizon', fontsize=18)
    ax.set_xlabel('Day into Forecast Horizon', fontsize=14)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14)
    ax.set_xticks(np.arange(1, HORIZON + 1, 2))  # 每隔两天显示一个刻度
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plot_path = f'{working_dir}error_vs_horizon.png'
    plt.savefig(plot_path)
    print(f"图表已保存至: {plot_path}")
    plt.show()


def plot_residuals_distribution(analysis_df, working_dir):
    """
    绘制预测误差（残差）的分布直方图。

    参数:
        analysis_df (pd.DataFrame): 包含 'sales' (真实值) 和 'prediction' 的DataFrame。
        working_dir (str): 保存图表的工作目录。
    """
    print("\n--- 正在生成: 预测误差分布图 ---")

    # 计算残差
    analysis_df['residuals'] = analysis_df['prediction'] - analysis_df['sales']

    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(analysis_df['residuals'], kde=True, ax=ax, color='skyblue', bins=100)

    mean_res = analysis_df['residuals'].mean()
    ax.axvline(mean_res, color='red', linestyle='--', label=f'Mean Error: {mean_res:.2f}')
    ax.axvline(0, color='black', linestyle='-', label='Ideal (Error = 0)')

    ax.set_title('Distribution of Prediction Errors (Residuals)', fontsize=18)
    ax.set_xlabel('Error (Prediction - Actual)', fontsize=14)
    ax.set_ylabel('Frequency / Density', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plot_path = f'{working_dir}residuals_distribution.png'
    plt.savefig(plot_path)
    print(f"图表已保存至: {plot_path}")
    plt.show()


def plot_performance_by_group(analysis_df, group_col, id_to_name_map, working_dir):
    """
    按给定的分组（如品类或州）分别绘制预测性能图。

    参数:
        analysis_df (pd.DataFrame): 包含所有必要列的DataFrame。
        group_col (str): 用于分组的列名 ('cat_id' or 'state_id')。
        id_to_name_map (dict): 将ID映射到可读名称的字典。
        working_dir (str): 保存图表的工作目录。
    """
    group_name = group_col.replace('_id', '').capitalize()
    print(f"\n--- 正在生成: 按 {group_name} 剖分的性能图 ---")

    unique_groups = analysis_df[group_col].unique()

    fig, axes = plt.subplots(len(unique_groups), 1, figsize=(15, 6 * len(unique_groups)), sharex=True)
    fig.suptitle(f'Aggregated Sales Performance by {group_name}', fontsize=20, y=0.93)

    for i, group_id in enumerate(unique_groups):
        ax = axes[i]
        group_df = analysis_df[analysis_df[group_col] == group_id]

        actuals_by_day = group_df.groupby('d')['sales'].sum()
        preds_by_day = group_df.groupby('d')['prediction'].sum()

        group_label = id_to_name_map.get(group_id, f"ID {group_id}")

        ax.plot(actuals_by_day.index, actuals_by_day.values, label='Actual Sales', color='black', marker='o',
                markersize=4)
        ax.plot(preds_by_day.index, preds_by_day.values, label='Ensemble Prediction', color='forestgreen',
                linestyle='--')

        ax.set_title(f'Performance for {group_label}', fontsize=16)
        ax.set_ylabel('Total Sales Volume', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--')

    axes[-1].set_xlabel('Forecast Day (d)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plot_path = f'{working_dir}performance_by_{group_name.lower()}.png'
    plt.savefig(plot_path)
    print(f"图表已保存至: {plot_path}")
    plt.show()


# (已修改)
def run_advanced_visualizations(working_dir, end_train, horizon):
    """
    运行所有高级可视化分析函数的主函数。
    此版本现在在验证集上运行所有分析。
    """
    print("\n" + "=" * 50 + "\n===== 开始生成高级可视化分析图表... =====\n" + "=" * 50)

    # --- 1. 加载用于分析的数据 ---
    print("\n--- 正在加载验证集数据用于分析 ---")

    # 加载包含真实值和元数据的基础数据
    base_grid = pd.read_pickle(f'{working_dir}grid_merged.pkl')

    # 加载已保存的验证集预测
    valid_preds_path = f'{working_dir}viz_data_validation_predictions.pkl'
    with open(valid_preds_path, 'rb') as f:
        valid_preds_list = pickle.load(f)

    pred1_val = valid_preds_list[0]  # 模型1 (store-dept) 的验证预测
    pred2_val = valid_preds_list[1]  # 模型2 (store-cat) 的验证预测

    # 计算集成的验证集预测
    avg_pred_val = pred1_val.copy()
    avg_pred_val['sales'] = (pred1_val['sales'] + pred2_val['sales']) / 2

    # 筛选出验证集周期内的真实数据
    val_mask = (base_grid['d'] > end_train - horizon) & (base_grid['d'] <= end_train)
    analysis_df = base_grid[val_mask].copy()

    # 重命名集成预测的 'sales' 列为 'prediction' 以便合并
    avg_pred_val.rename(columns={'sales': 'prediction'}, inplace=True)

    # 将集成预测合并到主分析 DataFrame 中
    analysis_df = analysis_df.merge(avg_pred_val[['id', 'd', 'prediction']], on=['id', 'd'], how='left')

    if analysis_df['prediction'].isnull().any():
        print("警告: 合并预测时出现缺失值，部分分析可能不准确。")
        analysis_df.dropna(subset=['prediction'], inplace=True)

    # a. 总体性能图 (现在作用于验证集)
    plot_prediction_performance(pred1_val, pred2_val, avg_pred_val, working_dir, end_train - horizon, horizon)

    # b. 误差 vs. 时域
    plot_error_vs_horizon(analysis_df.copy(), working_dir, end_train - horizon)

    # c. 残差分布
    plot_residuals_distribution(analysis_df.copy(), working_dir)

    # d. 按品类剖分
    cat_id_map = {0: 'HOBBIES', 1: 'HOUSEHOLD', 2: 'FOODS'}
    plot_performance_by_group(analysis_df, 'cat_id', cat_id_map, working_dir)

    # e. 按州剖分
    state_id_map = {0: 'CA', 1: 'TX', 2: 'WI'}
    plot_performance_by_group(analysis_df, 'state_id', state_id_map, working_dir)


if __name__ == '__main__':
    # 随机画5个商品的图
    visual_sanity_check(num_items_to_plot=5)
