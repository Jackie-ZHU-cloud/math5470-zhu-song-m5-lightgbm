import os
import gc
from typing import List, Tuple

import numpy as np
import pandas as pd

# 将魔术数字和字符串定义为常量，便于维护
LAG_DAYS = [28, 29, 30, 35, 42]
ROLLING_WINDOWS = [7, 14, 28, 56]
ID_COLUMNS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    通过将数值类型向下转型为更小的类型来显著降低DataFrame的内存占用。

    参数:
        df (pd.DataFrame): 需要优化内存的DataFrame。
        verbose (bool): 是否打印内存优化前后的对比信息。

    返回:
        pd.DataFrame: 优化内存后的DataFrame。
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()

            if str(col_type).startswith('int'):
                # 遍历整数类型，寻找能容纳数据范围的最小类型
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:  # 浮点数类型
                # 遍历浮点数类型
                for dtype in [np.float16, np.float32, np.float64]:
                    if c_min >= np.finfo(dtype).min and c_max <= np.finfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f'内存使用从 {start_mem:5.2f} Mb 减少到 {end_mem:5.2f} Mb ({reduction:.1f}% 减少)')

    return df


def load_source_data(input_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载所有必需的原始CSV文件。"""
    print("步骤 1: 加载原始数据...")
    sales = pd.read_csv(f'{input_dir}sales_train_evaluation.csv')
    prices = pd.read_csv(f'{input_dir}sell_prices.csv')
    calendar = pd.read_csv(f'{input_dir}calendar.csv')
    return sales, prices, calendar


def melt_and_extend_grid(sales_df: pd.DataFrame, end_train: int, horizon: int) -> pd.DataFrame:
    """
    将销售数据从宽格式转换为长格式（melt），并为预测期添加行。
    """
    print("步骤 2: 转换数据格式并为预测期创建网格...")

    # Melt操作
    grid = pd.melt(sales_df, id_vars=ID_COLUMNS, var_name='d', value_name='sales')

    # 为预测期添加空行
    future_days = [f'd_{i}' for i in range(end_train + 1, end_train + 1 + horizon)]
    future_df = pd.DataFrame(sales_df[ID_COLUMNS].drop_duplicates())
    future_df = future_df.reindex(future_df.index.repeat(len(future_days))).reset_index(drop=True)
    future_df['d'] = np.tile(future_days, len(sales_df[ID_COLUMNS].drop_duplicates()))
    future_df['sales'] = np.nan

    # 合并历史和未来数据
    grid = pd.concat([grid, future_df], ignore_index=True)

    # 释放内存
    del sales_df, future_df
    gc.collect()

    return grid


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """从日期列创建时间相关的特征。"""
    print("步骤 3.1: 创建时间特征...")
    df['date'] = pd.to_datetime(df['date'])
    df['tm_day'] = df['date'].dt.day.astype(np.int8)
    df['tm_week'] = df['date'].dt.isocalendar().week.astype(np.int8)
    df['tm_month'] = df['date'].dt.month.astype(np.int8)
    df['tm_year'] = (df['date'].dt.year - df['date'].dt.year.min()).astype(np.int8)
    df['tm_dayofweek'] = df['date'].dt.dayofweek.astype(np.int8)
    df['tm_is_weekend'] = (df['tm_dayofweek'] >= 5).astype(np.int8)
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建与价格相关的特征，如相对价格。"""
    print("步骤 3.2: 创建价格相关特征...")

    # 计算均值
    df['item_avg_price'] = df.groupby('item_id')['sell_price'].transform('mean')
    df['dept_avg_price'] = df.groupby('dept_id')['sell_price'].transform('mean')

    # 计算相对价格
    df['price_vs_item_avg'] = (df['sell_price'] / df['item_avg_price']).fillna(1.0)
    df['price_vs_dept_avg'] = (df['sell_price'] / df['dept_avg_price']).fillna(1.0)

    # 移除中间列
    df.drop(columns=['item_avg_price', 'dept_avg_price'], inplace=True)

    return df


def create_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建与历史销售额相关的特征，如目标编码。"""
    print("步骤 3.3: 创建销售额相关特征 (目标编码)...")
    df['item_avg_sales'] = df.groupby('item_id')['sales'].transform('mean')
    return df


def generate_lag_and_rolling_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    为销售额创建滞后特征和滚动窗口特征。
    """
    print("步骤 4: 创建滞后和滚动特征...")

    # 只选择必要的列以节省内存
    lag_df = df[['id', 'd', 'sales']].copy()

    # 创建滞后特征
    for lag in LAG_DAYS:
        lag_df[f'sales_lag_{lag}'] = lag_df.groupby('id')['sales'].shift(lag)

    # 创建滚动窗口特征，基于固定的28天偏移
    shift_days = horizon
    for window in ROLLING_WINDOWS:
        lag_df[f'rolling_mean_{window}'] = lag_df.groupby('id')['sales'].transform(
            lambda x: x.shift(shift_days).rolling(window).mean()
        )
        lag_df[f'rolling_std_{window}'] = lag_df.groupby('id')['sales'].transform(
            lambda x: x.shift(shift_days).rolling(window).std()
        )

    return lag_df.drop(columns=['sales'])


def process_and_feature_engineer(input_dir: str, working_dir: str, end_train: int, horizon: int) -> None:
    """
    统一的数据处理和特征工程主流程。
    该函数整合了加载、合并、特征创建和保存的全部逻辑。
    """
    print("===== [ETL Pipeline] 开始数据处理与特征工程 =====")
    os.makedirs(working_dir, exist_ok=True)

    # 定义输出文件路径
    merged_path = f'{working_dir}grid_merged.pkl'
    lags_path = f'{working_dir}features_lags.pkl'

    # 如果最终文件已存在，则跳过整个流程
    if os.path.exists(merged_path) and os.path.exists(lags_path):
        print("所有必需的特征文件已存在，跳过ETL流程。")
        print("===== [ETL Pipeline] 完毕 =====")
        return

    # 步骤 1 & 2: 加载数据，转换并扩展网格
    sales_df, prices_df, calendar_df = load_source_data(input_dir)
    grid = melt_and_extend_grid(sales_df, end_train, horizon)

    # 优化分类列和初始内存
    for col in ID_COLUMNS:
        grid[col] = grid[col].astype('category')
    grid = optimize_memory(grid)

    # 步骤 3: 合并数据并创建特征
    print("步骤 3: 合并日历和价格数据...")
    # 核心修复：在合并前确保连接键'd'的数据类型一致
    grid['d'] = grid['d'].str.replace('d_', '').astype(np.int16)
    calendar_df['d'] = calendar_df['d'].str.replace('d_', '').astype(np.int16)

    grid = pd.merge(grid, calendar_df, on='d', how='left')
    grid = pd.merge(grid, prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    del prices_df, calendar_df
    gc.collect()

    # 创建所有特征
    grid = create_time_features(grid)
    grid = create_price_features(grid)
    grid = create_sales_features(grid)

    # 再次优化内存并保存
    grid = optimize_memory(grid)
    grid.to_pickle(merged_path)
    print(f"合并后的主数据已保存至: {merged_path}")

    # 步骤 4: 创建计算密集的滞后和滚动特征
    lags_df = generate_lag_and_rolling_features(grid, horizon)
    lags_df = optimize_memory(lags_df)
    lags_df.to_pickle(lags_path)
    print(f"滞后与滚动特征已保存至: {lags_path}")

    print("===== [ETL Pipeline] 流程执行完毕！ =====")
