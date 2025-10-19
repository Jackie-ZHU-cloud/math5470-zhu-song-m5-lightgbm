import os

import numpy as np
import pandas as pd


def reduce_mem_usage(df, verbose=True):
    """
    通过将数值类型转换为更小的类型来降低DataFrame的内存占用。
    例如，将 float64 转换为 float32，int64 转换为 int16 等。

    参数:
        df (pd.DataFrame): 需要优化内存的DataFrame。
        verbose (bool): 是否打印内存优化信息。

    返回:
        pd.DataFrame: 优化后的DataFrame。
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            f'内存使用从 {start_mem:5.2f} Mb 减少到 {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% 减少)')
    return df


def merge_by_concat(df1, df2, merge_on):
    """
    通过 concat 方法合并两个DataFrame，以防止pandas自动转换数据类型。

    参数:
        df1 (pd.DataFrame): 主DataFrame。
        df2 (pd.DataFrame): 需要合并的DataFrame。
        merge_on (list): 合并的键。

    返回:
        pd.DataFrame: 合并后的DataFrame。
    """
    merged_gf = df1[merge_on].merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    return pd.concat([df1, merged_gf[new_columns]], axis=1)


# 数据预处理与特征工程
def prepare_data_and_features(WORKING_DIR, INPUT_DIR, HORIZON, END_TRAIN):
    print("===== 开始数据准备和特征工程 =====")

    # 步骤 1: 创建基础的销售数据网格
    if not os.path.exists(f'{WORKING_DIR}base.pkl'):
        print("\n步骤 1: 创建基础销售数据网格...")
        eva = pd.read_csv(f'{INPUT_DIR}sales_train_evaluation.csv')
        index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        grid = pd.melt(eva, id_vars=index_columns, var_name='d', value_name='sales')

        # 为预测期添加空行
        for i in range(1, HORIZON + 1):
            temp_df = eva[index_columns].drop_duplicates()
            temp_df['d'] = f'd_{END_TRAIN + i}'
            temp_df['sales'] = np.nan
            grid = pd.concat([grid, temp_df])

        grid = grid.reset_index(drop=True)
        for col in index_columns:
            grid[col] = grid[col].astype('category')
        grid = reduce_mem_usage(grid)
        grid.to_pickle(f'{WORKING_DIR}base.pkl')
        del grid, eva
    else:
        print("\n文件 'base.pkl' 已存在")

    # 步骤 2: 合并日历、价格信息，并创建时间特征
    if not os.path.exists(f'{WORKING_DIR}grid_merged.pkl'):
        print("\n步骤 2: 合并日历、价格信息...")
        grid = pd.read_pickle(f'{WORKING_DIR}base.pkl')
        calendar = pd.read_csv(f'{INPUT_DIR}calendar.csv')
        sell_prices = pd.read_csv(f'{INPUT_DIR}sell_prices.csv')

        grid['d'] = grid['d'].str.replace('d_', '').astype(int)
        calendar['d'] = calendar['d'].str.replace('d_', '').astype(int)

        # 合并日历
        grid = grid.merge(calendar, on='d', how='left')
        # 合并价格
        grid = grid.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

        # 创建日期相关特征
        grid['date'] = pd.to_datetime(grid['date'])
        grid['tm_d'] = grid['date'].dt.day.astype(np.int8)
        grid['tm_w'] = grid['date'].dt.isocalendar().week.astype(np.int8)
        grid['tm_m'] = grid['date'].dt.month.astype(np.int8)
        grid['tm_y'] = (grid['date'].dt.year - grid['date'].dt.year.min()).astype(np.int8)
        grid['tm_dw'] = grid['date'].dt.dayofweek.astype(np.int8)
        grid['tm_w_end'] = (grid['tm_dw'] >= 5).astype(np.int8)

        grid = reduce_mem_usage(grid)
        grid.to_pickle(f'{WORKING_DIR}grid_merged.pkl')
        del grid, calendar, sell_prices
    else:
        print("\n文件 'grid_merged.pkl' 已存在")

    # 步骤 3: 创建滞后和滚动特征
    if not os.path.exists(f'{WORKING_DIR}features_lags.pkl'):
        print("\n步骤 3: 创建滞后和滚动特征...")
        grid = pd.read_pickle(f'{WORKING_DIR}grid_merged.pkl')[['id', 'd', 'sales']]

        # 创建滞后特征 (shifted features)
        for lag in [28, 29, 30, 31, 32, 33, 34, 35]:
            grid[f'sales_lag_{lag}'] = grid.groupby('id')['sales'].transform(lambda x: x.shift(lag))

        # 创建滚动窗口特征 (rolling window features)
        for window in [7, 14, 28, 56]:
            grid[f'rolling_mean_{window}'] = grid.groupby('id')['sales'].transform(
                lambda x: x.shift(28).rolling(window).mean())
            grid[f'rolling_std_{window}'] = grid.groupby('id')['sales'].transform(
                lambda x: x.shift(28).rolling(window).std())

        grid = reduce_mem_usage(grid.drop(columns=['sales']))
        grid.to_pickle(f'{WORKING_DIR}features_lags.pkl')
        del grid
    else:
        print("\n文件 'features_lags.pkl' 已存在")

    print("\n===== 数据准备和特征工程完成！ =====")
