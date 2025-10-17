
使用 LightGBM GBDT，使用四类简单特征，四类复杂特征

简单特征主要分为以下四类：

**1. 基础ID与类别特征 (Basic ID and Categorical Features):**
*   这些特征主要用于让模型识别不同的实体。
    *   `item_id`: 商品ID
    *   `dept_id`: 部门ID
    *   `cat_id`: 品类ID
    *   `store_id`: 商店ID
    *   `state_id`: 州ID
    *   `event_name_1`, `event_type_1`: 事件名称和类型
    *   `event_name_2`, `event_type_2`: 第二事件名称和类型

**2. 日历与时间特征 (Calendar and Time Features):**
*   这些特征帮助模型捕捉与时间相关的模式，季节性、周期性。
    *   `tm_d`: 月份中的日期 (1-31)
    *   `tm_w`: 一年中的第几周 (1-53)
    *   `tm_m`: 月份 (1-12)
    *   `tm_y`: 年份 (从0开始的整数)
    *   `tm_dw`: 一周中的第几天 (0=周一, 6=周日)
    *   `tm_w_end`: 是否为周末

**3. 价格特征 (Price Feature):**
*   基础的价格特征。
    *   `sell_price`: 商品在当天的售价

**4. 时间序列特征 (Time Series Features):**
*   这些是基于历史销量计算得出的，用于捕捉商品的自身趋势和波动。
    *   **滞后特征 (Lag Features):**
        *   `sales_lag_28` 至 `sales_lag_35`: 过去第28天到第35天的单日销量。
    *   **滚动窗口特征 (Rolling Window Features):**
        *   `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`, `rolling_mean_56`: 基于过去不同时间窗口（7天、14天等）的**平均**销量。
        *   `rolling_std_7`, `rolling_std_14`, `rolling_std_28`, `rolling_std_56`: 基于过去不同时间窗口的销量**标准差**。

---
