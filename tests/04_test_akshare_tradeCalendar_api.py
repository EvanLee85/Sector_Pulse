import akshare as ak
import pandas as pd
from datetime import datetime, date, timedelta

# 获取数据并检查类型
df = ak.tool_trade_date_hist_sina()
print(f"数据类型: {df['trade_date'].dtype}")
print(f"总行数: {len(df)}")
print("前5行:")
print(df.head())
print("后5行:")
print(df.tail())

# 转换为datetime类型
df['trade_date'] = pd.to_datetime(df['trade_date'])

# 现在可以提取年份
df['year'] = df['trade_date'].dt.year
print(f"2024年交易日数量: {len(df[df['year'] == 2024])}")
print(f"2025年交易日数量: {len(df[df['year'] == 2025])}")

# 查看2025年的数据范围
df_2025 = df[df['year'] == 2025]
if len(df_2025) > 0:
    print(f"2025年第一个交易日: {df_2025['trade_date'].min()}")
    print(f"2025年最后一个交易日: {df_2025['trade_date'].max()}")

# 测试今天
today = datetime.now().date()
trading_dates = set(df['trade_date'].dt.date)
print(f"今天({today})是交易日: {today in trading_dates}")