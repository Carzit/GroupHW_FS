import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

# 假设我们已经有以下数据：
# 1. gap_events - Profita断层识别结果（包含stkcd, Annodt, Trddt等）
# 2. fin_data - 财务数据（包含stkcd, Annodt, Profita等）
# 3. price_data - 股票价格数据（包含stkcd, 日期, 收盘价等）
# 4. 以上数据已经预处理并存储为CSV文件

# 预处理财务数据
def preprocess_financial_data(fin_data):
    """
    预处理财务数据，计算盈利增速
    """
    fin_data = fin_data.sort_values(['Stkcd', 'Annodt'])
    
    # 计算单季度盈利增速
    fin_data['profit_growth'] = fin_data.groupby('Stkcd')['Profita'].pct_change()
    
    # 计算复合盈利增速（过去2年）
    fin_data['profit_growth_2y'] = fin_data.groupby('Stkcd')['Profita'].pct_change(periods=4)
    
    return fin_data

def strategy_consistent_growth(gap_events:pd.DataFrame, fin_data:pd.DataFrame):
    """
    策略一：Profita断层+业绩持续向好
    筛选标准：
    1. 发生Profita断层
    2. 过去2年盈利增速均为正
    3. 复合盈利增速>20%
    """
    # 合并断层事件与财务数据
    merged = pd.merge_asof(
        gap_events.sort_values('Annodt'),
        fin_data.dropna().sort_values('Annodt'),
        left_on='Annodt',
        right_on='Annodt',
        by='Stkcd',
        direction='backward'
    )
    
    # 获取历史财务数据用于判断持续增长
    fin_data_past = fin_data.copy()
    fin_data_past['Annodt'] = fin_data_past['Annodt'] + pd.DateOffset(months=3)  # 调整为下季度
    
    # 合并过去8个季度的数据
    for i in range(1, 9):
        temp = fin_data.copy()
        temp['Annodt'] = temp['Annodt'] + pd.DateOffset(months=3*i)
        temp = temp[['Stkcd', 'Annodt', 'profit_growth']]
        temp = temp.rename(columns={'profit_growth': f'growth_{i}q_ago'})
        merged = pd.merge(merged, temp, on=['Stkcd', 'Annodt'], how='left')
    
    # 筛选条件
    condition = (
        (merged['profit_growth_2y'] > 50)  # 复合增速>20%
        #(merged['growth_1q_ago'] > 0)        # 上季度增长
        #& (merged['growth_2q_ago'] > 0)        # 上上季度增长
        #&(merged['growth_3q_ago'] > 0)        # 过去一年增长
        #& (merged['growth_4q_ago'] > 0)          # 过去一年增长
    )
    
    selected = merged[condition].copy()
    return selected

def backtest_strategy(selected_stocks:pd.DataFrame, 
                      price_data:pd.DataFrame, 
                      hold_period:int=90, 
                      index_name:str='中证1000',
                      benchmark=None):
    """
    回测选股策略
    参数:
        selected_stocks: 选中的股票DataFrame
        price_data: 股票价格数据
        hold_period: 持有期(天)
        benchmark: 基准指数数据
    """
    # 预处理价格数据
    price_data['Trddt'] = pd.to_datetime(price_data['Trddt'])
    
    # 计算未来持有期收益率
    selected_stocks['buy_date'] = pd.to_datetime(selected_stocks['Trddt'])
    selected_stocks['sell_date'] = selected_stocks['buy_date'] + pd.Timedelta(days=hold_period)
    
    # 合并买入价格
    selected_stocks = pd.merge_asof(
        selected_stocks.sort_values('buy_date'),
        price_data.rename(columns={'Trddt': 'buy_date', 'Clsprc': 'buy_price'}).sort_values('buy_date'),
        left_on='buy_date',
        right_on='buy_date',
        by='Stkcd',
        direction='backward'
    )
    
    # 合并卖出价格
    selected_stocks = pd.merge_asof(
        selected_stocks.sort_values('sell_date'),
        price_data.rename(columns={'Trddt': 'sell_date', 'Clsprc': 'sell_price'}).sort_values('sell_date'),
        left_on='sell_date',
        right_on='sell_date',
        by='Stkcd',
        direction='forward'
    )
    
    # 计算收益率
    selected_stocks['return'] = selected_stocks['sell_price'] / selected_stocks['buy_price'] - 1
    
    # 按买入日期分组计算组合收益率
    portfolio_return = selected_stocks.groupby('buy_date')['return'].mean().reset_index()
    
    # 计算累计收益率
    portfolio_return['cum_return'] = (1 + portfolio_return['return']).cumprod() - 1
    
    # 如果有基准，计算基准收益率
    if benchmark is not None:
        benchmark['date'] = pd.to_datetime(benchmark['date'])
        benchmark = benchmark.sort_values('date')
        benchmark = pd.merge_asof(
            portfolio_return[['buy_date']].rename(columns={'buy_date': 'date'}),
            benchmark,
            on='date',
            direction='backward'
        )
        # 处理可能的缺失值
        benchmark[index_name+'_close'] = benchmark[index_name+'_close'].ffill()
        
        # 计算收益率 - 使用log收益率更稳定
        benchmark['benchmark_return'] = np.log(
            benchmark[index_name+'_close'] / benchmark[index_name+'_close'].shift(1))
        
        # 对于第一个交易日，收益率为0（因为没有前一天数据）
        benchmark['benchmark_return'] = benchmark['benchmark_return'].fillna(0)
        benchmark['benchmark_cum'] = (1 + benchmark['benchmark_return']).cumprod() - 1
        portfolio_return = pd.merge(
            portfolio_return,
            benchmark[['date', 'benchmark_return', 'benchmark_cum']],
            left_on='buy_date',
            right_on='date',
            how='left'
        )
    
    return portfolio_return




if __name__ == "__main__":
    
    gap_events = pd.read_csv('filtered_data.csv', parse_dates=['Annodt', 'Trddt'])
    fin_data = pd.read_csv('merged_pft_data.csv', parse_dates=['Accper', 'Annodt'])
    price_data = pd.read_csv('merged_stk_data.csv', parse_dates=['Trddt'])
    index_data = pd.read_csv(r'data\else_data\指数数据.csv', parse_dates=['时间'])  # 假设有基准数据
    index_data.rename(columns={'时间': 'date'}, inplace=True)
    fin_data = preprocess_financial_data(fin_data)
    consistent_growth_stocks = strategy_consistent_growth(gap_events, fin_data)
    print(consistent_growth_stocks)

    # 回测策略一
    backtest_consistent = backtest_strategy(consistent_growth_stocks, price_data, benchmark=index_data, hold_period=90)
    print(backtest_consistent)