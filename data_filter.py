import os
import sys
import json
import logging
import argparse
import warnings
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm
import matplotlib

import utils

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class BenchmarkDataConstructor:
    
    def __init__(self, 
                 stk_folder:str = r"data\stk_data",
                 pft_folder:str = r"data\pft_data",):
        """
        初始化StockMarket类
        """
        self.stk_folder:str = os.path.normpath(stk_folder)  # 股票数据文件夹路径
        self.pft_folder:str = os.path.normpath(pft_folder)  # 财务数据文件夹路径
        self.raw_data:Dict[str, pd.DataFrame] = {}  # 用于存储_rawdata数据
        self.processed_data:Dict[str, pd.DataFrame] = {}  # 用于存储处理后数据
        self.merged_data:pd.DataFrame = None  # 用于存储合并后的数据
        self.quarterly_data:pd.DataFrame = None  # 用于存储季度数据

        self.logger:logging.Logger = logging

    def set_logger(self, 
                   name:str="Benchmark DataProcessing",
                   log_file:Optional[str]=None):
        self.logger = utils.LoggerPreparer(name, 
                                           log_file=log_file, 
                                           console_level=logging.DEBUG).prepare()

    @utils.log_exceptions_inclass()
    def load_csv_files(self):
        """
        载入指定文件夹中的所有CSV文件：
        1. 读取每个CSV文件
        2. 创建{原始文件名}_rawdata副本存储在self.raw_data中，后续计算出错的话再从这里copy原始数据
        3. 创建{原始文件名}副本存储在self.processed_data中，后续用这里的数据进行计算
        4. 返回处理结果统计

        返回:
            dict: 包含处理文件的数量和名称
        """
        self.logger.info("开始载入CSV文件")

        for filename in tqdm(os.listdir(self.stk_folder), desc="载入数据"):
            if filename.endswith(".csv"):
                if filename.startswith("~$"):
                    # 跳过临时文件
                    self.logger.debug(f"跳过临时文件: {filename}")
                    continue
                try:
                    file_path = os.path.join(self.stk_folder, filename)

                    # 读取CSV文件
                    df = pd.read_csv(file_path, parse_dates=["Trddt"])
                    df["Stkcd"] = df["Stkcd"].astype(str).str.zfill(6)

                    # 存储{原始文件名}_rawdata副本
                    rawdata_key = f"{os.path.splitext(filename)[0]}"
                    self.raw_data[rawdata_key] = df.copy()
                    self.logger.info(f"载入成功: {filename} -> 存储为 {rawdata_key} 和 {filename}")

                except Exception as e:
                    self.logger.warning(f"载入失败: {filename} - 错误: {str(e)}")

                mem_info = utils.get_memory_usage()
                self.logger.debug(
                    f"当前内存占用: {mem_info['used_memory_mb']:.2f} MB ({mem_info['percent_of_total']:.2f}%)."
                )
        for filename in tqdm(os.listdir(self.pft_folder), desc="载入数据"):
            if filename.endswith(".csv"):
                if filename.startswith("~$"):
                    # 跳过临时文件
                    self.logger.debug(f"跳过临时文件: {filename}")
                    continue
                try:
                    file_path = os.path.join(self.pft_folder, filename)

                    # 读取CSV文件
                    df = pd.read_csv(file_path, parse_dates=["Accper", "Annodt"])
                    df["Stkcd"] = df["Stkcd"].astype(str).str.zfill(6)

                    # 存储{原始文件名}_rawdata副本
                    rawdata_key = f"{os.path.splitext(filename)[0]}"
                    self.raw_data[rawdata_key] = df.copy()
                    self.logger.info(f"载入成功: {filename} -> 存储为 {rawdata_key} 和 {filename}")

                except Exception as e:
                    self.logger.warning(f"载入失败: {filename} - 错误: {str(e)}")

                mem_info = utils.get_memory_usage()
                self.logger.debug(
                    f"当前内存占用: {mem_info['used_memory_mb']:.2f} MB ({mem_info['percent_of_total']:.2f}%)."
                )

    @utils.log_exceptions_inclass()
    def merge_all_data(self) -> None:
        """
        按照指定规则合并所有数据文件
        返回最终合并后的DataFrame
        """
        self.logger.info("开始合并所有数据")
        # 第一步: 合并TRD_Mnth和FS_Combas
        merged_df = pd.concat(
            [self.raw_data["TRD_Dalyr0"], self.raw_data["TRD_Dalyr1"], self.raw_data["TRD_Dalyr2"], self.raw_data["TRD_Dalyr3"], self.raw_data["TRD_Dalyr4"], self.raw_data["TRD_Dalyr5"]],
            axis=0
        )
        self.processed_data["stk_data"] = merged_df.copy()
        self.processed_data["pft_data"] = self.raw_data["IAR_Rept"].copy()

    def count_num(self) -> None:
        # 提取年份和季度
        df = self.processed_data["stk_data"]
        df['year'] = df['Trddt'].dt.year
        df['quarter'] = df['Trddt'].dt.quarter
        
        # 按年月季度分组，统计唯一公司数量
        result = df.groupby(['year', 'quarter'])['Stkcd'].nunique().reset_index()
        result.columns = ['Year', 'Quarter', 'Company_Count']
        
        # 按年份和季度排序
        result = result.sort_values(['Year', 'Quarter'])
        result.to_csv("company_count_by_quarter.csv", index=False)

    @utils.log_exceptions_inclass()
    def filter_data(self):

        # 准备存储净利润断层数据的列表
        net_profit_gap_list = []
        stk_data = self.processed_data["stk_data"].dropna()
        pft_data = self.processed_data["pft_data"].dropna()

        stk_data = stk_data.sort_values(['Stkcd', 'Trddt'])
        
        # 为股票数据添加前一日最高价(使用shift)
        stk_data['prev_Hiprc'] = stk_data.groupby('Stkcd')['Hiprc'].shift(1)
        
        # 合并数据 - 找到每个公告后的第一个交易日
        # 使用merge_asof找到每个公告日期后最近的交易日
        merged = pd.merge_asof(
            pft_data.sort_values('Annodt'),
            stk_data.sort_values('Trddt'),
            left_on='Annodt',
            right_on='Trddt',
            by='Stkcd',
            direction='forward'
        )
        
        # 筛选条件: 当日最低价 > 前一日最高价
        result = merged[merged['Loprc'] > merged['prev_Hiprc']].copy()
        
        # 添加跳空幅度计算
        result['gap_pct'] = (result['Loprc'] - result['prev_Hiprc']) / result['prev_Hiprc'] * 100

        return result[['Stkcd', 'Reptyp', 'Annodt', 'Trddt', 'Loprc', 'prev_Hiprc', 'gap_pct']]
                
           
    

if __name__ == "__main__":
    data_constructor = BenchmarkDataConstructor()
    data_constructor.set_logger(name="Benchmark DataProcessing")
    data_constructor.load_csv_files()
    data_constructor.merge_all_data()
    data_constructor.processed_data["stk_data"].to_csv("merged_stk_data.csv", index=False)
    data_constructor.processed_data["pft_data"].to_csv("merged_pft_data.csv", index=False)
    result = data_constructor.filter_data()
    result.to_csv("filtered_data.csv", index=False)
    data_constructor.logger.info("数据处理完成")