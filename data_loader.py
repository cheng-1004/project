"""
資料載入和處理模組
Author: Your Name
Date: 2024
這個模組負責載入和處理TX期貨的OHLCV資料
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, List
import streamlit as st

class DataLoader:
    """
    資料載入器類別，負責從各種來源載入和處理OHLCV資料
    """
    
    def __init__(self, file_path: str = "output/kline_60min.txt"):
        self.file_path = file_path
        self.supported_encodings = ['utf-8', 'utf-8-sig', 'big5', 'gbk', 'cp950', 'latin1']
        
    def load_from_text_file(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        if file_path is None:
            file_path = self.file_path
            
        try:
            if not os.path.exists(file_path):
                st.error(f"資料檔案不存在: {file_path}")
                return None
            
            df = self._try_different_encodings(file_path)
            if df is None:
                st.error("無法使用任何支援的編碼格式讀取檔案")
                return None
            
            df = self._process_columns(df)
            if df is None:
                return None
            
            df = self._clean_and_validate(df)
            if df is None or len(df) == 0:
                st.error("處理後沒有有效的資料")
                return None
            
            st.success(f"成功載入 {len(df)} 筆資料")
            return df
            
        except Exception as e:
            st.error(f"載入資料時發生錯誤: {str(e)}")
            return None
    
    def _try_different_encodings(self, file_path: str) -> Optional[pd.DataFrame]:
        for encoding in self.supported_encodings:
            try:
                df = pd.read_csv(file_path, sep='\s+', encoding=encoding, on_bad_lines='skip')
                return df
            except:
                continue
        return None
    
    def _process_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if len(df.columns) == 8:
                df.columns = ['日期', '時間', '開盤', '最高', '最低', '收盤', '成交量', '成交值']
            elif len(df.columns) == 7:
                df.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', '成交值']
            elif len(df.columns) == 6:
                df.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量']
            else:
                st.error(f"不支援的欄位數量: {len(df.columns)}")
                return None
            
            df = self._process_datetime(df)
            if df is None:
                return None
            
            column_mapping = {
                '完整時間': 'datetime',
                '日期': 'datetime',
                '時間': 'time_only',
                '開盤': 'open', 
                '最高': 'high',
                '最低': 'low',
                '收盤': 'close',
                '成交量': 'volume',
                '成交值': 'turnover'
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            return df
        except Exception as e:
            st.error(f"處理欄位時發生錯誤: {str(e)}")
            return None
    
    def _process_datetime(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if '日期' in df.columns and '時間' in df.columns:
                df['完整時間'] = df['日期'].astype(str) + ' ' + df['時間'].astype(str)
                datetime_col = '完整時間'
            elif '日期' in df.columns:
                datetime_col = '日期'
            else:
                return None
            
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            df = df.dropna(subset=[datetime_col])
            return df
        except Exception as e:
            st.error(f"處理日期時間時發生錯誤: {str(e)}")
            return None
    
    def _clean_and_validate(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            # 解決 'datetime' is not unique 問題
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            essential_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            df = df.dropna(subset=[col for col in essential_columns if col in df.columns])
            df = self._validate_ohlc(df)
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
        except Exception as e:
            st.error(f"清理資料時發生錯誤: {str(e)}")
            return None
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[
            (df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close']) &
            (df['open'] > 0) & (df['high'] > 0) &
            (df['low'] > 0) & (df['close'] > 0) &
            (df['volume'] >= 0)
        ]
        return df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            return {}
        return {
            'total_records': len(df),
            'date_range': {
                'start': df['datetime'].min(),
                'end': df['datetime'].max()
            },
            'price_range': {
                'min_low': df['low'].min(),
                'max_high': df['high'].max(),
                'current_price': df['close'].iloc[-1]
            }
        }

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df
        df_resampled = df.set_index('datetime').resample(timeframe).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        return df_resampled

def calculate_basic_metrics(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 2: return {}
    curr, prev = df['close'].iloc[-1], df['close'].iloc[-2]
    return {
        'current_price': curr,
        'price_change': curr - prev,
        'price_change_pct': ((curr - prev) / prev) * 100,
        'period_high': df['high'].max(),
        'period_low': df['low'].min(),
        'total_volume': df['volume'].sum()
    }

def create_test_data(num_bars: int = 100) -> pd.DataFrame:
    dates = pd.date_range('2024-01-01', periods=num_bars, freq='H')
    df = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(num_bars).cumsum() + 15000,
        'high': np.random.randn(num_bars).cumsum() + 15010,
        'low': np.random.randn(num_bars).cumsum() + 14990,
        'close': np.random.randn(num_bars).cumsum() + 15000,
        'volume': np.random.randint(100, 1000, num_bars)
    })
    return df