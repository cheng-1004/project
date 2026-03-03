"""
趨勢線和突破點檢測模組
Author: Your Name
Date: 2024

這個模組提供自動檢測趨勢線和突破點的功能
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime
MIN_TOUCHES = 3
MIN_BREAKOUT_PCT = 0.02


class TrendlineBreakoutDetector:
    """
    自動檢測趨勢線和突破點的類別
    
    參數:
    - swing_window: 搖擺點檢測的視窗大小 (預設: 3)
    - min_touches: 趨勢線最少接觸點數 (預設: 2)  
    - breakout_threshold: 突破閥值 (預設: 0.001, 即0.1%)
    - lookback_bars: 用於分析的最近K棒數量 (預設: 100)
    """
    
    def __init__(self, swing_window: int = 3, min_touches: int = 2, 
                 breakout_threshold: float = 0.001, lookback_bars: int = 100):
        self.swing_window = swing_window
        self.min_touches = min_touches
        self.breakout_threshold = breakout_threshold
        self.lookback_bars = lookback_bars
        
        # 驗證參數
        self._validate_parameters()
    
    def _validate_parameters(self):
        """驗證初始化參數的有效性"""
        if self.swing_window < 1:
            raise ValueError("swing_window must be at least 1")
        if self.min_touches < 2:
            raise ValueError("min_touches must be at least 2")
        if self.breakout_threshold <= 0:
            raise ValueError("breakout_threshold must be positive")
        if self.lookback_bars < 10:
            raise ValueError("lookback_bars must be at least 10")
    
    def find_swing_points(self, df: pd.DataFrame) -> Dict[str, List[Tuple]]:
        """
        識別搖擺高點和低點
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            Dict with 'highs' and 'lows' keys, each containing list of (index, datetime, price) tuples
        """
        if df.empty:
            return {'highs': [], 'lows': []}
        
        # 限制資料範圍到最近的lookback_bars
        if len(df) > self.lookback_bars:
            df = df.tail(self.lookback_bars).copy()
        
        df = df.reset_index(drop=True)
        swing_highs = []
        swing_lows = []
        
        # 確保有足夠的資料進行分析
        if len(df) <= 2 * self.swing_window:
            return {'highs': [], 'lows': []}
        
        for i in range(self.swing_window, len(df) - self.swing_window):
            # 檢查搖擺高點
            is_swing_high = True
            current_high = df.iloc[i]['high']
            
            # 檢查左側和右側的視窗
            for j in range(i - self.swing_window, i + self.swing_window + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append((i, df.iloc[i]['datetime'], current_high))
            
            # 檢查搖擺低點
            is_swing_low = True
            current_low = df.iloc[i]['low']
            
            # 檢查左側和右側的視窗
            for j in range(i - self.swing_window, i + self.swing_window + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append((i, df.iloc[i]['datetime'], current_low))
        
        return {'highs': swing_highs, 'lows': swing_lows}
    
    def calculate_line_params(self, point1: Tuple, point2: Tuple) -> Tuple[float, float]:
        """
        計算兩點間直線的參數 (斜率和截距)
        
        Args:
            point1, point2: (index, datetime, price) tuples
            
        Returns:
            (slope, intercept) tuple
        """
        x1, _, y1 = point1
        x2, _, y2 = point2
        
        if x2 - x1 == 0:
            return float('inf'), y1
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        return slope, intercept
    
    def get_line_value(self, slope: float, intercept: float, x: int) -> float:
        """
        根據直線參數計算給定x值的y值
        
        Args:
            slope: 直線斜率
            intercept: 直線截距
            x: x座標值
            
        Returns:
            對應的y座標值
        """
        if slope == float('inf'):
            return intercept
        return slope * x + intercept
    
    def find_trendlines(self, swing_points: List[Tuple]) -> List[Dict]:
        """
        從搖擺點中找出有效的趨勢線
        
        Args:
            swing_points: List of (index, datetime, price) tuples
            
        Returns:
            List of trendline dictionaries with keys: 'points', 'slope', 'intercept', 'touches'
        """
        if len(swing_points) < 2:
            return []
        
        trendlines = []
        
        # 檢查每對搖擺點
        for i in range(len(swing_points)):
            for j in range(i + 1, len(swing_points)):
                point1 = swing_points[i]
                point2 = swing_points[j]
                
                slope, intercept = self.calculate_line_params(point1, point2)
                
                # 計算有多少其他點接觸此趨勢線
                touches = [point1, point2]
                tolerance = abs(point1[2] * self.breakout_threshold)  # 使用價格的百分比作為容差
                
                for k, point in enumerate(swing_points):
                    if k != i and k != j:
                        expected_price = self.get_line_value(slope, intercept, point[0])
                        if abs(point[2] - expected_price) <= tolerance:
                            touches.append(point)
                
                # 如果接觸點數量達到最小要求，則加入趨勢線
                if len(touches) >= self.min_touches:
                    trendlines.append({
                        'points': touches,
                        'slope': slope,
                        'intercept': intercept,
                        'touches': len(touches),
                        'start_point': point1,
                        'end_point': point2,
                        'strength_score': self._calculate_strength_score(touches, slope)
                    })
        
        # 按接觸點數量和強度排序，優先返回最強的趨勢線
        trendlines.sort(key=lambda x: (x['touches'], x['strength_score']), reverse=True)
        
        return trendlines
    
    def _calculate_strength_score(self, touches: List[Tuple], slope: float) -> float:
        """
        計算趨勢線的強度評分
        
        Args:
            touches: 接觸點列表
            slope: 趨勢線斜率
            
        Returns:
            強度評分 (越高越強)
        """
        # 基礎分數基於接觸點數量
        base_score = len(touches)
        
        # 根據時間跨度調整分數 (時間跨度越長，趨勢線越可靠)
        if len(touches) >= 2:
            time_span = touches[-1][0] - touches[0][0]
            time_bonus = min(time_span / 50, 2.0)  # 最多加2分
        else:
            time_bonus = 0
        
        # 根據斜率調整分數 (適中的斜率更可靠)
        slope_penalty = min(abs(slope) / 10, 1.0)  # 過度陡峭的線扣分
        
        return base_score + time_bonus - slope_penalty
    
    def check_breakouts(self, df: pd.DataFrame,
                    support_lines: List[Dict],
                    resistance_lines: List[Dict]) -> List[Dict]:

        if df.empty:
            return []

        breakouts = []
        latest_bar = df.iloc[-1]
        latest_index = len(df) - 1

        # ========= 阻力突破 =========
        for resistance in resistance_lines:

            if resistance['touches'] < MIN_TOUCHES:
                continue

            resistance_price = self.get_line_value(
                resistance['slope'],
                resistance['intercept'],
                latest_index
            )

            breakout_pct = (latest_bar['close'] - resistance_price) / resistance_price
            if breakout_pct < MIN_BREAKOUT_PCT:
                continue

            if latest_bar['close'] > resistance_price * (1 + self.breakout_threshold):
                breakouts.append({
                    'datetime': latest_bar['datetime'],
                    'price': latest_bar['close'],
                    'direction': 'bullish_breakout',
                    'trendline_price': resistance_price,
                    'trendline_points': resistance['points'],
                    'strength': resistance['touches'],
                    'strength_score': resistance['strength_score'],
                    'breakout_magnitude': breakout_pct
                })

        # ========= 支撐跌破 =========
        for support in support_lines:

            if support['touches'] < MIN_TOUCHES:
                continue

            support_price = self.get_line_value(
                support['slope'],
                support['intercept'],
                latest_index
            )

            breakout_pct = (support_price - latest_bar['close']) / support_price
            if breakout_pct < MIN_BREAKOUT_PCT:
                continue

            if latest_bar['close'] < support_price * (1 - self.breakout_threshold):
                breakouts.append({
                    'datetime': latest_bar['datetime'],
                    'price': latest_bar['close'],
                    'direction': 'bearish_breakdown',
                    'trendline_price': support_price,
                    'trendline_points': support['points'],
                    'strength': support['touches'],
                    'strength_score': support['strength_score'],
                    'breakout_magnitude': breakout_pct
                })

        # ========= 同一根 K 棒只留最強 =========
        from collections import defaultdict

        grouped = defaultdict(list)
        for b in breakouts:
            grouped[b['datetime']].append(b)

        filtered = []
        for items in grouped.values():
            filtered.append(max(items, key=lambda x: x['breakout_magnitude']))

        return filtered


    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        主要分析函數 - 執行完整的趨勢線和突破點分析
        
        Args:
            df: DataFrame with columns ['datetime', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - swing_points: Dict with 'highs' and 'lows'
            - support_lines: List of support trendlines
            - resistance_lines: List of resistance trendlines  
            - breakouts: List of detected breakouts
            - summary: Analysis summary statistics
        """
        # 驗證輸入資料
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            return self._empty_analysis_result()
        
        # 確保資料按時間排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 1. 找出搖擺點
        swing_points = self.find_swing_points(df)
        
        # 2. 構建趨勢線
        support_lines = self.find_trendlines(swing_points['lows'])
        resistance_lines = self.find_trendlines(swing_points['highs'])
        
        # 3. 過濾出正確方向的趨勢線
        support_lines = [line for line in support_lines if line['slope'] >= -0.1]  # 允許略微下降的支撐線
        resistance_lines = [line for line in resistance_lines if line['slope'] <= 0.1]  # 允許略微上升的阻力線
        
        # 4. 檢查突破點
        breakouts = self.check_breakouts(df, support_lines, resistance_lines)
        
        # 5. 生成分析摘要
        summary = self._generate_summary(swing_points, support_lines, resistance_lines, breakouts)
        
        return {
            'swing_points': swing_points,
            'support_lines': support_lines,
            'resistance_lines': resistance_lines,
            'breakouts': breakouts,
            'summary': summary
        }
    
    def _empty_analysis_result(self) -> Dict:
        """返回空的分析結果"""
        return {
            'swing_points': {'highs': [], 'lows': []},
            'support_lines': [],
            'resistance_lines': [],
            'breakouts': [],
            'summary': {
                'swing_highs_count': 0,
                'swing_lows_count': 0,
                'support_lines_count': 0,
                'resistance_lines_count': 0,
                'breakouts_count': 0,
                'strongest_support_strength': 0,
                'strongest_resistance_strength': 0
            }
        }
    
    def _generate_summary(self, swing_points: Dict, support_lines: List, 
                         resistance_lines: List, breakouts: List) -> Dict:
        """生成分析摘要統計"""
        return {
            'swing_highs_count': len(swing_points['highs']),
            'swing_lows_count': len(swing_points['lows']),
            'support_lines_count': len(support_lines),
            'resistance_lines_count': len(resistance_lines),
            'breakouts_count': len(breakouts),
            'strongest_support_strength': max([line['touches'] for line in support_lines], default=0),
            'strongest_resistance_strength': max([line['touches'] for line in resistance_lines], default=0),
            'analysis_timeframe': f"Last {self.lookback_bars} bars",
            'swing_window': self.swing_window,
            'breakout_threshold': f"{self.breakout_threshold*100:.1f}%"
        }
    
    def get_trendline_coordinates(self, trendline: Dict, df_length: int, 
                                 extend_future: int = 10) -> List[Tuple]:
        """
        獲取趨勢線的座標點用於繪圖
        
        Args:
            trendline: Trendline dictionary from analyze()
            df_length: Length of the dataframe
            extend_future: Number of bars to extend the line into the future
            
        Returns:
            List of (index, price) tuples for plotting
        """
        if not trendline['points']:
            return []
        
        # 找出趨勢線的起始和結束索引
        start_idx = min(point[0] for point in trendline['points'])
        end_idx = min(df_length - 1 + extend_future, df_length + 50)  # 延伸到未來但有限制
        
        coordinates = []
        for idx in range(start_idx, end_idx + 1):
            price = self.get_line_value(trendline['slope'], trendline['intercept'], idx)
            if price > 0:  # 確保價格為正數
                coordinates.append((idx, price))
        
        return coordinates


def create_sample_data(num_bars: int = 100, start_price: float = 1000.0) -> pd.DataFrame:
    """
    創建範例OHLCV資料用於測試
    
    Args:
        num_bars: 資料筆數
        start_price: 起始價格
        
    Returns:
        包含OHLCV資料的DataFrame
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=num_bars, freq='H')
    
    # 模擬趨勢價格
    trend = np.linspace(0, 100, num_bars)
    noise = np.random.randn(num_bars) * 20
    base_prices = start_price + trend + noise
    
    # 生成OHLC資料
    opens = base_prices + np.random.randn(num_bars) * 5
    closes = opens + np.random.randn(num_bars) * 10
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(num_bars)) * 8
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(num_bars)) * 8
    volumes = np.random.randint(1000, 10000, num_bars)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df


if __name__ == "__main__":
    # 測試範例
    print("=== 趨勢線檢測器測試 ===")
    
    # 創建測試資料
    test_data = create_sample_data(150, 1000)
    print(f"創建了 {len(test_data)} 筆測試資料")
    
    # 初始化檢測器
    detector = TrendlineBreakoutDetector(
        swing_window=3,
        min_touches=2,
        breakout_threshold=0.005,  # 0.5%
        lookback_bars=100
    )
    
    # 執行分析
    results = detector.analyze(test_data)
    
    # 輸出結果摘要
    summary = results['summary']
    print(f"\n分析結果摘要:")
    print(f"- 搖擺高點: {summary['swing_highs_count']}")
    print(f"- 搖擺低點: {summary['swing_lows_count']}")
    print(f"- 支撐線: {summary['support_lines_count']}")
    print(f"- 阻力線: {summary['resistance_lines_count']}")
    print(f"- 突破點: {summary['breakouts_count']}")
    print(f"- 最強支撐線強度: {summary['strongest_support_strength']}")
    print(f"- 最強阻力線強度: {summary['strongest_resistance_strength']}")
    
    # 顯示突破點詳情
    if results['breakouts']:
        print(f"\n檢測到的突破點:")
        for i, breakout in enumerate(results['breakouts'], 1):
            print(f"{i}. {breakout['direction']} at {breakout['datetime']}")
            print(f"   價格: {breakout['price']:.2f}, 強度: {breakout['strength']}")
            print(f"   突破幅度: {breakout['breakout_magnitude']*100:.2f}%")
    else:
        print("\n未檢測到突破點")
    
    print("\n測試完成！")