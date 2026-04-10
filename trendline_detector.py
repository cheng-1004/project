"""
趨勢線和突破點檢測模組 - 優化版
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime

class TrendlineBreakoutDetector:
    """
    自動檢測趨勢線和突破點的類別
    """
    
    def __init__(self, swing_window: int = 3, min_touches: int = 2, 
                 breakout_threshold: float = 0.001, lookback_bars: int = 200):
        self.swing_window = swing_window
        self.min_touches = min_touches
        self.breakout_threshold = breakout_threshold
        self.lookback_bars = lookback_bars
        self._validate_parameters()
    
    def _validate_parameters(self):
        if self.swing_window < 1: self.swing_window = 1
        if self.min_touches < 2: self.min_touches = 2
        if self.breakout_threshold <= 0: self.breakout_threshold = 0.0005

    def find_swing_points(self, df: pd.DataFrame) -> Dict[str, List[Tuple]]:
        if df.empty: return {'highs': [], 'lows': []}
        
        # 為了保持計算與圖表索引一致，這裡不再重複 tail，由外部傳入切好的 df
        df_work = df.reset_index(drop=True)
        swing_highs = []
        swing_lows = []
        
        if len(df_work) <= 2 * self.swing_window:
            return {'highs': [], 'lows': []}
        
        for i in range(self.swing_window, len(df_work) - self.swing_window):
            current_high = df_work.iloc[i]['high']
            window_highs = df_work.iloc[i-self.swing_window : i+self.swing_window+1]['high']
            if current_high == window_highs.max():
                swing_highs.append((i, df_work.iloc[i]['datetime'], current_high))
            
            current_low = df_work.iloc[i]['low']
            window_lows = df_work.iloc[i-self.swing_window : i+self.swing_window+1]['low']
            if current_low == window_lows.min():
                swing_lows.append((i, df_work.iloc[i]['datetime'], current_low))
        
        return {'highs': swing_highs, 'lows': swing_lows}

    def calculate_line_params(self, point1: Tuple, point2: Tuple) -> Tuple[float, float]:
        x1, _, y1 = point1
        x2, _, y2 = point2
        if x2 - x1 == 0: return float('inf'), y1
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    def get_line_value(self, slope: float, intercept: float, x: int) -> float:
        if slope == float('inf'): return intercept
        return slope * x + intercept

    def find_trendlines(self, swing_points: List[Tuple]) -> List[Dict]:
        if len(swing_points) < 2: return []
        trendlines = []
        
        for i in range(len(swing_points)):
            for j in range(i + 1, len(swing_points)):
                p1, p2 = swing_points[i], swing_points[j]
                slope, intercept = self.calculate_line_params(p1, p2)
                
                touches = [p1, p2]
                # 容差設為價格的 0.1%，增加吸附感
                tolerance = abs(p1[2] * 0.001) 
                
                for k, point in enumerate(swing_points):
                    if k != i and k != j:
                        expected_price = self.get_line_value(slope, intercept, point[0])
                        if abs(point[2] - expected_price) <= tolerance:
                            touches.append(point)
                
                if len(touches) >= self.min_touches:
                    trendlines.append({
                        'points': touches,
                        'slope': slope,
                        'intercept': intercept,
                        'touches': len(touches),
                        'start_point': p1,
                        'end_point': p2,
                        'strength_score': self._calculate_strength_score(touches, slope)
                    })
        
        trendlines.sort(key=lambda x: (x['touches'], x['strength_score']), reverse=True)
        return trendlines

    def _calculate_strength_score(self, touches: List[Tuple], slope: float) -> float:
        base_score = len(touches)
        time_span = touches[-1][0] - touches[0][0]
        time_bonus = min(time_span / 20, 3.0) 
        # 移除過於嚴格的斜率懲罰，改為極端斜率才扣分
        slope_penalty = 1.0 if abs(slope) > 50 else 0 
        return base_score + time_bonus - slope_penalty

    def check_breakouts(self, df: pd.DataFrame, support_lines: List[Dict], resistance_lines: List[Dict]) -> List[Dict]:
        if df.empty: return []
        breakouts = []
        
        # 遍歷最近 20 根 K 棒，尋找有沒有任何一根產生突破
        # 而不是只看最後一根，這樣回測才有數據
        for i in range(len(df) - 20, len(df)):
            if i < 0: continue
            current_bar = df.iloc[i]
            
            # 阻力突破 (多單)
            for res in resistance_lines:
                if res['touches'] < self.min_touches: continue
                line_price = self.get_line_value(res['slope'], res['intercept'], i)
                
                # 判斷收盤價是否站上趨勢線 + 閥值
                if current_bar['close'] > line_price * (1 + self.breakout_threshold):
                    breakouts.append({
                        'datetime': current_bar['datetime'],
                        'price': current_bar['close'],
                        'direction': 'bullish_breakout',
                        'trendline_price': line_price,
                        'strength': res['touches'],
                        'breakout_magnitude': (current_bar['close'] - line_price) / line_price
                    })

            # 支撐跌破 (空單)
            for sup in support_lines:
                if sup['touches'] < self.min_touches: continue
                line_price = self.get_line_value(sup['slope'], sup['intercept'], i)
                
                if current_bar['close'] < line_price * (1 - self.breakout_threshold):
                    breakouts.append({
                        'datetime': current_bar['datetime'],
                        'price': current_bar['close'],
                        'direction': 'bearish_breakdown',
                        'trendline_price': line_price,
                        'strength': sup['touches'],
                        'breakout_magnitude': (line_price - current_bar['close']) / line_price
                    })

        # 同一時間點只取最強突破
        df_breakouts = pd.DataFrame(breakouts)
        if df_breakouts.empty: return []
        return df_breakouts.sort_values('breakout_magnitude', ascending=False).drop_duplicates('datetime').to_dict('records')

    def analyze(self, df: pd.DataFrame, swing_window: int = None, min_touches: int = None) -> Dict:
        if swing_window is not None: self.swing_window = swing_window
        if min_touches is not None: self.min_touches = min_touches
            
        df = df.sort_values('datetime').reset_index(drop=True)
        swing_points = self.find_swing_points(df)
        
        support_lines = self.find_trendlines(swing_points['lows'])
        resistance_lines = self.find_trendlines(swing_points['highs'])
        
        # 放寬斜率過濾：不再限制 0.1，改為合理的範圍 (台指期 60min)
        support_lines = [l for l in support_lines if l['slope'] >= -2.0]
        resistance_lines = [l for l in resistance_lines if l['slope'] <= 2.0]
        
        breakouts = self.check_breakouts(df, support_lines, resistance_lines)
        summary = self._generate_summary(swing_points, support_lines, resistance_lines, breakouts)
        
        return {
            'swing_points': swing_points,
            'support_lines': support_lines,
            'resistance_lines': resistance_lines,
            'breakouts': breakouts,
            'summary': summary
        }
    def get_trendline_coordinates(self, trendline: Dict, df_length: int, 
                                 extend_future: int = 10) -> List[Tuple]:
        """
        獲取趨勢線的座標點用於繪圖
        
        Args:
            trendline: 從 analyze() 得到的趨勢線字典
            df_length: 當前 DataFrame 的長度
            extend_future: 向未來延伸的 K 棒數量
            
        Returns:
            (索引, 價格) 的元組列表
        """
        if not trendline or 'points' not in trendline or not trendline['points']:
            return []
        
        # 找出趨勢線的起始索引（第一個接觸點）
        start_idx = min(point[0] for point in trendline['points'])
        
        # 設定結束索引：目前最後一根 K 棒再往後延展
        # 這裡的 df_length - 1 是當前最後一根的索引
        end_idx = df_length - 1 + extend_future
        
        coordinates = []
        for idx in range(start_idx, end_idx + 1):
            price = self.get_line_value(trendline['slope'], trendline['intercept'], idx)
            # 確保價格合理（不為負數），且不會畫到天文數字去
            if price > 0:
                coordinates.append((idx, price))
        
        return coordinates

    def _generate_summary(self, swing_points, support_lines, resistance_lines, breakouts) -> Dict:
        return {
            'swing_highs_count': len(swing_points['highs']),
            'swing_lows_count': len(swing_points['lows']),
            'support_lines_count': len(support_lines),
            'resistance_lines_count': len(resistance_lines),
            'breakouts_count': len(breakouts),
            'strongest_support_strength': max([l['touches'] for l in support_lines], default=0),
            'strongest_resistance_strength': max([l['touches'] for l in resistance_lines], default=0)
        }