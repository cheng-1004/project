import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class ChartVisualizer:
    """
    圖表視覺化器類別，負責創建各種交易圖表
    """
    
    def __init__(self, theme: str = 'dark'):
        """
        初始化圖表視覺化器
        
        Args:
            theme: 主題色彩 ('dark' 或 'light')
        """
        self.theme = theme
        self.colors = self._get_color_scheme()
        
    def _get_color_scheme(self) -> Dict[str, str]:
        """獲取色彩方案"""
        if self.theme == 'dark':
            return {
                'background': '#0e1117',
                'plot_bg': '#0e1117',
                'text': '#ffffff',
                'grid': '#333333',
                'up_candle': '#ff4444',      # 紅色代表上漲
                'down_candle': '#00ff00',    # 綠色代表下跌
                'volume': '#666666',
                'support': '#00ff00',        # 綠色支撐線
                'resistance': '#ff4444',     # 紅色阻力線
                'swing_high': '#ff6b6b',
                'swing_low': '#4ecdc4',
                'breakout_bull': '#00ff00',
                'breakout_bear': '#ff4444'
            }
        else:
            return {
                'background': '#ffffff',
                'plot_bg': '#ffffff',
                'text': '#000000',
                'grid': '#cccccc',
                'up_candle': '#26a69a',
                'down_candle': '#ef5350',
                'volume': '#999999',
                'support': '#26a69a',
                'resistance': '#ef5350',
                'swing_high': '#ff5722',
                'swing_low': '#2196f3',
                'breakout_bull': '#4caf50',
                'breakout_bear': '#f44336'
            }
    
    def create_basic_candlestick_chart(self, df: pd.DataFrame, 
                                     continuous: bool = True,
                                     title: str = "Price Chart") -> go.Figure:
        """
        創建基本K線圖
        """
        if df is None or len(df) == 0:
            st.error("沒有資料可以繪圖")
            return None
        
        # 資料清理
        df_clean = self._clean_chart_data(df)
        if df_clean is None:
            return None
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('價格', '成交量'),
            row_heights=[0.7, 0.3]
        )
        
        # 準備X軸資料
        if continuous:
            x_values = list(range(len(df_clean)))
            x_tickvals, x_ticktext = self._prepare_x_axis_labels(df_clean)
        else:
            x_values = df_clean['datetime']
            x_tickvals, x_ticktext = None, None
        
        # 添加K線圖
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=df_clean['open'].values,
                high=df_clean['high'].values,
                low=df_clean['low'].values,
                close=df_clean['close'].values,
                increasing_line_color=self.colors['up_candle'],
                decreasing_line_color=self.colors['down_candle'],
                increasing_fillcolor=self.colors['up_candle'],
                decreasing_fillcolor=self.colors['down_candle'],
                name='期貨價格',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 添加成交量
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df_clean['volume'].values,
                marker_color=self.colors['volume'],
                name='成交量',
                showlegend=False,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # --- 修正點：添加隱形散點圖以顯示時間資訊（確保時間格式正確） ---
        if continuous:
            # 使用 pd.to_datetime 確保 dt 訪問器可用，並轉為字串
            hover_times = pd.to_datetime(df_clean['datetime']).dt.strftime('%Y-%m-%d %H:%M')
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df_clean['close'].values,
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    text=hover_times,
                    hovertemplate='時間: %{text}<br>收盤: %{y:,.0f}<extra></extra>',
                    name='時間資訊',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 更新布局
        self._update_chart_layout(fig, title, continuous, x_tickvals, x_ticktext)
        
        return fig
    
    def create_trendline_chart(self, df: pd.DataFrame, 
                               trendline_analysis: Dict,
                               max_lines: int = 3) -> go.Figure:
        """
        創建包含趨勢線的K線圖
        """
        # 1. 先創建基本K線圖
        fig = self.create_basic_candlestick_chart(df, continuous=True, 
                                                 title="價格圖表 with 趨勢線分析")
        
        # 🌟 關鍵修正：如果 fig 是 None，直接回報並返回一個空的 Figure 物件，而不是返回 None
        if fig is None:
            return go.Figure() 
        
        # 確保資料清理後才進行後續標註
        df_clean = self._clean_chart_data(df)
        if df_clean is None:
            return fig
        
        # 2. 添加標記 (這部分維持原樣)
        try:
            self._add_swing_points(fig, trendline_analysis['swing_points'])
            self._add_trendlines(fig, trendline_analysis, df_clean, max_lines)
            self._add_breakout_markers(fig, trendline_analysis['breakouts'], df_clean)
        except Exception as e:
            st.warning(f"圖表標註時出錯: {e}")
        
        return fig
    
    def _clean_chart_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """清理圖表資料"""
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        
        if len(df_clean) == 0:
            st.error("清理後沒有有效資料")
            return None
        
        # 驗證OHLC資料
        df_clean = df_clean[
            (df_clean['high'] >= df_clean['low']) & 
            (df_clean['high'] >= df_clean['open']) & 
            (df_clean['high'] >= df_clean['close']) &
            (df_clean['low'] <= df_clean['open']) & 
            (df_clean['low'] <= df_clean['close'])
        ]
        
        if len(df_clean) == 0:
            st.error("驗證後沒有有效的K線資料")
            return None
        
        return df_clean.reset_index(drop=True)
    
    def _prepare_x_axis_labels(self, df: pd.DataFrame) -> Tuple[List[int], List[str]]:
        """
        準備X軸標籤 - 修正 strftime 錯誤
        """
        total_points = len(df)
        
        if total_points > 50:
            step = max(1, total_points // 12)
        else:
            step = max(1, total_points // 10)
        
        tick_positions = list(range(0, total_points, step))
        if len(tick_positions) > 0 and tick_positions[-1] != total_points - 1:
            tick_positions.append(total_points - 1)
        
        # --- 修正點：使用 pd.to_datetime 強制轉換，防止字串物件報錯 ---
        tick_labels = [
            pd.to_datetime(df.iloc[i]['datetime']).strftime('%m/%d %H:%M') 
            for i in tick_positions
        ]
        
        return tick_positions, tick_labels
    
    def _update_chart_layout(self, fig: go.Figure, title: str, continuous: bool,
                            x_tickvals: List[int] = None, x_ticktext: List[str] = None):
        """更新圖表佈局"""
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.colors['text']}
            },
            plot_bgcolor=self.colors['plot_bg'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode='x unified',
            dragmode='pan',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        # 更新X軸
        x_axis_config = {
            'showgrid': True,
            'gridcolor': self.colors['grid'],
            'gridwidth': 1,
            'showline': True,
            'linecolor': self.colors['grid'],
            'tickfont': dict(color=self.colors['text']),
        }
        
        if continuous and x_tickvals and x_ticktext:
            x_axis_config.update({
                'tickmode': 'array',
                'tickvals': x_tickvals,
                'ticktext': x_ticktext,
                'tickangle': 45
            })
        
        fig.update_xaxes(**x_axis_config)
        
        # 更新Y軸
        y_axis_config = {
            'showgrid': True,
            'gridcolor': self.colors['grid'],
            'gridwidth': 1,
            'showline': True,
            'linecolor': self.colors['grid'],
            'tickfont': dict(color=self.colors['text']),
            'title_font': dict(color=self.colors['text'])
        }
        
        fig.update_yaxes(title_text="價格", tickformat=',.0f', **y_axis_config, row=1, col=1)
        fig.update_yaxes(title_text="成交量", tickformat=',.0f', **y_axis_config, row=2, col=1)
    
    def _add_swing_points(self, fig: go.Figure, swing_points: Dict):
        """添加搖擺點標記"""
        if swing_points['highs']:
            highs_x = [point[0] for point in swing_points['highs']]
            highs_y = [point[2] for point in swing_points['highs']]
            
            fig.add_trace(
                go.Scatter(
                    x=highs_x,
                    y=highs_y,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=8,
                        color=self.colors['swing_high']
                    ),
                    name='搖擺高點',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if swing_points['lows']:
            lows_x = [point[0] for point in swing_points['lows']]
            lows_y = [point[2] for point in swing_points['lows']]
            
            fig.add_trace(
                go.Scatter(
                    x=lows_x,
                    y=lows_y,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color=self.colors['swing_low']
                    ),
                    name='搖擺低點',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    def _add_trendlines(self, fig: go.Figure, trendline_analysis: Dict, 
                        df_clean: pd.DataFrame, max_lines: int):
        """添加趨勢線"""
        try:
            from trendline_detector import TrendlineBreakoutDetector
            detector = TrendlineBreakoutDetector()
            
            # 支撐線
            for i, support in enumerate(trendline_analysis['support_lines'][:max_lines]):
                coords = detector.get_trendline_coordinates(support, len(df_clean))
                if coords:
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    fig.add_trace(
                        go.Scatter(
                            x=x_coords, y=y_coords, mode='lines',
                            line=dict(color=self.colors['support'], width=2),
                            name=f'支撐線 {i+1}', showlegend=True
                        ), row=1, col=1
                    )
            
            # 阻力線
            for i, resistance in enumerate(trendline_analysis['resistance_lines'][:max_lines]):
                coords = detector.get_trendline_coordinates(resistance, len(df_clean))
                if coords:
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    fig.add_trace(
                        go.Scatter(
                            x=x_coords, y=y_coords, mode='lines',
                            line=dict(color=self.colors['resistance'], width=2),
                            name=f'阻力線 {i+1}', showlegend=True
                        ), row=1, col=1
                    )
        except Exception as e:
            st.warning(f"添加趨勢線時出錯: {e}")
    
    def _add_breakout_markers(self, fig: go.Figure, breakouts: List[Dict], 
                             df_clean: pd.DataFrame):
        """添加突破點標記"""
        for breakout in breakouts:
            # 🌟 修正：嘗試從資料中找出對應的時間索引，否則才用最後一個
            try:
                # 假設你的 breakout 字典有 'datetime' 欄位
                target_dt = pd.to_datetime(breakout['datetime'])
                matches = df_clean[pd.to_datetime(df_clean['datetime']) == target_dt]
                if not matches.empty:
                    breakout_idx = matches.index[0]
                else:
                    breakout_idx = len(df_clean) - 1
            except:
                breakout_idx = len(df_clean) - 1
            
            # ... 後續畫圖邏輯維持原樣 ...

    def create_analysis_summary_chart(self, metrics: Dict) -> go.Figure:
        """創建分析摘要圖表"""
        fig = go.Figure()
        labels = ['當前價格', '價格變化%', '期間高點', '期間低點', '波動率%']
        values = [
            metrics.get('current_price', 0),
            metrics.get('price_change_pct', 0),
            metrics.get('period_high', 0),
            metrics.get('period_low', 0),
            metrics.get('volatility_pct', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=self.colors['text'],
            text=[f'{v:.2f}' for v in values],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='市場指標摘要',
            plot_bgcolor=self.colors['plot_bg'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=400
        )
        return fig


def create_metric_cards_html(metrics: Dict) -> str:
    """創建指標卡片的HTML"""
    current_price = metrics.get('current_price', 0)
    price_change = metrics.get('price_change', 0)
    price_change_pct = metrics.get('price_change_pct', 0)
    period_high = metrics.get('period_high', 0)
    period_low = metrics.get('period_low', 0)
    total_volume = metrics.get('total_volume', 0)
    data_points = metrics.get('data_points', 0)
    
    # TX期貨習慣：上漲紅色，下跌綠色
    change_color = "#ff4444" if price_change >= 0 else "#00ff00"
    
    html = f"""
    <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
        <div class="metric-container" style="flex: 1; margin: 0 0.5rem; background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="color: #888; font-size: 0.8rem;">當前價格</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">{current_price:.0f}</div>
            <div style="color: {change_color}; font-size: 0.8rem;">
                {price_change:+.0f} ({price_change_pct:+.2f}%)
            </div>
        </div>
        <div class="metric-container" style="flex: 1; margin: 0 0.5rem; background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="color: #888; font-size: 0.8rem;">期間高點</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">{period_high:.0f}</div>
        </div>
        <div class="metric-container" style="flex: 1; margin: 0 0.5rem; background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="color: #888; font-size: 0.8rem;">期間低點</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">{period_low:.0f}</div>
        </div>
        <div class="metric-container" style="flex: 1; margin: 0 0.5rem; background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="color: #888; font-size: 0.8rem;">總成交量</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">{total_volume:,.0f}</div>
        </div>
    </div>
    """
    return html