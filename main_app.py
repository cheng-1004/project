import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import os

# 確保你的自定義模組在同級目錄下
from chart_visualizer import ChartVisualizer
from trendline_detector import TrendlineBreakoutDetector

class TradingDashboard:
    def __init__(self):
        # 1. 頁面配置
        st.set_page_config(page_title="TX 期貨策略分析儀表板", layout="wide")
        
        # 2. 初始化核心物件
        self.visualizer = ChartVisualizer(theme='dark')
        self.detector = TrendlineBreakoutDetector()
        
        # 3. 初始化 Session State
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'analysis' not in st.session_state:
            st.session_state.analysis = None

    def run(self):
        self.render_sidebar()
        self.render_main_content()

    def render_sidebar(self):
        """渲染側邊欄 - 支援手動輸入路徑"""
        with st.sidebar:
            st.markdown("## ⚙️ 設定")
            st.markdown("### 📊 資料設定")
            
            data_source = st.selectbox(
                "資料來源模式",
                ["本地檔案", "測試資料"],
                help="選擇載入方式"
            )
            
            file_path = ""
            
            if data_source == "本地檔案":
                # 直接提供輸入框，讓你手動輸入路徑
                file_path = st.text_input(
                    "檔案路徑",
                    value="output/7652A_Hour.txt",
                    help="請輸入正確的相對或絕對路徑"
                )
                
                # 提示可用檔案
                if os.path.exists("output"):
                    ref_files = [f for f in os.listdir("output") if f.endswith(('.txt', '.csv'))]
                    if ref_files:
                        st.caption(f"💡 偵測到可用檔案: {', '.join(ref_files)}")
            else:
                # 測試資料模式
                file_path = "output/7652A_Hour.txt"
                st.info(f"使用預設測試路徑: {file_path}")

            st.markdown("---")
            
            # 偵測參數
            st.markdown("### 🔍 偵測參數")
            window = st.slider("格點視窗 (Swing Window)", 1, 10, 5)
            min_touches = st.slider("最少接觸點", 2, 5, 3)
            
            st.markdown("---")
            
            # 回測設定
            st.markdown("### 🎯 期望值回測設定")
            tp_ticks = st.slider("停利點數 (TP)", 10, 200, 60)
            sl_ticks = st.slider("停損點數 (SL)", 10, 100, 30)
            cost = st.number_input("交易成本 (點數)", 0.0, 5.0, 2.0, step=0.5)

            if st.button("載入/重新整理資料", type="primary"):
                if file_path:
                    self._process_data_loading(file_path, window, min_touches, tp_ticks, sl_ticks, cost)
                else:
                    st.error("❌ 請輸入檔案路徑")

    def _process_data_loading(self, file_path, window, min_touches, tp_ticks, sl_ticks, cost):
        """核心資料處理流程 - 已修正 volume 缺失問題"""
        try:
            if not os.path.exists(file_path):
                st.error(f"❌ 找不到檔案: {file_path}")
                return

            # 讀取資料 (處理分隔符號)
            sep = ',' if file_path.endswith('.csv') else '\s+'
            df_raw = pd.read_csv(file_path, sep=sep, encoding='big5')
            
            # 1. 欄位清理與映射
            df_raw.columns = [c.strip() for c in df_raw.columns]
            column_map = {
                '日期': 'date', '時間': 'time', 
                '開盤': 'open', '最高': 'high', '最低': 'low', '收盤': 'close',
                '成交量': 'volume', '成交量(口)': 'volume'
            }
            df_raw = df_raw.rename(columns=column_map)
            
            # 2. 🌟 關鍵修正：如果缺少 volume 欄位，手動補 0 (防止 Visualizer 崩潰)
            if 'volume' not in df_raw.columns:
                df_raw['volume'] = 0
            
            # 3. 時間轉換
            if 'date' in df_raw.columns and 'time' in df_raw.columns:
                df_raw['datetime'] = pd.to_datetime(df_raw['date'] + ' ' + df_raw['time'], errors='coerce')
            else:
                # 兼容只有單一日期欄位的格式
                df_raw['datetime'] = pd.to_datetime(df_raw.iloc[:, 0], errors='coerce')
            
            # 4. 數值轉換
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
            df_raw = df_raw.dropna(subset=['datetime', 'close'])

            if len(df_raw) > 0:
                # 只取最後 200 筆，並重置索引
                df_final = df_raw.tail(200).copy().reset_index(drop=True)
                
                # 執行偵測
                analysis = self.detector.analyze(df_final, swing_window=window, min_touches=min_touches)
                
                # 計算期望值
                ev_results = self.calculate_ev(df_final, analysis.get('breakouts', []), tp_ticks, sl_ticks, cost)
                analysis['ev_stats'] = ev_results
                
                # 存入 Session
                st.session_state.data = df_final
                st.session_state.analysis = analysis
                
                st.success(f"✅ 載入成功: {os.path.basename(file_path)}")
                st.rerun()
            else:
                st.error("❌ 檔案清理後無有效 K 線資料")
        except Exception as e:
            st.error(f"❌ 讀取失敗：{str(e)}")

    def calculate_ev(self, df, breakouts, tp, sl, cost):
        """回測期望值核心計算"""
        results = []
        df_reset = df.reset_index(drop=True)
        
        for b in breakouts:
            matches = df_reset[df_reset['datetime'] == b['datetime']]
            if matches.empty: continue
            
            idx = matches.index[0]
            post_df = df_reset.iloc[idx + 1:]
            
            outcome = None
            for i, (_, bar) in enumerate(post_df.iterrows()):
                if b['direction'] == 'bullish_breakout':
                    if bar['low'] <= b['price'] - sl:
                        outcome = -sl - cost
                        break
                    elif bar['high'] >= b['price'] + tp:
                        outcome = tp - cost
                        break
                else: 
                    if bar['high'] >= b['price'] + sl:
                        outcome = -sl - cost
                        break
                    elif bar['low'] <= b['price'] - tp:
                        outcome = tp - cost
                        break
                if i > 100: break # 最多往後看 100 根

            if outcome is not None:
                results.append(outcome)
        
        if not results:
            return {'ev': 0.0, 'win_rate': 0.0, 'pf': 0.0, 'total': 0}
            
        res = pd.Series(results)
        wins = res[res > 0]
        loss_sum = abs(res[res <= 0].sum())
        pf = wins.sum() / loss_sum if loss_sum > 0 else float(wins.sum())
        
        return {
            'ev': float(res.mean()),
            'win_rate': float(len(wins) / len(res)),
            'pf': float(pf),
            'total': len(results)
        }

    def render_main_content(self):
        """渲染主畫面"""
        if st.session_state.analysis is not None and st.session_state.data is not None:
            data = st.session_state.data
            ans = st.session_state.analysis
            ev_stats = ans.get('ev_stats', {'ev': 0, 'win_rate': 0, 'pf': 0, 'total': 0})

            # --- 第一排：指標 ---
            st.markdown("### 📊 策略期望值回測")
            e1, e2, e3, e4 = st.columns(4)
            ev_val = ev_stats['ev']
            e1.metric("期望值 (EV)", f"{ev_val:.2f} 點", delta=f"{ev_val:.1f}")
            e2.metric("模擬勝率", f"{ev_stats['win_rate']:.1%}")
            e3.metric("獲利因子 (PF)", f"{ev_stats['pf']:.2f}")
            e4.metric("總交易次數", f"{ev_stats['total']} 次")

            # --- 第二排：最新警報 ---
            if ans.get('breakouts'):
                last_b = ans['breakouts'][-1]
                st.info(f"🔔 最新訊號：{last_b['direction']} | 價格: {last_b['price']} | 時間: {last_b['datetime']}")

            # --- 第三排：圖表分頁 ---
            tab1, tab2, tab3 = st.tabs(["📈 趨勢分析圖", "📋 訊號清單", "📄 原始資料"])
            
            with tab1:
                fig = self.visualizer.create_trendline_chart(data, ans)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if ans.get('breakouts'):
                    st.dataframe(pd.DataFrame(ans['breakouts']).astype(str), use_container_width=True)
                else:
                    st.write("目前參數下無偵測到訊號。")
            
            with tab3:
                st.dataframe(data.tail(100), use_container_width=True)
        else:
            st.info("👋 歡迎！請在左側輸入檔案路徑並點擊執行開始分析。")

if __name__ == "__main__":
    app = TradingDashboard()
    app.run()