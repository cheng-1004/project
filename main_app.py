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
        st.set_page_config(page_title="TX 期貨策略分析儀表板 (進階過濾版)", layout="wide")
        
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
        """渲染側邊欄 - 包含進階過濾參數"""
        with st.sidebar:
            st.markdown("## ⚙️ 核心設定")
            st.markdown("### 📊 資料來源")
            
            data_source = st.selectbox("載入模式", ["本地檔案", "測試資料"])
            file_path = st.text_input("檔案路徑", value="output/7652A_Hour.txt") if data_source == "本地檔案" else "output/kline_60min.txt"

            st.markdown("---")
            st.markdown("### 🔍 趨勢偵測")
            window = st.slider("格點視窗 (Swing Window)", 1, 10, 5)
            min_touches = st.slider("最少接觸點", 2, 5, 3)
            
            st.markdown("---")
            st.markdown("### 🛡️ 假突破濾網 (Anti-False Breakout)")
            use_vol_filter = st.checkbox("啟用成交量過濾", value=True, help="突破時成交量需高於均值")
            vol_multiplier = st.slider("成交量放大倍數", 0.5, 3.0, 1.2, 0.1)
            use_close_confirm = st.checkbox("收盤價確認", value=True, help="必須收盤突破趨勢線才算數，排除長影線")
            
            st.markdown("---")
            st.markdown("### 🎯 回測參數")
            tp_ticks = st.slider("停利點數 (TP)", 10, 200, 60)
            sl_ticks = st.slider("停損點數 (SL)", 10, 100, 30)
            cost = st.number_input("交易成本 (點數)", 0.0, 5.0, 2.0, step=0.5)

            if st.button("啟動策略分析", type="primary"):
                self._process_data_loading(
                    file_path, window, min_touches, 
                    tp_ticks, sl_ticks, cost,
                    use_vol_filter, vol_multiplier, use_close_confirm
                )

    def _filter_breakouts(self, df, breakouts, use_vol, vol_mult, use_close):
        """
        假突破過濾邏輯：
        1. 收盤確認：避免只是影線刷過。
        2. 成交量驗證：突破必須伴隨量能。
        """
        filtered = []
        for b in breakouts:
            # 定位突破發生的 K 線
            target_rows = df.index[df['datetime'] == b['datetime']].tolist()
            if not target_rows: continue
            idx = target_rows[0]
            current_bar = df.iloc[idx]
            
            # 濾網 A: 收盤確認
            if use_close:
                if b['direction'] == 'bullish_breakout' and current_bar['close'] <= b['price']:
                    continue
                if b['direction'] == 'bearish_breakout' and current_bar['close'] >= b['price']:
                    continue

            # 濾網 B: 成交量驗證 (對比過去 5 根均量)
            if use_vol and idx >= 5:
                avg_vol = df['volume'].iloc[idx-5:idx].mean()
                if current_bar['volume'] < (avg_vol * vol_mult):
                    continue
            
            filtered.append(b)
        return filtered

    def _process_data_loading(self, file_path, window, min_touches, tp, sl, cost, use_vol, vol_mult, use_close):
        try:
            if not os.path.exists(file_path):
                st.error(f"❌ 找不到檔案: {file_path}")
                return

            # 讀取與清理
            sep = ',' if file_path.endswith('.csv') else r'\s+'
            df_raw = pd.read_csv(file_path, sep=sep, encoding='big5')
            df_raw.columns = [c.strip() for c in df_raw.columns]
            
            column_map = {
                '日期': 'date', '時間': 'time', 
                '開盤': 'open', '最高': 'high', '最低': 'low', '收盤': 'close',
                '成交量': 'volume', '成交量(口)': 'volume'
            }
            df_raw = df_raw.rename(columns=column_map)
            if 'volume' not in df_raw.columns: df_raw['volume'] = 0
            
            if 'date' in df_raw.columns and 'time' in df_raw.columns:
                df_raw['datetime'] = pd.to_datetime(df_raw['date'] + ' ' + df_raw['time'], errors='coerce')
            else:
                df_raw['datetime'] = pd.to_datetime(df_raw.iloc[:, 0], errors='coerce')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
            df_final = df_raw.dropna(subset=['datetime', 'close']).tail(500).copy().reset_index(drop=True)

            if len(df_final) > 0:
                # 1. 執行初步偵測
                analysis = self.detector.analyze(df_final, swing_window=window, min_touches=min_touches)
                
                # 2. 🌟 加入過濾邏輯
                raw_breakouts = analysis.get('breakouts', [])
                filtered_breakouts = self._filter_breakouts(df_final, raw_breakouts, use_vol, vol_mult, use_close)
                
                # 3. 更新分析結果中的訊號
                analysis['breakouts'] = filtered_breakouts
                
                # 4. 計算期望值
                ev_results = self.calculate_ev(df_final, filtered_breakouts, tp, sl, cost)
                analysis['ev_stats'] = ev_results
                
                st.session_state.data = df_final
                st.session_state.analysis = analysis
                st.success(f"✅ 成功載入並過濾訊號 (原 {len(raw_breakouts)} -> 剩 {len(filtered_breakouts)})")
                st.rerun()
        except Exception as e:
            st.error(f"❌ 處理錯誤：{str(e)}")

    def calculate_ev(self, df, breakouts, tp, sl, cost):
        """回測期望值核心計算 (加入基本滑價概念)"""
        results = []
        df_reset = df.reset_index(drop=True)
        
        for b in breakouts:
            matches = df_reset[df_reset['datetime'] == b['datetime']]
            if matches.empty: continue
            
            idx = matches.index[0]
            # 從下一根開始看結果
            post_df = df_reset.iloc[idx + 1:]
            
            outcome = None
            for i, (_, bar) in enumerate(post_df.iterrows()):
                if b['direction'] == 'bullish_breakout':
                    if bar['low'] <= b['price'] - sl: # 碰到停損
                        outcome = -sl - cost
                        break
                    elif bar['high'] >= b['price'] + tp: # 碰到停利
                        outcome = tp - cost
                        break
                else: # bearish_breakout
                    if bar['high'] >= b['price'] + sl:
                        outcome = -sl - cost
                        break
                    elif bar['low'] <= b['price'] - tp:
                        outcome = tp - cost
                        break
                if i > 120: break # 超時未觸及則放棄

            if outcome is not None:
                results.append(outcome)
        
        if not results:
            return {'ev': 0.0, 'win_rate': 0.0, 'pf': 0.0, 'total': 0, 'raw_results': []}
            
        res = pd.Series(results)
        wins = res[res > 0]
        losses = res[res <= 0]
        loss_sum = abs(losses.sum())
        pf = wins.sum() / loss_sum if loss_sum > 0 else float(wins.sum())
        
        return {
            'ev': float(res.mean()),
            'win_rate': float(len(wins) / len(res)),
            'pf': float(pf),
            'total': len(results),
            'raw_results': results
        }

    def render_main_content(self):
        """渲染主畫面"""
        if st.session_state.analysis is not None and st.session_state.data is not None:
            data = st.session_state.data
            ans = st.session_state.analysis
            ev_stats = ans.get('ev_stats', {})

            # --- 第一排：指標 ---
            st.markdown("### 📊 策略績效摘要")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("平均期望值 (EV)", f"{ev_stats['ev']:.2f} 點")
            e2.metric("模擬勝率", f"{ev_stats['win_rate']:.1%}")
            e3.metric("獲利因子 (PF)", f"{ev_stats['pf']:.2f}")
            e4.metric("有效訊號次數", f"{ev_stats['total']} 次")

            # --- 第二排：最新警報 ---
            if ans.get('breakouts'):
                last_b = ans['breakouts'][-1]
                color = "green" if last_b['direction'] == "bullish_breakout" else "red"
                st.markdown(f"**🔔 最新訊號：** :{color}[{last_b['direction']}] | 價格: **{last_b['price']}** | 時間: {last_b['datetime']}")

            # --- 第三排：分頁 ---
            tab1, tab2, tab3, tab4 = st.tabs(["📈 趨勢分析圖", "📋 訊號清單", "💰 累積損益", "📄 原始資料"])
            
            with tab1:
                fig = self.visualizer.create_trendline_chart(data, ans)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if ans.get('breakouts'):
                    st.dataframe(pd.DataFrame(ans['breakouts']), use_container_width=True)
                else:
                    st.info("目前的過濾條件下沒有訊號產生。")
            
            with tab3:
                if ev_stats.get('raw_results'):
                    equity = pd.Series(ev_stats['raw_results']).cumsum()
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(y=equity, mode='lines+markers', name='Equity'))
                    fig_equity.update_layout(title="累積點數曲線", template="plotly_dark", xaxis_title="交易序號", yaxis_title="點數")
                    st.plotly_chart(fig_equity, use_container_width=True)
                else:
                    st.write("尚無損益資料。")
            
            with tab4:
                st.dataframe(data.tail(100), use_container_width=True)
        else:
            st.info("👋 歡迎！請在左側設定參數後點擊「啟動策略分析」。")

if __name__ == "__main__":
    app = TradingDashboard()
    app.run()