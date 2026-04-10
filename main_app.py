import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# 假設你的模組名稱如下，請根據實際檔名調整
from chart_visualizer import ChartVisualizer
from trendline_detector import TrendlineBreakoutDetector

class TradingDashboard:
    def __init__(self):
        st.set_page_config(page_title="TX 期貨策略分析儀表板", layout="wide")
        self.visualizer = ChartVisualizer(theme='dark')
        self.detector = TrendlineBreakoutDetector()
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
        with st.sidebar:
            st.title("🛠️ 分析設定")
            window = st.slider("格點視窗", 1, 10, 5)
            min_touches = st.slider("最少接觸點", 2, 5, 3)
            
            st.markdown("---")
            st.markdown("### 🎯 期望值回測設定")
            tp_ticks = st.slider("停利點數 (TP)", 10, 200, 60)
            sl_ticks = st.slider("停損點數 (SL)", 10, 100, 30)
            cost = st.number_input("交易成本 (點數)", 0.0, 5.0, 2.0, step=0.5)

            if st.button("載入/重新整理資料", type="primary"):
                try:
                    # 1. 讀取與基本清理
                    df_raw = pd.read_csv("output/kline_60min.txt", sep='\s+', encoding='big5')
                    df_raw.columns = [c.strip() for c in df_raw.columns]
                    
                    column_map = {'日期': 'date', '時間': 'time', '開盤': 'open', '最高': 'high', '最低': 'low', '收盤': 'close', '成交量': 'volume'}
                    df_raw = df_raw.rename(columns=column_map)
                    df_raw['datetime'] = pd.to_datetime(df_raw['date'] + ' ' + df_raw['time'], errors='coerce')
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
                    
                    df_raw = df_raw.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])

                    if len(df_raw) > 0:
                        # 🌟 核心修正：只取最後 200 筆，並進行「深層複製」與「索引重置」
                        # 這會切斷與過去「遠古高點」的所有聯繫
                        df_final = df_raw.tail(200).copy().reset_index(drop=True)
                        
                        # 2. 執行分析
                        # 確保 detector 拿到的 dataframe 索引是從 0 到 199
                        analysis = self.detector.analyze(df_final, swing_window=window, min_touches=min_touches)
                        
                        # 3. 計算期望值
                        # 如果分析結果裡有 breakouts，計算才會有效
                        ev_results = self.calculate_ev(df_final, analysis.get('breakouts', []), tp_ticks, sl_ticks, cost)
                        analysis['ev_stats'] = ev_results
                        
                        # 4. 更新狀態
                        st.session_state.data = df_final
                        st.session_state.analysis = analysis
                        
                        st.success(f"✅ 局部分析完成！目前偵測到 {len(analysis.get('breakouts', []))} 個突破點")
                        st.rerun()
                    else:
                        st.error("❌ 資料清理後為空")

                except Exception as e:
                    st.error(f"❌ 載入失敗：{str(e)}")

    def calculate_ev(self, df, breakouts, tp, sl, cost):
        """計算期望值的完整邏輯 - 優化版"""
        results = []
        
        # 確保 df 的 index 是連續的整數，這對 iloc 操作很重要
        df_reset = df.reset_index(drop=True)
        
        for b in breakouts:
            # 找到突破點在 DataFrame 中的位置
            matches = df_reset[df_reset['datetime'] == b['datetime']]
            if matches.empty: continue
            
            idx = matches.index[0]
            # 取出突破後的 K 線資料
            post_df = df_reset.iloc[idx + 1:]
            
            outcome = None
            # 限制尋找範圍，例如最多往後看 100 根 K 線，避免過度跨越時空
            for i, (_, bar) in enumerate(post_df.iterrows()):
                if b['direction'] == 'bullish_breakout':
                    # 多頭突破：先看是否觸及停損，再看停利（保守估計）
                    if bar['low'] <= b['price'] - sl:
                        outcome = -sl - cost
                        break
                    elif bar['high'] >= b['price'] + tp:
                        outcome = tp - cost
                        break
                else: 
                    # 空頭跌破：先看是否觸及停損，再看停利
                    if bar['high'] >= b['price'] + sl:
                        outcome = -sl - cost
                        break
                    elif bar['low'] <= b['price'] - tp:
                        outcome = tp - cost
                        break
                
                # 如果超過 100 根 K 線還沒結果，可以選擇以當前收盤價結算或放棄這筆交易
                if i > 100: break 

            if outcome is not None:
                results.append(outcome)
        
        # 轉換為 Series 進行統計
        if not results:
            return {'ev': 0.0, 'win_rate': 0.0, 'pf': 0.0, 'total': 0}
            
        res = pd.Series(results)
        wins = res[res > 0]
        losses = res[res <= 0]
        
        # 計算獲利因子，避免除以零
        profit_sum = wins.sum()
        loss_sum = abs(losses.sum())
        pf = profit_sum / loss_sum if loss_sum > 0 else float(profit_sum)
        
        return {
            'ev': float(res.mean()),
            'win_rate': float(len(wins) / len(res)),
            'pf': float(pf),
            'total': len(results)
        }
    def render_main_content(self):
        # 檢查 session_state 是否有資料
        if st.session_state.analysis is not None and st.session_state.data is not None:
            data = st.session_state.data
            ans = st.session_state.analysis
            
            # 從分析結果中安全地取出期望值統計
            ev_stats = ans.get('ev_stats', {'ev': 0, 'win_rate': 0, 'pf': 0, 'total': 0})

            # --- 第一排：策略期望值回測指標 ---
            st.markdown("### 📊 策略期望值回測")
            e1, e2, e3, e4 = st.columns(4)
            
            ev_val = ev_stats.get('ev', 0)
            e1.metric("期望值 (EV)", f"{ev_val:.2f} 點", 
                      delta=f"{ev_val:.1f}", 
                      delta_color="normal" if ev_val > 0 else "inverse")
            e2.metric("模擬勝率", f"{ev_stats.get('win_rate', 0):.1%}")
            e3.metric("獲利因子 (PF)", f"{ev_stats.get('pf', 0):.2f}")
            e4.metric("總模擬交易", f"{ev_stats.get('total', 0)} 次")

            # --- 第二排：最新警報訊息 ---
            if ans.get('breakouts'):
                last_b = ans['breakouts'][-1]
                color = "green" if "bullish" in last_b['direction'] else "red"
                icon = "⬆️" if "bullish" in last_b['direction'] else "⬇️"
                st.info(f"最新訊號：{last_b['direction']} {icon} | 價格: {last_b['price']} | 時間: {last_b['datetime']}")

            # --- 第三排：分頁圖表與詳情 ---
            tab1, tab2, tab3 = st.tabs(["📈 主圖表分析", "📋 突破訊號紀錄", "📄 原始資料預覽"])
            
            with tab1:
                # 呼叫視覺化模組畫圖
                fig = self.visualizer.create_trendline_chart(data, ans)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if ans.get('breakouts'):
                    st.subheader("突破訊號明細")
                    # 先轉成 DataFrame
                    breakout_df = pd.DataFrame(ans['breakouts'])
                    
                    # 確保 DataFrame 不為空才顯示
                    if not breakout_df.empty:
                        # 🌟 關鍵：轉換成字串以避免之前的 ArrowInvalid 錯誤
                        st.dataframe(breakout_df.astype(str), use_container_width=True)
                    else:
                        st.write("目前沒有偵測到突破點")
                else:
                    st.write("目前參數設定下未偵測到任何突破訊號。")
            
            with tab3:
                st.subheader("最近 100 筆 K 線資料")
                st.dataframe(data.tail(100), use_container_width=True)
        else:
            # 初始狀態顯示提示
            st.info("👋 歡迎使用！請調整左側參數並點擊『載入/重新整理資料』開始分析。")
if __name__ == "__main__":
    app = TradingDashboard()
    app.run()