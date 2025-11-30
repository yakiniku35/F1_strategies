import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_race_data, get_driver_lap_data
from llm_helper import ask_groq_commentary

# 設定網頁標題與寬度
st.set_page_config(page_title="F1 GenAI Strategist", layout="wide")

st.title("🏎️ F1 GenAI Strategist")
st.caption("Powered by FastF1 & Groq AI")

# --- 側邊欄：設定比賽 ---
with st.sidebar:
    st.header("Race Settings")
    year = st.number_input("Year", min_value=2018, max_value=2024, value=2023)
    gp = st.text_input("Grand Prix", value="Monaco")
    session_type = st.selectbox("Session", ["R", "Q", "FP1", "FP2", "FP3"], index=0)
    
    load_btn = st.button("Load Race Data")

# --- 主畫面邏輯 ---
if 'session' not in st.session_state:
    st.session_state.session = None

if load_btn:
    with st.spinner(f"正在下載 {year} {gp} 的數據 (第一次會比較久)..."):
        session = load_race_data(year, gp, session_type)
        if isinstance(session, str):
            st.error(f"錯誤: {session}")
        else:
            st.session_state.session = session
            st.success("數據載入完成！")

# 如果數據已經載入，顯示儀表板
if st.session_state.session:
    session = st.session_state.session
    
    # 取得車手列表
    drivers = session.results['Abbreviation'].unique()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 圈速分析 (Lap Time Analysis)")
        selected_drivers = st.multiselect("選擇對比車手", drivers, default=drivers[:2])
        
        if selected_drivers:
            all_laps = []
            for d in selected_drivers:
                laps, _ = get_driver_lap_data(session, d)
                laps['Driver'] = d
                # 把圈速轉成秒數方便畫圖
                laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
                all_laps.append(laps)
            
            # 合併數據畫圖
            df_plot = pd.concat(all_laps)
            fig = px.line(df_plot, x='LapNumber', y='LapTimeSec', color='Driver', 
                          title='Lap Pace Comparison', markers=True)
            # Y軸反轉，因為時間越短越快
            fig.update_yaxes(autorange="reversed") 
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.subheader("🤖 AI 賽評台")
        
        # 讓使用者輸入當下狀況
        current_lap = st.slider("模擬圈數 (Current Lap)", min_value=1, max_value=int(session.total_laps), value=10)
        
        # 準備一些數據給 AI
        if st.button("Ask AI Commentator"):
            # 這裡我們簡單抓取這兩個車手在這一圈的數據
            context_info = f"Race: {year} {gp}. Current Lap: {current_lap}. \n"
            for d in selected_drivers:
                laps, _ = get_driver_lap_data(session, d)
                lap_data = laps[laps['LapNumber'] == current_lap]
                if not lap_data.empty:
                    time = lap_data.iloc[0]['LapTime'].total_seconds()
                    tyre = lap_data.iloc[0]['Compound']
                    context_info += f"Driver {d} just did a {time}s on {tyre} tyres.\n"
            
            with st.spinner("AI 正在思考..."):
                commentary = ask_groq_commentary(context_info, style="commentator")
                st.write(commentary)

        st.divider()
        st.subheader("💬 策略聊天室")
        user_q = st.text_input("問問策略長...")
        if user_q:
            with st.spinner("策略長分析中..."):
                answer = ask_groq_commentary(f"Context: {year} {gp}. Question: {user_q}", style="strategist")
                st.info(answer)