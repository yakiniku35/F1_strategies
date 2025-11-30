import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from groq import Groq
from dotenv import load_dotenv
from utils import load_session_data, process_replay_data

# --- 初始化 ---
load_dotenv()
st.set_page_config(page_title="F1 GenAI Strategist", layout="wide", page_icon="🏎️")

# CSS 優化：讓介面更像儀表板
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    .stMetric {background-color: #1e1e1e; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# AI 初始化
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Session State 管理播放狀態 ---
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_time' not in st.session_state:
    st.session_state.current_time = 0
if 'ai_chat_history' not in st.session_state:
    st.session_state.ai_chat_history = []

# --- AI 功能函數 ---
def get_ai_commentary(context, style="commentator"):
    try:
        if style == "commentator":
            sys_prompt = "You are a hyped F1 commentator like David Croft. Provide short, dramatic, play-by-play commentary based on the data."
        else:
            sys_prompt = "You are a F1 Strategy Engineer. Analyze the tyre strategy and gaps logically."
            
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": context}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=100
        )
        return completion.choices[0].message.content
    except Exception as e:
        return "AI Radio Check... (Connection Error)"

# --- 側邊欄：設定與模擬 ---
with st.sidebar:
    st.title("🏎️ 設定控制台")
    year = st.number_input("年份", 2021, 2024, 2023)
    gp = st.text_input("大獎賽 (英文)", "Monaco")
    session_type = st.selectbox("賽程", ["R", "Q"], index=0)
    
    if st.button("載入比賽數據"):
        st.session_state.data_loaded = False # 重置
        with st.spinner("正在下載並處理遙測數據 (這需要一點時間)..."):
            session = load_session_data(year, gp, session_type)
            if session:
                replay_data, time_range = process_replay_data(session)
                st.session_state.session = session
                st.session_state.replay_data = replay_data
                st.session_state.start_time = time_range[0]
                st.session_state.end_time = time_range[1]
                st.session_state.current_time = time_range[0] # 重置時間
                st.session_state.data_loaded = True
                st.success("數據準備完成！")
            else:
                st.error("找不到該比賽數據")

    st.divider()
    st.subheader("🛠️ 策略模擬實驗室")
    target_driver = st.text_input("目標車手 (例如 VER)", "VER")
    new_tyre = st.selectbox("更換輪胎", ["SOFT", "MEDIUM", "HARD"])
    pit_lap = st.slider("模擬進站圈數", 1, 70, 20)
    if st.button("執行策略模擬"):
        st.toast(f"正在計算 {target_driver} 使用 {new_tyre} 於第 {pit_lap} 圈進站的結果...", icon="🤖")
        # 這裡會呼叫 utils 裡的模擬函數 (未來擴充用)
        # 目前先讓 AI 針對這個設定做評論
        ai_response = get_ai_commentary(f"User wants to simulate {target_driver} pitting on lap {pit_lap} for {new_tyre}s.", style="strategist")
        st.session_state.ai_chat_history.append({"role": "assistant", "content": f"📊 模擬分析: {ai_response}"})

# --- 主畫面 ---
st.title(f"F1 GenAI Live Replay: {year} {gp}")

if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    
    # 建立版面：左邊是賽道圖，右邊是資訊與AI
    col_map, col_info = st.columns([2, 1])
    
    # --- 播放控制器 ---
    # 使用 slider 讓使用者也能手動拉時間
    curr_t = st.slider("比賽時間軸 (Session Time)", 
                       min_value=int(st.session_state.start_time), 
                       max_value=int(st.session_state.end_time), 
                       value=int(st.session_state.current_time))
    
    st.session_state.current_time = curr_t # 同步
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        if st.button("▶️ 開始播放 / 暫停"):
            st.session_state.is_playing = not st.session_state.is_playing
    
    # --- 核心邏輯：過濾當下數據 ---
    # 找出「現在這一秒」所有車手在哪裡
    df_now = st.session_state.replay_data[
        (st.session_state.replay_data['TimeSec'] >= st.session_state.current_time) & 
        (st.session_state.replay_data['TimeSec'] < st.session_state.current_time + 2) # 取2秒區間避免沒數據
    ].drop_duplicates(subset=['Driver']) # 每個車手只取一點
    
    # --- 左側：動態賽道圖 ---
    with col_map:
        # 繪製賽道圖
        fig = go.Figure()
        
        # 1. 畫出所有車手當前位置
        fig.add_trace(go.Scatter(
            x=df_now['X'], y=df_now['Y'],
            mode='markers+text',
            text=df_now['Driver'],
            textposition="top center",
            marker=dict(size=12, color='red', line=dict(width=2, color='white')),
            name='Drivers'
        ))
        
        # 2. 畫出賽道背景 (用所有數據畫一條淡灰色的線)
        # 為了效能，我們只畫一次或是抽樣
        # 這裡簡化：Plotly 會自動適應座標，所以不畫背景線也可以看到相對位置
        
        fig.update_layout(
            width=800, height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor='#262730',
            paper_bgcolor='#262730',
            font=dict(color='white'),
            title=f"Live Track Map - T+{int(st.session_state.current_time)}s"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 右側：排行榜 & AI ---
    with col_info:
        # 排行榜 (根據當前進度排序)
        st.subheader("🏆 Leaderboard")
        # 簡單邏輯：誰跑的距離長/位置？(這裡暫時用資料順序模擬，真實需要 lap count)
        leaderboard_df = df_now[['Driver', 'Speed', 'nGear']].sort_values(by='Speed', ascending=False)
        st.dataframe(leaderboard_df, hide_index=True)
        
        # AI 轉播區
        st.subheader("🎙️ AI Live Commentary")
        
        # 自動觸發 AI：每過 60 秒 (模擬時間) 觸發一次，或者是按下按鈕
        if st.button("🎙️ 生成即時賽評"):
            # 整理當前前三名數據給 AI
            top_3 = leaderboard_df.head(3)['Driver'].tolist()
            context = f"Race Time: {int(st.session_state.current_time)}s. Top 3 drivers are: {top_3}. Drivers are pushing hard."
            commentary = get_ai_commentary(context, style="commentator")
            st.session_state.ai_chat_history.insert(0, {"role": "ai", "content": commentary})
            
        # 顯示對話紀錄
        chat_container = st.container(height=300)
        for msg in st.session_state.ai_chat_history:
            if msg['role'] == 'ai':
                chat_container.chat_message("assistant").write(msg['content'])
            elif msg['role'] == 'assistant': # 策略師
                 chat_container.chat_message("assistant", avatar="🛠️").write(msg['content'])

    # --- 自動播放邏輯 (Auto-Play Loop) ---
    # 這是 Streamlit 模擬動畫的關鍵：使用 st.rerun()
    if st.session_state.is_playing:
        st.session_state.current_time += 10 # 每次更新快轉 10 秒 (加速播放)
        if st.session_state.current_time >= st.session_state.end_time:
            st.session_state.is_playing = False
        time.sleep(0.1) # 控制更新頻率
        st.rerun() # 強制刷新畫面，產生動畫效果

else:
    st.info("👈 請在左側輸入比賽資訊並載入數據")