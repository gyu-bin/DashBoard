import os
import pandas as pd
import streamlit as st
import plotly.express as px

# CSV 파일 경로
CSV_FILE = "health_data_example.csv"

###########################################
# CSV 파일 로드 및 저장 함수
###########################################
def load_health_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "날짜" in df.columns and not df.empty:
            df["날짜"] = pd.to_datetime(df["날짜"])
        return df
    else:
        # 지정한 열 이름에 맞춰 빈 DataFrame 생성
        return pd.DataFrame(columns=["날짜", "체중", "수면시간", "운동시간(분)", "물섭취량(ml)"])

def save_health_data(df):
    df_to_save = df.copy()
    if not df_to_save.empty and "날짜" in df_to_save.columns:
        # datetime 형식의 날짜를 문자열로 변환 후 저장
        df_to_save["날짜"] = df_to_save["날짜"].dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(CSV_FILE, index=False)

###########################################
# Streamlit 페이지 설정
###########################################
st.set_page_config(
    page_title="개인 건강 관리 대시보드",
    page_icon="💪",
    layout="wide"
)

st.title("💪 개인 건강 관리 대시보드")
st.markdown("CSV 파일에서 불러온 건강 데이터를 시각화하고, 새로운 데이터를 추가 및 저장할 수 있습니다.")

###########################################
# CSV 파일 업로드 (옵션)
###########################################
uploaded_csv = st.sidebar.file_uploader("CSV 파일 불러오기 (선택)", type=["csv"])
if uploaded_csv is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_csv)
        if not uploaded_df.empty and "날짜" in uploaded_df.columns:
            uploaded_df["날짜"] = pd.to_datetime(uploaded_df["날짜"])
        st.session_state.health_data = uploaded_df
        st.sidebar.success("CSV 파일 업로드 및 로드 완료!")
    except Exception as e:
        st.sidebar.error("CSV 파일 로드 오류: " + str(e))
else:
    if "health_data" not in st.session_state:
        st.session_state.health_data = load_health_data()

###########################################
# 사이드바: 건강 데이터 입력
###########################################
st.sidebar.header("오늘의 건강 데이터 입력")
date = st.sidebar.date_input("날짜", pd.Timestamp.today())
weight = st.sidebar.number_input("체중 (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
sleep_hours = st.sidebar.slider("수면 시간 (시간)", min_value=0, max_value=12, value=7)
exercise_minutes = st.sidebar.number_input("운동 시간 (분)", min_value=0, max_value=300, value=30, step=5)
water_intake = st.sidebar.number_input("물 섭취량 (ml)", min_value=0, max_value=5000, value=2000, step=100)

if st.sidebar.button("데이터 추가"):
    new_data = pd.DataFrame({
        "날짜": [pd.to_datetime(date)],  # 날짜를 pd.Timestamp로 변환
        "체중": [weight],
        "수면시간": [sleep_hours],
        "운동시간(분)": [exercise_minutes],
        "물섭취량(ml)": [water_intake]
    })
    st.session_state.health_data = pd.concat([st.session_state.health_data, new_data], ignore_index=True)
    st.sidebar.success("새 데이터가 추가되었습니다!")

if st.sidebar.button("CSV에 저장하기"):
    save_health_data(st.session_state.health_data)
    st.sidebar.success("CSV 파일이 업데이트되었습니다!")

###########################################
# 메인 화면: 저장된 데이터 표시
###########################################
st.subheader("저장된 건강 데이터")
if st.session_state.health_data.empty:
    st.info("아직 데이터가 없습니다. 사이드바에서 데이터를 입력하거나 CSV 파일을 불러오세요.")
else:
    df_display = st.session_state.health_data.copy()
    df_display["날짜"] = pd.to_datetime(df_display["날짜"])
    df_display = df_display.sort_values("날짜")
    st.dataframe(df_display)

###########################################
# 데이터 시각화: Plotly 차트
###########################################
if not st.session_state.health_data.empty:
    st.subheader("건강 데이터 시각화")
    df_viz = st.session_state.health_data.copy().sort_values("날짜")
    # Plotly가 날짜형을 인식할 수 있도록 확인
    if not pd.api.types.is_datetime64_any_dtype(df_viz["날짜"]):
        df_viz["날짜"] = pd.to_datetime(df_viz["날짜"])
    
    col1, col2 = st.columns(2)
    with col1:
        fig_weight = px.line(df_viz, x="날짜", y="체중", title="날짜별 체중 변화", markers=True)
        st.plotly_chart(fig_weight, use_container_width=True)
    with col2:
        fig_sleep = px.line(df_viz, x="날짜", y="수면시간", title="날짜별 수면 시간", markers=True)
        st.plotly_chart(fig_sleep, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        fig_exercise = px.bar(df_viz, x="날짜", y="운동시간(분)", title="날짜별 운동 시간")
        st.plotly_chart(fig_exercise, use_container_width=True)
    with col4:
        fig_water = px.bar(df_viz, x="날짜", y="물섭취량(ml)", title="날짜별 물섭취량")
        st.plotly_chart(fig_water, use_container_width=True)

###########################################
# 푸터
###########################################
st.markdown("---")
st.caption("© 2023 개인 건강 관리 대시보드")
