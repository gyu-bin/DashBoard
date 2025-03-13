import streamlit as st

# 페이지 설정 (아이콘과 제목 설정)
st.set_page_config(
    page_title="내 소개",
    page_icon="👋",
    layout="wide"
)

# 개선된 CSS 스타일 (디자인 변경)
st.markdown("""
    <style>
    /* 전체 배경과 기본 폰트 설정 */
    .main {
        background: linear-gradient(135deg, #e0f7fa, #e8f5e9);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        padding: 30px;
    }
    /* 제목 스타일 - 좀 더 부드러운 느낌과 그림자 효과 추가 */
    .title {
        color: #00695c;
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }
    /* 헤더 스타일 */
    .header {
        color: #2e7d32;
        font-size: 2.2rem;
        margin-top: 25px;
        border-bottom: 2px solid #b2dfdb;
        padding-bottom: 5px;
    }
    /* 강조박스 스타일 - 둥근 테두리와 부드러운 그림자 */
    .highlight {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    /* 방문자 인사말 스타일 */
    .greeting-box {
        background-color: #f1f8e9;
        padding: 25px;
        border-radius: 10px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 앱 제목
st.markdown("<div class='title'>✨ 내 소개 ✨</div>", unsafe_allow_html=True)

# 메인 콘텐츠 영역에 CSS 클래스 적용
st.markdown("<div class='main'>", unsafe_allow_html=True)

# 프로필 정보를 2개 컬럼으로 표시
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("<div style='text-align: center; font-size: 7rem;'>👍💻</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<h2 class='header'>기본 정보</h2>", unsafe_allow_html=True)
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("**이름**: 문규빈")
    st.markdown("**전공**: 보건계열")
    st.markdown("**거주지**: 서울시 강서구")
    st.markdown("</div>", unsafe_allow_html=True)

# 취미 섹션
st.markdown("<h2 class='header'>관심 프로젝트/도메인</h2>", unsafe_allow_html=True)
st.markdown("<div class='highlight'>", unsafe_allow_html=True)
st.markdown("""
* 🏃‍♂️ **1순위**: 건강 관련 (예: 피트니스 앱, 웨어러블 기기)
* 📚 **2순위**: 교육 및 학습 플랫폼
* 🎸 **3순위**: 음악/문화 관련 프로젝트
""")
st.markdown("</div>", unsafe_allow_html=True)

# 자기소개 섹션
st.markdown("<h2 class='header'>자기소개</h2>", unsafe_allow_html=True)
st.markdown("<div class='highlight'>", unsafe_allow_html=True)
st.write("""
안녕하세요! 저는 최신 기술과 창의적 아이디어에 관심이 많은 개발자입니다.
앱 및 웹 개발을 해왔으며, 운동과 음악에도 열정을 가지고 있습니다.
새로운 프로젝트에 도전하며 지속적으로 성장하고자 노력하고 있습니다.
""")
st.markdown("</div>", unsafe_allow_html=True)

# 방문자 인사말 섹션
st.markdown("<h2 class='header'>방명록</h2>", unsafe_allow_html=True)
greeting = st.text_input("인사말을 남겨주세요 👋")
if greeting:
    st.markdown("<div class='greeting-box'>", unsafe_allow_html=True)
    st.success(f"감사합니다, {greeting}! 방문해주셔서 기쁩니다. 😊")
    st.markdown("</div>", unsafe_allow_html=True)

# 푸터
st.markdown("</div>", unsafe_allow_html=True)  # main div 닫기
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #424242;'>© 2025 멋쟁이사자처럼 프로젝트</p>", unsafe_allow_html=True)
