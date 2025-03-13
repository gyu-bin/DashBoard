import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="카페 주문 시스템",
    page_icon="☕",
    layout="centered"
)

# 앱 제목 (요구사항 1)
st.title("☕ 카페 주문 시스템")
st.markdown("원하시는 음료를 선택하고 주문해주세요.")

# 주문자 정보 섹션
st.subheader("주문자 정보")

# 주문자 이름 입력 (요구사항 2)
customer_name = st.text_input("이름을 입력해주세요")

# 음료 선택 섹션
st.subheader("음료 선택")

# 음료 가격 정보 (도전 과제)
drink_prices = {
    "아메리카노": 2000,
    "카페라떼": 2500,
    "카푸치노": 3000,
    "에스프레소": 1500
}

# 음료 선택 (요구사항 3)
drink = st.selectbox(
    "음료를 선택해주세요",
    list(drink_prices.keys())
)

# 음료 크기 선택 (요구사항 4)
size = st.radio(
    "크기를 선택해주세요",
    ["Small", "Medium", "Large"],
    horizontal=True
)

# 크기별 추가 가격 (도전 과제)
size_prices = {
    "Small": 0,
    "Medium": 500,
    "Large": 1000
}

# 옵션 선택 섹션
st.subheader("옵션 선택")

# 옵션 체크박스 (요구사항 5)
col1, col2 = st.columns(2)

with col1:
    extra_shot = st.checkbox("샷 추가 (+500원)")
    whipped_cream = st.checkbox("휘핑크림 추가 (+500원)")

with col2:
    ice = st.checkbox("얼음 추가 (무료)")
    takeout = st.checkbox("테이크아웃")

# 수량 선택 (요구사항 6)
st.subheader("수량 선택")
quantity = st.slider("수량을 선택해주세요", min_value=1, max_value=30, value=1)

# 주문 버튼 (요구사항 7)
st.markdown("---")
if st.button("주문하기"):
    # 주문 정보가 있는지 확인
    if not customer_name:
        st.error("이름을 입력해주세요!")
    else:
        # 총 가격 계산 (도전 과제)
        base_price = drink_prices[drink]
        size_extra = size_prices[size]
        option_price = 0
        
        if extra_shot:
            option_price += 500
        if whipped_cream:
            option_price += 500
            
        unit_price = base_price + size_extra + option_price
        total_price = unit_price * quantity
        
        # 주문 정보 표시
        st.success("주문이 완료되었습니다!")
        
        # 주문 내역 요약 표시
        st.subheader("📝 주문 내역")
        
        # 주문 정보 테이블 생성
        order_info = {
            "항목": ["주문자", "음료", "크기", "옵션", "수량", "단가", "총 금액"],
            "내용": [
                customer_name,
                drink,
                size,
                ", ".join([opt for opt, selected in {
                    "샷 추가": extra_shot,
                    "휘핑크림 추가": whipped_cream,
                    "얼음 추가": ice,
                    "테이크아웃": takeout
                }.items() if selected]) or "없음",
                f"{quantity}잔",
                f"{unit_price:,}원",
                f"{total_price:,}원"
            ]
        }
        
        # 주문 내역 표시
        for i in range(len(order_info["항목"])):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{order_info['항목'][i]}**")
            with col2:
                st.markdown(f"{order_info['내용'][i]}")
        
        # 영수증 스타일로 총액 표시
        st.markdown("---")
        st.markdown(f"### 💰 결제 금액: {total_price:,}원")
        
        # 감사 메시지
        st.markdown("---")
        st.markdown(f"### {customer_name}님, 주문해주셔서 감사합니다! ☕")
        
        # 도전 과제: 주문 내역 상세 표시
        with st.expander("주문 상세 내역 보기"):
            st.markdown(f"**기본 가격 ({drink})**: {base_price:,}원")
            if size != "Small":
                st.markdown(f"**크기 추가 ({size})**: +{size_extra:,}원")
            if extra_shot:
                st.markdown("**샷 추가**: +500원")
            if whipped_cream:
                st.markdown("**휘핑크림 추가**: +500원")
            st.markdown(f"**단가**: {unit_price:,}원")
            st.markdown(f"**수량**: {quantity}잔")
            st.markdown("---")
            st.markdown(f"**총 금액**: {total_price:,}원")

# 푸터
st.markdown("---")
st.caption("© 2025 멋쟁이사자처럼 프로젝트")