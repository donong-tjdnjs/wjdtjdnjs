import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="공공건축물 에너지 소비량 분석 및 예측",
    layout="wide",
    page_icon="🏛️"
)

st.title("🏛️ 공공건축물 특성별 에너지 소비량 분석 및 예측 플랫폼")
st.markdown("##### 국토안전관리원 공공건축물 데이터를 활용한 건축공학 데이터 사이언스 프로젝트")
st.markdown("---")

@st.cache_data
def load_and_clean_data():
    df_raw = pd.read_csv("국토안전관리원_공공건축물_에너지_소비량_20260331.csv", encoding="utf-8")
    raw_count = len(df_raw)
    
    # '미제출' 제거 및 숫자형 변환
    df = df_raw[df_raw['연간 단위면적당 1차 에너지 소비량 3개년 평균'] != '미제출'].copy()
    target_col = '연간 단위면적당 1차 에너지 소비량 3개년 평균'
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df['연면적'] = pd.to_numeric(df['연면적'], errors='coerce')
    df['규모(지상)'] = pd.to_numeric(df['규모(지상)'], errors='coerce')
    
    # 결측치 및 이상치 정제
    df = df.dropna(subset=[target_col, '연면적', '규모(지상)', '지역구분', '용도'])
    df = df[(df[target_col] > 0) & (df[target_col] <= 5000)]
    
    return df, raw_count

try:
    df, raw_count = load_and_clean_data()
    target_col = '연간 단위면적당 1차 에너지 소비량 3개년 평균'
    
    st.subheader("📌 1. 데이터 전처리 및 현황 요약")
    st.info("원시 데이터에서 에너지 소비량 '미제출' 항목과 극단적 이상치를 정제하여 신뢰도 높은 데이터셋을 구축했습니다.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("원본 데이터 수", f"{raw_count:,} 개")
    with col2:
        st.metric("정제된 유효 건물 수", f"{len(df):,} 개", delta=f"-{raw_count - len(df):,} 개 정제됨", delta_color="off")
    with col3:
        st.metric("평균 단위면적당 소비량", f"{df[target_col].mean():.1f} kWh/㎡")
    with col4:
        st.metric("평균 연면적", f"{df['연면적'].mean():,.0f} ㎡")
        
    st.markdown("---")
    
    st.subheader("🧭 웹앱 탐구 및 시뮬레이터 안내 (좌측 사이드바에서 이동)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        * **1단계 [데이터 정제 현황]**: 결측치 정제 결과 및 전체 통계 요약 *(현재 페이지)*
        * **2단계 [지역별 에너지 분석]**: 기후 권역별 단위면적당 에너지 소비량 편차 분석
        * **3단계 [용도별 에너지 분석]**: 건축물 용도별 에너지 집약도 및 멀티 비교
        """)
    with c2:
        st.markdown("""
        * **4단계 [머신러닝 예측 모델]**: 선형회귀(Linear Regression) 성능 검증 ($R^2$, MSE)
        * **5단계 [실시간 시뮬레이터]**: 슬라이더 조작을 통한 건물 규모별 에너지 예측
        * **6단계 [시각화 게이지]**: 예측 결과를 계기판 형태로 직관적 확인
        """)
        
    st.markdown("---")
    st.subheader("📋 정제 완료된 공공건축물 데이터 미리보기")
    display_cols = ['지역구분', '용도', '건축물명', '연면적', '규모(지상)', '사용승인연도', target_col]
    st.dataframe(df[display_cols].head(10), use_container_width=True)

except Exception as e:
    st.error(f"데이터 파일을 불러오는 중 오류가 발생했습니다: {e}")
