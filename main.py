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
    
    # 결측치 및 비정상적인 극단값(0 이하 또는 5000 초과) 정제
    df = df.dropna(subset=[target_col, '연면적', '규모(지상)', '지역구분', '용도'])
    df = df[(df[target_col] > 0) & (df[target_col] <= 5000)]
    
    return df, raw_count

try:
    df, raw_count = load_and_clean_data()
    target_col = '연간 단위면적당 1차 에너지 소비량 3개년 평균'
    
    # 소질문 1: 데이터 정제 및 요약 통계 대시보드
    st.subheader("📌 1. 데이터 정제 현황 및 핵심 통계 요약")
    st.info("원본 데이터에서 에너지 소비량 '미제출' 항목과 통계적 극단 이상치를 정제하여 분석의 신뢰도를 확보했습니다.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("원본 데이터 수", f"{raw_count:,} 개")
    with col2:
        st.metric("정제 후 유효 건물 수", f"{len(df):,} 개", delta=f"-{raw_count - len(df):,} 개 정제됨", delta_color="off")
    with col3:
        st.metric("평균 단위면적당 소비량", f"{df[target_col].mean():.1f} kWh/㎡")
    with col4:
        st.metric("평균 연면적", f"{df['연면적'].mean():,.0f} ㎡")
        
    st.markdown("---")
    
    # 프로젝트 로드맵 안내
    st.subheader("🧭 프로젝트 탐구 로드맵 (좌측 사이드바에서 분석 페이지 이동)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        * **소질문 1 [데이터 정제]**: 결측치('미제출') 및 이상치 제거 후 통계 분포 확인 *(현재 페이지)*
        * **소질문 2 [지역별 특성]**: 기후 권역(지역구분)별 단위면적당 에너지 소비량 차이 비교
        * **소질문 3 [용도별 특성]**: 건축물 주용도별 에너지 집약도 비교 및 필터링 분석
        """)
    with c2:
        st.markdown("""
        * **소질문 4 [머신러닝 평가]**: 선형회귀(Linear Regression) 모델 학습 및 성능($R^2$, MSE) 검증
        * **소질문 5 [실시간 예측]**: 연면적 및 지상 층수 슬라이더 조작을 통한 소비량 예측
        * **소질문 6 [시각화 극대화]**: 예측 결과를 직관적으로 보여주는 실시간 게이지(계기판) 구현
        """)
        
    st.markdown("---")
    st.subheader("📋 정제 완료된 공공건축물 데이터 미리보기")
    display_cols = ['지역구분', '용도', '건축물명', '연면적', '규모(지상)', '사용승인연도', target_col]
    st.dataframe(df[display_cols].head(10), use_container_width=True)

except Exception as e:
    st.error(f"데이터 파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.warning("깃허브 저장소 루트 폴더에 '국토안전관리원_공공건축물_에너지_소비량_20260331.csv' 파일이 업로드되어 있는지 확인해주세요.")
