import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="공공건축물 에너지 소비량 분석 시뮬레이터",
    layout="wide",
    page_icon="🏢"
)

st.title("🏢 공공건축물 특성별 에너지 소비량 분석 및 예측 웹앱")
st.markdown("---")

st.markdown("""
### 💡 프로젝트 소개
본 웹앱은 **국토안전관리원 공공건축물 에너지 소비량 데이터**를 활용하여, 
건물의 **연면적, 지상 층수, 용도, 지역구분** 등의 건축적 특성이 
**연간 단위면적당 1차 에너지 소비량**에 미치는 영향을 분석하고 선형회귀 모델로 예측하기 위해 제작되었습니다.

---
### 🧭 페이지 안내
* **1. 에너지 분석 및 예측 (`pages/1_에너지_분석.py`)**:
  - 결측치 및 '미제출' 데이터를 정제한 데이터 탐색적 분석 (Plotly 시각화)
  - Scikit-learn 선형회귀(Linear Regression) 모델 학습 및 성능 평가 ($R^2$, MSE)
  - 사용자 맞춤형 공공건축물 조건 입력에 따른 실시간 에너지 소비량 예측 시뮬레이터
""")

@st.cache_data
def load_data():
    df = pd.read_csv("국토안전관리원_공공건축물_에너지_소비량_20260331.csv", encoding="utf-8")
    # '미제출' 데이터 정제 및 숫자형 변환
    df = df[df['연간 단위면적당 1차 에너지 소비량 3개년 평균'] != '미제출'].copy()
    df['연간 단위면적당 1차 에너지 소비량 3개년 평균'] = pd.to_numeric(df['연간 단위면적당 1차 에너지 소비량 3개년 평균'])
    df['연면적'] = pd.to_numeric(df['연면적'], errors='coerce')
    df['규모(지상)'] = pd.to_numeric(df['규모(지상)'], errors='coerce')
    return df.dropna(subset=['연간 단위면적당 1차 에너지 소비량 3개년 평균', '연면적', '규모(지상)'])

try:
    df = load_data()
    st.success("✅ 공공건축물 에너지 데이터가 성공적으로 로드되었습니다!")
    st.subheader("📋 정제된 원본 데이터 미리보기 (처음 5행)")
    st.dataframe(df.head())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("유효 건물 샘플 수", f"{len(df)}개")
    col2.metric("전체 변수(컬럼) 수", f"{df.shape[1]}개")
    col3.metric("예측 타겟 변수", "단위면적당 에너지 소비량")
    
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
