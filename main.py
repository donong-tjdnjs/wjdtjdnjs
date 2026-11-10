import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="건축물 에너지 효율 분석 시뮬레이터",
    layout="wide",
    page_icon="🏢",
)

st.title("🏢 건축물 설계 요소에 따른 에너지 효율 예측 웹앱")
st.markdown("---")

st.markdown(
    """
### 💡 프로젝트 소개
본 웹앱은 **UCI 에너지 효율(Energy Efficiency) 데이터셋**을 활용하여, 
건물의 형태적 설계 요소(밀집도, 표면적, 벽면적, 지붕면적, 높이, 방향, 창문 면적 등)가 
건물의 **냉난방 에너지 부하(Heating & Cooling Load)**에 미치는 영향을 분석하고 예측하기 위해 제작되었습니다.

---
### 🧭 페이지 안내
* **1. 에너지 분석 및 회귀 모델 (`pages/1_에너지_분석.py`)**: 
  - 데이터 탐색(EDA) 및 시각화 (Plotly)
  - 선형회귀 모델을 통한 냉난방 부하 예측 학습 및 성능 평가
  - 사용자 맞춤형 설계 조건 입력에 따른 실시간 에너지 부하 예측 시뮬레이터
"""
)

# 데이터 로드 확인용 캐시 함수
@st.cache_data
def load_data():
    # UCI Energy Efficiency 데이터셋 (공개된 정제 소스 활용)
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/energy_efficiency.csv"
    try:
        df = pd.read_csv(url)
    except:
        # 혹시 모를 로드 실패 시 예시 구조 대응
        df = None
    return df

df = load_data()

if df is not None:
    st.success("✅ 데이터가 성공적으로 로드되었습니다!")
    st.subheader("📋 원본 데이터 미리보기 (처음 5행)")
    st.dataframe(df.head())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 데이터 샘플 수", f"{len(df)}개")
    col2.metric("특성(컬럼) 수", f"{df.shape[1]}개")
    col3.metric("타겟 변수", "냉방/난방 부하")
else:
    st.error("데이터를 불러오는 중 문제가 발생했습니다. 인터넷 연결이나 URL을 확인해주세요.")
