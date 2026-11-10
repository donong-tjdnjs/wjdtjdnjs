import streamlit as st
import pandas as pd

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
건물의 형태적 설계 요소(상대적 밀집도, 표면적, 벽면적, 지붕면적, 건물 높이, 유리창 면적 비율 등)가 
건물의 **난방 에너지 부하(Heating Load)**에 미치는 영향을 분석하고 선형회귀 모델로 예측하기 위해 제작되었습니다.

---
### 🧭 페이지 안내
* **1. 에너지 분석 (`pages/1_에너지_분석.py`)**: 
  - 설계 요소와 에너지 부하 간의 탐색적 데이터 분석 (Plotly 시각화)
  - Scikit-learn 선형회귀(Linear Regression) 모델 학습 및 $R^2$, MSE 성능 평가
  - 사용자 맞춤형 건물 설계 수치 입력에 따른 실시간 난방 부하 예측 시뮬레이터
"""
)

# 데이터 URL (인터넷에서 바로 읽어오는 Raw CSV 링크)
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/energy-efficiency.csv"

@st.cache_data
def load_data():
    # 데이터 로드 (컬럼명이 없거나 다른 경우를 대비해 표준 건축 용어로 정리)
    df = pd.read_csv(DATA_URL)
    
    # 데이터 컬럼명 한글/영문 매핑 표준화
    # X1: Relative Compactness, X2: Surface Area, X3: Wall Area, X4: Roof Area, 
    # X5: Overall Height, X6: Orientation, X7: Glazing Area, X8: Glazing Area Distribution, Y1: Heating Load, Y2: Cooling Load
    rename_dict = {
        'X1': 'Relative_Compactness',
        'X2': 'Surface_Area',
        'X3': 'Wall_Area',
        'X4': 'Roof_Area',
        'X5': 'Overall_Height',
        'X6': 'Orientation',
        'X7': 'Glazing_Area',
        'X8': 'Glazing_Area_Distribution',
        'Y1': 'Heating_Load',
        'Y2': 'Cooling_Load'
    }
    df = df.rename(columns=rename_dict)
    return df

try:
    df = load_data()
    st.success("✅ 인터넷 데이터셋(Raw CSV)이 성공적으로 로드되었습니다!")
    st.subheader("📋 원본 데이터 미리보기 (처음 5행)")
    st.dataframe(df.head())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("전체 건물 샘플 수", f"{len(df)}개")
    col2.metric("전체 변수(컬럼) 수", f"{df.shape[1]}개")
    col3.metric("예측 타겟 변수", "Heating Load (난방 부하)")
    
    st.markdown("---")
    st.caption(f"**데이터 출처 URL:** `{DATA_URL}`")
    
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
