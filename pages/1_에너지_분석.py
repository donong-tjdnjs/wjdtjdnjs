import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="에너지 분석 및 회귀 모델", layout="wide", page_icon="📊")

st.title("📊 공공건축물 데이터 탐색 및 에너지 소비량 예측")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("국토안전관리원_공공건축물_에너지_소비량_20260331.csv", encoding="utf-8")
    df = df[df['연간 단위면적당 1차 에너지 소비량 3개년 평균'] != '미제출'].copy()
    df['연간 단위면적당 1차 에너지 소비량 3개년 평균'] = pd.to_numeric(df['연간 단위면적당 1차 에너지 소비량 3개년 평균'])
    df['연면적'] = pd.to_numeric(df['연면적'], errors='coerce')
    df['규모(지상)'] = pd.to_numeric(df['규모(지상)'], errors='coerce')
    return df.dropna(subset=['연간 단위면적당 1차 에너지 소비량 3개년 평균', '연면적', '규모(지상)'])

df = load_data()

# 1. 시각화 섹션
st.header("📌 1. 데이터 탐색적 분석 (Plotly 시각화)")
st.write("건물의 **연면적**과 **단위면적당 에너지 소비량** 간의 관계를 지역구분에 따라 시각화합니다.")

fig = px.scatter(
    df,
    x="연면적",
    y="연간 단위면적당 1차 에너지 소비량 3개년 평균",
    color="지역구분",
    title="연면적에 따른 단위면적당 에너지 소비량 분포",
    labels={
        "연면적": "연면적 (㎡)",
        "연간 단위면적당 1차 에너지 소비량 3개년 평균": "단위면적당 에너지 소비량",
        "지역구분": "지역 구분"
    },
    range_y=[0, 3000] # 극단적 이상치로 인한 시각화 왜곡 방지
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 2. 선형회귀 모델 섹션
st.header("🤖 2. 선형회귀 모델(Linear Regression) 학습 및 평가")
st.write("연면적과 지상 층수를 독립 변수로 사용하여 에너지 소비량을 예측하는 회귀 모델을 학습합니다.")

features = ['연면적', '규모(지상)']
target = '연간 단위면적당 1차 에너지 소비량 3개년 평균'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("테스트 데이터 $R^2$ Score (결정계수)", f"{r2:.4f}")
col2.metric("평균 제곱 오차 (MSE)", f"{mse:.4f}")

st.markdown("---")

# 3. 실시간 예측 시뮬레이터
st.header("🎛️ 3. 건축물 조건에 따른 에너지 소비량 실시간 예측")
st.write("슬라이더를 조절하여 건물의 연면적과 지상 층수를 입력하고 예상되는 에너지 소비량을 확인해보세요.")

c1, c2 = st.columns(2)
with c1:
    v_area = st.slider("연면적 (㎡)", float(X['연면적'].min()), float(min(X['연면적'].max(), 50000.0)), float(X['연면적'].mean()))
with c2:
    v_floor = st.slider("지상 층수 (층)", float(X['규모(지상)'].min()), float(min(X['규모(지상)'].max(), 30.0)), float(X['규모(지상)'].mean()))

input_df = pd.DataFrame([[v_area, v_floor]], columns=features)
pred_val = model.predict(input_df)[0]

st.success(f"💡 **예상 연간 단위면적당 에너지 소비량:** `{pred_val:.2f}` kWh/㎡")
