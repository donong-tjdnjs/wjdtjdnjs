import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="공공건축물 에너지 분석 및 시뮬레이터", layout="wide", page_icon="📊")

st.title("📊 공공건축물 데이터 탐색 및 에너지 소비량 예측 시뮬레이터")
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

# 1. 데이터 탐색 및 시각화 섹션
st.header("📌 1. 건축물 특성별 데이터 탐색 (Plotly 시각화)")
st.write("공공건축물의 **연면적**과 **지상 층수**에 따른 단위면적당 에너지 소비량 분포를 인터랙티브 산점도로 확인합니다.")

fig_scatter = px.scatter(
    df,
    x="연면적",
    y="연간 단위면적당 1차 에너지 소비량 3개년 평균",
    color="지역구분",
    size="규모(지상)",
    hover_name="건축물명",
    title="[인터랙티브] 연면적 및 층수에 따른 에너지 소비량 분포",
    labels={
        "연면적": "연면적 (㎡)",
        "연간 단위면적당 1차 에너지 소비량 3개년 평균": "단위면적당 에너지 소비량 (kWh/㎡)",
        "지역구분": "지역 구분",
        "규모(지상)": "지상 층수"
    },
    range_y=[0, 3000] # 이상치 제외 시각화 최적화
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# 2. 선형회귀 모델 학습 섹션
st.header("🤖 2. 선형회귀(Linear Regression) 예측 모델 성능 평가")
st.write("건물의 물리적 특성(연면적, 지상 층수)을 독립 변수로 학습한 회귀 모델의 평가 지표입니다.")

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
with col1:
    st.metric(label="🎯 결정계수 ($R^2$ Score)", value=f"{r2:.4f}", help="1에 가까울수록 모델의 예측력이 우수함을 의미합니다.")
with col2:
    st.metric(label="📉 평균 제곱 오차 (MSE)", value=f"{mse:.2f}")

st.markdown("---")

# 3. 실시간 인터랙티브 예측 시뮬레이터 (게이지 차트 적용)
st.header("🎛️ 3. 건축물 설계 조건별 실시간 에너지 소비량 예측 시뮬레이터")
st.write("슬라이더를 움직여 건물의 규모를 변경할 때, **실시간으로 변화하는 에너지 소비량과 게이지 시각화**를 확인해보세요.")

c1, c2 = st.columns(2)

with c1:
    st.subheader("🛠️ 설계 조건 입력")
    v_area = st.slider("연면적 (㎡)", float(X['연면적'].min()), float(min(X['연면적'].max(), 50000.0)), float(X['연면적'].mean()))
    v_floor = st.slider("지상 층수 (층)", float(X['규모(지상)'].min()), float(min(X['규모(지상)'].max(), 30.0)), float(X['규모(지상)'].mean()))
    
    # 예측 수행
    input_df = pd.DataFrame([[v_area, v_floor]], columns=features)
    pred_val = model.predict(input_df)[0]

with c2:
    st.subheader("📈 실시간 예측 결과 게이지")
    # Plotly 게이지 차트 생성 (시각적 완성도를 극대화하는 요소)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_val,
        title = {'text': "예상 에너지 소비량 (kWh/㎡)"},
        gauge = {
            'axis': {'range': [0, 1500]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 300], 'color': "lightgreen"},
                {'range': [300, 700], 'color': "yellow"},
                {'range': [700, 1500], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': pred_val
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.info(f"💡 **시뮬레이션 분석 결과:** 연면적 `{v_area:,.1f}㎡`, 지상 ` {v_floor:.0f}층` 규모의 공공건축물 설계 시, 예상되는 연간 단위면적당 1차 에너지 소비량은 약 **`{pred_val:.2f} kWh/㎡`** 입니다.")
