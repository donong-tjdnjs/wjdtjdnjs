import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="에너지 분석 및 회귀 모델", layout="wide")

st.title("📊 건축물 설계 데이터 탐색 및 에너지 부하 예측 모델")
st.markdown("---")

# 데이터 로드
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/energy_efficiency.csv"
    return pd.read_csv(url)

df = load_data()

# 1. 데이터 탐색 섹션
st.header("📌 1. 데이터 탐색적 분석 (EDA & Plotly 시각화)")
st.write("건물의 벽면적(`Wall Area`)과 냉난방 부하(`Heating Load`) 간의 관계를 산점도로 확인합니다.")

fig_scatter = px.scatter(
    df,
    x="Wall Area",
    y="Heating Load",
    color="Overall Height",
    title="벽면적에 따른 난방 부하 분포",
    labels={"Wall Area": "벽면적 (m²)", "Heating Load": "난방 부하", "Overall Height": "건물 높이"},
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# 2. 선형회귀 모델 학습 섹션
st.header("🤖 2. 선형회귀 모델(Linear Regression) 학습 및 평가")

# 독립 변수와 종속 변수 설정 (여기서는 난방 부하 예측을 타겟으로 설정)
# 컬럼명: Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution, Heating Load, Cooling Load
feature_cols = ["Relative Compactness", "Surface Area", "Wall Area", "Roof Area", "Overall Height", "Glazing Area"]
target_col = "Heating Load"

X = df[feature_cols]
y = df[target_col]

# 데이터 분할 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("📌 테스트 세트 결정계수 ($R^2$ Score)", f"{r2:.4f}")
col2.metric("📌 평균 제곱 오차 (MSE)", f"{mse:.4f}")

st.markdown("---")

# 3. 사용자 맞춤형 예측 시뮬레이터
st.header("🎛️ 3. 건축물 설계 조건별 난방 부하 실시간 예측 시뮬레이터")
st.write("슬라이더를 조절하여 건물의 설계 수치를 바꾸고 예상되는 난방 부하를 확인해보세요.")

# 슬라이더 입력 위젯 구성
rel_comp = st.slider("상대적 밀집도 (Relative Compactness)", float(X["Relative Compactness"].min()), float(X["Relative Compactness"].max()), float(X["Relative Compactness"].mean()))
surf_area = st.slider("표면적 (Surface Area)", float(X["Surface Area"].min()), float(X["Surface Area"].max()), float(X["Surface Area"].mean()))
wall_area = st.slider("벽면적 (Wall Area)", float(X["Wall Area"].min()), float(X["Wall Area"].max()), float(X["Wall Area"].mean()))
roof_area = st.slider("지붕면적 (Roof Area)", float(X["Roof Area"].min()), float(X["Roof Area"].max()), float(X["Roof Area"].mean()))
height = st.selectbox("건물 높이 (Overall Height)", options=sorted(X["Overall Height"].unique()))
glazing = st.slider("유리창 면적 비율 (Glazing Area)", float(X["Glazing Area"].min()), float(X["Glazing Area"].max()), float(X["Glazing Area"].mean()))

# 입력값 예측
input_data = pd.DataFrame([[rel_comp, surf_area, wall_area, roof_area, height, glazing]], columns=feature_cols)
predicted_load = model.predict(input_data)[0]

st.success(f"💡 **예상 난방 부하 (Heating Load):** `{predicted_load:.2f}` kW")
