import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="에너지 분석 및 시뮬레이터", layout="wide", page_icon="📊")

st.title("📊 공공건축물 에너지 소비량 심층 분석 및 예측 시뮬레이터")
st.markdown("---")

@st.cache_data
def load_clean_data():
    df = pd.read_csv("국토안전관리원_공공건축물_에너지_소비량_20260331.csv", encoding="utf-8")
    target_col = '연간 단위면적당 1차 에너지 소비량 3개년 평균'
    df = df[df[target_col] != '미제출'].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df['연면적'] = pd.to_numeric(df['연면적'], errors='coerce')
    df['규모(지상)'] = pd.to_numeric(df['규모(지상)'], errors='coerce')
    df = df.dropna(subset=[target_col, '연면적', '규모(지상)', '지역구분', '용도'])
    df = df[(df[target_col] > 0) & (df[target_col] <= 5000)]
    return df

df = load_clean_data()
target_col = '연간 단위면적당 1차 에너지 소비량 3개년 평균'

# ==========================================
# 1단계: 지역별 및 용도별 특성 비교 시각화
# ==========================================
st.header("📌 1. 공간적(지역) 및 건축 용도별 에너지 소비 특성 분석")
st.write("기후 권역(지역구분)과 건축물의 용도에 따라 단위면적당 에너지 소비량이 어떤 차이를 보이는지 분석합니다.")

tab1, tab2 = st.tabs(["🗺️ 지역구분별 소비량 비교", "🏢 건축 용도별 에너지 집약도 비교"])

with tab1:
    region_avg = df.groupby('지역구분')[target_col].mean().reset_index().sort_values(by=target_col, ascending=False)
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig_box = px.box(
            df,
            x="지역구분",
            y=target_col,
            color="지역구분",
            title="지역구분별 단위면적당 에너지 소비량 분포",
            labels={target_col: "단위면적당 에너지 소비량 (kWh/㎡)", "지역구분": "기후 권역(지역)"}
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with c2:
        fig_bar = px.bar(
            region_avg,
            x="지역구분",
            y=target_col,
            color=target_col,
            color_continuous_scale="Blues",
            title="지역구분별 평균 에너지 소비량 순위",
            labels={target_col: "평균 에너지 소비량 (kWh/㎡)", "지역구분": "기후 권역(지역)"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.success("💡 **[데이터 분석 결과 및 시사점]**: 지역별 기후 특성(외기 온도 등)에 따라 난방 및 공조 부하가 달라지며, 특정 기후 권역에서 단위면적당 에너지 소비량이 더 높게 나타나는 경향을 확인할 수 있습니다.")

with tab2:
    # 용도 선택 인터랙티브 필터 (기본값을 상위 3개로 설정하여 데이터가 한눈에 보이도록 최적화)
    usage_list = sorted(df['용도'].unique())
    default_selected = usage_list[:3] if len(usage_list) >= 3 else usage_list
    
    selected_usages = st.multiselect(
        "비교할 건축물 용도를 선택하세요 (여러 개 선택 가능):",
        options=usage_list,
        default=default_selected
    )
    
    if selected_usages:
        filtered_df = df[df['용도'].isin(selected_usages)]
        
        # [수정 완료] 축 고정 범위를 없애고 자동 조정(autoscale)되도록 수정하여 데이터가 사라지지 않음
        fig_scatter = px.scatter(
            filtered_df,
            x="연면적",
            y=target_col,
            color="용도",
            size="규모(지상)",
            hover_name="건축물명",
            title="선택된 용도별 연면적 대비 에너지 소비량 분산 비교",
            labels={"연면적": "연면적 (㎡)", target_col: "단위면적당 에너지 소비량 (kWh/㎡)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.success("💡 **[데이터 분석 결과 및 시사점]**: 건축물 용도(예: 업무시설 vs 문화시설 등)에 따라 내부 설비 가동 시간과 에너지 집약도가 다르며, 연면적이 커질수록 규모의 경제로 인해 단위면적당 소비량이 수렴하는 패턴을 도출할 수 있습니다.")
    else:
        st.warning("비교를 위해 최소 1개 이상의 건축물 용도를 선택해 주세요.")

st.markdown("---")

# ==========================================
# 2단계: 머신러닝 모델 학습 및 성능 평가
# ==========================================
st.header("🤖 2. 선형회귀(Linear Regression) 예측 모델 학습 및 성능 평가")
st.write("건물의 물리적 규모 특성(**연면적**, **지상 층수**)을 독립 변수로 활용하여 에너지 소비량을 예측하는 AI 모델을 학습하고 정확도를 검증합니다.")

features = ['연면적', '규모(지상)']
X = df[features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("학습/테스트 데이터 분할", "80% : 20%", f"테스트 {len(X_test):,}건")
with m2:
    st.metric("🎯 결정계수 ($R^2$ Score)", f"{r2:.4f}", help="모델이 데이터의 경향성을 설명하는 정도")
with m3:
    st.metric("📉 평균 제곱 오차 (MSE)", f"{mse:,.2f}", help="예측 오차의 크기")

st.info("💡 **[모델 평가 결과 및 시사점]**: 연면적과 층수만으로도 건물의 에너지 소비 트렌드를 일정 부분 예측할 수 있음을 확인했으며, 향후 창호 비율이나 단열 성능 등 추가 변수를 결합하면 예측력을 더욱 향상시킬 수 있습니다.")

st.markdown("---")

# ==========================================
# 3단계: 실시간 시뮬레이터 및 게이지 차트
# ==========================================
st.header("🎛️ 3. 건축물 설계 조건별 실시간 에너지 소비량 예측 시뮬레이터")
st.write("슬라이더를 조절하여 건물의 규모를 변경하면, 학습된 회귀 모델이 실시간으로 연산한 예상 에너지 소비량과 게이지 시각화를 제공합니다.")

sim_col1, sim_col2 = st.columns([1, 1.2])

with sim_col1:
    st.subheader("🛠️ 가상 건축물 설계 파라미터 입력")
    
    v_area = st.slider(
        "📐 건물 연면적 (㎡)",
        min_value=float(X['연면적'].min()),
        max_value=50000.0,
        value=float(X['연면적'].median()),
        step=500.0
    )
    
    v_floor = st.slider(
        "🏢 지상 층수 (층)",
        min_value=1.0,
        max_value=30.0,
        value=float(X['규모(지상)'].median()),
        step=1.0
    )
    
    input_df = pd.DataFrame([[v_area, v_floor]], columns=features)
    pred_val = max(0.0, float(model.predict(input_df)[0]))
    avg_val = float(df[target_col].mean())
    diff_val = pred_val - avg_val
    
    st.markdown("---")
    st.metric(
        label="공공건축물 전체 평균 대비 예측 차이",
        value=f"{pred_val:.2f} kWh/㎡",
        delta=f"{diff_val:+.2f} kWh/㎡ (전체 평균: {avg_val:.1f})",
        delta_color="inverse"
    )

with sim_col2:
    st.subheader("📈 실시간 에너지 효율 게이지 차트")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "예상 연간 단위면적당 1차 에너지 소비량 (kWh/㎡)", 'font': {'size': 16}},
        delta={'reference': avg_val, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 250], 'color': "#d4edda"},
                {'range': [250, 500], 'color': "#fff3cd"},
                {'range': [500, 1000], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_val
            }
        }
    ))
    fig_gauge.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

if pred_val < avg_val:
    st.success(f"✅ **[시뮬레이션 결론]**: 연면적 `{v_area:,.0f}㎡`, 지상 `{v_floor:.0f}층` 조건은 공공건축물 평균보다 에너지 소비가 적어 **친환경 고효율 설계**에 적합합니다.")
else:
    st.warning(f"⚠️ **[시뮬레이션 결론]**: 해당 규모의 건축물은 평균보다 에너지 소비량이 높을 것으로 예측되므로, **고효율 단열재 및 에너지 절감 설비 설계**가 추가로 요구됩니다.")
