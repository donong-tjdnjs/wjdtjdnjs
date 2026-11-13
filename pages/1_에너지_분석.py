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
# 소질문 2 & 3: 지역별 및 용도별 특성 비교 시각화
# ==========================================
st.header("📌 1. 공간적(지역) 및 건축 용도별 에너지 소비 특성 비교")
st.write("기후 권역(지역구분)과 건축물의 주된 용도에 따라 단위면적당 에너지 소비량이 어떻게 달라지는지 시각적으로 비교합니다.")

tab1, tab2 = st.tabs(["🗺️ 소질문 2: 지역구분별 소비량 비교", "🏢 소질문 3: 건축 용도별 에너지 집약도 필터링"])

with tab1:
    # 지역별 평균 계산
    region_avg = df.groupby('지역구분')[target_col].mean().reset_index().sort_values(by=target_col, ascending=False)
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig_box = px.box(
            df,
            x="지역구분",
            y=target_col,
            color="지역구분",
            title="지역구분별 단위면적당 에너지 소비량 분포 (Box Plot)",
            labels={target_col: "단위면적당 에너지 소비량 (kWh/㎡)", "지역구분": "기후 권역(지역)"},
            range_y=[0, 1500]
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

with tab2:
    # 용도 선택 인터랙티브 필터
    usage_list = sorted(df['용도'].unique())
    selected_usages = st.multiselect(
        "비교할 건축물 용도를 선택하세요 (기본: 상위 주요 용도):",
        options=usage_list,
        default=usage_list[:5] if len(usage_list) >= 5 else usage_list
    )
    
    if selected_usages:
        filtered_df = df[df['용도'].isin(selected_usages)]
        fig_scatter = px.scatter(
            filtered_df,
            x="연면적",
            y=target_col,
            color="용도",
            size="규모(지상)",
            hover_name="건축물명",
            title="선택된 용도별 연면적 대비 에너지 소비량 분포",
            labels={"연면적": "연면적 (㎡)", target_col: "단위면적당 에너지 소비량 (kWh/㎡)"},
            range_y=[0, 2000],
            range_x=[0, 60000]
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("최소 1개 이상의 건축물 용도를 선택해 주세요.")

st.markdown("---")

# ==========================================
# 소질문 4: 머신러닝 모델 학습 및 성능 평가
# ==========================================
st.header("🤖 2. 소질문 4: 선형회귀(Linear Regression) 예측 모델 학습 및 평가")
st.write("공공건축물의 물리적 규모 특성(**연면적**, **지상 층수**)을 독립 변수로 활용하여 에너지 소비량 예측 모델을 학습합니다.")

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
    st.metric("학습/테스트 데이터 비율", "80% : 20%", f"테스트 {len(X_test):,}건")
with m2:
    st.metric("🎯 결정계수 ($R^2$ Score)", f"{r2:.4f}", help="모델이 데이터의 분산을 설명하는 비율")
with m3:
    st.metric("📉 평균 제곱 오차 (MSE)", f"{mse:,.2f}", help="예측값과 실제값 차이의 제곱 평균")

st.markdown("---")

# ==========================================
# 소질문 5 & 6: 실시간 시뮬레이터 및 게이지 차트
# ==========================================
st.header("🎛️ 3. 소질문 5 & 6: 건축물 설계 조건별 실시간 예측 시뮬레이터")
st.write("슬라이더를 움직여 건물의 물리적 규모를 조절하면, AI 모델이 연산한 예상 에너지 소비량이 우측 계기판(게이지)에 실시간으로 반영됩니다.")

sim_col1, sim_col2 = st.columns([1, 1.2])

with sim_col1:
    st.subheader("🛠️ 설계 파라미터 입력")
    st.markdown("아래 슬라이더를 좌우로 움직여 수치를 변경해 보세요.")
    
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
    
    # 예측 실행
    input_df = pd.DataFrame([[v_area, v_floor]], columns=features)
    pred_val = max(0.0, float(model.predict(input_df)[0])) # 음수 방지
    avg_val = float(df[target_col].mean())
    diff_val = pred_val - avg_val
    
    st.markdown("---")
    st.metric(
        label="전체 공공건축물 평균 대비 예상 소비량 차이",
        value=f"{pred_val:.2f} kWh/㎡",
        delta=f"{diff_val:+.2f} kWh/㎡ (전체 평균: {avg_val:.1f})",
        delta_color="inverse" # 소비량이 적을수록 초록색(긍정적) 표시
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
                {'range': [0, 250], 'color': "#d4edda"},   # 우수 (연녹색)
                {'range': [250, 500], 'color': "#fff3cd"}, # 보통 (연노랑)
                {'range': [500, 1000], 'color': "#f8d7da"} # 과소비 주의 (연분홍)
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

# 건축공학적 해석 가이드 자동 출력
if pred_val < avg_val:
    st.success(f"✅ **에너지 효율 우수 설계안:** 입력하신 조건(연면적 `{v_area:,.0f}㎡`, 지상 `{v_floor:.0f}층`)은 국내 공공건축물 평균(`{avg_val:.1f} kWh/㎡`)보다 단위면적당 에너지 소비량이 낮아 친환경·고효율 건축 기준에 유리합니다.")
else:
    st.warning(f"⚠️ **에너지 절감 보완 필요:** 입력하신 조건(연면적 `{v_area:,.0f}㎡`, 지상 `{v_floor:.0f}층`)은 국내 공공건축물 평균(`{avg_val:.1f} kWh/㎡`)보다 소비량이 높게 예측됩니다. 단열 성능 강화나 신재생에너지 설비 도입이 권장됩니다.")
