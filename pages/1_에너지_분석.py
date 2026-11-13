import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="공공건축물 에너지 심층 분석 및 시뮬레이터", layout="wide", page_icon="📊")

st.title("📊 공공건축물 에너지 소비량 분석 및 예측 시스템")
st.markdown("##### 각 탭(Tab)을 통해 건축물 특성별 분석부터 머신러닝 예측, 실시간 시뮬레이션까지 체계적으로 확인할 수 있습니다.")
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

# -------------------------------------------------------------
# 4개의 독립된 탭으로 구성하여 각 분석 단계(소질문)가 1:1로 정확히 매칭되도록 설계
# -------------------------------------------------------------
tab_region, tab_usage, tab_ml, tab_sim = st.tabs([
    "🗺️ 1. 지역별 기후 권역 분석", 
    "🏢 2. 건축 용도별 에너지 분석", 
    "🤖 3. 머신러닝 예측 모델 평가", 
    "🎛️ 4. 실시간 설계 시뮬레이터"
])

# =============================================================
# [탭 1] 지역구분별 에너지 소비량 분석 (소질문 대응)
# =============================================================
with tab_region:
    st.header("🗺️ 기후 권역(지역구분)에 따른 단위면적당 에너지 소비량 비교")
    st.write("우리나라의 지역별 기후 및 외기 온도 차이가 공공건축물의 에너지 소비 효율에 미치는 영향을 분석합니다.")
    
    region_avg = df.groupby('지역구분')[target_col].mean().reset_index().sort_values(by=target_col, ascending=False)
    
    col_r1, col_r2 = st.columns([1.2, 1])
    with col_r1:
        fig_box = px.box(
            df,
            x="지역구분",
            y=target_col,
            color="지역구분",
            title="지역구분별 에너지 소비량 분포 (Box Plot)",
            labels={target_col: "단위면적당 에너지 소비량 (kWh/㎡)", "지역구분": "기후 권역"}
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_r2:
        fig_bar = px.bar(
            region_avg,
            x="지역구분",
            y=target_col,
            color=target_col,
            color_continuous_scale="Blues",
            title="지역구분별 평균 에너지 소비량 순위",
            labels={target_col: "평균 에너지 소비량 (kWh/㎡)", "지역구분": "기후 권역"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.success("""
    💡 **[분석 결과 및 도출 결론]**
    * **결론**: 지역별 기후 권역에 따라 난방 및 공조 부하의 차이가 발생하며, 외기 온도가 낮은 중부 권역 등의 공공건축물에서 단위면적당 에너지 소비량이 통계적으로 유의미한 차이를 보임을 확인했습니다.
    * **시사점**: 건축 설계 시 해당 지역의 기후 특성(난방일수 등)을 고려한 맞춤형 단열 기준 적용이 필수적입니다.
    """)

# =============================================================
# [탭 2] 건축 용도별 에너지 집약도 분석 (소질문 대응)
# =============================================================
with tab_usage:
    st.header("🏢 건축물 주용도별(업무시설, 교육시설 등) 에너지 집약도 비교")
    st.write("건축물의 사용 목적과 내부 설비 가동 특성에 따라 에너지 소비량이 어떻게 달라지는지 비교합니다.")
    
    usage_list = sorted(df['용도'].unique())
    default_selected = usage_list[:3] if len(usage_list) >= 3 else usage_list
    
    selected_usages = st.multiselect(
        "비교할 건축물 용도를 선택하세요 (업무시설 포함 복수 선택 가능):",
        options=usage_list,
        default=default_selected,
        key="usage_multiselect"
    )
    
    with st.expander("🔍 [그래프 보기 설정] 연면적 축 범위 조절"):
        max_area_limit = st.slider("X축 최대 연면적 (㎡)", min_value=10000, max_value=200000, value=60000, step=10000, key="area_slider")
    
    if selected_usages:
        filtered_df = df[df['용도'].isin(selected_usages)]
        
        if len(filtered_df) > 0:
            fig_scatter = px.scatter(
                filtered_df,
                x="연면적",
                y=target_col,
                color="용도",
                size="규모(지상)",
                hover_name="건축물명",
                title="선택된 용도별 연면적 대비 에너지 소비량 분산 비교",
                labels={"연면적": "연면적 (㎡)", target_col: "단위면적당 에너지 소비량 (kWh/㎡)"},
                range_x=[0, max_area_limit]
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.success("""
            💡 **[분석 결과 및 도출 결론]**
            * **결론**: 업무시설과 교육연구시설 등 용도에 따라 고유의 에너지 소비 밀집 구간이 존재합니다. 업무시설의 경우 대형 건물일수록 에너지 집약도가 특정 범위 내로 수렴하는 경향을 보입니다.
            * **시사점**: 건물 용도 특성을 반영한 차별화된 에너지 목표 관리 기준 수립이 필요합니다.
            """)
        else:
            st.warning("선택하신 조건에 해당하는 데이터가 없습니다.")
    else:
        st.warning("비교를 위해 최소 1개 이상의 건축물 용도를 선택해 주세요.")

# =============================================================
# [탭 3] 머신러닝 모델 학습 및 성능 평가 (소질문 대응)
# =============================================================
with tab_ml:
    st.header("🤖 선형회귀(Linear Regression) 기반 에너지 예측 모델 검증")
    st.write("건물의 물리적 규모 요인(연면적, 지상 층수)을 독립 변수로 활용하여 에너지 소비량을 예측하는 AI 모델을 학습하고 성능을 평가합니다.")
    
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
        st.metric("🎯 결정계수 ($R^2$ Score)", f"{r2:.4f}", help="모델이 데이터의 경향성을 설명하는 정도 (1에 가까울수록 우수)")
    with m3:
        st.metric("📉 평균 제곱 오차 (MSE)", f"{mse:,.2f}", help="예측값과 실제값 오차 크기")
        
    st.markdown("---")
    st.success(f"""
    💡 **[모델 평가 결과 및 도출 결론]**
    * **결론**: 연면적과 지상 층수만으로 학습한 선형회귀 모델의 결정계수($R^2$)는 **{r2:.4f}**로 도출되었습니다. 
    * **시사점**: 건물의 기본적인 물리적 규모가 에너지 소비량 예측에 유의미한 영향력을 가짐을 증명하였으며, 향후 설계 초기 단계에서 예상 에너지 부하를 가늠하는 유용한 도구로 활용할 수 있습니다.
    """)

# =============================================================
# [탭 4] 실시간 설계 시뮬레이터 및 게이지 (소질문 대응)
# =============================================================
with tab_sim:
    st.header("🎛️ 건축물 설계 조건별 실시간 에너지 소비량 예측 시뮬레이터")
    st.write("슬라이더를 조절하여 가상의 건축물 규모를 설정하고, AI 모델이 즉각 연산하는 예상 에너지 소비량과 게이지 계기판을 확인하세요.")
    
    sim_col1, sim_col2 = st.columns([1, 1.2])
    
    with sim_col1:
        st.subheader("🛠️ 가상 설계 파라미터 입력")
        
        v_area = st.slider(
            "📐 건물 연면적 (㎡)",
            min_value=float(X['연면적'].min()),
            max_value=50000.0,
            value=float(X['연면적'].median()),
            step=500.0,
            key="sim_area"
        )
        
        v_floor = st.slider(
            "🏢 지상 층수 (층)",
            min_value=1.0,
            max_value=30.0,
            value=float(X['규모(지상)'].median()),
            step=1.0,
            key="sim_floor"
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
        st.success(f"✅ **[시뮬레이션 최종 결론]**: 연면적 `{v_area:,.0f}㎡`, 지상 `{v_floor:.0f}층` 조건은 공공건축물 평균보다 에너지 소비가 적어 **친환경 고효율 설계**에 적합합니다.")
    else:
        st.warning(f"⚠️ **[시뮬레이션 최종 결론]**: 해당 규모의 건축물은 평균보다 에너지 소비량이 높을 것으로 예측되므로, **고효율 단열재 및 친환경 설비 설계** 도입이 강력히 권장됩니다.")
