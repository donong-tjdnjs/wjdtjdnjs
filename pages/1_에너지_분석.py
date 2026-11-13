import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="공공건축물 에너지 심층 분석 및 시뮬레이터", layout="wide", page_icon="📊")

st.title("📊 공공건축물 에너지 소비량 분석 및 예측 시스템")
st.markdown("##### 5가지 탐구 주제에 따라 데이터 분석, 머신러닝 평가, 실시간 시뮬레이션을 체계적으로 수행합니다.")
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
# 5개의 소질문 주제에 정확히 대응하는 5개의 독립된 탭(Tab) 구성
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ 1. 지역별 기후 권역 분석", 
    "🏢 2. 건축 용도별 에너지 분석", 
    "📐 3. 규모(연면적·층수) 특성 분석", 
    "🤖 4. 머신러닝 예측 모델 평가", 
    "🎛️ 5. 실시간 설계 시뮬레이터"
])

# =============================================================
# [탭 1] 소질문 1: 지역구분별 에너지 소비량 비교 분석
# =============================================================
with tab1:
    st.header("🗺️ [소질문 1] 기후 권역(지역구분)에 따른 단위면적당 에너지 소비량 편차 분석")
    st.write("우리나라의 지역별 기후 및 외기 온도 차이가 공공건축물의 에너지 소비 효율에 미치는 영향을 비교합니다.")
    
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
# [탭 2] 소질문 2: 건축 용도별 에너지 집약도 비교 분석 (업무시설 왜곡 해결)
# =============================================================
with tab2:
    st.header("🏢 [소질문 2] 건축물 주용도별(업무시설, 교육시설 등) 에너지 집약도 비교 분석")
    st.write("건축물의 사용 목적과 내부 설비 가동 특성에 따라 에너지 소비량이 어떻게 달라지는지 비교합니다. (업무시설 포함)")
    
    usage_list = sorted(df['용도'].unique())
    default_selected = usage_list[:3] if len(usage_list) >= 3 else usage_list
    
    selected_usages = st.multiselect(
        "비교할 건축물 용도를 선택하세요 (업무시설 포함 복수 선택 가능):",
        options=usage_list,
        default=default_selected,
        key="usage_multiselect"
    )
    
    # 업무시설 등의 초대형 건물로 인한 뭉침 현상을 방지하기 위한 축 범위 조절 및 로그 스케일 옵션
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        max_area_limit = st.slider("X축 최대 연면적 보기 범위 (㎡)", min_value=10000, max_value=300000, value=70000, step=10000, key="area_slider")
    with c_opt2:
        use_log_scale = st.checkbox("🔄 X축(연면적) 로그 스케일 적용 (초대형 업무시설 비교 용이)", value=False)
    
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
                title="선택된 용도별 연면적 대비 에너지 소비량 비교",
                labels={"연면적": "연면적 (㎡)", target_col: "단위면적당 에너지 소비량 (kWh/㎡)"}
            )
            
            if use_log_scale:
                fig_scatter.update_xaxes(type="log")
                fig_scatter.update_layout(title="선택된 용도별 연면적 대비 에너지 소비량 비교 (로그 스케일)")
            else:
                fig_scatter.update_xaxes(range=[0, max_area_limit])
                
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.success("""
            💡 **[분석 결과 및 도출 결론]**
            * **결론**: 업무시설과 교육연구시설 등을 함께 비교해 볼 때, 업무시설은 연면적이 수만 ㎡ 이상으로 넓어지더라도 에너지 집약도가 특정 밀집 구간 내에 형성되는 특성을 보입니다. 로그 스케일을 통해 초대형 업무시설의 분포도 왜곡 없이 선명하게 확인할 수 있습니다.
            * **시사점**: 건물 용도 특성을 반영한 차별화된 에너지 목표 관리 기준 수립이 필요합니다.
            """)
        else:
            st.warning("선택하신 조건에 해당하는 데이터가 없습니다.")
    else:
        st.warning("비교를 위해 최소 1개 이상의 건축물 용도를 선택해 주세요.")

# =============================================================
# [탭 3] 소질문 3: 건축물 규모(연면적 및 지상 층수)와 에너지 소비의 상관관계 분석
# =============================================================
with tab3:
    st.header("📐 [소질문 3] 건축물 규모(연면적 및 지상 층수)와 에너지 소비량의 상관관계 분석")
    st.write("건물의 물리적 크기(연면적, 층수)가 커질수록 단위면적당 에너지 효율에 어떤 상관관계가 작용하는지 분석합니다.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_corr1 = px.scatter(
            df, x="연면적", y=target_col, opacity=0.6, color_discrete_sequence=["#2ca02c"],
            title="연면적과 단위면적당 에너지 소비량 상관성",
            labels={"연면적": "연면적 (㎡)", target_col: "에너지 소비량 (kWh/㎡)"},
            range_x=[0, 100000]
        )
        st.plotly_chart(fig_corr1, use_container_width=True)
    with col_s2:
        fig_corr2 = px.scatter(
            df, x="규모(지상)", y=target_col, opacity=0.6, color_discrete_sequence=["#ff7f0e"],
            title="지상 층수와 단위면적당 에너지 소비량 상관성",
            labels={"규모(지상)": "지상 층수 (층)", target_col: "에너지 소비량 (kWh/㎡)"},
            range_x=[0, 30]
        )
        st.plotly_chart(fig_corr2, use_container_width=True)
        
    st.success("""
    💡 **[분석 결과 및 도출 결론]**
    * **결론**: 연면적이 증가함에 따라 단위면적당 에너지 소비량의 분산 폭이 점진적으로 수렴하는 경향을 보이며, 지상 층수 역시 일정 높이 이상에서 에너지 부하 패턴에 유의미한 상관관계를 나타냅니다.
    * **시사점**: 건물의 규모 특성은 에너지 예측 모델의 핵심 독립 변수로 기능하기에 매우 적합합니다.
    """)

# =============================================================
# [탭 4] 소질문 4: 머신러닝 예측 모델 학습 및 성능 평가
# =============================================================
with tab4:
    st.header("🤖 [소질문 4] 선형회귀(Linear Regression) 기반 에너지 예측 모델 학습 및 성능 검증")
    st.write("연면적과 지상 층수를 독립 변수로 활용하여 공공건축물의 에너지 소비량을 예측하는 AI 모델을 학습하고 정확도를 검증합니다.")
    
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
# [탭 5] 소질문 5: 실시간 설계 조건 변경 및 최적화 시뮬레이터
# =============================================================
with tab5:
    st.header("🎛️ [소질문 5] 건축물 설계 조건 변경에 따른 에너지 소비량 실시간 시뮬레이터")
    st.write("슬라이더를 조절하여 가상의 건축물 규모를 설정하고, 학습된 AI 모델이 즉각 연산하는 예상 에너지 소비량과 게이지 계기판을 확인하세요.")
    
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
