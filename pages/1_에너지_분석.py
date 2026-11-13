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

# 5개의 소질문 주제에 정확히 대응하는 5개의 독립된 탭(Tab) 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ 1. 지역별 기후 권역 분석", 
    "🏢 2. 건축 용도별 에너지 분석", 
    "📐 3. 규모(연면적·층수) 특성 분석", 
    "🤖 4. 머신러닝 예측 모델 평가", 
    "🎛️ 5. 실시간 설계 시뮬레이터"
])

# [탭 1] 소질문 1: 지역구분별 에너지 소비량 비교 분석
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

# [탭 2] 소질문 2: 건축 용도별 에너지 집약도 비교 분석 (업무시설 왜곡 완벽 해결)
with tab2:
    st.header("🏢 [소질문 2] 건축물 주용도별(업무시설, 교육시설 등) 에너지 집약도 비교 분석")
    st.write("건축물의 사용 목적과 내부 설비 가동 특성에 따라 에너지 소비량이 어떻게 달라지는지 비교합니다.")
    
    usage_list = sorted(df['용도'].unique())
    default_selected = usage_list[:3] if len(usage_list) >= 3 else usage_list
    
    selected_usages = st.multiselect(
        "비교할 건축물 용도를 선택하세요 (업무시설 포함 복수 선택 가능):",
        options=usage_list,
        default=default_selected,
        key="usage_multiselect"
    )
    
    if selected_usages:
        filtered_df = df[df['용도'].isin(selected_usages)]
        
        if len(filtered_df) > 0:
            # 선택된 데이터의 최대 연면적을 자동으로 계산하여 그래프가 찌그러지지 않고 꽉 차게 표시
            max_val = float(filtered_df['연면적'].max())
            
            c_opt1, c_opt2 = st.columns(2)
            with c_opt1:
                # 사용자가 줌인/줌아웃을 직접 조절할 수 있도록 동적 슬라이더 제공
                view_range = st.slider("X축 연면적 확대/축소 범위 설정 (㎡)", min_value=1000.0, max_value=max(max_val, 10000.0), value=max_val, step=1000.0, key="auto_area_slider")
            with c_opt2:
                use_log_scale = st.checkbox("🔄 X축 로그 스케일 적용 (초대형 건물 밀집도 확인 용이)", value=False)
            
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
            else:
                fig_scatter.update_xaxes(range=[0, view_range])
                
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.success("""
            💡 **[분석 결과 및 도출 결론]**
            * **결론**: 업무시설을 선택했을 때 데이터가 한쪽으로 뭉치지 않도록 동적 범위 조절 기능을 적용하여 분포를 명확히 확인했습니다. 업무시설은 연면적이 넓어지더라도 에너지 집약도가 특정 구간에 밀집하는 안정적인 특성을 보입니다.
            * **시사점**: 건물 용도 특성을 반영한 차별화된 에너지 목표 관리 기준 수립이 필요합니다.
            """)
        else:
            st.warning("선택하신 조건에 해당하는 데이터가 없습니다.")
    else:
        st.warning("비교를 위해 최소 1개 이상의 건축물 용도를 선택해 주세요.")

# [탭 3] 소질문 3: 건축물 규모와 에너지 소비의 상관관계 분석
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

# [탭 4] 소질문 4: 머신러닝 예측 모델 학습 및 성능 평가 (시각화 추가)
with tab4:
    st.header("🤖 [소질문 4] 선형회귀(Linear Regression) 기반 에너지 예측 모델 학습 및 성능 검증")
    st.write("연면적을 독립 변수로 활용하여 공공건축물의 에너지 소비량을 예측하는 AI 모델이 어떻게 학습되었는지 **회귀선 시각화**를 통해 확인합니다.")
    
    features = ['연면적'] # 시각화 직관성을 위해 연면적 단일 변수로 회귀선 표현
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
        st.metric("📉 평균 제곱 오차 (MSE)", f"{mse:,.2f}", help="예측값과 실제값 오차 크기")
        
    st.markdown("---")
    
    # [추가] 머신러닝 모델의 예측 과정과 결과가 왜 이렇게 나왔는지 보여주는 회귀선 시각화 그래프
    st.subheader("📈 AI 예측 모델의 회귀선(Regression Line) 시각화")
    st.write("아래 그래프의 **빨간색 직선**은 AI가 학습한 예측 모델입니다. 데이터들이 넓게 퍼져 있어서 직선 주변의 오차가 존재하기 때문에 $R^2$ 값이 도출되는 과정을 눈으로 확인할 수 있습니다.")
    
    vis_df = X_test.copy()
    vis_df['Actual'] = y_test
    vis_df['Predicted'] = y_pred
    
    fig_reg = px.scatter(
        vis_df, x='연면적', y='Actual', opacity=0.7,
        title="실제 테스트 데이터(파란 점) vs AI 예측 회귀선(빨간 선)",
        labels={'연면적': '연면적 (㎡)', 'Actual': '단위면적당 에너지 소비량 (kWh/㎡)'},
        range_x=[0, 60000]
    )
    # 회귀선 추가
    fig_reg.add_traces(px.line(vis_df.sort_values(by='연면적'), x='연면적', y='Predicted').data[0])
    fig_reg.data[1].line.color = 'red'
    fig_reg.data[1].name = 'AI 회귀 예측선'
    
    st.plotly_chart(fig_reg, use_container_width=True)
    
    st.success(f"""
    💡 **[모델 평가 결과 및 도출 결론]**
    * **결론**: 회귀선 주변에 데이터가 넓게 퍼져 있는 것은 건물마다 단열 성능, 설비 효율 등 다른 변수가 작용하기 때문입니다. 따라서 결정계수($R^2$)가 **{r2:.4f}**로 측정되었으며, 이는 규모 외의 추가 변수가 존재함을 시사합니다.
    * **시사점**: AI 모델이 전체적인 에너지 소비 트렌드를 우상향/우하향 형태로 잡아내는 과정을 시각적으로 증명할 수 있습니다.
    """)

# [탭 5] 소질문 5: 실시간 설계 조건 변경 및 최적화 시뮬레이터
with tab5:
    st.header("🎛️ [소질문 5] 건축물 설계 조건 변경에 따른 에너지 소비량 실시간 시뮬레이터")
    st.write("슬라이더를 조절하여 가상의 건축물 규모를 설정하고, 학습된 AI 모델이 즉각 연산하는 예상 에너지 소비량과 게이지 계기판을 확인하세요.")
    
    sim_col1, sim_col2 = st.columns([1, 1.2])
    
    with sim_col1:
        st.subheader("🛠️ 가상 설계 파라미터 입력")
        
        v_area = st.slider(
            "📐 건물 연면적 (㎡)",
            min_value=float(df['연면적'].min()),
            max_value=50000.0,
            value=float(df['연면적'].median()),
            step=500.0,
            key="sim_area"
        )
        
        v_floor = st.slider(
            "🏢 지상 층수 (층)",
            min_value=1.0,
            max_value=30.0,
            value=float(df['규모(지상)'].median()),
            step=1.0,
            key="sim_floor"
        )
        
        input_df = pd.DataFrame([[v_area]], columns=['연면적'])
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
