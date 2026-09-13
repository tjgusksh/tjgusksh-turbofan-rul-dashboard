import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="Turbofan RUL Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS 디자인
# =========================================================
st.markdown("""
<style>

    /* 전체 배경 */
    .stApp {
        background-color: #f5f7fb;
    }

    /* 상단 제목 */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #687386;
        margin-bottom: 30px;
    }

    /* 카드 */
    .dashboard-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e5e9f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    /* 상태 박스 */
    .status-normal {
        background: #e9f7ef;
        border-left: 6px solid #27ae60;
        padding: 15px 20px;
        border-radius: 10px;
        color: #176b3a;
        font-weight: 700;
    }

    .status-warning {
        background: #fff6df;
        border-left: 6px solid #f2a900;
        padding: 15px 20px;
        border-radius: 10px;
        color: #805b00;
        font-weight: 700;
    }

    .status-danger {
        background: #fdecec;
        border-left: 6px solid #e74c3c;
        padding: 15px 20px;
        border-radius: 10px;
        color: #8a2117;
        font-weight: 700;
    }

    /* 섹션 제목 */
    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #172033;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background-color: #172033;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# 모델 / 데이터 불러오기
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("rf_model_compressed.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_data
def load_data():
    return pd.read_csv("test_data.csv")


rf_model, scaler = load_model()
test_data = load_data()

all_sensors = [f"sensor_{i}" for i in range(1, 22)]


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:

    st.markdown("## ✈️ TURBOFAN")
    st.markdown("### RUL MONITORING")

    st.divider()

    st.markdown("### ⚙️ Engine Selection")

    engine_ids = sorted(test_data["engine_id"].unique())

    target_engine = st.selectbox(
        "분석할 엔진",
        engine_ids
    )

    st.divider()

    st.markdown("### 📌 System Information")

    st.write("**Dataset**")
    st.write("NASA C-MAPSS FD001")

    st.write("**Model**")
    st.write("Random Forest")

    st.write("**XAI Method**")
    st.write("SHAP")

    st.divider()

    st.caption("Predictive Maintenance Dashboard")


# =========================================================
# 선택 엔진 데이터
# =========================================================
engine_data = test_data[
    test_data["engine_id"] == target_engine
].iloc[-1:]

X_engine = engine_data[all_sensors]

X_engine_scaled = scaler.transform(X_engine)

predicted_rul = rf_model.predict(X_engine_scaled)[0]

current_cycle = int(engine_data["cycle"].iloc[0])


# =========================================================
# 상태 판단
# =========================================================
if predicted_rul >= 50:
    status_text = "● NORMAL"
    status_class = "status-normal"
    status_description = "현재 예측 RUL이 비교적 충분한 상태입니다."

elif predicted_rul >= 20:
    status_text = "● CAUTION"
    status_class = "status-warning"
    status_description = "잔여수명이 감소하고 있어 상태 확인이 필요합니다."

else:
    status_text = "● ATTENTION"
    status_class = "status-danger"
    status_description = "예측 잔여수명이 낮아 점검 우선순위를 높일 필요가 있습니다."


# =========================================================
# 메인 헤더
# =========================================================
st.markdown(
    '<div class="main-title">✈️ Turbofan RUL Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'NASA C-MAPSS 기반 터보팬 엔진 잔여수명 예측 및 XAI 분석'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 상태 표시
# =========================================================
st.markdown(
    f'<div class="{status_class}">{status_text} &nbsp; | &nbsp; {status_description}</div>',
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# 핵심 지표
# =========================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🔧 Engine",
        f"#{target_engine}"
    )

with col2:
    st.metric(
        "⏱ Current Cycle",
        f"{current_cycle}"
    )

with col3:
    st.metric(
        "📉 Predicted RUL",
        f"{predicted_rul:.1f}",
        "cycles"
    )

with col4:
    st.metric(
        "🤖 Model",
        "Random Forest"
    )


# =========================================================
# SHAP 분석
# =========================================================
st.markdown(
    '<div class="section-title">🔍 XAI Analysis</div>',
    unsafe_allow_html=True
)

explainer = shap.TreeExplainer(rf_model)
shap_exp = explainer(X_engine_scaled)


# SHAP 값 정리
shap_values = shap_exp.values[0]

shap_df = pd.DataFrame({
    "Sensor": all_sensors,
    "SHAP": shap_values,
    "Sensor Value": X_engine.iloc[0].values
})

shap_df["Abs_SHAP"] = np.abs(shap_df["SHAP"])

shap_df = shap_df.sort_values(
    "Abs_SHAP",
    ascending=False
)


# =========================================================
# SHAP + Top Sensors
# =========================================================
left, right = st.columns([1.7, 1])

with left:

    st.markdown(
        '<div class="dashboard-card">',
        unsafe_allow_html=True
    )

    st.markdown("#### SHAP Waterfall")

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.plots.waterfall(
        shap_exp[0],
        show=False
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


with right:

    st.markdown(
        '<div class="dashboard-card">',
        unsafe_allow_html=True
    )

    st.markdown("#### 📊 주요 영향 센서")

    top_sensors = shap_df.head(8).copy()

    top_sensors["Direction"] = top_sensors["SHAP"].apply(
        lambda x: "RUL ↑" if x > 0 else "RUL ↓"
    )

    display_df = top_sensors[
        ["Sensor", "Sensor Value", "SHAP", "Direction"]
    ].copy()

    display_df["Sensor Value"] = display_df["Sensor Value"].round(3)
    display_df["SHAP"] = display_df["SHAP"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 센서 데이터
# =========================================================
st.markdown(
    '<div class="section-title">📡 Sensor Monitoring</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs([
    "📋 현재 센서값",
    "📊 SHAP 영향도"
])


with tab1:

    sensor_table = pd.DataFrame({
        "Sensor": all_sensors,
        "Current Value": X_engine.iloc[0].values
    })

    sensor_table["Current Value"] = sensor_table[
        "Current Value"
    ].round(4)

    st.dataframe(
        sensor_table,
        use_container_width=True,
        hide_index=True
    )


with tab2:

    chart_df = shap_df.head(10).sort_values(
        "Abs_SHAP",
        ascending=True
    )

    st.bar_chart(
        chart_df.set_index("Sensor")["SHAP"]
    )


# =========================================================
# 엔진 원본 정보
# =========================================================
with st.expander("🔎 엔진 상세 데이터 보기"):

    st.dataframe(
        engine_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 하단 설명
# =========================================================
st.divider()

st.caption(
    "※ RUL(Residual Useful Life)은 엔진의 잔여수명을 의미합니다. "
    "SHAP 값은 각 센서가 해당 RUL 예측에 미친 영향을 나타냅니다."
)

st.caption(
    "NASA C-MAPSS FD001 데이터셋과 Random Forest 모델을 기반으로 구현한 "
    "예지정비 연구용 대시보드입니다."
)
