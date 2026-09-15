import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="Turbofan RUL Prediction",
    page_icon="✈️",
    layout="wide"
)

# =========================================================
# 2. CSS 디자인
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f6f8;
}

.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 25px;
}

.dashboard-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #1f2937;
    margin-top: 10px;
    margin-bottom: 15px;
}

.status-normal {
    background-color: #dcfce7;
    color: #166534;
    padding: 15px;
    border-radius: 10px;
    font-weight: 600;
    text-align: center;
}

.status-warning {
    background-color: #fef3c7;
    color: #92400e;
    padding: 15px;
    border-radius: 10px;
    font-weight: 600;
    text-align: center;
}

.status-danger {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 15px;
    border-radius: 10px;
    font-weight: 600;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. 모델 및 데이터 불러오기
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

# =========================================================
# 4. 센서 목록
# =========================================================

all_sensors = [f"sensor_{i}" for i in range(1, 22)]

# =========================================================
# 5. 데이터 확인
# =========================================================

required_columns = ["engine_id", "cycle"] + all_sensors

missing_columns = [
    col for col in required_columns
    if col not in test_data.columns
]

if missing_columns:
    st.error(
        "새 데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()

# =========================================================
# 6. 사이드바
# =========================================================

st.sidebar.title("⚙️ Dashboard Settings")

engine_ids = sorted(
    test_data["engine_id"].dropna().unique()
)

if len(engine_ids) == 0:
    st.error("데이터에 engine_id가 없습니다.")
    st.stop()

target_engine = st.sidebar.selectbox(
    "엔진 선택",
    engine_ids
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset")
st.sidebar.write("NASA C-MAPSS FD001")

st.sidebar.markdown("### Model")
st.sidebar.write("Random Forest")

st.sidebar.markdown("### XAI")
st.sidebar.write("SHAP")

# =========================================================
# 7. 선택한 엔진 데이터
# =========================================================

engine_data = test_data[
    test_data["engine_id"] == target_engine
].copy()

engine_data = engine_data.sort_values("cycle")

if engine_data.empty:
    st.error("선택한 엔진의 데이터가 없습니다.")
    st.stop()

# 가장 최근 cycle의 데이터
current_data = engine_data.iloc[-1:]

current_cycle = current_data["cycle"].iloc[0]

# =========================================================
# 8. RUL 예측
# =========================================================

X_engine = current_data[all_sensors].copy()

X_engine_scaled = scaler.transform(X_engine)

predicted_rul = float(
    rf_model.predict(X_engine_scaled)[0]
)

# 음수 RUL 방지
predicted_rul = max(0, predicted_rul)

# =========================================================
# 9. 상태 판단
# =========================================================

if predicted_rul >= 50:
    status = "NORMAL"
    status_class = "status-normal"
    status_text = "정상 상태"
elif predicted_rul >= 20:
    status = "CAUTION"
    status_class = "status-warning"
    status_text = "주의 필요"
else:
    status = "ATTENTION"
    status_class = "status-danger"
    status_text = "점검 필요"

# =========================================================
# 10. 제목
# =========================================================

st.markdown(
    '<div class="main-title">✈️ Turbofan Engine RUL Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'NASA C-MAPSS 기반 터보팬 엔진 잔여수명 예측 및 XAI 분석'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# 11. 상태
# =========================================================

st.markdown(
    f'<div class="{status_class}">'
    f'현재 상태 : {status_text}'
    f'</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================================
# 12. 주요 지표
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🔧 현재 엔진",
        f"Engine {int(target_engine)}"
    )

with col2:
    st.metric(
        "🔄 Current Cycle",
        f"{int(current_cycle)}"
    )

with col3:
    st.metric(
        "⏳ Predicted RUL",
        f"{predicted_rul:.2f} cycles"
    )

with col4:
    st.metric(
        "🌲 Model",
        "Random Forest"
    )

# =========================================================
# 13. 엔진 운용 데이터 그래프
# =========================================================

st.markdown(
    '<div class="section-title">📈 Engine RUL Monitoring</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-card">',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# 새로운 데이터가 들어오면 이 그래프도 자동으로 변경
# ---------------------------------------------------------

plot_data = engine_data.copy()

# 실제 RUL이 존재하는 경우
if "RUL_capped" in plot_data.columns:

    st.line_chart(
        plot_data.set_index("cycle")["RUL_capped"],
        height=300
    )

elif "RUL" in plot_data.columns:

    st.line_chart(
        plot_data.set_index("cycle")["RUL"],
        height=300
    )

else:

    st.line_chart(
        plot_data.set_index("cycle")[all_sensors[:3]],
        height=300
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 14. SHAP XAI 분석
# =========================================================

st.markdown(
    '<div class="section-title">🔍 XAI Analysis — SHAP</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-card">',
    unsafe_allow_html=True
)

try:

    # -----------------------------------------------------
    # SHAP TreeExplainer
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(rf_model)

    shap_exp = explainer(X_engine_scaled)

    shap_values = np.asarray(
        shap_exp.values
    ).flatten()

    # -----------------------------------------------------
    # SHAP 데이터 정리
    # -----------------------------------------------------

    shap_df = pd.DataFrame({
        "Sensor": all_sensors,
        "SHAP Value": shap_values,
        "Sensor Value": X_engine.iloc[0].values
    })

    shap_df["Absolute Impact"] = np.abs(
        shap_df["SHAP Value"]
    )

    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    # -----------------------------------------------------
    # SHAP Waterfall
    # -----------------------------------------------------

    st.markdown("#### 🔎 개별 엔진 SHAP 분석")

    fig = plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        shap_exp[0],
        show=False,
        max_display=10
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close()

    # -----------------------------------------------------
    # 주요 영향 센서
    # -----------------------------------------------------

    st.markdown("#### 📊 주요 영향 센서")

    top_shap = shap_df.head(8).copy()

    top_shap["Direction"] = np.where(
        top_shap["SHAP Value"] > 0,
        "RUL ↑",
        "RUL ↓"
    )

    display_df = top_shap[
        [
            "Sensor",
            "Sensor Value",
            "SHAP Value",
            "Direction"
        ]
    ].copy()

    display_df["Sensor Value"] = display_df[
        "Sensor Value"
    ].round(3)

    display_df["SHAP Value"] = display_df[
        "SHAP Value"
    ].round(3)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

except Exception as e:

    st.warning(
        "SHAP 분석을 표시하는 과정에서 문제가 발생했습니다."
    )

    st.code(str(e))

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 15. 센서 모니터링
# =========================================================

st.markdown(
    '<div class="section-title">📡 Sensor Monitoring</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(
    ["현재 센서값", "SHAP 영향도"]
)

# =========================================================
# Tab 1 : 센서값
# =========================================================

with tab1:

    sensor_display = pd.DataFrame({
        "Sensor": all_sensors,
        "Current Value": X_engine.iloc[0].values
    })

    sensor_display["Current Value"] = sensor_display[
        "Current Value"
    ].round(4)

    st.dataframe(
        sensor_display,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# Tab 2 : SHAP 영향도
# =========================================================

with tab2:

    try:

        shap_chart = shap_df.head(10).copy()

        shap_chart = shap_chart[
            [
                "Sensor",
                "SHAP Value"
            ]
        ]

        shap_chart = shap_chart.set_index(
            "Sensor"
        )

        st.bar_chart(
            shap_chart,
            height=400
        )

    except Exception:

        st.warning(
            "SHAP 영향도 그래프를 표시할 수 없습니다."
        )

# =========================================================
# 16. 엔진 상세 데이터
# =========================================================

with st.expander("🔎 엔진 상세 데이터 보기"):

    st.dataframe(
        engine_data,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# 17. 안내
# =========================================================

st.markdown("---")

st.caption(
    "※ 본 시스템의 상태 기준은 연구 및 시각화를 위한 임의 기준이며 "
    "실제 항공기 정비 판단 기준이 아닙니다."
)

st.caption(
    "※ 새로운 데이터는 기존 모델과 동일한 sensor_1~sensor_21 "
    "구조 및 측정 단위를 사용하는 것을 전제로 합니다."
)
