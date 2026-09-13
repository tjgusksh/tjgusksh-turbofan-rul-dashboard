import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="Turbofan RUL Dashboard",
    page_icon="✈️",
    layout="wide"
)

# =========================
# 디자인
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #172033;
}

.sub-title {
    font-size: 17px;
    color: #687386;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e9f0;
    margin-bottom: 20px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #172033;
    margin-top: 20px;
    margin-bottom: 15px;
}

.normal {
    background-color: #e9f7ef;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #27ae60;
}

.caution {
    background-color: #fff6df;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #f2a900;
}

.attention {
    background-color: #fdecec;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #e74c3c;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 파일 불러오기
# =========================
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


# =========================
# 사이드바
# =========================
with st.sidebar:

    st.markdown("## ✈️ TURBOFAN")
    st.markdown("### RUL MONITORING")

    st.divider()

    engine_ids = sorted(
        test_data["engine_id"].unique()
    )

    target_engine = st.selectbox(
        "분석할 엔진 번호",
        engine_ids
    )

    st.divider()

    st.markdown("### SYSTEM")

    st.write("Dataset")
    st.write("NASA C-MAPSS FD001")

    st.write("Model")
    st.write("Random Forest")

    st.write("XAI")
    st.write("SHAP")


# =========================
# 선택된 엔진 데이터
# =========================
engine_data = test_data[
    test_data["engine_id"] == target_engine
].iloc[-1:]

X_engine = engine_data[all_sensors]

X_engine_scaled = scaler.transform(X_engine)


# =========================
# RUL 예측
# =========================
predicted_rul = float(
    rf_model.predict(X_engine_scaled)[0]
)

current_cycle = int(
    engine_data["cycle"].iloc[0]
)


# =========================
# 상태
# =========================
if predicted_rul >= 50:
    status = "NORMAL"
    status_class = "normal"
    description = "예측 잔여수명이 비교적 충분합니다."

elif predicted_rul >= 20:
    status = "CAUTION"
    status_class = "caution"
    description = "잔여수명이 감소하고 있어 상태 확인이 필요합니다."

else:
    status = "ATTENTION"
    status_class = "attention"
    description = "예측 잔여수명이 낮아 점검 우선순위를 높일 필요가 있습니다."


# =========================
# 헤더
# =========================
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


# =========================
# 상태
# =========================
st.markdown(
    f'<div class="{status_class}">'
    f'<b>● {status}</b>　{description}'
    f'</div>',
    unsafe_allow_html=True
)

st.write("")


# =========================
# 핵심 지표
# =========================
st.markdown(
    '<div class="section-title">📊 Engine Status</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🔧 현재 엔진",
        f"Engine {target_engine}"
    )

with col2:
    st.metric(
        "⏱ 현재 Cycle",
        f"{current_cycle}"
    )

with col3:
    st.metric(
        "📉 예측 RUL",
        f"{predicted_rul:.1f} cycles"
    )

with col4:
    st.metric(
        "🤖 예측 모델",
        "Random Forest"
    )


# =========================
# XAI
# =========================
st.markdown(
    '<div class="section-title">🔍 XAI Analysis</div>',
    unsafe_allow_html=True
)

try:

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(
        X_engine_scaled
    )

    # SHAP 결과 형태 대응
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    # 데이터프레임 생성
    shap_df = pd.DataFrame({
        "Sensor": all_sensors,
        "SHAP Value": shap_values,
        "Sensor Value": X_engine.iloc[0].values
    })

    shap_df["Absolute SHAP"] = np.abs(
        shap_df["SHAP Value"]
    )

    shap_df = shap_df.sort_values(
        "Absolute SHAP",
        ascending=False
    )

    # -------------------------
    # TOP 센서
    # -------------------------
    left, right = st.columns([1.4, 1])

    with left:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### SHAP 영향도")

        chart_data = shap_df.head(10).copy()

        chart_data = chart_data[
            ["Sensor", "SHAP Value"]
        ]

        chart_data = chart_data.set_index(
            "Sensor"
        )

        st.bar_chart(
            chart_data,
            use_container_width=True
        )

        st.caption(
            "양수: RUL을 증가시키는 방향"
        )

        st.caption(
            "음수: RUL을 감소시키는 방향"
        )

        st.markdown("</div>", unsafe_allow_html=True)


    with right:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### 🔥 주요 영향 센서 TOP 10")

        top10 = shap_df.head(10).copy()

        top10["방향"] = top10[
            "SHAP Value"
        ].apply(
            lambda x: "RUL ↑" if x > 0 else "RUL ↓"
        )

        top10["SHAP Value"] = top10[
            "SHAP Value"
        ].round(2)

        top10["Sensor Value"] = top10[
            "Sensor Value"
        ].round(3)

        st.dataframe(
            top10[
                [
                    "Sensor",
                    "Sensor Value",
                    "SHAP Value",
                    "방향"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


except Exception as e:

    st.error("SHAP 분석 중 오류가 발생했습니다.")

    st.code(str(e))


# =========================
# 센서 데이터
# =========================
st.markdown(
    '<div class="section-title">📡 Sensor Monitoring</div>',
    unsafe_allow_html=True
)

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


# =========================
# 상세 데이터
# =========================
with st.expander("🔎 선택된 엔진의 상세 데이터"):

    st.dataframe(
        engine_data,
        use_container_width=True,
        hide_index=True
    )


# =========================
# 설명
# =========================
st.divider()

st.caption(
    "RUL(Residual Useful Life)은 엔진의 잔여수명을 의미합니다."
)

st.caption(
    "SHAP 값은 각 센서가 개별 RUL 예측에 미친 영향을 나타냅니다."
)

st.caption(
    "※ NORMAL / CAUTION / ATTENTION 기준은 본 연구의 대시보드 시각화를 "
    "위해 설정한 기준이며 실제 항공기 정비 기준을 의미하지 않습니다."
)
