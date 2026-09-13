import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(
    page_title="Turbofan RUL Prediction",
    page_icon="✈️",
    layout="wide"
)

# 제목
st.title("✈️ Turbofan RUL Prediction")
st.write("NASA C-MAPSS 기반 터보팬 엔진 잔여수명(RUL) 예측 및 XAI 분석")

# 파일 불러오기
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

# 센서 목록
all_sensors = [f"sensor_{i}" for i in range(1, 22)]

# 엔진 선택
engine_ids = sorted(test_data["engine_id"].unique())

target_engine = st.selectbox(
    "분석할 엔진을 선택하세요",
    engine_ids
)

# 선택한 엔진의 마지막 사이클 데이터
engine_data = test_data[
    test_data["engine_id"] == target_engine
].iloc[-1:]

X_engine = engine_data[all_sensors]

# 스케일링
X_engine_scaled = scaler.transform(X_engine)

# RUL 예측
predicted_rul = rf_model.predict(X_engine_scaled)[0]

# 결과 표시
st.subheader("🔧 RUL 예측 결과")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "예측 잔여수명",
        f"{predicted_rul:.1f} cycles"
    )

with col2:
    st.metric(
        "현재 Cycle",
        f"{int(engine_data['cycle'].iloc[0])}"
    )

# SHAP 분석
st.subheader("🔍 XAI 분석 — SHAP")

explainer = shap.TreeExplainer(rf_model)
shap_exp = explainer(X_engine_scaled)

# SHAP Waterfall
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_exp[0], show=False)
st.pyplot(fig, clear_figure=True)

st.write(
    "SHAP 값이 양수이면 해당 센서가 모델의 RUL 예측을 증가시키는 방향으로, "
    "음수이면 감소시키는 방향으로 영향을 주었음을 의미합니다."
)

# 현재 센서값
st.subheader("📊 현재 센서 데이터")

sensor_table = pd.DataFrame({
    "Sensor": all_sensors,
    "Value": X_engine.iloc[0].values
})

st.dataframe(
    sensor_table,
    use_container_width=True
)
