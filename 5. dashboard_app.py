import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import WaferCNN
from rca_mapping import CLASS_NAMES, get_process_guide
import gpio_control


MODEL_PATH = "wafer_cnn.pth"
DATA_PATH = "wm811k_64_fault8.npz"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.csv")


@st.cache_resource
def load_model():
    device = torch.device("cpu")

    model = WaferCNN(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device


@st.cache_data
def load_data():
    data = np.load(DATA_PATH, allow_pickle=True)
    return data["X"], data["y"]


def predict(model, device, img):
    img_input = img.astype(np.float32) / 2.0

    input_tensor = torch.tensor(img_input, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(prob, 1)

    pred_label = CLASS_NAMES[pred_idx.item()]
    confidence_value = confidence.item() * 100
    guide = get_process_guide(pred_label)

    return pred_label, confidence_value, guide


def save_log(sample_idx, true_label, pred_label, confidence, guide, alarm_status):
    os.makedirs(LOG_DIR, exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "time",
                "sample_idx",
                "true_label",
                "pred_label",
                "confidence",
                "guide",
                "alarm_status"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sample_idx,
            true_label,
            pred_label,
            f"{confidence:.2f}",
            guide,
            alarm_status
        ])


def main():
    st.set_page_config(
        page_title="Wafer Defect AI System",
        layout="wide"
    )

    st.title("Jetson Orin Nano 기반 AI 웨이퍼 불량 분석 시스템")
    st.write(
        "CNN 모델을 이용해 웨이퍼 불량 패턴을 예측하고, "
        "공정 점검 가이드 및 알람 상태를 출력합니다."
    )

    try:
        model, device = load_model()
        X, y = load_data()
    except Exception as e:
        st.error("모델 또는 데이터 로딩 실패")
        st.exception(e)
        return

    st.success("시스템 준비 완료: 모델 및 데이터 로딩 성공")

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.metric("실행 장치", str(device))

    with col_info2:
        st.metric("전체 데이터 수", f"{len(X)}개")

    with col_info3:
        st.metric("분류 클래스", "8종 불량 패턴")

    st.info(
        "현재 모델은 정상 데이터를 제외한 8가지 불량 패턴 분류 모델입니다. "
        "추후 정상 none 클래스를 추가하면 정상/불량 판별 기능으로 확장할 수 있습니다."
    )

    st.sidebar.header("분석 설정")

    max_idx = len(X) - 1

    select_mode = st.sidebar.radio(
        "샘플 선택 방식",
        ["숫자 직접 입력", "슬라이더 선택"]
    )

    if select_mode == "숫자 직접 입력":
        sample_idx = st.sidebar.number_input(
            "샘플 번호 직접 입력",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1
        )
    else:
        sample_idx = st.sidebar.slider(
            "샘플 번호 슬라이더 선택",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1
        )

    sample_idx = int(sample_idx)

    st.sidebar.write(f"현재 선택된 샘플 번호: {sample_idx}")

    auto_alarm = st.sidebar.checkbox("AI 분석 후 자동 알람 출력", value=True)
    confidence_threshold = st.sidebar.slider("알람 신뢰도 기준(%)", 0, 100, 70)

    img = X[sample_idx]
    true_label = CLASS_NAMES[int(y[sample_idx])]

    if "result" not in st.session_state:
        st.session_state.result = None

    if st.sidebar.button("AI 분석 실행"):
        pred_label, confidence, guide = predict(model, device, img)

        alarm_status = "ALARM_OFF"

        if auto_alarm and confidence >= confidence_threshold:
            ok, msg = gpio_control.alarm_beep(1.0)

            if ok:
                alarm_status = "AUTO_ALARM_ON"
            else:
                alarm_status = "SOFTWARE_ALARM_ONLY"

        save_log(
            sample_idx=sample_idx,
            true_label=true_label,
            pred_label=pred_label,
            confidence=confidence,
            guide=guide,
            alarm_status=alarm_status
        )

        st.session_state.result = {
            "sample_idx": sample_idx,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "guide": guide,
            "alarm_status": alarm_status
        }

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("입력 웨이퍼 맵")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.subheader("AI 분석 결과")

        if st.session_state.result is None:
            st.warning("왼쪽 사이드바에서 샘플 번호를 선택한 뒤 [AI 분석 실행] 버튼을 누르세요.")
        else:
            result = st.session_state.result

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric("실제 라벨", result["true_label"])

            with r2:
                st.metric("예측 결과", result["pred_label"])

            with r3:
                st.metric("신뢰도", f"{result['confidence']:.2f}%")

            if result["true_label"] == result["pred_label"]:
                st.success("예측 결과가 실제 라벨과 일치합니다.")
            else:
                st.warning("예측 결과가 실제 라벨과 다릅니다. 추가 검토가 필요합니다.")

            st.subheader("공정 점검 가이드")
            st.info(result["guide"])

            st.subheader("알람 상태")

            if result["alarm_status"] == "AUTO_ALARM_ON":
                st.error("불량 패턴 예측 결과에 따라 LED/부저 알람이 자동 출력되었습니다.")
            elif result["alarm_status"] == "SOFTWARE_ALARM_ONLY":
                st.warning("PCB 미연결 또는 GPIO 사용 불가 상태입니다. 소프트웨어 알람 상태만 기록되었습니다.")
            else:
                st.info("알람이 출력되지 않았습니다.")

            if st.button("수동 알람 OFF"):
                ok, msg = gpio_control.alarm_off()

                if ok:
                    st.info("LED/부저 알람을 수동으로 OFF 했습니다.")
                else:
                    st.warning(msg)

    st.divider()
    st.subheader("분석 결과 저장 로그")

    if os.path.exists(LOG_FILE):
        try:
            log_df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
            st.dataframe(log_df.tail(10), use_container_width=True)
        except Exception as e:
            st.error("로그 파일을 읽는 중 오류가 발생했습니다.")
            st.exception(e)
    else:
        st.write("아직 저장된 분석 결과가 없습니다.")

    st.caption(
        "본 화면은 Jetson Orin Nano 기반 CNN 추론, Streamlit 대시보드, "
        "CSV 로그 저장, GPIO LED/부저 알람 연동 구조를 통합한 테스트 화면입니다."
    )


if __name__ == "__main__":
    main()
