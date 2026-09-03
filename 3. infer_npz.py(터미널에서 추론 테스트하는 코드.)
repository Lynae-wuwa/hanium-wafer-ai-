import argparse
import numpy as np
import torch

from model import WaferCNN
from rca_mapping import CLASS_NAMES, get_process_guide


MODEL_PATH = "wafer_cnn.pth"
DATA_PATH = "wm811k_64_fault8.npz"


def load_model(device):
    model = WaferCNN(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, device, img):
    img = img.astype(np.float32) / 2.0

    input_tensor = torch.tensor(img, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(prob, 1)

    pred_label = CLASS_NAMES[pred_idx.item()]
    confidence_value = confidence.item() * 100
    guide = get_process_guide(pred_label)

    return pred_label, confidence_value, guide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="분석할 웨이퍼 샘플 번호")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("사용 장치:", device)

    model = load_model(device)
    print("모델 로딩 성공")

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    print("데이터 로딩 성공")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    sample_idx = args.idx

    if sample_idx < 0 or sample_idx >= len(X):
        raise ValueError(f"sample_idx는 0부터 {len(X) - 1} 사이여야 합니다.")

    img = X[sample_idx]
    true_label = CLASS_NAMES[int(y[sample_idx])]

    pred_label, confidence, guide = predict(model, device, img)

    print("================================")
    print("샘플 번호:", sample_idx)
    print("실제 라벨:", true_label)
    print("예측 결과:", pred_label)
    print("신뢰도:", f"{confidence:.2f}%")
    print("점검 가이드:", guide)
    print("================================")


if __name__ == "__main__":
    main()
