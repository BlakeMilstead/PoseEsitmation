import cv2
import torch
import joblib
import numpy as np
from xgboost import XGBClassifier
from PoseModYOLO import load_yolo_model, PoseDetector
from FrameDataset import SquatKneeFrameDataset
from mlp import SquatMLP
from BaseClassifier import PhaseLSTM
import torch.nn.functional as F

# Load pretrained models
def load_models():
    # MLP
    mlp_model = SquatMLP(input_dim=4)
    mlp_model.load_state_dict(torch.load("models/mlp_model.pth", map_location=torch.device('cpu')))
    mlp_model.eval()

    # Random Forest
    rf_model = joblib.load("models/best_rf_model.pkl")

    # XGBoost
    xgb_model = joblib.load("models/best_xgb_model.pkl")

    # LSTM
    lstm_model = PhaseLSTM(input_size=4)
    lstm_model.load_state_dict(torch.load("models/lstm_model.pth", map_location=torch.device('cpu')))
    lstm_model.eval()

    return mlp_model, rf_model, xgb_model, lstm_model

# Feature extraction from MediaPipe landmarks
def extract_features(lmList):
    joints = {}
    for lm in lmList:
        id, _, _, x, y, z, world_x, world_y, world_z = lm
        joints[id] = np.array([world_x, world_y, world_z])

    try:
        left_thigh = joints[23] - joints[25]
        left_shin = joints[27] - joints[25]
        right_thigh = joints[24] - joints[26]
        right_shin = joints[28] - joints[26]
        torso = (joints[11] + joints[12]) / 2 - (joints[23] + joints[24]) / 2

        def safe_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return np.clip(np.arccos(cos), 0, np.pi)

        left_knee = safe_angle(left_thigh, left_shin)
        right_knee = safe_angle(right_thigh, right_shin)
        left_hip = safe_angle(torso, left_thigh)
        torso_lean = np.arctan2(torso[1], abs(torso[2]))

        return np.array([left_knee, right_knee, left_hip, torso_lean])
    except:
        return None

# Temporal buffer for LSTM
from collections import deque
lstm_buffer = deque(maxlen=30)

# Main real-time prediction function
def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    yolo_net, output_layers, class_names = load_yolo_model()
    mlp_model, rf_model, xgb_model, lstm_model = load_models()

    scaler = joblib.load("models/scaler.pkl")
    print("Scaler loaded successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)

        height, width = frame.shape[:2]
        boxes = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_names[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])

        if boxes:
            frame_center = np.array([width // 2, height // 2])
            min_dist = float('inf')
            best_box = None

            for (x, y, w, h) in boxes:
                box_center = np.array([x + w // 2, y + h // 2])
                dist = np.linalg.norm(box_center - frame_center)
                if dist < min_dist:
                    min_dist = dist
                    best_box = (x, y, w, h)
                        
            if best_box is not None:
                x, y, w, h = best_box
                x, y = max(0, x), max(0, y)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = frame[y:y+h, x:x+w]
                pose_img = detector.findPose(cropped)
                lmList = detector.findPosition(cropped)

            if lmList:
                features = extract_features(lmList)
                if features is not None:
                    # Scale features
                    scaled_features = scaler.transform(features.reshape(1, -1))

                    # Temporal features for RF/XGB
                    temp_features = scaled_features.flatten()

                    # MLP Prediction
                    mlp_pred = torch.softmax(mlp_model(torch.tensor(scaled_features, dtype=torch.float32)), dim=1)
                    mlp_label = "UP" if mlp_pred.argmax() == 1 else "DOWN"

                    # Random Forest Prediction
                    rf_pred = rf_model.predict(temp_features.reshape(1, -1))[0]
                    rf_label = "UP" if rf_pred == 1 else "DOWN"

                    # XGBoost Prediction
                    xgb_pred = xgb_model.predict(temp_features.reshape(1, -1))[0]
                    xgb_label = "UP" if xgb_pred == 1 else "DOWN"

                    # LSTM Prediction
                    lstm_buffer.append(features)
                    if len(lstm_buffer) == 30:
                        lstm_input = torch.tensor(np.stack(lstm_buffer)).unsqueeze(0).float()
                        lstm_out = torch.softmax(lstm_model(lstm_input), dim=1)
                        lstm_label = "UP" if lstm_out.argmax() == 1 else "DOWN"
                    else:
                        lstm_label = "---"

                    y_offset = 30
                    colors = {
                        "MLP": (255, 0, 0),   # Blue
                        "RF": (0, 255, 0),    # Green
                        "XGB": (0, 0, 255),   # Red
                        "LSTM": (0, 255, 255) # Yellow
                    }

                    for model_name, pred_label in zip(["MLP", "RF", "XGB", "LSTM"], [mlp_label, rf_label, xgb_label, lstm_label]):
                        color = colors.get(model_name, (255, 255, 255))  # Default white if missing
                        cv2.putText(frame, f"{model_name}: {pred_label}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset += 30

        resized = cv2.resize(frame, (800, 600))
        cv2.imshow("Squat Prediction", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video("AllVids/Fort3.mov")  # <<< Replace with your video path!