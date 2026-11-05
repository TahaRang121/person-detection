import cv2
import torch
import sys
from pathlib import Path

# YOLOX directory path
YOLOX_DIR = Path(__file__).resolve().parent / "YOLOX"
sys.path.insert(0, str(YOLOX_DIR))

from yolox.exp import get_exp
from tools.demo import Predictor


# 1. Load YOLOX configuration
exp = get_exp("exps/default/yolox_nano.py", None)

exp.test_conf = 0.25
exp.nmsthre = 0.45
exp.test_size = (416, 416)


# 2. Initialize model
model = exp.get_model()
model.eval()


# 3. Load pretrained weights
checkpoint = torch.load(
    Path(__file__).resolve().parent / "models" / "yolox_nano.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model"])


# 4. Initialize predictor
predictor = Predictor(
    model,
    exp,
    device="cpu",
    fp16=False
)


# 5. Open video
video_path = Path(__file__).resolve().parent / "test_videos" / "6387-191695740_medium.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError("Video could not be opened!")


# 6. Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = Path(__file__).resolve().parent / "outputs" / "detected_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)


# 7. Process video frame by frame
while True:

    ret, frame = cap.read()

    if not ret:
        break

    outputs, img_info = predictor.inference(frame)

    if outputs[0] is not None:

        detections = outputs[0].cpu().numpy()

        for detection in detections:

            x1, y1, x2, y2 = detection[:4]

            confidence = detection[4] * detection[5]
            class_id = int(detection[6])

            if class_id == 0:

                ratio = img_info["ratio"]

                x1, y1, x2, y2 = (
                    int(x1 / ratio),
                    int(y1 / ratio),
                    int(x2 / ratio),
                    int(y2 / ratio)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person: {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    writer.write(frame)


# 8. Release resources
cap.release()
writer.release()

print("Video detection completed!")
print(f"Output saved at: {output_path}")