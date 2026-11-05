import cv2
import torch
import sys
from pathlib import Path

# Add YOLOX directory to Python path
YOLOX_DIR = Path(__file__).resolve().parent / "YOLOX"
sys.path.insert(0, str(YOLOX_DIR))

from yolox.exp import get_exp
from tools.demo import Predictor


# 1. Load YOLOX experiment configuration
exp = get_exp("exps/default/yolox_nano.py", None)

exp.test_conf = 0.25
exp.nmsthre = 0.45
exp.test_size = (416, 416)


# 2. Initialize model
model = exp.get_model()
model.eval()


# 3. Load pretrained weights
checkpoint = torch.load(
    "../models/yolox_nano.pth",
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


# 5. Read input image
image = cv2.imread("../test_images/people.jpg")

if image is None:
    raise FileNotFoundError("Input image not found!")


# 6. Run inference
outputs, img_info = predictor.inference(image)


# 7. Filter persons and draw bounding boxes
if outputs[0] is not None:

    detections = outputs[0].cpu().numpy()

    for detection in detections:

        x1, y1, x2, y2 = detection[:4]

        confidence = detection[4] * detection[5]
        class_id = int(detection[6])

        # COCO dataset: Person class ID = 0
        if class_id == 0:

            ratio = img_info["ratio"]

            x1, y1, x2, y2 = (
                int(x1 / ratio),
                int(y1 / ratio),
                int(x2 / ratio),
                int(y2 / ratio)
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Person: {confidence:.2f}"

            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


# 8. Save output
cv2.imwrite("../outputs/detected_people.jpg", image)

print("Person detection completed!")
print("Output saved successfully!")