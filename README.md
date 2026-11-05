# TraceVision – Person Detection POC

A Computer Vision Proof of Concept (POC) for detecting people in images and videos using **YOLOX-Nano** and **PyTorch**.

This module is the first building block of TraceVision, a larger computer vision project focused on detecting and identifying matching persons or objects across video footage.

---

## 1. Overview

The Person Detection module uses a pretrained YOLOX-Nano model to detect people in images and video files.

It processes input images or videos, identifies persons, draws bounding boxes around detected individuals, and generates annotated output files.

The POC supports execution on:

* CPU – Local machine (Windows).
* GPU – Google Colab with NVIDIA Tesla T4.

## 2. Features

* Person detection in static images.
* Person detection in video files.
* Bounding box visualization.
* Confidence score display.
* CPU-based inference.
* GPU-based inference.
* Processed video generation.
* Performance measurement using FPS and processing time.

## 3. Tech Stack

| Technology      | Purpose                    |
| --------------- | -------------------------- |
| Python          | Main programming language  |
| PyTorch         | Deep learning framework    |
| YOLOX-Nano      | Object detection model     |
| OpenCV          | Image and video processing |
| NumPy           | Numerical operations       |
| Google Colab    | GPU-based inference        |
| NVIDIA Tesla T4 | GPU acceleration           |

## 4. Model Information

**Model:** YOLOX-Nano

YOLOX-Nano is a lightweight object detection model suitable for resource-constrained environments.

For this POC, we use pretrained weights rather than training the model from scratch.

### Configuration

| Parameter            | Value      |
| -------------------- | ---------- |
| Model                | YOLOX-Nano |
| Input Size           | 416 × 416  |
| Confidence Threshold | 0.25       |
| NMS Threshold        | 0.45       |
| Person Class ID      | 0          |
| Inference Mode       | CPU / GPU  |

The model uses the COCO dataset's class mapping, where class ID `0` represents a person.

## 5. Project Structure

```text
person-detection/
│
├── models/
│   └── yolox_nano.pth
│
├── test_images/
│   └── people.jpg
│
├── test_videos/
│   └── 6387-191695740_medium.mp4
│
├── outputs/
│   ├── detected_people.jpg
│   └── detected_video.mp4
│
├── YOLOX/
│   └── YOLOX source repository
│
├── detect.py
├── detect_video.py
└── README.md
```

## 6. Installation and Setup

### Step 1: Clone YOLOX

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install dependencies

Install PyTorch and the required dependencies according to your hardware and compatible Python environment.

Install additional packages:

```bash
pip install opencv-python numpy loguru tqdm thop ninja tabulate psutil tensorboard pycocotools onnx
```

Install YOLOX:

```bash
cd YOLOX
pip install -e . --no-deps
```

### Step 4: Download pretrained weights

Download the YOLOX-Nano pretrained checkpoint and place it inside:

```text
models/yolox_nano.pth
```

## 7. Running the POC

### Image Detection

Run from the project root:

```bash
python detect.py
```

The script processes the input image and saves the annotated result inside the `outputs/` directory.

### Video Detection

Run:

```bash
python detect_video.py
```

The script processes the video frame by frame and generates an annotated output video.

## 8. GPU Inference – Google Colab

GPU inference was successfully tested using Google Colab with an NVIDIA Tesla T4 GPU.

The YOLOX-Nano model and pretrained checkpoint were loaded into the Colab environment, and video inference was performed using CUDA.

### GPU Configuration

| Parameter      | Value           |
| -------------- | --------------- |
| Platform       | Google Colab    |
| GPU            | NVIDIA Tesla T4 |
| Framework      | PyTorch         |
| CUDA Available | True            |
| Model          | YOLOX-Nano      |

## 9. Performance Results

The following results were recorded during GPU-based video inference.

### GPU Benchmark

| Metric          |          Result |
| --------------- | --------------: |
| Total Frames    |             341 |
| Processing Time |   13.17 seconds |
| Average FPS     |           25.88 |
| GPU             | NVIDIA Tesla T4 |

### Validation Results

| Test                                 | Result |
| ------------------------------------ | ------ |
| Person Detection                     | Passed |
| Bounding Box Accuracy (Visual Check) | Passed |
| Output Video Playback                | Passed |

**Note:** These are results from a single test video and should not be treated as universal performance benchmarks.

## 10. Current Limitations

This is an initial Proof of Concept. The current implementation has the following limitations:

* Detects persons independently in each frame.
* Does not assign persistent IDs to detected persons.
* Does not perform person tracking.
* Does not support image-to-video similarity search yet.
* Uses pretrained model weights without custom training.

## 11. Future Scope

The Person Detection module will serve as the foundation for upcoming TraceVision components.

Planned improvements include:

1. Person tracking using ByteTrack.
2. Persistent ID assignment across video frames.
3. Person feature extraction using appearance embeddings.
4. Vector database integration for similarity search.
5. Image-to-video person matching.
6. Object detection and matching.
7. Integration with the main TraceVision application.

## 12. Project Status

**Current Status: Person Detection POC – Completed**

* [x] YOLOX-Nano setup.
* [x] Image-based person detection.
* [x] Video-based person detection.
* [x] CPU inference.
* [x] GPU inference using Google Colab.
* [x] GPU performance benchmarking.
* [ ] Person tracking.
* [ ] Image-to-video similarity search.

---

**TraceVision**
Computer Vision | Person Detection | Video Analytics
