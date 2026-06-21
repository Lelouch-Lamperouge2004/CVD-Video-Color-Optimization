# Video Color Optimization for Color Blind People using Conditional GAN

A deep learning-based video enhancement system that improves color distinguishability for people with Color Vision Deficiency (CVD) using a Conditional GAN (cGAN). The application provides a lightweight screening test, suggests an enhancement mode, processes uploaded videos frame-by-frame, and generates an optimized output video while preserving visual realism.

---

## Features

### Color Vision Deficiency Support

The system supports three major color vision deficiency types:

* Protan (Red deficiency)
* Deutan (Green deficiency)
* Tritan (Blue-Yellow deficiency)

---

### Dynamic Screening Test

* Synthetic Ishihara-style screening plates
* Red-Green and Blue-Yellow axis evaluation
* Automatic enhancement type suggestion
* One-click generation of new randomized screening tests
* CSV-based answer synchronization

---

### Video Enhancement Pipeline

* Upload MP4 videos
* Frame-by-frame enhancement using Conditional GAN
* Temporal smoothing to reduce flickering
* LAB color-space enhancement
* Adjustable enhancement strength
* Adjustable sharpening controls

---

### Result Visualization

* Side-by-side video comparison
* Original vs Enhanced preview
* Frame-level comparison

  * Beginning frame
  * Middle frame
  * Ending frame
* Download enhanced video

---

## System Architecture

```text
Screening Test
      ↓
Deficiency Estimation
      ↓
Type Selection
      ↓
Video Upload
      ↓
Conditional GAN
      ↓
Frame Enhancement
      ↓
Video Reconstruction
      ↓
Result Comparison
      ↓
Download Output
```

---

## Project Workflow

### Step 1 – Screening Test

The user answers synthetic Ishihara-style plates.

### Step 2 – Deficiency Estimation

The application analyzes mistakes on:

* Red-Green plates
* Blue-Yellow plates

and suggests an enhancement type.

### Step 3 – Type Selection

The user may:

* Accept the suggested type
* Manually select Protan, Deutan, or Tritan

### Step 4 – Video Upload

The user uploads an MP4 video.

### Step 5 – Video Processing

Each frame is:

* Extracted
* Resized
* Conditioned with the selected CVD type
* Passed through the trained Conditional GAN

### Step 6 – Enhancement

Additional processing includes:

* Temporal smoothing
* LAB color enhancement
* Edge sharpening

### Step 7 – Result Generation

Enhanced frames are reconstructed into a processed video.

### Step 8 – Comparison

The application displays:

* Original vs Enhanced Video
* Representative Frame Comparisons

### Step 9 – Download

The enhanced video can be downloaded locally.

---

# Technology Stack

## Frontend

### Streamlit

Used for:

* User Interface
* Screening Test
* Video Upload
* Progress Tracking
* Comparison Dashboard
* Download Functionality

---

## Backend

### Python

Used for:

* Application Logic
* Video Processing
* Inference Pipeline
* File Management

---

## Deep Learning

### PyTorch

Used for:

* Conditional GAN
* U-Net Generator
* PatchGAN Discriminator
* GPU Acceleration
* Checkpoint Loading

---

## Computer Vision

### OpenCV

Used for:

* Video Reading
* Frame Extraction
* Frame Resizing
* Video Writing

---

## Image Processing

### Pillow (PIL)

Used for:

* Synthetic Plate Generation
* Image Rendering
* Plate Visualization

---

## Numerical Computing

### NumPy

Used for:

* Matrix Operations
* Pixel Processing
* Image Masks

---

## Video Encoding

### imageio-ffmpeg

Used for:

* Browser-compatible MP4 generation
* Streamlit video playback support

---

# Project Structure

```text
CVD-Video-Color-Optimization/
│
├── app/
│   ├── app.py
│   ├── generate_synthetic_plates.py
│   └── plates/
│
├── checkpoints/
│
├── dataset/
│
├── inference/
│   └── video_pipeline.py
│
├── preprocessing/
│
├── simulate/
│
├── training/
│
├── utils/
│
├── outputs/
│
├── README.md
│
└── requirements.txt
```

---

# Folder Description

## app/

Contains:

* Streamlit application
* Screening test logic
* User interface

### Important Files

#### app.py

Main application entry point.

#### generate_synthetic_plates.py

Generates randomized screening plates and updates plate configuration.

---

## checkpoints/

Stores pretrained model checkpoints.

Example:

```text
checkpoints/
└── enhance_gan_conditional/
    └── latest.pth
```

---

## inference/

Contains the runtime video enhancement pipeline.

### video_pipeline.py

Responsible for:

* Loading the trained model
* Reading videos
* Processing frames
* Writing enhanced output

---

## preprocessing/

Dataset preprocessing utilities.

---

## simulate/

Generates enhancement targets used during training.

---

## training/

Contains:

* GAN architecture
* Training pipeline
* Loss functions

---

## utils/

Shared helper functions and checkpoint utilities.

---

# Model Architecture

## Generator

U-Net Generator

Advantages:

* Skip Connections
* Better Detail Preservation
* Stable Image Reconstruction

---

## Discriminator

PatchGAN Discriminator

Advantages:

* Better Texture Learning
* Improved Local Feature Detection
* Stable GAN Training

---

# Training Configuration

| Parameter     | Value         |
| ------------- | ------------- |
| Dataset Size  | 15,000 Images |
| Resolution    | 256×256       |
| Epochs        | 40            |
| Batch Size    | 6             |
| Learning Rate | 0.0002        |
| Optimizer     | Adam          |
| Generator     | U-Net         |
| Discriminator | PatchGAN      |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Lelouch-Lamperouge2004/CVD-Video-Color-Optimization.git

cd CVD-Video-Color-Optimization
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

```bash
.\.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install PyTorch

### CUDA Version

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU Version

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Install FFmpeg Wrapper

```bash
pip install imageio-ffmpeg
```

---

# Download Pretrained Model

Download the checkpoint from:

https://www.kaggle.com/datasets/lelouchlamperouge69/cvd-pretrained-cgan-model-for-vc-optimization

Place the file at:

```text
checkpoints/enhance_gan_conditional/latest.pth
```

---

# Running the Application

Generate screening plates:

```bash
python app/generate_synthetic_plates.py
```

Run Streamlit:

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

# Running Direct Inference

```bash
python -m inference.video_pipeline \
  --ckpt checkpoints/enhance_gan_conditional/latest.pth \
  --in_video input.mp4 \
  --out_video output.mp4 \
  --type protan
```

Supported types:

* protan
* deutan
* tritan

---

# Challenges Addressed

### No Real Training Dataset

No public dataset exists containing:

```text
Original Image
→
Color-Blind Optimized Image
```

Solution:

Synthetic enhancement targets were generated using LAB color-space transformations.

---

### Video Flickering

Challenge:

Frame-by-frame GAN inference can introduce flicker.

Solution:

Temporal smoothing was added.

---

### Soft Outputs

Challenge:

GAN outputs may appear slightly blurry.

Solution:

Sharpening controls were introduced.

---

### Browser Video Compatibility

Challenge:

Videos generated by OpenCV may not always play inside browsers.

Solution:

FFmpeg-based encoding support was added.

---

### Screening Test Memorization

Challenge:

Users could memorize fixed plate answers.

Solution:

Randomized plate generation with automatic configuration updates.

---

# Future Improvements

* Real-time video enhancement
* Mobile application deployment
* User-specific adaptive enhancement
* Clinical validation with CVD participants
* Cloud deployment

---

# Author

Aditya Dnyandeo Ingale

Final Year Computer Science Engineering Project

Video Color Optimization for Color Blind People using Conditional GAN
