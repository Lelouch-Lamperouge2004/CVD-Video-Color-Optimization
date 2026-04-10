# Video Color Optimization for Color Blind People

A local Streamlit application for enhancing uploaded videos for users with color vision deficiency using a conditional GAN.

The project supports three deficiency types:

- Protan
- Deutan
- Tritan

The application flow is:

1. User takes a lightweight screening test
2. User selects or confirms the deficiency type
3. User uploads a video
4. The model enhances the video
5. User downloads the processed result

---

# Project Structure

```text
CVD-Video-Color-Optimization/
│
├── app/
├── checkpoints/
├── inference/
├── preprocessing/
├── simulate/
├── training/
├── utils/
├── .gitignore
├── README.md
└── requirements.txt
```

Important folders:

- `app/` → Streamlit application
- `checkpoints/` → pretrained model checkpoint goes here
- `inference/` → video processing pipeline
- `preprocessing/` → image resize utilities
- `simulate/` → target generation and color-deficiency simulation
- `training/` → model training scripts
- `utils/` → helper functions

---

# System Requirements

## Minimum
- Windows 10 or 11
- Python 3.10
- Git
- 8 GB RAM

## Recommended
- NVIDIA GPU
- Latest NVIDIA driver
- Microsoft Visual C++ Redistributable (x64)

PyTorch’s official install page recommends installing the build that matches the user’s system from the official selector, and current stable Windows support is for Python 3.10 or later.

Microsoft provides the current Visual C++ Redistributable downloads for Windows on its official page. 
---

# Before You Start

You need two things:

1. The GitHub project code
2. The pretrained checkpoint file from Kaggle

The checkpoint file must be placed manually after downloading.

---

# Step 1 - Clone the Repository

Open PowerShell or Command Prompt and run:

```powershell
git clone https://github.com/Lelouch-Lamperouge2004/CVD-Video-Color-Optimization.git
cd CVD-Video-Color-Optimization
```

# Step 2 - Create a Virtual Environment

Create the virtual environment **inside the project folder**:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

You should now see:

```text
(.venv)
```

---

# Step 3 - Install Project Dependencies

Install the Python packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

# Step 4 - Install PyTorch Separately

PyTorch should be installed separately from the rest of the requirements so users can choose the correct build for their system.

PyTorch provides official install commands for CPU and CUDA builds on its install page.

## Option A - NVIDIA GPU users

Open the official PyTorch installation page and select:

- OS: Windows
- Package: Pip
- Language: Python
- Compute Platform: CUDA

Then run the command shown there.

Example CUDA command:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Option B - CPU-only users

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

# Step 5 - Verify PyTorch Installation

Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:
- It should print the installed PyTorch version
- It should print `True` for GPU systems or `False` for CPU-only systems

---

# Step 6 - Download the Pretrained Model from Kaggle

Download the pretrained checkpoint file from the Kaggle dataset link provided in this repository.

https://www.kaggle.com/datasets/lelouchlamperouge69/cvd-pretrained-cgan-model-for-vc-optimization

After downloading, create this folder inside the project if it does not already exist:

```text
checkpoints/enhance_gan_conditional/
```

Place the model file here:

```text
checkpoints/enhance_gan_conditional/best.pth
```

Final expected path:

```text
CVD-Video-Color-Optimization/checkpoints/enhance_gan_conditional/latest.pth
```
Here both can be used latest.pth or best.pth but latest is recommended.
---

# Step 7 - Edit `app.py` Checkpoint Path

Open:

```text
app/app.py
```

Find the checkpoint path section.

Use a **relative path** like this:

```python
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT_DIR / "checkpoints" / "enhance_gan_conditional" / "best.pth"
PLATE_DIR = ROOT_DIR / "app" / "plates"
OUTPUT_DIR = ROOT_DIR / "outputs" / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

Do not use a personal absolute path like:

```python
CKPT_PATH = r"D:\CVD_GAN\checkpoints\enhance_gan_conditional\latest.pth"
```

because that will only work on your own machine.

---

# Step 8 - Run the Application

From the project root:

```powershell
streamlit run app/app.py
```

After a few seconds, Streamlit will print a local URL such as:

```text
http://localhost:8501
```

Open that URL in your browser.

---

# Step 9 - Use the Application

Inside the app:

1. Take the screening test
2. Select or confirm the CVD type
3. Upload an MP4 video
4. Click **Process Video**
5. Download the processed output

Output videos are saved in:

```text
outputs/videos/
```
or can be directly downloaded.
---

# Running the App on a Fresh Windows Machine - Full Command Order

Use these commands in this order:

```powershell
git clone https://github.com/YOUR_USERNAME/CVD-Video-Color-Optimization.git
cd CVD-Video-Color-Optimization
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
streamlit run app/app.py
```

For CPU-only systems, replace the PyTorch command with:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

# Optional: Run Video Processing Without Streamlit

You can run the inference pipeline directly from terminal.

Example:

```powershell
python -m inference.video_pipeline `
  --ckpt "checkpoints/enhance_gan_conditional/latest.pth" `
  --in_video "inputs/videos/test.mp4" `
  --out_video "outputs/videos/output.mp4" `
  --type protan
```

Replace `protan` with `deutan` or `tritan` when needed.

---

# Optional: Train the Model from Scratch

This section is for advanced users only.

## 1. Resize raw images

```powershell
python -m preprocessing.resize_to_256 `
  --src "dataset/coco_raw" `
  --dst "dataset/original_256" `
  --limit 15000
```

## 2. Generate enhancement targets

```powershell
python -m simulate.enhance_target `
  --in_dir "dataset/original_256" `
  --out_dir "dataset/enh_targets_256" `
  --limit 15000
```

## 3. Train the conditional GAN

```powershell
python -m training.train_enhance_gan `
  --x_dir "dataset/original_256" `
  --y_dir "dataset/enh_targets_256" `
  --out_dir "checkpoints/enhance_gan_conditional" `
  --limit 15000 `
  --epochs 40 `
  --batch 6 `
  --lr 0.0002 `
  --lambda_l1 100 `
  --lambda_id 10 `
  --lambda_gan 1 `
  --seed 123
```

---

# Troubleshooting

## 1. `requirements.txt` not found
Make sure you are inside the project root before running:

```powershell
pip install -r requirements.txt
```

## 2. PyTorch DLL error on Windows (`c10.dll`, `WinError 1114`)
Common fix steps:

1. Install Microsoft Visual C++ Redistributable x64
2. Recreate the virtual environment
3. Install PyTorch using the official command from the PyTorch website

PyTorch’s official installation guidance and Microsoft’s runtime download page are the correct references for this issue. 

## 3. Checkpoint not found
Make sure the model file exists exactly here:

```text
checkpoints/enhance_gan_conditional/best.pth
```

## 4. Streamlit runs but app cannot find files
Check that:
- `app/plates/` exists
- `checkpoints/enhance_gan_conditional/best.pth` exists
- `app.py` uses relative paths

## 5. GPU not detected
Run:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, reinstall PyTorch using the correct GPU command from the official PyTorch install page. 

---

# Notes

- Dataset files are not included in this repository
- Pretrained checkpoints are not stored in GitHub
- Checkpoints must be downloaded separately from Kaggle
- The project is intended to run locally

---

# Author
Aditya Dnyandeo Ingale , 
Final Year CSE Project ,
Video Color Optimization for Color Blind People
