# Pig Cough Recognition Using Ensemble Deep Learning Frameworks

This project implements an ensemble of deep learning frameworks for **automatic pig cough recognition** in complex piggery environments. It leverages **transfer learning from both ImageNet and AudioSet** to improve the robustness of acoustic-based cough detection.

## 🌟 Key Features

- 🎯 Dual transfer learning approach:
  - **ImageNet-based models** via PyTorch's pretrained image classification networks.
  - **AudioSet-based models** for audio signal representation and classification.
- ⚙️ Flexible classifier ensemble: easily combine and evaluate models of your choice.
- 📊 Includes tools for generating Cough Wavelet Packets (CWP) and Mel Spectrograms.
- 🐷 Specifically tailored to noisy, real-world piggery environments.

## 🧠 Model Components

| File | Description |
|------|-------------|
| `audioset_models.py` | Transfer learning models based on [AudioSet](https://research.google.com/audioset/). |
| `audioset_train-test.py` | Training and testing pipeline for AudioSet-based models. |
| `PaintCWPT.m` | MATLAB script for visualizing **CWP (Cough Wavelet Packet)** features. |
| `PaintMelSP.m` | MATLAB script for generating **Mel Spectrogram** features. |
| PyTorch built-in models | Used for transfer learning from **ImageNet** (e.g., ResNet, VGG, etc.). |

## 🛠️ Installation & Setup

### Python Environment
Make sure you have **Python ≥ 3.7** and install the required packages:

```bash
pip install torch==1.13.0 torchvision
# Add other required packages as needed (e.g., numpy, matplotlib, etc.)
