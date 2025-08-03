# Deep Audio Classification & Diagnostics on ESC-50 Dataset

## Project

This project is an end-to-end audio classification and diagnostics system built from the ground up. It ingests environmental sound recordings, preprocesses and augments them, trains a ResNet-style convolutional neural network on mel spectrogram representations, and exposes real-time inference via a robust API. The entire inference pipeline is deployed serverlessly using Modal’s cloud infrastructure, leveraging **NVIDIA A10G** GPUs for scalable, on-demand audio classification. Comprehensive observability (via TensorBoard) and interpretability tooling are integrated, and an interactive Tableau dashboard surfaces performance, confidence, distribution shift, and failure modes—turning model behavior (including real-world degradation) into actionable insight.

### Project Architecture

1. **Data Ingestion & Preprocessing**  
   Raw audio clips (EXC50/ESC-50 style environmental sounds) are loaded, normalized, and converted into mel spectrograms—effectively turning audio into image-like representations suitable for deep CNNs. Waveform and spectrogram visualizations are generated for transparency.

2. **Data Augmentation**  
   To improve generalization, advanced augmentations are applied: Mixup and SpecAugment-style time/frequency masking. These introduce robustness to variations and mimic real-world noise/distortions.

3. **Model & Training**  
   A ResNet-style CNN with residual blocks is trained on the augmented spectrograms. Training is optimized using:
   - **AdamW optimizer** for decoupled weight decay  
   - **OneCycleLR learning rate scheduler** for efficient convergence  
   - **Batch Normalization** for stable and faster training  
   Full training dynamics—including learning rate progression, training/validation accuracy, and loss curves—are tracked via **TensorBoard** for experiment transparency and debugging.

4. **Inference Pipeline**  
   The trained model is exposed via a **FastAPI**  endpoint. Requests are schema-validated using **Pydantic** to ensure robustness. Real-time audio inputs are processed, classified, and returned with confidence scores.

5. **Cloud Deployment & Scalability**  
   Inference is deployed **serverlessly** using **Modal**, tapping into Modal’s orchestration to run GPU-backed workloads on **NVIDIA A10G** hardware. This design delivers scalable, low-latency classification without persistent infrastructure cost—spinning up GPU resources on demand.

6. **Interpretability & Diagnostics**  
   - Internal CNN feature maps are visualized to give insight into what the network is learning.  
   - Confidence-aware predictions highlight uncertainty.  
   - Distribution shift between training data and real-world inference inputs is analyzed to explain degraded performance and surface unreliability.

7. **Performance Visualization**  
   An **interactive Tableau dashboard** consolidates all diagnostics:
   - Per-class metrics (precision, recall, F1), confusion matrix  
   - Confidence calibration and reliability analysis  
   - Failure mode exploration (high-confidence errors, common confusions)  
   - Drift visualization comparing training vs inference feature distributions  
   - Sample-level inspector showing waveform/spectrogram, true vs predicted label, confidence, and feature map thumbnails  

---

## Features

- 🧠 ResNet-style deep CNN with residual blocks for robust audio feature extraction  
- 🎼 Mel spectrogram audio-to-image conversion  
- 🎛️ Advanced augmentation: Mixup and time/frequency masking (SpecAugment-style)  
- ⚙️ Optimized training pipeline: AdamW optimizer, OneCycleLR scheduler, and Batch Normalization  
- 📈 TensorBoard logging capturing learning rate schedule, accuracy curves, loss convergence, and training dynamics  
- 🚀 FastAPI inference endpoint with strict request validation using Pydantic  
- ⚡ Serverless GPU inference powered by Modal on **NVIDIA A10G** hardware for scalable, on-demand classification  
- 👁️ Visualization of internal CNN feature maps for interpretability  
- 📊 Confidence-aware real-time predictions  
- 🌊 Waveform and spectrogram visualizations for input transparency  
- 📋 Interactive Tableau dashboard for analysis of  11 specific wav files
- 🔍 Systematic analysis of distribution shift between training and real inference data  

---

## Skills

- Deep learning architecture design (ResNet-style CNN)  
- Audio preprocessing and representation. Converting from waverform to Mel Spectogram. (mel spectrograms)  
- Data augmentation strategies for robustness (Mixup, SpecAugment)  
- Training optimization and scheduler tuning (AdamW, OneCycleLR, BatchNorm)  
- Experiment tracking and model observability with TensorBoard (Included accuracy,validation and Loss Function Graph for CNN Project. 
- API design and validation (FastAPI + Pydantic)  
- Serverless cloud deployment and GPU orchestration using Modal on NVIDIA A10G  
- Model interpretability (feature maps, confidence, accuracy, bath norm, shortcut , relu)   
- Data product design via interactive Tableau dashboard  

---

## Results & Dashboard

- Validation accuracy: **84%** (example; replace with your actual metric)  
- Interactive dashboard: [Insert Tableau Public link here]  

---

## Next Steps / Known Limitations

- Frontend/dashboard integration (UI planned with modern stacks)  
- Confidence calibration and out-of-distribution detection  
- Automated regression tests on curated “gold” audio samples  
- Further domain adaptation to reduce inference degradation from distribution shift  

---

## Tech Stack

Python, PyTorch (or TensorFlow), FastAPI, Pydantic, Modal (serverless GPU), NVIDIA A10G, Mel Spectrograms, ResNet-style CNN, Mixup, Time/Frequency Masking, AdamW, OneCycleLR, Batch Normalization, TensorBoard, Tableau.



## Setup

```bash
# Clone
git clone https://github.com/AymanM7/CNN-Audio-Project
cd CNN-Audio-Project

# Python environment
python -m venv .venv
# PowerShell
.venv\Scripts\Activate.ps1
# or CMD
.venv\Scripts\activate.bat

# Backend dependencies
pip install -r requirements.txt

# Modal setup for Training Network
modal run train.py


# Modal setup and usage

## Modal Setup & Connecting Cloud Infrastructure from VS Code

This project uses **Modal** to run inference serverlessly on **NVIDIA A10G** GPUs. Below are the full steps to connect your local VS Code environment to Modal’s cloud infrastructure and deploy the backend.

### 1. Sign up for Modal
Create an account at Modal (e.g., via the web UI). Once logged in you’ll have access to your dashboard.
modal setup
modal run main.py       # run locally on Modal
modal deploy main.py    # deploy backend

##Frontend
Install dependencies:

cd CNN--Project
npm i

Run :
npm install

npm run dev



