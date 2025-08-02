## Overview

This project is an end-to-end audio classification prototype built from scratch. It trains a ResNet-style convolutional neural network on environmental sound data (EXC50/ESC-50)  dataset using mel spectrogram inputs and advanced augmentation techniques to recognize sounds like bird chirps. rain sound. The training pipeline incorporates Data Mixing and time/frequency masking, and is optimized with AdamW, OneCycleLR, and batch normalization, with full experiment tracking via TensorBoard. Inference is exposed via a FastAPI endpoint with Pydantic validation and executed on demand using serverless GPU infrastructure (Modal) backed by NVIDIA A10-class hardware. The system includes interpretability tooling—visualizing internal CNN feature maps, confidence scores, waveforms, and spectrograms—and a clean Tableau dashboard for diagnosing per-class performance, calibration, distribution shift, and failure modes. Real-world inference gaps are analyzed and framed as actionable insights.

##  Project Features

## Overview

This project is an end-to-end audio classification and diagnostics prototype. It trains a ResNet-style convolutional neural network on environmental sound data (EXC50/ESC-50) using mel spectrogram inputs and advanced augmentation (Mixup, time/frequency masking) to recognize real-world sounds. Training is optimized with AdamW, OneCycleLR learning rate scheduling, and batch normalization, with full observability via TensorBoard (tracking learning rate progression, training/validation accuracy, loss curves, and convergence behavior). Inference is exposed through a FastAPI endpoint with Pydantic validation and executed on-demand using serverless cloud GPU infrastructure—Modal powering **NVIDIA A10G-class** GPUs for scalable, cost-efficient real-time classification. The system includes interpretability tooling (internal CNN feature map visualization, confidence scoring, waveform/spectrogram transparency) and an interactive Tableau dashboard to surface per-class performance, calibration, distribution shift, and failure modes. Real-world inference gaps are diagnosed and presented as actionable insights.

## Features

- 🧠 ResNet-style deep CNN with residual blocks for audio classification  
- 🎼 Mel Spectrogram audio-to-image conversion  
- 🎛️ Robust data augmentation: Mixup and time/frequency masking (SpecAugment-style)  
- ⚙️ Optimized training pipeline: AdamW optimizer, OneCycleLR scheduler, and Batch Normalization  
- 📈 TensorBoard integration capturing learning rate schedule, training/validation accuracy, loss curves, and convergence behavior  
- 🚀 FastAPI inference endpoint with schema-validated requests via Pydantic  
- ⚡ Serverless GPU inference on Modal using **NVIDIA A10G** hardware for scalable, on-demand prediction  
- 👁️ Internal CNN feature map visualization for model interpretability  
- 📊 Confidence-aware real-time classification outputs  
- 🌊 Waveform and spectrogram visualization for input transparency  
- 📋 Interactive Tableau dashboard for diagnosing performance (confusion matrix, calibration, drift, and failure modes)  
- 🔍 Analysis of distribution shift between training and inference to surface and explain degraded real-world behavior  
