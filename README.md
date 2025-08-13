# Deep Audio Classification & Diagnostics on ESC-50 Dataset

## Executive Summary
This project is an end-to-end deep learning audio classification system, I built for the ESC-50 environmental sound dataset applying CNNs(Nueral Networks).
It transforms raw audio into mel spectrogram images, trains a ResNet-style CNN for robust feature extraction, and deploys the inference pipeline serverlessly on NVIDIA A10G GPUs via Modal.


## Problem Statement
Environmental sound recognition is crucial in domains like smart cities, wildlife monitoring, industrial safety, and assistive technologies. However, real-world audio classification faces several key challenges:

Spectral Similarity — Similar frequency patterns (e.g., wind vs. rain) can confuse models.

Noise & Variability — Background sounds, microphone quality, and recording conditions can reduce accuracy.

Distribution Shift — Models trained on clean datasets often struggle with unseen real-world data.

Scalability & Latency — Real-time, high-volume inference demands significant compute resources.

These challenges have direct implications in critical sectors. In smart city infrastructure, accurate sound recognition can enable automated detection of hazardous events such as sirens, explosions, or breaking glass, supporting faster emergency response. In wildlife conservation, it can identify endangered species through acoustic monitoring, enabling non-invasive population tracking. In industrial settings, it can detect early signs of equipment malfunction, preventing costly downtime and improving safety. For assistive technologies, such as devices for the hearing-impaired, reliable classification provides real-time alerts for important environmental cues.

In this project, I address these real-world problems by converting audio to mel spectrograms, applying robust augmentations (Mixup, SpecAugment), training a ResNet-style CNN, and deploying inference serverlessly on GPUs for scalable, low-latency classification

## Table of Contents
### [1.) Overview](#sec-overview)
### [2.) Dataset](#sec-dataset)
### [3.) Project Architecture](#sec-architecture)
### [4.) Project Properties](#sec-properties)
### [5.) Skills](#sec-skills)
### [6.) Results From Training the Convolutional Nueral Network](#sec-results-cnn)
### [7.) CNN Training Process](#sec-training-process)
### [8.) Running Inference with Modal](#sec-inference-modal)
### [9.) Modal Deployment & Inference Status](#sec-deploy-status)
### [10.) Active Application Status](#sec-active-status)
### [11.) Results Table](#sec-results-table)
### [12.) Dashboard Visulization](#sec-dashboard)
### [13.) Challenges Faced](#sec-challenges)
### [14.) What I Learned](#sec-learned)
### [15.) Next Steps/(Plan)](#sec-next)
### [16.) Tech Stack](#sec-tech-stack)
### [17.) Project Setup](#sec-setup)



### 1.) Overview
I built an environmental sound classification system that:

Converts .wav audio into mel spectrograms

Applies augmentation for robustness

Trains a ResNet-18 CNN

Serves predictions through a FastAPI endpoint with GPU acceleration

Monitoring and interpretabilit and anlysis of specific .wav files  are handled via TensorBoard and Tableau dashboards.


### 2.) Dataset
Dataset
ESC-50: 50 balanced classes of environmental sounds (e.g., rain, wind, thunder, insects, dog barking,etc)


### 3.)  Project Architecture

1. **Data Ingestion & Preprocessing**  
   Raw audio clips (EXC50/ESC-50 style environmental sounds) are loaded, normalized, and converted into mel spectrograms—effectively turning audio into image-like representations suitable for deep CNNs. Waveform and spectrogram visualizations are generated for transparency.

2. **Data Augmentation**  
   To improve generalization, advanced augmentations are applied: Mixup and SpecAugment-style time/frequency masking. These introduce robustness to variations and mimic real-world noise/distortions.

3. **Model & Training**


ResNet-18 backbone (pretrained on ImageNet, adapted for 1-channel spectrogram input).

Optimization:

AdamW optimizer (decoupled weight decay)

OneCycleLR scheduler (cyclical learning rate for fast convergence)

Batch Normalization** for stable and faster training  

Label smoothing

Training metrics (loss, accuracy, learning rate progression) logged in TensorBoard.


4. **Inference Pipeline**  
   The trained model is exposed via a **FastAPI**  endpoint. Requests are schema-validated using **Pydantic** to ensure robustness. Real-time audio inputs are processed, classified, and returned with confidence scores.
   Accepts base64-encoded audio → preprocesses to mel spectrogram → predicts top-3 classes with confidence scores.

6. **Cloud Deployment & Scalability**  
   Inference is deployed **serverlessly** using **Modal**, tapping into Modal’s orchestration to run GPU-backed workloads on **NVIDIA A10G** GPU . This design delivers scalable, low-latency classification without persistent infrastructure cost—spinning up GPU resources on demand.

      
8. **Interpretability & Diagnostics**  
   - Internal CNN feature maps are visualized to give insight into what the network is learning.  
   - Confidence-aware predictions highlight uncertainty.  
   - Distribution shift between training data and real-world inference inputs is analyzed to explain degraded performance and surface unreliability.

9. **Dashboard Visualization**
   The Tableau dashboard provides an acoustic feature analysis of 9 selected ESC-50 sound clips, helping to understand and contextualize the data that the CNN model processes.

How it relates to the CNN project:

These features help explain why certain sounds may be harder to classify — for example, “wind” and “rain” may have similar spectral profiles, leading to model confusion.

By filtering by true class in Tableau, you can compare within-class and between-class acoustic variability.

The metrics provide insight into potential data preprocessing or augmentation strategies — e.g., balancing loud/quiet samples, augmenting tonal diversity.



10. **CNN Application/Worflow in Project**

    
    ![Screenshot_12-8-2025_125413_chatgpt com](https://github.com/user-attachments/assets/4e802143-fba8-4ff5-8fe6-5ce642097641)



12. **Project Arhitecture Diagram/ Workflow**

    
![Screenshot_12-8-2025_182314_chatgpt com](https://github.com/user-attachments/assets/aac7ec37-a5f4-4088-a77a-02692921e6c4)

    


## 4.) Project Properties 

-  ResNet-style deep CNN with residual blocks for robust audio feature extraction  
-  Mel spectrogram audio-to-image conversion  
-  Advanced augmentation: Mixup and time/frequency masking (SpecAugment-style)  
-  Optimized training pipeline: AdamW optimizer, OneCycleLR scheduler, and Batch Normalization  
-  TensorBoard logging capturing learning rate schedule, accuracy curves, loss convergence, and training dynamics  
-  FastAPI inference endpoint with strict request validation using Pydantic  
-   GPU inference powered by Modal on **NVIDIA A10G** hardware for scalable, on-demand classification  
-  Visualization of internal CNN feature maps for interpretability  
-  Confidence-aware real-time predictions  
-  Waveform and spectrogram visualizations for input transparency  
-  Interactive Tableau dashboard for analysis of  9 specific wav files from the ESC-50 Dataset

---

## 5.) Skills

- Deep learning architecture design (ResNet-style CNN)  
- Audio preprocessing and representation. Converting from waverform to Mel Spectogram. (mel spectrograms)  
- Data augmentation strategies for robustness (Mixup, SpecAugment)  
- Training optimization and scheduler tuning (AdamW, OneCycleLR, BatchNorm)  
- Experiment tracking and model observability with TensorBoard (Included accuracy,validation and Loss Function Graph for CNN Project. 
- API design and validation (FastAPI + Pydantic)  
- Serverless cloud deployment and GPU orchestration using Modal on NVIDIA A10G  
- Model interpretability (feature maps, confidence, accuracy, bath norm, shortcut , relu)   
- Visulization via interactive Tableau dashboard  

---

## 6.) Results From Training the Convolutional Nueral Network 

- This includes Accuracy/Validation , Learning Rate , The general Loss Function Graph for the Training and Validation Set for the Nueral Network coming from Tensorboard.

 
![Screenshot_12-8-2025_13116_docs google com](https://github.com/user-attachments/assets/6f0db1aa-f3dc-4632-9b6b-e4f4c61dfff9)





![Screenshot_12-8-2025_132250_docs google com](https://github.com/user-attachments/assets/7fb68381-41f8-4bb1-8983-f37204be6f2b)




![Screenshot_12-8-2025_132346_docs google com](https://github.com/user-attachments/assets/3fa43db6-8c18-4ac4-9a49-612055b2bd12)




## 7.) CNN Training Process
• The Convolutional Neural Network (CNN) was trained over 100 epochs using Modal’s cloud infrastructure on NVIDIA A10G GPUs.

Before starting the training process, Modal must be set up by following the official documentation on their website and running the necessary configuration commands in the terminal. Once configured, training is initiated by executing:  modal run train.py


This command launches the full training pipeline, including data preprocessing, augmentation, and model optimization. Throughout the 100 training epochs, TensorBoard logs  training accuracy and validation loss for performance monitoring.

Final Results From My Training:

Validation Accuracy: 83.75%

Validation Loss: 1.2997

These metrics indicate that the CNN achieved strong generalization on the ESC-50 test fold, with relatively low classification error given the dataset’s diversity.







![Screenshot_12-8-2025_141339_docs google com](https://github.com/user-attachments/assets/0e1d15ee-ee12-45ab-a6c8-b8956d0905bf)



##  8.) Running Inference with Modal

Once the CNN model has been trained and validated, it can be deployed to serve real-time predictions using Modal’s serverless GPU infrastructure.


1.) Deploy the Model :    modal deploy main.py (run this command in your terminal)

This command packages the trained model, configures the FastAPI inference endpoint, and deploys it to Modal’s cloud environment. The endpoint is then accessible via a unique URL.


2.) Run the Model Locally on Modal : modal run main.py (run this command in your terminal)

This allows testing the deployed inference pipeline directly from your development environment without fully publishing it as web app.


3.) How Predictions Work:

The API  endpoint accepts a base64-encoded .wav audio file as input.

The file is converted to a mel spectrogram and passed through the trained CNN.

The output contains the top-3 predicted classes with their probabilites.


# Example 1  : Thunderstorm.wav audio file
-- Model correctly predicts thunderstorm with high confidence.


![Screenshot_12-8-2025_161439_docs google com](https://github.com/user-attachments/assets/1b7f7375-bf7a-4bd2-8b5e-ee0cf0418d7d)







# Example 2 : Airplane.wav audio file
-- Model prediction is less accurate, highlighting that similar spectral patterns in different environmental sounds can cause classification challenges.

![Screenshot_12-8-2025_16170_discord com](https://github.com/user-attachments/assets/5fa88f4e-6751-4a14-be73-50ae74aa4d7d)






## 9.)  Modal Deployment & Inference Status

- This screenshot displays the deployment history for the audio-cnn-inference application hosted on Modal.
- Each entry represents a separate version of the deployed inference service.
- Across the 25 deployments shown, I was able to iteratively test and validate the model’s performance on 25 different .wav audio files, ensuring the classifier could handle a wide range of environmental sounds under real-world conditions.
- Out of these, 18 audio files were accurately classified with the correct top prediction, showing strong model reliability while also highlighting areas where further robustness is needed for the remaining cases as this Convoluntional Nueral Network.
- Misclassifications highlight challenges such as overfitting and the model’s occasional difficulty in providing confident, accurate predictions under all conditions for all files and not picking up trends on unseen data.



![Screenshot_12-8-2025_163044_docs google com](https://github.com/user-attachments/assets/c6c70070-60aa-4da8-bcc1-d885809ee106)



## 10.)  Active Application Status 

- Application Name: audio-cnn-inference

- Active Service: AudioClassifier.*

- GPU Hardware: Running on NVIDIA A10G for high-performance inference.

- Web Endpoint: Indicates that the model is live and accessible via a cloud-based web API endpoint.


![Screenshot_12-8-2025_163054_docs google com](https://github.com/user-attachments/assets/aec6dc71-a5b3-430f-b203-2e56d6e7bba5)

## 11.) Results Table 

Metric Values
Validation Accuracy	:83.75%
Validation Loss	 :1.2997
Inference Latency	:<2s (GPU)




## 12.) Dashboard Visulization 


- Interactive Analysis of 9 selected ESC-50 environmental sound clips, focusing on key acoustic properties.

- Frequency Spread chart highlights “bright” vs. “dark” sounds based on spectral bandwidth.

- Mel-Spectrogram Heatmap visualizes average dB energy distribution across frequencies.

- Clip Feature Distribution plots spectral centroid for class brightness comparisons.

- RMS Energy Comparison shows loud vs. quiet sounds for amplitude-based separability.

- If the CNN misclassifies a clip, you can cross-reference this dashboard to see if low RMS energy, narrow bandwidth, or low spectral centroid might have contributed.




![Screenshot_12-8-2025_125529_public tableau com](https://github.com/user-attachments/assets/4b46a76b-6c32-4506-b4be-7ca9ca0b88ce)


Link to View Full Dashboard in Detail :   <a href="https://public.tableau.com/app/profile/ayman.mohammad/viz/ESC-50AudioSampleInsightsDashboard/ESC-50AudioSampleInsights" target="_blank">View Entire Dashboard</a>
></a>



---

## 13.) Challenges Faced 
After deployment on Modal, some audio files (e.g., bird vs. rain) gave inconsistent results because small differences in preprocessing (normalization/scaling, background noise, etc.) and runtime variability like cold starts changed how the model “saw” the sound.

The system always returns the top 3 label guesses with confidence scores (e.g., Bird: 75%, Wind: 10%, Insect: 15%), so outputs are probabilistic—not simply right or wrong—and similar-sounding clips can produce different mixes of scores.

Fixing this meant tightening consistency between training and inference, adding input validation/sanity checks, and making the model more robust to real-world variation.

Different sounds behaved differently: Even if two clips looked similar, some caused errors—showing that real-world audio varies a lot and can confuse the system.

Added toughness to the model: To handle messy or unusual sounds better, we introduced techniques that make the model more resilient to variability.

Early training failed: The first few full training attempts gave poor accuracy because the learning settings weren’t tuned right.

Small steps fixed things: Instead of big changes, gradual tweaks (like adjusting learning speed and adding more varied examples) led to stable, better performance.



## 14.) What I Learned

This project reinforced the importance of aligning the training and inference pipelines to ensure consistent model performance. I learned how small variations in preprocessing, scaling, and augmentation can significantly affect classification accuracy in real-world deployment. Through iterative debugging and hyperparameter tuning and altering, I developed a deeper understanding of  strategies for CNN-based audio models, the trade-offs between latency and accuracy in GPU inference, and the value of rigorous validation across diverse input conditions. Additionally, integrating interpretability tools and dashboard analytics taught me how to communicate model behavior to both technical and non-technical stakeholders, which is crucial for deploying AI solutions in practical domains.





## 15.) Next Steps/(Plan)

- Frontend/dashboard integration (UI planned with modern stacks) with deployment to the web
- Confidence calibration and out-of-distribution detection  
- Further domain adaptation to reduce inference degradation from distribution shift  

---

## 16.) Tech Stack

Python, PyTorch (or TensorFlow), FastAPI, Pydantic, Modal (GPU), NVIDIA A10G, Mel Spectrograms, ResNet-style CNN, Mixup, Time/Frequency Masking, AdamW, OneCycleLR, Batch Normalization, TensorBoard, Tableau.



##  17.) Project Setup

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
Create an account at Modal via your github account (e.g., via the web UI). Once logged in you’ll have access to your dashboard.
modal setup
modal run main.py       # run locally on Modal(cloud)
modal deploy main.py    # deploy backend

##Frontend
Install dependencies:

cd CNN--Project
npm i

Run :
npm install

npm run dev



