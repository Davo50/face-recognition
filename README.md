# Face Recognition Project

## Overview
This project implements a machine learning face recognition system.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Davo50/face-recognition.git
cd face-recognition
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
- Place your face images in `data/people_dataset/`
- Each person should have a separate subdirectory
- Example structure:
  ```
  data/people_dataset/
  ├── person1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  ├── person2/
  │   ├── image1.jpg
  │   └── image2.jpg
  ```

### 5. Train the Model
```bash
python -m src.train
```

### 6. Run Inference
```bash
python -m src.inference
```

## Project Structure
- `src/`: Source code modules
- `data/`: Dataset directory
- `results/`: Trained models and logs
- `notebooks/`: Jupyter notebooks for exploration

## Key Components
- Data Loading and Augmentation
- Transfer Learning with ResNet50V2
- Model Training with Early Stopping
- Face Recognition Inference

## Requirements
- Python 3.8+
- TensorFlow 2.x
- GPU (recommended but optional)