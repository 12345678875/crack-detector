# AI Crack Detection System

A computer vision project for **automatic crack detection** using **OpenCV and PyTorch CNN**.

This project implements a full pipeline including:

* Image preprocessing
* Crack feature extraction
* CNN-based crack classification
* Gradio visualization interface
* Automatic detection reports

This system can be applied to:

* Road crack inspection
* Concrete structure monitoring
* Bridge health monitoring
* Infrastructure safety inspection

---

# Project Demo

## Input Image

Crack image captured from the dataset.

## Processing Pipeline

Image preprocessing includes:

* Grayscale conversion
* Gaussian filtering
* Edge detection (Canny)
* Morphological processing

## Detection Result

The system automatically detects cracks and outputs detection results.

---

# Tech Stack

This project uses the following technologies:

* Python
* OpenCV
* PyTorch
* NumPy
* Gradio
* SciPy
* Pillow

---

# Project Structure

```
crack-detector
│
├── app
│   └── app.py
│
├── models
│   └── crack_cnn.py
│
├── training
│   ├── train.py
│   └── dataset.py
│
├── image_processing
│   └── crack_detection.py
│
├── docs
│   └── images
│
├── results
│   └── reports
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/12345678875/crack-detector.git
cd crack-detector
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Train the Model

Run the training script:

```
python training/train.py
```

The CNN model will be trained on the CrackForest dataset.

---

# Run the Detection System

Start the Gradio interface:

```
python run_app.py
```

Then open the browser:

```
http://127.0.0.1:7860
```

Upload an image to perform crack detection.

---

# Dataset

This project uses the **CrackForest Dataset**, a commonly used dataset for road crack detection research.

Dataset link:

https://github.com/cuilimeng/CrackForest-dataset

---

# Author

Yan Shiyi
AI Crack Detection Project
