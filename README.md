# AI Crack Detection System
An AI-assisted concrete crack detection system built with **Python, OpenCV, and Gradio**.
This project detects cracks in uploaded images and estimates their **length and width**, providing a visual detection result and a textual report.

## Demo
Upload a crack image and the system will:

* Detect crack contours
* Estimate crack **length** and **average width**
* Display the **annotated image**
* Generate a **detection report**

## Features

* Image preprocessing (grayscale, Gaussian blur)
* Edge detection using **Canny**
* Adaptive threshold segmentation
* Morphological operations for noise reduction
* Crack contour detection
* Crack length and width estimation
* Visual result output
* Automatic detection report
* Web interface built with **Gradio**

## Project Structure

crack-detector
│
├── app.py              # Main application (Gradio interface)
├── image/              # Test images
├── results/            # Output results (optional)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies

## Installation
Clone this repository:
bash
git clone https://github.com/12345678875/crack-detector.git
cd crack-detector

Install dependencies:
bash
pip install -r requirements.txt

If you don't have a requirements file yet, install manually:

bash
pip install opencv-python numpy gradio

## Usage
Run the application:
bash
python app.py

After running, open the local web interface:

http://127.0.0.1:7860

Upload a crack image and click **Start Detection**.

The system will display:

* Detected crack contours
* Crack length and width estimation
* Detection report

## Detection Pipeline

The crack detection workflow:

Input Image
    ↓
Grayscale Conversion
    ↓
Gaussian Blur
    ↓
Canny Edge Detection
    ↓
Adaptive Threshold
    ↓
Morphological Closing
    ↓
Contour Detection
    ↓
Crack Measurement

## Crack Measurement

The crack length and width are estimated using contour properties:

* **Length:** contour perimeter
* **Width:** calculated from area and length approximation

Conversion:

mm = pixel / PIXELS_PER_MM

Note:
PIXELS_PER_MM` must be calibrated according to the real camera setup.

## Example Output

Detection result includes:

* Crack contour visualization
* Crack count
* Crack length (mm)
* Average crack width (mm)
* Detection timestamp

## Technologies Used

* Python
* OpenCV
* NumPy
* Gradio

## Author

Bang Bang
North China University of Water Resources and Electric Power

GitHub:
https://github.com/12345678875/crack-detector

## Future Improvements

* Crack segmentation using **Deep Learning (U-Net / YOLOv8)**
* Crack width precision improvement
* Batch image detection
* Dataset training support
* Deployment as a web service
