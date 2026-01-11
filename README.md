# CIFAR-10 Image Preprocessing & Visualization Pipeline

## Overview
This project demonstrates a complete **image preprocessing pipeline** on the CIFAR-10 dataset using Python, OpenCV, NumPy, and Matplotlib. It focuses on understanding how different preprocessing techniques affect image quality and pixel distributions through **visual comparison, histograms, and difference analysis**.

The project is designed to support **computer vision and deep learning workflows**, especially before feeding images into CNN models.

---

## Objectives
- Load CIFAR-10 images from CSV format
- Reconstruct RGB images from raw pixel values
- Apply and visualize common image preprocessing techniques
- Analyze pixel distribution changes using histograms
- Measure preprocessing impact using difference heatmaps

---

## Technologies Used
- Python
- NumPy
- Pandas
- OpenCV
- Matplotlib

---

## Dataset
- **Dataset:** CIFAR-10 (CSV format)
- **Image Size:** 32×32 RGB
- **Classes:** 10 object categories
- **Input Format:**  
  - Column 0 → Label  
  - Columns 1–3072 → Pixel values (R, G, B channels)

---

## Preprocessing Techniques Implemented
The following **five preprocessing steps** are applied sequentially:

1. **Resizing**  
   - Converts images from `32×32` to `64×64`

2. **Normalization**  
   - Scales pixel values to range `[0, 1]`

3. **Denoising**  
   - Gaussian Blur to reduce noise

4. **Contrast Enhancement**  
   - Pixel intensity scaling using alpha factor

5. **Data Augmentation**  
   - Horizontal flip  
   - 90° rotation

---

## Visual Analysis
The project provides:
- Side-by-side comparison of original vs processed images
- Pixel intensity histograms before and after preprocessing
- Difference heatmaps between original and processed images
- Pixel-wise difference distribution plots

These visualizations help in understanding how preprocessing transforms image data.

---

## Pipeline Structure

### File 1: Step-by-Step Visualization
- Loads a single CIFAR-10 image
- Applies preprocessing **one step at a time**
- Displays:
  - Original image
  - Processed image
  - Histograms for both

### File 2: Unified Preprocessing Function
- Implements all preprocessing steps inside a reusable function
- Compares original and processed images
- Generates:
  - Difference heatmap
  - Difference histogram

---

## How to Run

### 1️Install dependencies
```bash
pip install pandas numpy matplotlib opencv-python

Update the CSV path inside the script:

df = pd.read_csv("path/to/train.csv")

Run the script
python cifar10_preprocessing.py
