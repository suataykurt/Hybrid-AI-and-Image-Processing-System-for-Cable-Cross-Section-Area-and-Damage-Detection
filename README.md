# Hybrid-AI-and-Image-Processing-System-for-Cable-Inspection

A hybrid inspection system combining MATLAB’s classic image processing (morphological analysis, GLCM, pixel-to-mm scaling) with Python-based Deep Learning (YOLOv11) for real-time cable damage detection and cross-section measurement.

## 📋 Project Overview
This system automates the inspection of industrial cables. It identifies physical defects (burns, thinning, deformation) and calculates accurate cross-section areas, even in challenging environments, using a custom-trained YOLOv11 model.

## 🛠 Technical Pipeline
* **Preprocessing:** CLAHE (Contrast Limited Adaptive Histogram Equalization) for illumination normalization and Gaussian filtering for noise reduction.
* **Segmentation:** Lab* color space conversion combined with Canny edge detection and morphological closing/filling to create precise cable masks.
* **AI Integration:** Python bridge (`yolo_kopru.py`) triggers YOLOv11 inference on the MATLAB-processed images.
* **Feature Extraction:** Skeletonization for precise diameter/area measurement and GLCM-based entropy filtering for texture-based damage detection.
* **Data Management:** Lossless image compression using Huffman coding for efficient storage.

## 🚀 Key Features
* **Hybrid Architecture:** Bridges classic computer vision (MATLAB) with Deep Learning (Python) for robust decision-making.
* **Precise Measurement:** Implements pixel-to-millimeter scaling with automatic exclusion of damaged zones to prevent measurement bias.
* **Advanced Analytics:** Detects defects using texture entropy and color-space analysis (HSV) beyond simple classification.
* **Efficient Archiving:** Custom Huffman coding implementation for space-saving log management.

## 📂 Project Structure
* `main_code.m`: MATLAB core processing pipeline and AI integration bridge.
* `yolo_kopru.py`: Python-based YOLOv11 inference script.
* `best.pt`: Trained weight file for cable defect classification.
* `images/`: Sample dataset containing various cable types and damage conditions.
* `report.pdf`: Detailed technical report on the hybrid methodology and experimental results.

## 💡 How to Run
1. Ensure `Ultralytics` and `MATLAB` support for Python are configured.
2. Place your image dataset in `images/`.
3. Run `main_code.m` in MATLAB; it will automatically trigger the Python script and generate log files (Excel/CSV) and Huffman-compressed `.mat` results.

---
*Developed by Suat AYKURT | Zonguldak Bülent Ecevit University*