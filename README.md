# 👁️ Florence-2 Manga & Webtoon Analyst

A GPU-accelerated local AI tool for analyzing Manga and Webtoons.

This application uses **Microsoft Florence-2-Large** for detailed visual descriptions and **EasyOCR / Manga-OCR** for text extraction. It features a modular design that allows you to toggle between OCR and Visual Analysis to save memory and speed up processing.

## ✨ Features
* **Modular AI:** Toggle **OCR** and **Visual Description** independently. Unchecking a module prevents it from loading into VRAM.
* **Smart Slicing:** Automatically cuts long webtoons into 1500px segments (with overlap) to ensure high-resolution analysis.
* **Detailed Captions:** Uses Florence-2's `<MORE_DETAILED_CAPTION>` mode to describe character actions and scenes.
* **Dual OCR Support:**
    * **English:** Uses EasyOCR (GPU-accelerated).
    * **Japanese:** Uses Manga-OCR (specialized for vertical text).
* **RTX 50-Series Ready:** Includes instructions for running on the latest Blackwell architecture.

## 📋 Prerequisites
* **OS:** Windows 10/11
* **GPU:** NVIDIA GPU (6GB+ VRAM recommended).
* **Python:** Python 3.10 or newer.
* **CUDA:**
    * Standard Cards (RTX 30/40 series): CUDA 12.1
    * New Cards (RTX 50 series): CUDA 12.8 (Nightly)

## 🛠️ Installation

### 1. Clone the Repository
```
https://github.com/Eldiablozzz/Manga-to-Prompt.git
```

### 2. Create a Virtual Environment
```python
python -m venv venv
.\venv\Scripts\activate
```
### 3. Install Torch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
### 4. Install Dependencies
```
pip install -r requirements.txt
```
### 5. Run the App
```
python app.py
```
Open Browser: Go to http://127.0.0.1:7860.

Analyze:

  Upload images.

  Check/Uncheck OCR or Visual Description based on what you need.

  Click Analyze Manga.

