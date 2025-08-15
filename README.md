# Visual Grounding & OCR Demo

## Project Overview

This project demonstrates an end-to-end AI application for interactive visual grounding and text extraction. Users can upload any image, provide a text prompt to find an object (e.g., "a billboard," "a person in a red shirt"), see all matching objects highlighted and numbered, select one, and receive highly accurate text extracted from that specific object.

## Features

- **Visual Grounding:** Locate objects in an image based on natural language prompts using state-of-the-art models.
- **Interactive Selection:** All detected objects are highlighted and numbered; users select which region to analyze.
- **Advanced OCR:** Extract text from the selected region using powerful OCR models.
- **Modern Web UI:** Built with Streamlit for a fast, interactive, and user-friendly experience.

## Core Technologies

- **Framework:** Streamlit
- **AI/ML Libraries:** PyTorch, Hugging Face Transformers
- **Models:**
  - **Visual Grounding:** [google/owlvit-base-patch32 (OWL-ViT)](https://huggingface.co/google/owlvit-base-patch32)
  - **OCR:** [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large), TrOCR, EasyOCR, PaddleOCR
- **Image Processing:** Pillow
- **Environment:** Python venv

## Implementation Journey

### Phase 1: Kaggle & TransVG
- Attempted to run the TransVG model on the RefCOCOg dataset in Kaggle.
- Faced critical dependency and compatibility issues due to outdated code and library versions.
- Resolved some issues by programmatically patching files and installing legacy dependencies, but ultimately hit environment limitations.

### Phase 2: Pivot to OWL-ViT
- Switched to using the OWL-ViT model for zero-shot, text-conditioned object detection.
- Built a Streamlit app for interactive visual grounding.

### Phase 3: Integrating OCR
- Initial OCR attempts with EasyOCR produced poor results on complex, real-world images.
- Upgraded to Florence-2 and TrOCR for more robust text extraction.
- Implemented image preprocessing (resizing, contrast enhancement) to improve OCR accuracy.

### Phase 4: Final Architecture
- Combined OWL-ViT for region detection and multiple OCR models for text extraction.
- Added a user interface for selecting detected regions and comparing OCR results from different models.
- Resolved major dependency issues by downgrading Python to 3.11 and carefully managing package versions.

## Challenges & Solutions

- **Dependency Hell:** Faced numerous issues with incompatible library versions (notably numpy and torch). Solved by downgrading Python and pinning package versions.
- **Model Compatibility:** Some models required legacy dependencies or specific preprocessing steps.
- **OCR Accuracy:** Improved by experimenting with different models and preprocessing techniques (resizing, contrast, margin around crops).

## How to Run

1. **Clone the repository and set up a Python 3.11 virtual environment.**
2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
3. **Run the Streamlit app:**
	```bash
	streamlit run app.py
	```
4. **Upload an image, enter a prompt, and interact with the results!**

## Example Use Case

- Upload a photo of a busy street.
- Enter a prompt like "a billboard" or "a person in a red shirt".
- See all matching objects highlighted and numbered.
- Select a region to extract and compare text using different OCR models.

## Lessons Learned

- Modern AI projects often require careful environment and dependency management.
- Preprocessing and model selection are critical for robust OCR in real-world images.
- Interactive demos are invaluable for showcasing and validating AI capabilities.
# Visual-Grounding-Scene-Localization-