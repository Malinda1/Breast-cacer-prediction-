Search files...
Ran tool
Ran tool
Ran tool
I have gathered information from your codebase and project files. Here’s a comprehensive README.md draft for your Breast Cancer Prediction project, including all major components, usage, and references to Hugging Face model fine-tuning:

---

# Breast Cancer Prediction

A professional, end-to-end AI-powered system for breast cancer detection from medical images, featuring robust data preparation, state-of-the-art model fine-tuning using Hugging Face, and a modern web/API interface for predictions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training & Fine-Tuning](#model-training--fine-tuning)
- [Web & API Usage](#web--api-usage)
- [Prediction Example](#prediction-example)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project provides a complete pipeline for breast cancer prediction from medical images (e.g., MRI scans). It includes:

- Automated data download and preprocessing
- Advanced image organization and cleaning
- Fine-tuning of a Vision Transformer (ViT) model from Hugging Face for binary classification (cancer/no cancer)
- A FastAPI-based web server and modern web UI for easy predictions

---

## Features

- **Automated Data Handling:** Download and organize large datasets from Kaggle.
- **Image Preprocessing:** Clean, filter, and structure images for robust training.
- **Model Fine-Tuning:** Uses Hugging Face’s `google/vit-base-patch16-224` model, adapted for breast cancer detection.
- **Web & API Interface:** Upload images and get instant predictions via a user-friendly web app or REST API.
- **Professional Recommendations:** Each prediction includes actionable recommendations and risk assessment.

---

## Project Structure

```
Brest_cancer_prediction/
│
├── app/                  # FastAPI app and web UI
│   ├── main.py           # Main FastAPI server
│   └── template/         # HTML templates (index.html)
│
├── data/                 # Datasets and CSVs
│   ├── raw_data/         # Raw images from Kaggle
│   ├── preprocessed_data/# Organized images for training
│   ├── cleaned_extracted_data.csv
│   └── unrecognized_images.csv
│
├── model/
│   ├── raw_model/        # Downloaded Hugging Face model
│   └── fine_tuning_model/# Fine-tuned/adjusted models
│
├── notebooks/            # Jupyter notebooks for data prep, training, etc.
│
├── src/                  # Scripts for dataset loading, model download, etc.
│
├── requirements.txt      # All dependencies
└── run_requirement_install.py # Script to install requirements
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Brest_cancer_prediction
   ```

2. **Install dependencies:**
   ```bash
   python run_requirement_install.py
   ```
   Or manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets:**
   - Use the provided scripts/notebooks in `src/` and `notebooks/` to download and organize data from Kaggle.

4. **Download and prepare the model:**
   - Run the scripts in `src/model_download.ipynb` to fetch the ViT model from Hugging Face.

---

## Data Preparation

- Data is downloaded from Kaggle (`olegbaryshnikov/rsna-roi-512x512-pngs`).
- Images are organized and cleaned using scripts in `notebooks/` and `src/`.
- Preprocessing includes:
  - Removing unrecognized or corrupt images
  - Organizing images into `cancer` and `no_cancer` folders
  - Generating CSVs for training/validation splits

---

## Model Training & Fine-Tuning

- The project uses Hugging Face’s `google/vit-base-patch16-224` Vision Transformer.
- The model is adjusted for binary classification (cancer/no cancer).
- Fine-tuning is performed using PyTorch and Hugging Face Transformers.
- Training scripts and logs are available in `notebooks/Model_fine_tuning.ipynb`.
- The best model is saved in `model/fine_tuning_model/`.

**Key Steps:**
1. Download the base model from Hugging Face.
2. Adjust the classifier head for 2 classes.
3. Fine-tune on the preprocessed dataset.
4. Save the best model for inference.

---

## Web & API Usage

### Web Interface

- Start the FastAPI server:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```
- Open your browser and go to [http://localhost:8000](http://localhost:8000)
- Upload a medical image (JPG, PNG, TIFF, BMP, max 10MB) and get instant predictions.

### API Endpoints

- **POST `/predict`**: Upload an image and receive a prediction.
- **GET `/health`**: Check server and model status.
- **GET `/prediction/{prediction_id}`**: Retrieve a previous prediction result.
- **GET `/class-mapping`**: Get class label mapping.

See `/docs` for full OpenAPI documentation.

---

## Prediction Example

- Upload an image via the web UI or API.
- Receive a prediction:
  - **Prediction:** Cancer Detected / No Cancer
  - **Confidence:** e.g., 97.5%
  - **Risk Level:** High/Medium/Low
  - **Recommendations:** Next steps and medical advice

---

## Disclaimer

⚠️  This tool is for research and educational purposes only.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult with qualified healthcare professionals for any medical concerns.

---

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for the ViT model and Transformers library
- [Kaggle](https://www.kaggle.com/) for the dataset
- [PyTorch](https://pytorch.org/) and [FastAPI](https://fastapi.tiangolo.com/) for the core framework

---

Let me know if you want to customize any section or add more details! I can now write this to your README.md file. Would you like me to proceed?