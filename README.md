# Toxicity Detection System

## Description

This project implements an automated toxicity detection system for online gaming chat using machine learning and NLP.

It compares a traditional TF-IDF + Logistic Regression model with a transformer-based DistilBERT model. The final deployed system uses DistilBERT because it provides much better recall for toxic content.

The system is deployed as a FastAPI backend for real-time toxicity prediction.

## Key Features

- Toxic vs Non-Toxic classification
- DistilBERT-based NLP model
- FastAPI backend
- Real-time prediction through REST API
- Confidence score output
- Swagger UI testing

## Project Structure

- `backend/` contains the API code
- `distilbert_toxicity_model/` contains the trained model files

## Installation

1. Open terminal inside the `backend` folder.
2. Create a virtual environment:
   `python -m venv venv`
3. Activate the environment:
   - Windows PowerShell: `.\venv\Scripts\Activate`
4. Install dependencies:
   `pip install -r requirements.txt`

## Run Instructions

From the `backend` folder, run:

`uvicorn api:app --reload`

Then open:

`http://127.0.0.1:8000/docs`

## Example Inputs

- `you are trash` → Toxic
- `good game everyone` → Non-Toxic

## System Requirements

- Python 3.9 or newer
- Required Python packages listed in `requirements.txt`

## Attribution

- Jigsaw Toxic Comment Classification Dataset
- Hugging Face Transformers
- FastAPI
- PyTorch

## Model Files

The trained DistilBERT model is not included in this repository due to size limitations.

To run the project, place the model files in the following directory:

distilbert_toxicity_model/

Alternatively, the model can be downloaded from Hugging Face or provided separately.

## Author

Syed Owais Haider Kazmi
University of New Brunswick
