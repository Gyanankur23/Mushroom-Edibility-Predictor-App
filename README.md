# Mushroom ML Predictor App

## Overview
A Streamlit application that predicts mushroom edibility using a RandomForest model. Enhanced with interactive sidebars, styled metric cards, and live graph fluctuations for a professional dashboard experience.

## Features
- Sidebar controls for selecting mushroom features (cap shape, cap color, odor).
- Prediction output with confidence score (Edible or Poisonous).
- Styled metric cards using custom CSS for selected features.
- Live probability graph with fluctuations to simulate dynamic ML behavior.
- Clean layout with no background boxes, blending visuals into the canvas.

## Requirements
- streamlit==1.32.0
- pandas==2.2.0
- numpy==1.26.4
- scikit-learn==1.4.0
- matplotlib==3.8.2

## Deployment
1. Save the app code as `app.py`.
2. Create a `requirements.txt` file with the listed dependencies.
3. Run locally with:

stremwlit run app.py
