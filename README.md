# Electronics
Spatio-Temporal Attention-Based Deep Learning for Smart Grid Demand Prediction

# ST-CALNet: A Hybrid Spatio-Temporal CNN-Attentive LSTM Network for Short-Term Load Forecasting

ST-CALNet is a hybrid deep learning model designed for short-term electricity load forecasting in smart grids with high renewable energy penetration. The model integrates Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) units, and an attention mechanism to effectively capture spatial-temporal dependencies and improve interpretability.

## Features

- Spatio-temporal forecasting architecture combining CNN and LSTM
  
- Temporal attention mechanism to highlight influential time steps
  
- Sliding window preprocessing of multivariate time-series data
  
- Residual attention fusion and normalisation for stable training
  
- Support for electricity load, solar PV, and wind generation inputs

## Project Structure

├── data/ # Raw and processed time-series data
├── models/ # Trained model checkpoints (optional)
├── src/ # Source code for model architecture and training
│ ├── preprocessing.py # Data normalisation and windowing
│ ├── stcalnet.py # Model architecture definition
│ ├── train.py # Training pipeline
│ ├── evaluate.py # Evaluation and metrics
├── results/ # Forecast outputs, plots, and error analyses
├── figures/ # Visualisations used in the paper
├── README.md # This file
├── requirements.txt # Python dependencies
└── main.py # Entry point script to train and evaluate the model
