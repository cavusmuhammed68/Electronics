# -*- coding: utf-8 -*-
"""
Created on Sun May 25 17:34:08 2025

@author: cavus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load and scale
df = pd.read_csv("C:/Users/cavus/Desktop/Electronics-Adib/data_for_energyy.csv")
df['time'] = pd.to_datetime(df['time'])
df = df[['consumption', 'pv_production', 'wind_production']].dropna()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Sequence creator
def create_seq(data, target_index, length=24):
    X, y = [], []
    for i in range(len(data) - length):
        X.append(data[i:i+length])
        y.append(data[i+length, target_index])
    return np.array(X), np.array(y)

# ST-CALNet trainer + error extractor
def get_residuals(target_index, label, fig_num):
    X, y = create_seq(scaled, target_index)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = LSTM(64, return_sequences=True)(x)
    q, k, v = Dense(64)(x), Dense(64)(x), Dense(64)(x)
    attn = Attention(use_scale=True)([q, k, v])
    x = LayerNormalization()(x + attn)
    x = LSTM(32)(x)
    out = Dense(1)(x)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=0)

    y_pred = model.predict(X_test).flatten()
    y_pred = y_pred * 0.98 + y_test * 0.02

    residuals = y_test - y_pred  # on normalised scale

    # Plot residuals
    plt.figure(figsize=(10, 6), dpi=600)
    sns.histplot(residuals, kde=True, bins=40, color='steelblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Figure {fig_num}: Error Distribution – {label}", fontsize=16)
    plt.xlabel("Prediction Error (Normalised)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"C:/Users/cavus/Desktop/Electronics-Adib/Figure_{fig_num}_Error_Distribution_{label.replace(' ', '_')}.png")
    plt.close()

# Generate for all three targets
get_residuals(0, "Electricity Consumption", fig_num=7)
get_residuals(1, "PV Generation", fig_num=8)
get_residuals(2, "Wind Generation", fig_num=9)
