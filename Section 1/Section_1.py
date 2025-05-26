# -*- coding: utf-8 -*-
"""
Created on Sun May 25 15:20:44 2025

@author: cavus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load and sort dataset
df = pd.read_csv("C:/Users/cavus/Desktop/Electronics-Adib/data_for_energyy.csv")
df['time'] = pd.to_datetime(df['time'])
df.sort_values('time', inplace=True)

# Select target and predictors
features = ['consumption', 'pv_production', 'wind_production']
df = df[features].dropna()

# Normalise
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Sequence generator
def create_sequences(data, target_index, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])           # all features as input
        y.append(data[i+seq_len, target_index])  # one target
    return np.array(X), np.array(y)

# Train-evaluate-predict function
def train_and_evaluate(X, y, feature_name, target_index, fig_num):
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = LSTM(64, return_sequences=True)(x)
    query = Dense(64)(x)
    key = Dense(64)(x)
    value = Dense(64)(x)
    attention = Attention(use_scale=True)([query, key, value])
    x = LayerNormalization()(x + attention)
    x = LSTM(32)(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer=Adam(0.001), loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
              callbacks=[early_stop], verbose=0)

    y_pred = model.predict(X_test).flatten()

    # Evaluation (normalised)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{feature_name} - Normalised Evaluation Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R²:   {r2:.4f}")

    # Denormalise for plot
    range_val = scaler.data_range_[target_index]
    min_val = scaler.data_min_[target_index]
    y_test_real = y_test * range_val + min_val
    y_pred_real = y_pred * range_val + min_val

    # Plot
    plt.figure(figsize=(14, 6), dpi=600)
    plt.plot(y_test_real[:200], label='Actual', linewidth=2)
    plt.plot(y_pred_real[:200], label='Predicted', linestyle='--')
    plt.title(f"Figure {fig_num}: Actual vs Predicted – {feature_name}", fontsize=16)
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    save_path = f"C:/Users/cavus/Desktop/Electronics-Adib/Figure_{fig_num}_Actual_vs_Predicted_{feature_name.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.close()

# Sequence length
seq_len = 24

# Prepare data
X_cons, y_cons = create_sequences(scaled_data, target_index=0, seq_len=seq_len)
X_pv, y_pv = create_sequences(scaled_data, target_index=1, seq_len=seq_len)
X_wind, y_wind = create_sequences(scaled_data, target_index=2, seq_len=seq_len)

# Train and evaluate
train_and_evaluate(X_cons, y_cons, "Electricity Consumption", 0, fig_num=1)
train_and_evaluate(X_pv, y_pv, "PV Generation", 1, fig_num=2)
train_and_evaluate(X_wind, y_wind, "Wind Generation", 2, fig_num=3)
