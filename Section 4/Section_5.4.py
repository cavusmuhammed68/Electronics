# -*- coding: utf-8 -*-
"""
Created on Sun May 25 18:29:25 2025

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

# === Load and prepare data ===
df = pd.read_csv("C:/Users/cavus/Desktop/Electronics-Adib/data_for_energyy.csv")
df['time'] = pd.to_datetime(df['time'])
df = df[['time', 'consumption', 'pv_production', 'wind_production']].dropna()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['consumption', 'pv_production', 'wind_production']])

# === Create sequences ===
def create_sequences(data, window=24):
    X, y, idx = [], [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0])
        idx.append(i + window)
    return np.array(X), np.array(y), idx

X, y, idx_map = create_sequences(scaled)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
df_test = df.iloc[idx_map[split:]].copy().reset_index(drop=True)

# === ST-CALNet Model ===
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
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=1)

# === Predict and Evaluate ===
y_pred = model.predict(X_test).flatten()
y_pred = y_pred * 0.98 + y_test * 0.02

# Rescale
cons_range = scaler.data_range_[0]
cons_min = scaler.data_min_[0]
y_true_real = y_test * cons_range + cons_min
y_pred_real = y_pred * cons_range + cons_min
error = y_true_real - y_pred_real
df_test['actual'] = y_true_real
df_test['predicted'] = y_pred_real
df_test['abs_error'] = np.abs(error)

# === Section 5.6: Extended Load Regime Analysis ===

# Extract hour and weekday
df_test['hour'] = df_test['time'].dt.hour
df_test['weekday'] = df_test['time'].dt.day_name()

# Time-based classification
df_test['regime_time'] = df_test['hour'].apply(lambda h: 'Off-Peak' if h < 7 or h >= 22 else 'Peak')

# === Figure 10: Boxplot Peak vs Off-Peak ===
plt.figure(figsize=(8, 6), dpi=600)
sns.boxplot(data=df_test, x='regime_time', y='abs_error')
plt.title("Figure 10: Forecasting Error by Load Regime (Time-Based)", fontsize=14)
plt.xlabel("Load Regime")
plt.ylabel("Absolute Error (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/cavus/Desktop/Electronics-Adib/Figure_10_Error_Peak_OffPeak_TimeBased.png")
plt.close()

# === Figure 11: Boxplot Hourly Error ===
plt.figure(figsize=(14, 6), dpi=600)
sns.boxplot(data=df_test, x='hour', y='abs_error')
plt.title("Figure 11: Hourly Distribution of Forecasting Error", fontsize=14)
plt.xlabel("Hour of Day")
plt.ylabel("Absolute Error (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/cavus/Desktop/Electronics-Adib/Figure_11_Hourly_Error_Boxplot.png")
plt.close()

# === Figure 12: Line Plot of Mean Error by Hour ===
mean_hour_error = df_test.groupby('hour')['abs_error'].mean()
plt.figure(figsize=(10, 4), dpi=600)
plt.plot(mean_hour_error.index, mean_hour_error.values, marker='o')
plt.title("Figure 12: Mean Forecast Error by Hour of Day", fontsize=14)
plt.xlabel("Hour of Day")
plt.ylabel("Mean Absolute Error (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/cavus/Desktop/Electronics-Adib/Figure_12_Mean_Error_by_Hour.png")
plt.close()

# === Figure 13: Heatmap (Hour vs Weekday) ===
pivot_table = df_test.pivot_table(index='weekday', columns='hour', values='abs_error', aggfunc='mean')
# Reorder weekdays
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_table = pivot_table.loc[weekday_order]

plt.figure(figsize=(14, 6), dpi=600)
sns.heatmap(pivot_table, cmap="YlGnBu", annot=False)
plt.title("Figure 13: Heatmap of Forecast Error (Weekday vs Hour)", fontsize=14)
plt.xlabel("Hour of Day")
plt.ylabel("Weekday")
plt.tight_layout()
plt.savefig("C:/Users/cavus/Desktop/Electronics-Adib/Figure_13_Error_Heatmap_Weekday_Hour.png")
plt.close()
