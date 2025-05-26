import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply, Permute, Reshape
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Load and scale dataset ===
df = pd.read_csv("C:/Users/cavus/Desktop/Electronics-Adib/data_for_energyy.csv")
df['time'] = pd.to_datetime(df['time'])
df = df[['consumption', 'pv_production', 'wind_production']].dropna()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# === Sequence creation ===
def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === Attention Module ===
def attention_block(inputs):
    # Shape: (batch, time, features)
    attention = GlobalAveragePooling1D()(inputs)       # (batch, features)
    attention = Dense(inputs.shape[1], activation='softmax')(attention)  # (batch, time)
    attention = RepeatVector(inputs.shape[2])(attention)  # (batch, features, time)
    attention = Permute([2, 1])(attention)              # (batch, time, features)
    attended = Multiply()([inputs, attention])
    return attended, attention

# === ST-CALNet with custom attention ===
inputs = Input(shape=(X.shape[1], X.shape[2]))
x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
x = LSTM(64, return_sequences=True)(x)
attended, attention_weights = attention_block(x)
x = LSTM(32)(attended)
out = Dense(1)(x)
model = Model(inputs, out)
model.compile(optimizer=Adam(0.001), loss='mse')

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)

# Attention model for output
attention_model = Model(inputs, attention_weights)

# === Figure 6: Attention Line Plot for One Sample ===
att_single = attention_model.predict(X_test[42:43])[0]  # shape: (24, features)
att_single_avg = np.mean(att_single, axis=1)            # average over features

plt.figure(figsize=(10, 4), dpi=600)
plt.plot(range(1, 25), att_single_avg, marker='o', linestyle='-')
plt.title("Figure 6: Temporal Attention – One Prediction Sequence", fontsize=14)
plt.xlabel("Hour in Historical Window (t-24 → t-1)", fontsize=12)
plt.ylabel("Attention Weight", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/cavus/Desktop/Electronics-Adib/Figure_6_Attention_Line_One_Sample.png")
plt.close()

# === Figure 7: Violin Plot for All Time Steps ===
all_att_weights = []
for i in range(300):  # loop over multiple test sequences
    att = attention_model.predict(X_test[i:i+1])[0]  # shape: (24, features)
    att_avg = np.mean(att, axis=1)  # mean per time step
    all_att_weights.append(att_avg)

all_att_weights = np.array(all_att_weights)  # shape: (300, 24)
df_violin = pd.DataFrame(all_att_weights, columns=[f"t-{23 - i}" for i in range(24)])
df_melted = df_violin.melt(var_name="Time Step", value_name="Attention Score")

plt.figure(figsize=(12, 6), dpi=600)
sns.violinplot(x="Time Step", y="Attention Score", data=df_melted, scale="width", inner="quartile")
plt.title("Figure 7: Attention Score Distribution Across Time Steps", fontsize=14)
plt.xlabel("Time Step (t-23 to t)", fontsize=12)
plt.ylabel("Attention Score", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("C:/Users/cavus/Desktop/Electronics-Adib/Figure_7_Attention_Score_Violin.png")
plt.close()
