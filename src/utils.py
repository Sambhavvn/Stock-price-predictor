import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_lstm_data(df, feature_cols, target_col, sequence_length=50):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    X, y = [], []

    for i in range(sequence_length, len(df)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df[target_col].iloc[i])

    return np.array(X), np.array(y), scaler