import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import matplotlib.pyplot as plt

def process_ais_data(file_path, mmsi_to_filter, sequence_length=5, epochs=50, batch_size=32):
    # Load AIS data into a DataFrame
    data = pd.read_csv(file_path)
    
    # Convert BaseDateTime to datetime format
    data['BaseDateTime'] = pd.to_datetime(data['BaseDateTime'])
    
    # Filter by a specific MMSI
    data = data[data['MMSI'] == mmsi_to_filter]
    
    # Sort data by timestamp
    data.sort_values(by='BaseDateTime', inplace=True)
    
    # Reset the index
    data.reset_index(drop=True, inplace=True)
    
    # Select relevant features and target variables
    features = ['LAT', 'LON', 'Heading']
    targets = ['SOG', 'COG']
    
    # Normalize the features and targets
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(data[features])
    targets_scaled = target_scaler.fit_transform(data[targets])
    
    # Create sequences for time series prediction
    def create_sequences(features, targets, sequence_length):
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(features_scaled, targets_scaled, sequence_length)
    
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=False, input_shape=(sequence_length, len(features))),
        Dense(2)  # Predicting SOG and COG
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    
    # Predict and inverse transform the results
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)
    y_test_actual = target_scaler.inverse_transform(y_test)
    
    # Calculate the prediction errors for SOG and COG
    sog_errors = y_test_actual[:, 0] - predictions[:, 0]  # SOG errors
    cog_errors = y_test_actual[:, 1] - predictions[:, 1]  # COG errors
    
    # Get the corresponding LAT, LON, and BaseDateTime values for the test set
    lat_lon_datetime_test = data[['LAT', 'LON', 'BaseDateTime']].iloc[train_size + sequence_length:].reset_index(drop=True)
    
    # Function to detect outliers using IQR
    def detect_outliers_iqr(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outlier_indices
    
    # Detect outliers for SOG and COG errors
    sog_outlier_indices = detect_outliers_iqr(sog_errors)
    cog_outlier_indices = detect_outliers_iqr(cog_errors)
    
    # Collect SOG outliers
    sog_outliers = lat_lon_datetime_test.iloc[sog_outlier_indices]
    
    # Collect COG outliers
    cog_outliers = lat_lon_datetime_test.iloc[cog_outlier_indices]
    
    return {
        'SOG_Outliers': sog_outliers[['LON', 'LAT', 'BaseDateTime']].reset_index(drop=True),
        'COG_Outliers': cog_outliers[['LON', 'LAT', 'BaseDateTime']].reset_index(drop=True)
    }
