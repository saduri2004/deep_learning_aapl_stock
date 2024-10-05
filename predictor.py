import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import cv2

# Download AAPL historical data
print("Downloading AAPL historical data...")
data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')
print("Data download complete.")
close_prices = data['Close']

# Normalize the data using MinMaxScaler
print("Normalizing the data...")
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(np.array(close_prices).reshape(-1, 1))
print("Data normalization complete.")

# Convert data into sequences for the ResNet-50 input (using image representation)
def create_sequences(data, window_size=30):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i: i + window_size])
    return np.array(sequences)

print("Creating sequences...")
window_size = 36  # Using 36 to create 6x6 images
sequences = create_sequences(close_prices_scaled, window_size)
print(f"Total sequences created: {len(sequences)}")

# Generate fake RGB images
print("Generating fake RGB images...")
# Reshape sequences into 6x6 images
image_size = int(np.sqrt(window_size))
images_2d = sequences.reshape(-1, image_size, image_size)

# Convert to 3 channels by stacking
images_rgb = np.stack((images_2d,)*3, axis=-1)  # Shape: (num_samples, image_size, image_size, 3)

# Resize images to (32, 32, 3)
images_resized = np.array([cv2.resize(img, (32, 32)) for img in images_rgb])
print(f"Generated images shape: {images_resized.shape}")

# Add the time dimension
images_resized = images_resized.reshape(images_resized.shape[0], 1, 32, 32, 3)
print(f"Images with time dimension added: {images_resized.shape}")

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
train_size = int(len(images_resized) * 0.8)
train_images, test_images = images_resized[:train_size], images_resized[train_size:]
train_labels = close_prices_scaled[window_size:train_size + window_size]
test_labels = close_prices_scaled[train_size + window_size:]
print(f"Training set size: {len(train_images)}, Testing set size: {len(test_images)}")

# Load the pre-trained ResNet-50 model without the top layer
print("Loading pre-trained ResNet-50 model...")
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
print("ResNet-50 model loaded.")

# Create a Sequential model
print("Creating Sequential model...")
model = Sequential()

# Add ResNet-50 as a feature extractor
print("Adding ResNet-50 as a feature extractor...")
model.add(TimeDistributed(resnet, input_shape=(None, 32, 32, 3)))
model.add(TimeDistributed(Flatten()))

# Add LSTM layer for forecasting
print("Adding LSTM layer...")
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

# Compile the model
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
print("Model compilation complete.")

# Train the model
print("Training the model...")
model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(test_images, test_labels))
print("Model training complete.")

# Make predictions
print("Making predictions...")
predictions = model.predict(test_images)
print("Predictions complete.")

# Inverse transform the predictions
print("Inverse transforming the predictions...")
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(test_labels)
print("Inverse transformation complete.")

# Plot the predictions vs actual values
print("Plotting the predictions vs actual values...")
plt.plot(actual, label='Actual AAPL Prices')
plt.plot(predictions, label='Predicted AAPL Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
print("Plotting complete.")