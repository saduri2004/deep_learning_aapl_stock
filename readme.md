# Deep Learning Trading Bot

## Overview

I developed this project as a deep learning-based trading bot that uses a two-stage time series forecasting model to predict future stock prices for Apple Inc. (AAPL). The model combines the feature extraction capabilities of a pre-trained ResNet-50 Convolutional Neural Network (CNN) with the sequential learning power of a Long Short-Term Memory (LSTM) network.

The approach consists of two main steps:

1. **Feature Extraction**: Use ResNet-50 to extract meaningful features from sequences of stock price data represented as images.
2. **Time Series Forecasting**: Feed the extracted features into an LSTM network to predict future stock prices.

## Project Structure

The key stages of the project include:

- Data Collection
- Data Preprocessing
- Image Generation for Feature Extraction
- Model Creation and Training
- Evaluation and Visualization of Predictions

## Requirements

The following Python packages are required to run this project:

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- yfinance
- opencv-python

To install all dependencies, you can use the following command:

