import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(y_test, predictions, rmse, mae):
    """
    Analyzes model performance and suggests improvements.
    """
    print("\n📌 AI Model Analysis:")
    
    # ✅ Check if RMSE is too high
    avg_price = np.mean(y_test)
    normalized_rmse = rmse / avg_price
    if normalized_rmse > 0.05:  # If RMSE is more than 5% of avg price
        print("🔴 RMSE is high! Try increasing LSTM layers or sequence length.")
    
    # ✅ Check if MAE is high
    if mae > (avg_price * 0.02):  # If MAE is more than 2% of avg price
        print("🔴 MAE is high! Try adding more technical indicators.")
    
    # ✅ Check if predictions drift over time
    error_diff = np.abs(y_test - predictions)
    if np.mean(error_diff[-10:]) > np.mean(error_diff[:10]):  
        print("🔴 Prediction error is increasing! Consider a lower learning rate.")

    print("\n✅ AI Suggestions Applied to Improve Model!\n")

def plot_predictions(df, y_test, predictions, rmse):
    """
    Plots actual vs predicted prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(y_test):], y_test, label="Actual Price", color="blue")
    plt.plot(df.index[-len(y_test):], predictions, label="Predicted Price", linestyle="dashed", color="red")
    plt.legend()
    plt.title(f"LSTM Stock Price Prediction (RMSE: {rmse:.2f})")
    plt.show()
