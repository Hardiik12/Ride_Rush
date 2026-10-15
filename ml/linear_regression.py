"""
Simple Linear Regression with Gradient Descent Module for NYC Taxi Big Data.

Implements a custom SimpleLinearRegressionGD estimator trained via Gradient Descent,
evaluates model metrics (MSE, RMSE, R2), and generates regression diagnostic plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "charts")


class SimpleLinearRegressionGD:
    """
    Simple Linear Regression trained using Vectorized Gradient Descent.
    
    Model Hypothesis: y_hat = w * x + b
    Loss Function (MSE): J(w, b) = (1 / 2m) * sum((y_hat - y)^2)
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0.0  # Weight (Slope) for raw features
        self.b = 0.0  # Bias (Intercept) for raw features
        self.w_std = 0.0
        self.b_std = 0.0
        self.x_mean = 0.0
        self.x_std = 1.0
        self.cost_history = []

    def fit(self, X, y):
        """
        Fits linear regression parameters (w, b) using Gradient Descent on standardized X.

        Parameters:
        -----------
        X : numpy.ndarray (1D or 2D) - Feature values (e.g. trip_distance)
        y : numpy.ndarray (1D) - Target values (e.g. fare_amount)
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        m = len(y)

        # Standardize X for gradient descent numerical stability
        self.x_mean = np.mean(X)
        self.x_std = np.std(X)
        if self.x_std == 0:
            self.x_std = 1.0
        
        X_scaled = (X - self.x_mean) / self.x_std

        # Initialize parameters
        w = 0.0
        b = np.mean(y)
        self.cost_history = []

        for i in range(self.iterations):
            y_pred = w * X_scaled + b
            error = y_pred - y

            # Calculate gradients
            dw = (1.0 / m) * np.sum(error * X_scaled)
            db = (1.0 / m) * np.sum(error)

            # Update parameters
            w -= self.learning_rate * dw
            b -= self.learning_rate * db

            # Calculate Cost J(w, b)
            cost = (1.0 / (2.0 * m)) * np.sum(error ** 2)
            self.cost_history.append(float(cost))

        self.w_std = w
        self.b_std = b

        # Convert standardized parameters back to unscaled raw features scale:
        # y_hat = w_std * ((x - mu) / sigma) + b_std
        #       = (w_std / sigma) * x + (b_std - w_std * mu / sigma)
        self.w = float(self.w_std / self.x_std)
        self.b = float(self.b_std - (self.w_std * self.x_mean / self.x_std))

        return self

    def predict(self, X):
        """
        Predicts target values for given feature values X.
        """
        X = np.asarray(X, dtype=np.float64)
        return self.w * X + self.b

    def evaluate(self, X, y):
        """
        Computes evaluation metrics: MSE, RMSE, MAE, R2 score.
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)

        mse = float(np.mean((y - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y - y_pred)))

        # R^2 Score = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

        return {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4)
        }


def train_and_save_regression_model(df_clean, learning_rate=0.01, iterations=1000):
    """
    Trains Simple Linear Regression on trip_distance -> fare_amount,
    generates regression plots in static/charts/, and returns evaluation summary.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Sample 5% clean dataset for fast training & visualization
    sample_df = df_clean.sample(frac=0.05, random_state=42).compute()

    # Filter reasonable ranges for training stability
    sub = sample_df[(sample_df['trip_distance'] > 0) & (sample_df['trip_distance'] <= 30) &
                    (sample_df['fare_amount'] >= 2.5) & (sample_df['fare_amount'] <= 150)]

    X = sub['trip_distance'].values
    y = sub['fare_amount'].values

    # Train model
    model = SimpleLinearRegressionGD(learning_rate=learning_rate, iterations=iterations)
    model.fit(X, y)
    metrics = model.evaluate(X, y)

    # 1. Plot Regression Line Fit
    plt.figure(figsize=(9, 4.5))
    plt.scatter(X, y, alpha=0.25, color='#58a6ff', s=12, label="Actual Trips")

    x_line = np.linspace(0, 30, 100)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='#f0883e', linewidth=2.5, label=f"Fit: Fare = ${model.w:.2f} * Dist + ${model.b:.2f}")

    plt.title("Simple Linear Regression Fit: Trip Distance vs. Fare Amount", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Fare Amount ($)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "regression_fit.png"), dpi=200)
    plt.close()

    # 2. Plot Gradient Descent Cost Convergence
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, len(model.cost_history) + 1), model.cost_history, color='#3fb950', linewidth=2)
    plt.title("Gradient Descent Cost Function Convergence J(w, b)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Iterations")
    plt.ylabel("Cost J(w, b)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "cost_convergence.png"), dpi=200)
    plt.close()

    formula_str = f"Fare ($) = {model.w:.2f} × Distance + ${model.b:.2f}"

    return {
        "model_parameters": {
            "weight": round(model.w, 4),
            "bias": round(model.b, 4),
            "learning_rate": learning_rate,
            "iterations": iterations,
            "formula": formula_str
        },
        "metrics": metrics,
        "sample_size": len(X),
        "cost_initial": round(model.cost_history[0], 4),
        "cost_final": round(model.cost_history[-1], 4)
    }
