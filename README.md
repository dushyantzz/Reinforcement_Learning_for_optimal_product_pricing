# Reinforcement Learning for Optimal Product Pricing

## Overview
This project implements a Reinforcement Learning (RL) module to predict the optimal price for a product that maximizes sales while also rewarding higher price points. The RL model balances exploration (testing new price points) and exploitation (optimizing based on historical data) to find the best pricing strategy.

## Problem Statement
Given historical sales data with product prices, conversion rates, and total sales, the goal of this RL model is to:
- Predict the ideal product price for the next day.
- Increase product prices while maintaining or improving total sales.
- Prevent the model from getting stuck at historically optimal prices (e.g., if $16 was the best price in the past, it should attempt to push beyond $16 to find potentially better price points).

## Dataset Description
The dataset contains the following columns:
- **Report Date:** The date for which the data point belongs.
- **Product Price:** The price at which the product was sold.
- **Organic Conversion Percentage:** Conversion rate for organic traffic.
- **Ad Conversion Percentage:** Conversion rate for ad-driven traffic.
- **Total Profit:** Total profit made on that day at the given price.
- **Total Sales:** Number of units sold at that price.
- **Predicted Sales:** Forecasted sales for future dates at the same price point (useful for RL simulation).

## Approach
### 1. Data Preprocessing
- Load and clean the dataset.
- Normalize features using **MinMaxScaler** to ensure stable RL training.
- Extract key features: `Product Price`, `Total Sales`, `Organic Conversion`, and `Ad Conversion`.

### 2. Environment Setup
- Implement a custom **OpenAI Gym environment** for price optimization.
- Define the **state space**: Historical price, sales, conversion rates, and predicted sales.
- Define the **action space**: Adjusting the product price up or down.
- Implement a **reward function** based on:
  - Higher total sales (rewarded positively).
  - Higher conversion rates (optional but helpful for tuning the model).
  - Higher product prices (encouraging price increases).
  - Punishment for sales drop compared to predictions.

### 3. Reinforcement Learning Model
- Implement a **Deep Q-Network (DQN)** or **Proximal Policy Optimization (PPO)** agent.
- Train the agent using historical data as the simulated environment.
- Balance **exploration vs exploitation**:
  - Exploration: Testing new price points.
  - Exploitation: Optimizing based on best historical price points.

### 4. Training and Evaluation
- Train the RL model over multiple episodes to find the best pricing strategy.
- Evaluate performance using:
  - Sales increase over baseline historical sales.
  - Conversion rate improvements.
  - Revenue and profit growth.
- Visualize results with **Matplotlib**.

### 5. Deployment
- The trained model predicts the optimal price for the next day.
- Integrate with a pricing engine for real-time decision-making.

## Reward & Punishment System
### Rewards:
- Higher sales than historical median → Higher reward.
- Increased product price without a drop in sales → Reward.
- Higher organic/ad conversion rates → Additional reward.

### Punishments:
- Sales lower than predicted → Penalty.
- Price drop leading to lower total revenue → Penalty.

## Example Pricing Simulation
| Input Price | Output Price |
|-------------|-------------|
| $14.0       | $14.8       |
| $15.0       | $14.7       |
| $15.7       | $16.2       |
| $16.2       | $16.6       |
| $16.6       | $16.5       |
| $16.5       | $16.6       |

## 🛠️ Project Structure
├── Reinforcement_Learning_project.ipynb # Main Jupyter notebook containing the implementation ├── ppo_model.pkl # Saved model in pickle format ├── ppo_model.zip # Compressed model file └── README.md # Project documentation


## 🎯 Features
- Implementation of PPO algorithm for price optimization
- Dynamic pricing strategy learning
- Market simulation environment
- Model persistence and portability
- Interactive Jupyter notebook implementation

## 🔧 Technical Stack
- Python
- PyTorch (for deep learning)
- Stable Baselines3 (for RL implementation)
- Jupyter Notebook
- Pandas (for data handling)
- NumPy (for numerical computations)

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required Python packages:
  ```bash
  pip install stable-baselines3 torch numpy pandas jupyter
