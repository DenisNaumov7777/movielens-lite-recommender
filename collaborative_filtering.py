"""
🎬 MovieLens-Lite Collaborative Filtering
Author: Denis Naumov
GitHub: https://github.com/DenisNaumov7777
Description:
    Simple collaborative filtering recommender system implemented from scratch.
    Learns both user preferences and movie features using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt

# === 1️⃣ Data Setup ===
Y = np.array([
    [5, 5, 0, 0],  # Love at Last
    [5, 0, 0, 4],  # Romance Forever
    [0, 4, 0, 0],  # Cute Puppies of Love
    [0, 0, 5, 5],  # Nonstop Car Chases
    [0, 0, 5, 0],  # Sword vs Karate
])
R = (Y != 0).astype(int)

movies = ["Love at Last", "Romance Forever", "Cute Puppies of Love",
           "Nonstop Car Chases", "Sword vs Karate"]
users = ["Alice", "Bob", "Carol", "Dave"]

num_movies, num_users = Y.shape
num_features = 2
lambda_ = 0.1
alpha = 0.01
epochs = 2000

# === 2️⃣ Parameter Initialization ===
np.random.seed(42)
X = np.random.randn(num_movies, num_features)  # Movie features
W = np.random.randn(num_users, num_features)   # User preferences
b = np.zeros(num_users)                        # User biases

# === 3️⃣ Cost Function ===
def cost(X, W, b, Y, R, lambda_):
    pred = X @ W.T + b
    error = (pred - Y) * R
    J = 0.5 * np.sum(error**2)
    J += (lambda_/2) * (np.sum(W**2) + np.sum(X**2))
    return J

# === 4️⃣ Training Loop ===
cost_history = []
for epoch in range(epochs):
    pred = X @ W.T + b
    error = (pred - Y) * R

    X_grad = error @ W + lambda_ * X
    W_grad = error.T @ X + lambda_ * W
    b_grad = np.sum(error, axis=0)

    X -= alpha * X_grad
    W -= alpha * W_grad
    b -= alpha * b_grad

    J = cost(X, W, b, Y, R, lambda_)
    cost_history.append(J)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Cost: {J:.4f}")

# === 5️⃣ Visualize Cost Function ===
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color='blue')
plt.title("Cost Function over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost J(X, W, b)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6️⃣ Visualize Movie Feature Space ===
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=120, color='red', label='Movies')
for i, title in enumerate(movies):
    plt.text(X[i, 0] + 0.02, X[i, 1], title, fontsize=9)
plt.title("Learned Movie Feature Space")
plt.xlabel("Feature 1 (Romance)")
plt.ylabel("Feature 2 (Action)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 7️⃣ Visualize User Preference Space ===
plt.figure(figsize=(7, 5))
plt.scatter(W[:, 0], W[:, 1], s=120, color='green', label='Users')
for j, name in enumerate(users):
    plt.text(W[j, 0] + 0.02, W[j, 1], name, fontsize=9)
plt.title("Learned User Preference Space")
plt.xlabel("Feature 1 (Romance)")
plt.ylabel("Feature 2 (Action)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 8️⃣ Predictions ===
predicted_ratings = X @ W.T + b
print("\n🎬 Predicted Ratings Matrix:\n")
print(np.round(predicted_ratings, 2))
