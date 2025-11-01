# 🎬 MovieLens-Lite Collaborative Filtering  
### *A Hands-On Recommender System Built from Scratch*  
**Author:** [Denis Naumov](https://github.com/DenisNaumov7777)

---

## 📖 Overview

This project implements a **Collaborative Filtering Recommender System** completely from scratch —  
no `scikit-learn`, no `surprise`, just **NumPy + gradient descent**.  

The goal is to show how movie recommendations (like Netflix or Spotify) actually work under the hood —  
how both **user preferences** and **movie features** are *learned simultaneously* from rating data.

---

## 🧠 Core Idea

We assume we have a set of users and movies, but no predefined “movie features” (like *romance*, *action*, etc).  
Instead, the algorithm **learns** hidden features automatically — for example:

| Hidden Feature 1 | Hidden Feature 2 |
|------------------|------------------|
| Romance ❤️ | Action 💥 |

Each user and movie are represented as vectors in the same latent feature space.  
The predicted rating is the dot product between these vectors:

$$
\hat{y}^{(i,j)} = w^{(j)} \cdot x^{(i)} + b^{(j)}
$$
---

## 🧩 Cost Function

We minimize the mean squared error for all user–movie pairs with known ratings:

\[
J(X, W, b) =
\frac{1}{2}
\sum_{(i,j):r(i,j)=1}
(w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2
+ \frac{\lambda}{2}
\left(
\sum_i ||x^{(i)}||^2 +
\sum_j ||w^{(j)}||^2
\right)
\]

where  
- \( X \): learned movie features  
- \( W \): learned user preferences  
- \( b \): user biases  
- \( \lambda \): regularization factor  

---

## 🚀 Training

We optimize all parameters (`X`, `W`, and `b`) via **batch gradient descent**,  
tracking the cost function to ensure convergence.

---

## 🧰 Technologies Used

- 🐍 Python 3.11+
- 📦 NumPy
- 📊 Matplotlib
- 💡 Jupyter-ready structure for easy visualization

---

## 📂 Project Structure

MovieLens-Lite/
│
├── collaborative_filtering.py # Core training & visualization script
├── README.md # Project documentation (this file)
└── requirements.txt # Dependencies (NumPy, Matplotlib)


---

## 💻 How to Run

```bash
# Clone repository
git clone https://github.com/DenisNaumov7777/movielens-lite-recommender.git
cd MovieLens-Lite

# Install dependencies
pip install -r requirements.txt

# Run training
python collaborative_filtering.py

📈 Visualizations

During training, the notebook displays:

Cost function over epochs (training convergence)

Learned movie feature space (romance vs action)

User preference space (each user’s learned taste)

Predicted ratings matrix

These visuals make the hidden structure of the data fully interpretable.

🧩 Example Output
Movie	Alice	Bob	Carol	Dave
Love at Last	4.8	4.9	0.3	0.6
Romance Forever	4.7	4.2	0.5	1.1
Cute Puppies	4.5	3.8	0.6	0.9
Car Chases	0.2	0.4	5.0	4.7
Sword vs Karate	0.1	0.3	4.8	4.5
🧮 Mathematical Summary
Symbol	Meaning

𝑌
Y	User–movie rating matrix

𝑅
R	Indicator matrix (1 if rating exists)

𝑋
X	Learned movie feature matrix

𝑊
W	Learned user preference matrix

𝑏
b	User bias vector

𝐽
J	Cost function to minimize

𝜆
λ	Regularization term

𝛼
α	Learning rate
🧑‍💻 Author

👋 Denis Naumov
AI Engineer • Data Scientist • ML Researcher

GitHub: @DenisNaumov7777

Location: Cologne, Germany 🇩🇪
