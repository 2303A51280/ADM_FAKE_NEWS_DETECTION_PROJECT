# 📰 Fake News Detection Project

This is a simple web application that predicts whether a news article or headline is *Real* or *Fake, built using **Python, **Flask, **scikit-learn, and **HTML/CSS*.

---
## 📌 Features
- Upload any news text and get instant prediction.
- Beautiful gradient background that changes color based on prediction:
  - ✅ Green for *Real* news
  - ❌ Red for *Fake* news
- Clean and responsive design.
- Lightweight and fast Flask server.
---
## ⚙ Technologies Used

- Python 3
- Flask
- scikit-learn
- Pandas
- HTML5
- Inline CSS Styling
---
## 📂 Project Structure

project-folder/

│

├── app.py             # Flask backend server

├── train_model.py     # Script to train and save the machine learning model

├── model.pkl          # Trained Logistic Regression model (saved using pickle)

├── vectorizer.pkl     # Saved TfidfVectorizer (used for text transformation)

├── train.csv          # Dataset used for training the model

│
└── templates/

    └── index.html     # Frontend HTML page with inline CSS styling
