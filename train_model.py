import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Load the dataset
df = pd.read_csv("FA-KES-Dataset.csv", encoding='ISO-8859-1')  # no /content/ if local file

# 2. Prepare data
X = df['article_content']
y = df['labels']

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# 7. Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("\n✅ Model and Vectorizer saved successfully!")

