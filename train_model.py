import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

# Load the dataset
df = pd.read_csv("/content/FA-KES-Dataset.csv.zip", encoding='ISO-8859-1')

# Use article_content for classification
X = df['article_content']
y = df['labels']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
# The model should be fit with the transformed data (X_train_tfidf) not the text directly (X_train)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm=confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"confusion matrix:{cm}")
print("Classification Report:\n", report)

# Example: Take the first article in the dataset
example_text = df['article_content'].iloc[0]
print("News article:\n", example_text)

# Transform the text using the same TF-IDF vectorizer
example_vector = vectorizer.transform([example_text])

# Predict using the trained model
prediction = model.predict(example_vector)[0]
label = "Fake" if prediction == 1 else "Real"

print("\nPrediction:", label)

# Function to predict real or fake news
def predict_news(text):
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)[0]
    return "Fake" if prediction == 1 else "Real"

# Example: user inputs news content
user_input = input("Paste a news article or headline:\n")
result = predict_news(user_input)
print("\nPrediction:", result)
