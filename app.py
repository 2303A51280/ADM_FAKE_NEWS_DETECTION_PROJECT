from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    result = "Fake" if prediction == 1 else "Real"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
