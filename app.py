from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news_clean = clean(news)
    vec = vectorizer.transform([news_clean])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    label = "Real 📰" if prediction == 1 else "Fake ⚠️"
    return render_template("index.html", prediction_text=label, prob=round(prob * 100, 2), review=news)

if __name__ == '__main__':
    app.run(debug=True)
