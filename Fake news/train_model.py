import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load datasets
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df['content'] = df['title'] + " " + df['text']

# Text cleaning
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

df['clean_content'] = df['content'].apply(clean)

# Vectorize with reduced features to save space
X = df['clean_content']
y = df['label']
vectorizer = TfidfVectorizer(max_features=1000)  # was 5000, reduced to 1000
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train a lightweight model
model = LogisticRegression(solver='liblinear')  # smaller & faster
model.fit(X_train, y_train)

# Save model & vectorizer with compression
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl', compress=9)
joblib.dump(vectorizer, 'model/vectorizer.pkl', compress=9)

print(" Model and vectorizer saved successfully.")
