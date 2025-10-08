# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
fake = pd.read_csv('C:\\Users\\Cap\\Downloads\\Fake.csv')
real = pd.read_csv('C:\\Users\\Cap\\Downloads\\True.csv')

fake['label'] = 0  # Fake = 0
real['label'] = 1  # Real = 1

data = pd.concat([fake[['text','label']], real[['text','label']]], axis=0)
data = data.sample(frac=1, random_state=42)  # Shuffle

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Function to predict new news
def predict_news(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return "Real" if prediction==1 else "Fake"

# Example
news = "The government has announced a new education policy."
print(f"News Prediction: {predict_news(news)}")
