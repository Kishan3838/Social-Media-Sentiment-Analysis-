from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

# Load and preprocess data
df = pd.read_csv('proper tweet data.csv')
data = {'text': df['selected_text'], 'sentiment': df['sentiment']}
df = pd.DataFrame(data)
df = df.dropna()
df1 = df[df['text'].str.contains(r'^[a-zA-Z\s]*$')]

# Train the model
X = df1['text']
y = df1['sentiment']
tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getCleanedText(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    clean_text = " ".join(stemmed_tokens)
    return clean_text

X_clean = [getCleanedText(i) for i in X]
cv = CountVectorizer(ngram_range=(1, 2))
X_vec = cv.fit_transform(X_clean).toarray()

mn = MultinomialNB()
mn.fit(X_vec, y)

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        cleaned_text = getCleanedText(text)
        text_vectorized = cv.transform([cleaned_text]).toarray()

        # Get probabilities of each class
        proba = mn.predict_proba(text_vectorized)

        # Get predicted class
        prediction = mn.predict(text_vectorized)

        # Combine predicted class and confidence scores
        classes = mn.classes_
        results = [{'sentiment': cls, 'confidence': round(score * 100, 2)} for cls, score in zip(classes, proba[0])]

        return jsonify({'prediction': prediction[0], 'confidence_scores': results})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
	