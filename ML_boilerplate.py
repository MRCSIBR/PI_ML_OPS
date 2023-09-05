import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Cargar el dataset
df = pd.read_json('user_reviews.json')

# Step 2: Text Preprocessing (you can customize this as needed)
df['reviews'] = df['reviews'].str.lower()
df['reviews'] = df['reviews'].str.replace('[^a-zA-Z\s]', '')
df['reviews'] = df['reviews'].str.split()

# Step 3: Feature Extraction (using TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X = tfidf_vectorizer.fit_transform(df['reviews'].apply(lambda x: ' '.join(x)))

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42)

# Step 5: Train a Sentiment Analysis Model (using Naive Bayes as an example)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Step 7: Add Sentiment Analysis Results to the Dataset
df['sentiment_analysis'] = clf.predict(X)

# Save the updated dataset
df.to_json('user_reviews_with_sentiment.json', orient='records')
