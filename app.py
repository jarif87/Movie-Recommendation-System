import pandas as pd
from flask import Flask, render_template, request
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Define the clean_text function
def clean_text(text):
    cleaned_text = text.replace('\'', '').replace('[', '').replace(']', '').replace(',', '')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

# Define the stem function
def stem(text):
    ret = []
    words = text.split(" ")
    for word in words:
        ret.append(stemmer.stem(word))
    return " ".join(ret)

# Load the DataFrame
df = pd.read_csv("my_data.csv")  # Replace "your_dataframe.csv" with the actual file name/path

# Preprocess the text data
df = df[["movie_id", "title", "text"]]
df["text"] = df["text"].apply(clean_text)
stemmer = PorterStemmer()
df["text"] = df["text"].apply(stem)

# Vectorize the text data
cv = CountVectorizer(max_features=10000, stop_words="english")
data = cv.fit_transform(df["text"]).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(data)
sim_df = pd.DataFrame(similarity, index=df['title'], columns=df['title'])

# Define recommendation function
def make_recommendation(movie_name):
    # Selecting similar scores for the given movie
    similar_scores = sim_df[movie_name].sort_values(ascending=False)[1:11]
    
    # Merging similar scores with the main dataframe
    similar_movies = pd.DataFrame(similar_scores).merge(df, on="title")
    
    # Extracting recommended movie titles
    recommended_movies = list(similar_movies["title"])
    
    return recommended_movies

# Example usage of make_recommendation function
recommended_movies = make_recommendation("The Dark Knight")
print(recommended_movies)  # Print or return the recommended movies

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    recommendations = make_recommendation(movie_name)
    return render_template('recommendations.html', movie_name=movie_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
