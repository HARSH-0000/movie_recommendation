import pandas as pd
import pickle
import gzip
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk

app = Flask(__name__)

# Load the movie list
with gzip.open('movie_list.pkl', 'rb') as f:
    movie_dict = pickle.load(f)
movies = pd.DataFrame(movie_dict)

@app.route('/')
def index():
    return render_template('index.html', movies=movies['title'].values)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get('movie')

    # On-the-fly model generation
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    new_df = movies.copy()
    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=1000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    movie_index = new_df[new_df['title'] == selected_movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)

    return render_template('recommend.html', movie=selected_movie, recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)