import streamlit as st
import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    List = []
    for i in ast.literal_eval(obj):
        List.append(i['name'])
    return List

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    List = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter > 2:
            break
        else:
            List.append(i['name'])
            counter += 1
    return List

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    List = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            List.append(i['name'])
            break
    return List

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_movies

st.title("Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie:", new_df['title'].values)
if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("Movies similar to", selected_movie, "are:")
    for movie in recommendations:
        st.write(movie)