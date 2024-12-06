
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    file_path = '/Users/liambowen/Desktop/principles_of_programming/hw9_Liam_Bowen/movies.csv'
    movies = pd.read_csv(file_path, nrows=20000)
    vectorizer = CountVectorizer()
    genre_vectors = vectorizer.fit_transform(movies['genres'])
    similarity_matrix = cosine_similarity(genre_vectors)
    return movies, similarity_matrix

movies, similarity_matrix= load_data()

def recommendation(movie_title, movies, similarity_matrix, num_recommendations = 4):
    try:
        movie_index = movies[movies['title'].str.contains(movie_title, case = False, na = False)].index[0]
    except IndexError:
        return ["Movie not found in the dataset"]
    
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    similar_movies = sorted(similarity_scores, key = lambda x: x[1], reverse = True)
    top_indices = [i[0] for i in similar_movies[1:num_recommendations + 1]]
    return movies.iloc[top_indices]['title'].tolist()

st.title('Movie Recommendation Algorithm')
movie_title = st.text_input("What is a movie that you enjoyed?")

if movie_title.strip():
    recommendations = recommendation(movie_title, movies, similarity_matrix)
    if "Movie not found in the dataset" in recommendations:
        st.write(recommendations[0])
    else:
        st.write("RECOMMENDATIONS")
        for idx, movie in enumerate(recommendations, start = 1):
            st.write(f"{idx}, {movie}")

st.dataframe(movies)
