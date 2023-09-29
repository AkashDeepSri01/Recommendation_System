import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the movie dataset (you can replace this with your own dataset)
# Load the movie dataset (you can replace this with your own dataset)
# Load the movie dataset (you can replace this with your own dataset)
@st.cache_data
def load_data():
    file_path = r"D:\Projects\movie_recommender\tmdb_5000_movies.csv"
    df = pd.read_csv(file_path)
    return df


# Calculate cosine similarity between movie genres
def calculate_similarity(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Recommend movies based on user input
def recommend_movies(movie_title, cosine_sim, data):
    idx = data.loc[data['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

# Streamlit UI
def main():
    st.title("Movie Recommendation System")

    st.sidebar.header("User Input")
    movie_title = st.sidebar.selectbox("Select a movie:", data['title'])

    if st.sidebar.button("Recommend"):
        st.subheader("Recommended Movies:")
        recommendations = recommend_movies(movie_title, cosine_sim, data)
        st.write(recommendations)

if __name__ == "__main__":
    data = load_data()
    cosine_sim = calculate_similarity(data)
    main()
