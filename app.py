# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# MUAT DATA & MODEL
# ----------------------------
@st.cache_resource
def load_data():
    # Muat data film
    df = pd.read_csv("processed_movies.csv")
    
    # Muat matriks kemiripan (hasil SBERT + genre)
    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    
    # Buat mapping judul → index
    indices = pd.Series(df.index, index=df['clean_title'])
    
    return df, cosine_sim, indices

df, cosine_sim, indices = load_data()

# ----------------------------
# FUNGSI REKOMENDASI
# ----------------------------
def get_recommendations(title, top_n=5):
    if title not in indices.index:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:top_n+1]]  # skip diri sendiri
    return df.iloc[movie_indices]

# ----------------------------
# TAMPILAN STREAMLIT
# ----------------------------
st.set_page_config(page_title="🎥 Movie Recommender", layout="wide")
st.title("🎥 Movie Recommender System")
st.markdown("Rekomendasi film berdasarkan **kemiripan isi (genre + sinopsis)**.")

# Pilih film
title_input = st.selectbox(
    "Pilih film favoritmu:",
    df['clean_title'].tolist(),
    index=0
)

# Tampilkan rekomendasi
if title_input:
    recs = get_recommendations(title_input, top_n=5)
    if recs is not None:
        st.subheader(f"Rekomendasi untuk: **{title_input}**")
        for _, row in recs.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                if pd.notna(row['poster_path']):
                    poster_url = f"https://image.tmdb.org/t/p/w300{row['poster_path']}"
                    st.image(poster_url, width=140)
                else:
                    st.write("🖼️")
            with col2:
                st.markdown(f"### {row['clean_title']} ({row['year']})")
                st.markdown(f"**Genre**: {row['genres_movielens']}")
                st.text(row['overview'][:200] + "...")
            st.divider()

st.markdown("---")
st.caption("Dibuat dengan MovieLens + TMDB API | Rekomendasi Berbasis Konten: SBERT + Genre")