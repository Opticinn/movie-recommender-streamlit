# app.py
import streamlit as st
import pandas as pd
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
import numpy as np

# ----------------------------
# 1. SETUP & MUAT DATA
# ----------------------------
@st.cache_resource
def load_all_data():
    # Muat data film
    df = pd.read_csv("processed_movies.csv")
    
    # Muat cosine similarity (SBERT)
    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    
    # Muat model SVD
    with open("svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    
    # Muat mapping
    with open("movie_id_to_title.json", "r") as f:
        movie_id_to_title = json.load(f)
    with open("title_to_movie_id.json", "r") as f:
        title_to_movie_id = json.load(f)
    
    # Mapping index
    indices = pd.Series(df.index, index=df['clean_title'])
    
    return df, cosine_sim, svd_model, movie_id_to_title, title_to_movie_id, indices

df, cosine_sim, svd_model, movie_id_to_title, title_to_movie_id, indices = load_all_data()

# ----------------------------
# 2. FUNGSI REKOMENDASI
# ----------------------------
def get_content_recommendations(title, top_n=5):
    if title not in indices.index:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def get_hybrid_recommendations(user_id, top_n=5):
    # Ambil film yang disukai user (simulasi dari rating tinggi)
    # Di dunia nyata, ambil dari database
    # Untuk demo, asumsikan user suka "Toy Story" dan "Shrek"
    liked_titles = ["Toy Story", "Shrek"]
    candidate_movies = set()
    for title in liked_titles:
        if title in indices.index:
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            for i, _ in sim_scores[1:6]:
                candidate_movies.add(df.iloc[i]['movieId'])
    
    # Prediksi rating SVD
    scored = []
    for mid in candidate_movies:
        pred = svd_model.predict(user_id, mid).est
        scored.append((mid, pred))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    top_mids = [mid for mid, _ in scored[:top_n]]
    
    return df[df['movieId'].isin(top_mids)]

# ----------------------------
# 3. TAMPILAN STREAMLIT
# ----------------------------
st.set_page_config(page_title="🎥 Movie Recommender", layout="wide")
st.title("🎥 Movie Recommender System")
st.markdown("Rekomendasi film berdasarkan **kemiripan isi** atau **riwayat menontonmu**.")

tab1, tab2 = st.tabs(["Berdasarkan Judul Film", "Berdasarkan User ID"])

# Tab 1: Rekomendasi Berdasarkan Judul
with tab1:
    title_input = st.selectbox(
        "Pilih film favoritmu:",
        df['clean_title'].tolist(),
        index=0
    )
    if title_input:
        recs = get_content_recommendations(title_input, top_n=5)
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

# Tab 2: Rekomendasi Berdasarkan User ID
with tab2:
    user_id = st.number_input("Masukkan User ID", min_value=1, max_value=100000, value=1)
    if st.button("Dapatkan Rekomendasi"):
        recs = get_hybrid_recommendations(user_id, top_n=5)
        if not recs.empty:
            st.subheader(f"Rekomendasi Personal untuk User {user_id}")
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
        else:
            st.warning("Tidak ada rekomendasi untuk user ini.")

st.markdown("---")
st.caption("Dibuat dengan MovieLens + TMDB API | Sistem Hybrid: SBERT + SVD")