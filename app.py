import ast
import hashlib
import pickle
import urllib.parse
import numpy as np
import pandas as pd
import requests
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Hugging Face Repository Configuration
# ---------------------------------------------------------------------------
# Apna Hugging Face repo ID yahan update karein (e.g., "username/movie-recommender")
HF_REPO_ID= "shivam878/movie-recommender-model"


# ---------------------------------------------------------------------------
# Movie Recommender Pipeline Definition
# ---------------------------------------------------------------------------
class MovieRecommenderPipeline:

    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None

    def load_from_pickle(self, movies_dict_path: str, similarity_path: str):
        # Pickle files load karein
        with open(movies_dict_path, "rb") as f:
            movies_dict = pickle.load(f)
            # Support if pickled object is dict or DataFrame
            if isinstance(movies_dict, dict):
                self.movies_df = pd.DataFrame(movies_dict)
            else:
                self.movies_df = movies_dict

        with open(similarity_path, "rb") as f:
            self.similarity_matrix = pickle.load(f)

    def recommend(self, movie_title: str, top_n: int = 5) -> list:
        if self.similarity_matrix is None or self.movies_df is None:
            raise ValueError("Model is not loaded properly.")

        movie_title_lower = movie_title.lower()
        matching_indices = self.movies_df[
            self.movies_df["title"].str.lower() == movie_title_lower
        ].index

        if len(matching_indices) == 0:
            return []

        index = matching_indices[0]
        distances = self.similarity_matrix[index]
        movies_list = sorted(
            list(enumerate(distances)), key=lambda x: x[1], reverse=True
        )[1 : top_n + 1]

        recommendations = []
        for i in movies_list:
            recommendations.append(self.movies_df.iloc[i[0]].title)

        return recommendations


# ---------------------------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Design tokens
BG = "#101013"
SURFACE = "#17171B"
BORDER = "#26262C"
TEXT = "#ECEBE8"
MUTED = "#8B8A90"
ACCENT = "#E8A23B"

PLACEHOLDER_TONES = [
    "#3A2E22",
    "#22303A",
    "#2E2238",
    "#233A2A",
    "#3A2229",
    "#2A2A3A",
]

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: {BG}; color: {TEXT}; }}
        #MainMenu, header, footer {{ visibility: hidden; }}
        .block-container {{ padding-top: 2rem; max-width: 1100px; }}

        .topbar {{
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            border-bottom: 1px solid {BORDER};
            padding-bottom: 1rem;
            margin-bottom: 1.6rem;
        }}
        .topbar .mark {{ display: inline-flex; gap: 3px; }}
        .topbar .mark span {{
            width: 6px; height: 6px; border-radius: 50%;
            background: {ACCENT}; display: inline-block;
        }}
        .topbar .mark span:nth-child(2) {{ opacity: 0.7; }}
        .topbar .mark span:nth-child(3) {{ opacity: 0.4; }}
        .topbar h1 {{ font-size: 1.15rem; font-weight: 700; margin: 0; letter-spacing: 0.2px; }}
        .topbar .tag {{ color: {MUTED}; font-size: 0.85rem; }}

        div[data-baseweb="select"] > div {{
            background-color: {BG} !important;
            border-color: {BORDER} !important;
            color: {TEXT} !important;
            border-radius: 4px;
        }}

        .stButton > button {{
            background: {ACCENT};
            color: #1A1206;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1.4rem;
            font-weight: 600;
            width: 100%;
        }}
        .stButton > button:hover {{ filter: brightness(1.08); color: #1A1206; }}

        .section-label {{ font-size: 0.8rem; color: {MUTED}; margin: 1.5rem 0 1rem 0.1rem; }}

        a.movie-link {{
            text-decoration: none !important;
            color: inherit !important;
            display: block;
            transition: transform 0.2s ease;
        }}
        a.movie-link:hover {{
            transform: translateY(-4px);
        }}

        .poster {{
            border-radius: 4px;
            overflow: hidden;
            background: {SURFACE};
            border: 1px solid {BORDER};
            position: relative;
        }}
        .poster img {{
            width: 100%;
            display: block;
            aspect-ratio: 2 / 3;
            object-fit: cover;
        }}
        .poster .fallback {{
            width: 100%;
            aspect-ratio: 2 / 3;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            font-weight: 700;
            color: {TEXT};
        }}
        .poster .rating {{
            position: absolute;
            top: 6px;
            right: 6px;
            background: rgba(16,16,19,0.85);
            border: 1px solid {BORDER};
            color: {ACCENT};
            font-size: 0.72rem;
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 3px;
        }}

        .poster-title {{ margin-top: 0.5rem; font-size: 0.85rem; line-height: 1.3; color: {TEXT}; font-weight: 600; }}
        .poster-meta {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.15rem; }}

        .selected-preview {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 1rem;
            margin-top: 1rem;
            display: flex;
            gap: 1.2rem;
            align-items: center;
        }}
        .selected-preview img {{
            width: 80px;
            height: 120px;
            border-radius: 4px;
            object-fit: cover;
        }}

        .empty-state {{ color: {MUTED}; font-size: 0.9rem; padding: 1rem 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def fallback_tone(title: str) -> str:
    idx = int(hashlib.md5(title.encode()).hexdigest(), 16) % len(
        PLACEHOLDER_TONES
    )
    return PLACEHOLDER_TONES[idx]


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_details(title: str, api_key: str = ""):
    imdb_url = (
        f"https://www.imdb.com/find/?q={urllib.parse.quote(title)}&s=tt&exact=true"
    )

    if api_key:
        try:
            search = requests.get(
                f"{TMDB_BASE}/search/movie",
                params={"api_key": api_key, "query": title},
                timeout=4,
            )
            search.raise_for_status()
            results = search.json().get("results") or []
            if results:
                movie = results[0]
                direct_imdb = (
                    f"https://www.imdb.com/title/{movie.get('imdb_id')}"
                    if movie.get("imdb_id")
                    else imdb_url
                )

                return {
                    "poster_url": (
                        f"{TMDB_IMG}{movie['poster_path']}"
                        if movie.get("poster_path")
                        else None
                    ),
                    "rating": movie.get("vote_average"),
                    "year": (movie.get("release_date") or "")[:4],
                    "imdb_url": direct_imdb,
                }
        except Exception:
            pass

    try:
        omdb_resp = requests.get(
            f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey=trilogy",
            timeout=4,
        )
        if omdb_resp.status_code == 200:
            data = omdb_resp.json()
            if data.get("Response") == "True":
                poster = (
                    data.get("Poster") if data.get("Poster") != "N/A" else None
                )
                imdb_id = data.get("imdbID")
                direct_imdb = (
                    f"https://www.imdb.com/title/{imdb_id}/"
                    if imdb_id
                    else imdb_url
                )
                return {
                    "poster_url": poster,
                    "rating": (
                        float(data.get("imdbRating"))
                        if data.get("imdbRating") != "N/A"
                        else None
                    ),
                    "year": data.get("Year"),
                    "imdb_url": direct_imdb,
                }
    except Exception:
        pass

    return {"poster_url": None, "rating": None, "year": None, "imdb_url": imdb_url}


def render_poster(title: str, details: dict):
    imdb_link = details.get("imdb_url") if details else "#"

    if details and details.get("poster_url"):
        media = f'<img src="{details["poster_url"]}" alt="{title}">'
    else:
        tone = fallback_tone(title)
        initial = title.strip()[0].upper() if title.strip() else "?"
        media = (
            f'<div class="fallback" style="background:{tone};">{initial}</div>'
        )

    rating_badge = ""
    if details and details.get("rating"):
        rating_badge = f'<div class="rating">★ {details["rating"]:.1f}</div>'

    meta_line = ""
    if details and details.get("year"):
        meta_line = f'<div class="poster-meta">{details["year"]}</div>'

    st.markdown(
        f"""
        <a href="{imdb_link}" target="_blank" class="movie-link" title="Click to view {title} on IMDb">
            <div class="poster">{media}{rating_badge}</div>
            <div class="poster-title">{title}</div>
            {meta_line}
        </a>
        """,
        unsafe_allow_html=True,
    )


# Hugging Face hub se download aur caching setup
@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Model 1: Movies list/dict download
    movies_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="movie_dict.pkl"
    )
    # Model 2: Similarity matrix download
    similarity_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="similarity.pkl"
    )

    pipeline = MovieRecommenderPipeline()
    pipeline.load_from_pickle(movies_path, similarity_path)
    return pipeline


# ---------------------------------------------------------------------------
# App Execution
# ---------------------------------------------------------------------------
inject_css()

with st.sidebar:
    st.caption("Enter TMDB Key for official HD posters & details.")
    tmdb_key = st.text_input(
        "TMDB API key", type="password", label_visibility="collapsed"
    )

st.markdown(
    """
    <div class="topbar">
        <span class="mark"><span></span><span></span><span></span></span>
        <h1>Recommender</h1>
        <span class="tag">— pick one film, find five more</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Downloading & loading models from Hugging Face..."):
    pipeline = load_pipeline()

col_select, col_btn = st.columns([5, 1])
with col_select:
    movie_list = pipeline.movies_df["title"].values
    selected_movie = st.selectbox(
        "Your favorite movie", movie_list, label_visibility="collapsed"
    )
with col_btn:
    recommend_clicked = st.button("Recommend")

if selected_movie:
    selected_details = fetch_movie_details(selected_movie, tmdb_key)
    poster_src = selected_details.get("poster_url")
    preview_img = (
        f'<img src="{poster_src}">'
        if poster_src
        else f'<div class="fallback" style="background:{fallback_tone(selected_movie)}; width:80px; height:120px; border-radius:4px; display:flex; align-items:center; justify-content:center; font-weight:700;">{selected_movie[0]}</div>'
    )

    st.markdown(
        f"""
        <div class="selected-preview">
            {preview_img}
            <div>
                <span style="font-size: 0.75rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.5px;">Currently Selected</span>
                <h3 style="margin: 0.2rem 0; font-size: 1.1rem; color: {TEXT};">{selected_movie}</h3>
                <a href="{selected_details['imdb_url']}" target="_blank" style="color: {ACCENT}; font-size: 0.8rem; text-decoration: none;">View on IMDb ↗</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if recommend_clicked:
    recommendations = pipeline.recommend(selected_movie, top_n=5)

    if not recommendations:
        st.markdown(
            '<div class="empty-state">No close matches found — try a different title.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="section-label">Because you picked something similar (Click card to open IMDb)</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(recommendations))
        for col, movie in zip(cols, recommendations):
            with col:
                details = fetch_movie_details(movie, tmdb_key)
                render_poster(movie, details)