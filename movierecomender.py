import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommenderPipeline:

    def __init__(self, movies_path: str = None, credits_path: str = None):
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies_df = None
        self.similarity_matrix = None
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    @staticmethod
    def _parse_list(obj_str: str) -> list:
        try:
            items = ast.literal_eval(obj_str)
            return [i["name"] for i in items if "name" in i]
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def _parse_cast(obj_str: str, limit: int = 3) -> list:
        try:
            items = ast.literal_eval(obj_str)
            return [i["name"] for i in items[:limit] if "name" in i]
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def _fetch_director(obj_str: str) -> list:
        try:
            items = ast.literal_eval(obj_str)
            for i in items:
                if i.get("job") == "Director":
                    return [i["name"]]
            return []
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def _collapse_spaces(L: list) -> list:
        return [i.replace(" ", "") for i in L]

    def load_from_pickle(
        self,
        movies_pkl_path: str = "movie_dict.pkl",
        similarity_pkl_path: str = "similarity.pkl",
    ):
        """Loads precomputed dataframe dictionary and similarity matrix."""
        with open(movies_pkl_path, "rb") as f:
            movies_dict = pickle.load(f)
            self.movies_df = pd.DataFrame(movies_dict)

        with open(similarity_pkl_path, "rb") as f:
            self.similarity_matrix = pickle.load(f)

    def load_and_preprocess(self):
        movies = pd.read_csv(self.movies_path)
        credits = pd.read_csv(self.credits_path)

        df = movies.merge(credits, on="title")
        df = df[
            [
                "movie_id",
                "title",
                "overview",
                "genres",
                "keywords",
                "cast",
                "crew",
            ]
        ].copy()
        df.dropna(inplace=True)

        df["genres"] = df["genres"].apply(self._parse_list)
        df["keywords"] = df["keywords"].apply(self._parse_list)
        df["cast"] = df["cast"].apply(lambda x: self._parse_cast(x, limit=3))
        df["crew"] = df["crew"].apply(self._fetch_director)
        df["overview"] = df["overview"].apply(
            lambda x: x.split() if isinstance(x, str) else []
        )

        df["genres"] = df["genres"].apply(self._collapse_spaces)
        df["keywords"] = df["keywords"].apply(self._collapse_spaces)
        df["cast"] = df["cast"].apply(self._collapse_spaces)
        df["crew"] = df["crew"].apply(self._collapse_spaces)

        df["tags"] = (
            df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]
        )

        self.movies_df = df[["movie_id", "title", "tags"]].copy()
        self.movies_df["tags"] = self.movies_df["tags"].apply(
            lambda x: " ".join(x).lower()
        )

    def fit(self):
        if self.movies_df is None:
            raise ValueError(
                "Data is not preprocessed. Call 'load_and_preprocess()' first."
            )

        vectors = self.vectorizer.fit_transform(self.movies_df["tags"]).toarray()
        self.similarity_matrix = cosine_similarity(vectors)

    def recommend(self, movie_title: str, top_n: int = 5) -> list:
        if self.similarity_matrix is None:
            raise ValueError(
                "Model is not fitted. Call 'fit()' or 'load_from_pickle()' first."
            )

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