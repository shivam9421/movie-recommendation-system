import pickle
from movierecomender import MovieRecommenderPipeline

# Initialize and compute
pipeline = MovieRecommenderPipeline(
    movies_path="tmdb_5000_movies.csv", credits_path="tmdb_5000_credits.csv"
)
pipeline.load_and_preprocess()
pipeline.fit()

# Export precomputed objects
with open("movie_dict.pkl", "wb") as f:
    pickle.dump(pipeline.movies_df.to_dict(), f)

with open("similarity.pkl", "wb") as f:
    pickle.dump(pipeline.similarity_matrix, f)

import pickle
from movierecomender import MovieRecommenderPipeline
pipeline = MovieRecommenderPipeline(
    movies_path="tmdb_5000_movies.csv", credits_path="tmdb_5000_credits.csv"
)
pipeline.load_and_preprocess()
pipeline.fit()

# Export precomputed objects
with open("movie_dict.pkl", "wb") as f:
    pickle.dump(pipeline.movies_df.to_dict(), f)

with open("similarity.pkl", "wb") as f:
    pickle.dump(pipeline.similarity_matrix, f)


print("Pickle files successfully generated!")