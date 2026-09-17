# 🎬 Movie Recommendation System

> Content-based movie recommender using NLP and cosine similarity | Dataset: TMDB 5000

## 📌 About
This project recommends 5 similar movies based on a given movie title.
It uses Natural Language Processing techniques — Porter Stemming and Cosine Similarity —
to find movies with similar genres, cast, crew, keywords, and overview.

## ⚙️ How It Works
1. Merged TMDB movies and credits datasets on movie title
2. Extracted useful features: genres, keywords, cast (top 3), director, overview
3. Combined all features into a single `tags` column
4. Applied **Porter Stemmer** to normalize words
5. Used **CountVectorizer** to convert tags into vectors (top 5000 features)
6. Computed **Cosine Similarity** between all movie vectors
7. Returns top 5 most similar movies for any given title

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-lightgreen?style=flat)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)


## 📂 Dataset
Dataset is not included due to file size limits.
Download it from Kaggle:
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

After downloading, place files in the `data/` folder:
- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`

## 📊 Example Output
```python
recommend('Newlyweds')
# Output: 5 most similar movies based on content
```

## 🚀 How to Run
1. Clone this repo
2. Upload both CSV files to your Google Drive
3. Open `Movie_recomender_system.ipynb` in Google Colab
4. Update the file paths to match your Drive location
5. Run all cells

## 📚 Reference
Project inspired by CampusX tutorial on content-based filtering.

---
⭐️ Star this repo if you found it helpful!
