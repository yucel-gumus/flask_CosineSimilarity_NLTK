import pandas as pd 
import ast
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Verisetini yükle kaggle da var 
movies = pd.read_csv('tmdb_5000_movies.csv') 
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'genres', 'cast', 'overview', 'keywords', 'crew']]
movies.dropna(inplace=True)

# Genres, keywords, cast, crew ve overview sütunlarındaki verileri düzenle
def transform(obj):
    return [item['name'] for item in ast.literal_eval(obj)]

def transform1(obj):
    return [item['name'] for i, item in enumerate(ast.literal_eval(obj)) if i < 3]

def fetch_director(text):
    return [item['name'] for item in ast.literal_eval(text) if item['job'] == 'Director']

movies['genres'] = movies['genres'].apply(transform)
movies['keywords'] = movies['keywords'].apply(transform)
movies['cast'] = movies['cast'].apply(transform1)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] 

# Gerekli sütunları seç
df = movies[['movie_id', 'title', 'tags']]

df.loc[:, 'tags'] = df['tags'].apply(lambda x: " ".join(x))
df.loc[:, 'tags'] = df['tags'].apply(lambda x: x.lower())

# nltk.download('punkt') yalnızca bir kez çalıştırılır
if 'punkt' not in nltk.data.path:
    nltk.download('punkt')

ps = PorterStemmer()
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(df['tags'])
feature_names = cv.get_feature_names_out()

# Her kelimenin kökünü al ve benzersiz kökleri bul
stemmed_feature_names = [" ".join([ps.stem(word) for word in word_tokenize(feature)]) for feature in feature_names]
unique_stemmed_feature_names = list(set(stemmed_feature_names))

# Cosine Similarity hesapla
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_lower = movie.lower()
    df.loc[:, 'title_lower'] = df['title'].str.lower()
    movie_index1 = df[df['title_lower'] == movie_lower].index

    if not movie_index1.empty:
        movie_index1 = movie_index1[0]
        distances = similarity[movie_index1]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
        # Benzer filmleri listele
        for i in movies_list:
            print(df.iloc[i[0]].title)
        
    df.drop('title_lower', axis=1, inplace=True)  # Drop the temporary column

# 'thor' filmine benzer filmleri öner
recommend('thor')

import pickle
pickle.dump(df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))