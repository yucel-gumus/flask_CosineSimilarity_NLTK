import pickle

def load_data():
    # movie_list.pkl ve similarity.pkl dosyalarını yükle
    df = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))

    # 'title_lower' sütununu oluştur
    df['title_lower'] = df['title'].str.lower()

    return df, similarity

def recommend_similar_movies(movie_title, df, similarity):
    # Kullanıcının girdiği film adını küçük harfe çevir
    movie_title_lower = movie_title.lower()

    # DataFrame'den film indexini bul
    movie_index = df[df['title_lower'] == movie_title_lower].index

    if not movie_index.empty:
        movie_index = movie_index[0]

        # Benzerlik matrisinden ilgili satırı çek
        distances = similarity[movie_index]

        # Benzer filmleri sırala ve en yakın 5 filmi seç
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        # Benzer filmleri ekrana yazdır
        print(f"Similar movies to {df.iloc[movie_index].title}:")
        for i in movies_list:
            similar_movie_title = df.iloc[i[0]].title
            similarity_score = i[1]
            print(f"{similar_movie_title} - Similarity Score: {similarity_score}")

    else:
        print(f"Movie '{movie_title}' not found in the dataset.")

# Verileri yükle
df, similarity = load_data()
# Kullanıcıdan bir film adı al ve benzer filmleri öner
user_input = input("Enter a movie title: ")
recommend_similar_movies(user_input, df, similarity)
