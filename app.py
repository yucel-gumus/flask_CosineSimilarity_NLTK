from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

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
        similar_movies = []
        for i in movies_list:
            similar_movie_title = df.iloc[i[0]].title
            similarity_score = i[1]
            similar_movies.append({"title": similar_movie_title, "score": similarity_score})

        return similar_movies

    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    movie_title = None
    similar_movies = None

    if request.method == 'POST':
        user_input = request.form['movie_title']

        # Verileri yükle
        df, similarity = load_data()

        # Benzer filmleri öner
        similar_movies = recommend_similar_movies(user_input, df, similarity)
        movie_title = user_input

    return render_template('index.html', movie_title=movie_title, similar_movies=similar_movies)

if __name__ == '__main__':
    app.run(debug=True)
