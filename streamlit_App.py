import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from textblob import TextBlob
from ml_pipeline import get_sentiment, predict

st.set_page_config(page_title="BoxOfficeBot  🎬", page_icon = "🎥")
st.title("🎥 BoxOfficeBot : Movie Hit/Flop Predictor")

st.markdown("""
Welcome, 
Feed Me your - budget, runtime, release year and genre and I'll tell if your movie is destined for glory or doom
""")

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

@st.cache_data
def load_and_train_model():
    df = pd.read_csv(r'c:\Users\Administrator\Downloads\tmdb_5000_movies\tmdb_50000_movies.csv')
    
    
    df.dropna(subset=['release_date', 'revenue'], inplace=True)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['overview'] = df['overview'].fillna("")
    df['sentiment_score'] = df['overview'].apply(get_sentiment)
    df.dropna(subset=['release_year'], inplace = True)
    
    #Define Target
    threshold = df['revenue'].median()
    df['is_hit'] = (df['revenue'] >= threshold).astype(int)
    
    # Features
    features = ['budget', 'runtime', 'release_year', 'main_genre', 'sentiment_score']
    X = df[features]
    y = df['is_hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    numeric_feature = ['budget', 'runtime', 'release_year', 'sentiment_score']
    categorial_feature = ['main_genre']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]),numeric_feature),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]),categorial_feature)
    ])


    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    model.fit(X_train, y_train)
    return model

model = load_and_train_model()


st.subheader("Enter Movie Details")
budget = st.number_input("Bugdet ($)", min_value= 1000000, max_value=5000000, step=1000000)
runtime = st.slider("Runtime (minutes)", 60, 240, 120)
release_year = st.slider("Release Year", 1980, 2025, 2020)
genre = st.selectbox("Main Genre", ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama',
                                     'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

overview = st.text_area("Movie Overview (Short Description)")

#--- Prediction ---

if st.button("Predict"):
    sentiment_score = get_sentiment(overview)
    input_data = pd.DataFrame([{
        'budget': budget,
        'runtime': runtime,
        'release_year': release_year,
        'main_genre': genre,
        'sentiment_score': sentiment_score
    }])
    prediction = model.predict(input_data)[0]
    label = "🎉 HIT!" if prediction == 1 else "💔 FLOP!"
    st.success(f"Prediction: **{label}**")
    st.info(f"Sentiment Score (from overview): `{round(sentiment_score, 3)}`")


