# %%
import numpy as np 
import pandas as pd
from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity


def predict(model, input_df):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    return prediction, prediction_proba 

    



df = pd.read_csv(r'c:\Users\Administrator\Downloads\tmdb_5000_movies\tmdb_50000_movies.csv')

df.head()

# %%
print("Shape of the dataset", df.shape)
df.info()

# %%
print("Missisng values per columns")
print(df.isnull().sum())
df.describe()

df['sentiment_score'] = df['overview'].apply(get_sentiment)

# %%
cols_to_drop = ['homepage', 'tagline', 'keywords','production_countries', 'status']
df = df.drop(columns= cols_to_drop, errors='ignore')


#filling missing values
#For numerical columns like 'budget' or 'revenue', replace missing with 0   
df['budget'] = df['budget'].fillna(0)
df['revenue'] = df['revenue'].fillna(0)

df['release_date'] = pd.to_datetime(df['release_date'], errors = 'coerce')

# 4 Remove duplicates if any 
df = df.drop_duplicates()

# 5. features engineering example:
# Create 'profit' = revenue - budget
df['profit'] = df['revenue'] - df['budget']

#Create a target variable 'hit' (1 if revenue > budget, else 0)
df['hit'] = np.where(df['revenue'] > df['budget'], 1, 0)
# 6. Extract release month from release_date
df['release_month'] = df['release_date'].dt.month

# 7. Reset index
df= df.reset_index(drop = True)
print(df.head())

# %%
df['roi'] = df['profit'] / df['budget'].replace(0, np.nan)
df['roi'] = df['roi'].fillna(0)

df['is_big_budget'] = df['budget'].apply(lambda x: 1 if x >= 100000000 else 0)

df['popularity_bin'] = pd.qcut(df['popularity'], 4, labels = False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(x = 'budget', y = 'revenue', data = df)
plt.title('Bugdet vs Revenue')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.show()

# %%
sns.countplot(x='hit', data = df)
plt.title('Hit Vs Flop Distribution')
plt.xlabel('0 = Flop | 1 = Hit')
plt.ylabel('Number Of Movies')
plt.show()

# %%
import ast

df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
df['main_genre'] = df['genres'].apply(lambda x: x[0]['name'] if x else None)
df['main_genre'].fillna('Unknown', inplace=True)




genre_revenue = df.groupby('main_genre')['revenue'].mean().sort_values(ascending=False)


plt.figure(figsize=(12, 6))
genre_revenue.plot(kind = 'bar')
plt.title('Average Revenue by Genre')
plt.ylabel('Mean Revenue')
plt.xlabel('Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
df['release_year'] = df['release_date'].dt.year
df['release_year'].value_counts().sort_index().plot(kind = 'line', figsize = (10, 5))
plt.title('Number of Movies Released Each Year')
plt.xlabel('Year')
plt.ylabel('Movie Count')
plt.show()

# %%
#Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only = True), annot=True, cmap= 'coolwarm')
plt.title('Feature Correlation heatmap')
plt.show()

# %%
#Profit Distribution
sns.histplot(df['profit'], bins = 50, kde=True)
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.show()


# %%

threshold = df['revenue'].median()
df['is_hit'] = (df['revenue'] >= threshold.astype(int))

features = ['budget', 'runtime', 'release_year', 'main_genre']
X = df[features]
y = df['is_hit']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report


# Create the target variable
threshold = df['revenue'].median()
df['is_hit'] = (df['revenue'] >= threshold).astype(int)

# Define features and target
features = ['budget', 'runtime', 'release_year', 'main_genre', 'sentiment_score']
X = df[features]
y = df['is_hit']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['budget', 'runtime', 'release_year', 'sentiment_score']
categorial_features = ['main_genre']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ]), categorial_features)
])

# Full pipeline with Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the model
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
def explain_prediction_with_lime(model, X_train, X_test, numeric_features, categorical_features, instance_index=0):
    import lime
    import lime.lime_tabular

    # Transform the training and test data
    preprocessed_X_train = model.named_steps['preprocessor'].transform(X_train)
    preprocessed_X_test = model.named_steps['preprocessor'].transform(X_test)

    # Extract feature names from OneHotEncoder
    onehot = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    onehot_feature_names = onehot.get_feature_names_out(categorical_features)

    # Combine all final feature names
    final_feature_names = numeric_features + list(onehot_feature_names)

    # Access the classifier
    classifier = model.named_steps['classifier']

    # Create the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=preprocessed_X_train,
        feature_names=final_feature_names,
        class_names=['Flop', 'Hit'],
        mode='classification'
    )

    # Explain a specific instance
    explanation = explainer.explain_instance(
        data_row=preprocessed_X_test[instance_index],
        predict_fn=classifier.predict_proba
    )

    # Show explanation in notebook
    explanation.show_in_notebook(show_table=True)