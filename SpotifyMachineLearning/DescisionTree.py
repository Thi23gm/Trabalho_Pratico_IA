import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', encoding='ISO-8859-1')
    return df

def handle_missing_data(df):
    df.fillna({'streams': df['streams'].mean(), 'key': '0', 'mode': '0', 'season': '0', 'track_name': '0', 'artist(s)_name': '0'}, inplace=True)
    return df

def normalize_artist_names(df):
    def normalize_artist_name(artist_name):
        matches = re.search(r'([A-Z][a-z]+)\s([A-Z][a-z]+)', artist_name)
        if matches:
            return f"{matches.group(1)} {matches.group(2)}"
        else:
            return artist_name

    df['artist(s)_name'] = df['artist(s)_name'].apply(normalize_artist_name)
    return df

def encode_categorical_variables(df):
    df = pd.get_dummies(df, columns=['key', 'mode'])
    return df

def standardize_numeric_values(df, numeric_columns):
    for column in numeric_columns:
        if df[column].dtype == 'O':
            df[column] = df[column].str.replace(',', '.').astype(float)

    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def detect_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    outlier_indices = np.where(z_scores > threshold)[0]
    df = df.drop(outlier_indices, axis=0).reset_index(drop=True)
    return df

def create_popularity_column(df):
    df['Popularidade'] = np.where(df['streams'] > 0, '1', '0')
    return df

def preprocess_data(df):
    df = handle_missing_data(df)
    df = normalize_artist_names(df)
    df = encode_categorical_variables(df)

    numeric_columns = ['streams', 'in_deezer_playlists', 'artist_count', 'in_spotify_playlists', 'in_spotify_charts',
                       'in_apple_playlists', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts', 'bpm',
                       'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%',
                       'speechiness_%']
    
    df = standardize_numeric_values(df, numeric_columns)
    df = detect_outliers(df, 'streams')
    df = create_popularity_column(df)

    df['release_date'] = pd.to_datetime(df['released_year'].astype(str) + '-' + df['released_month'].astype(str) + '-' + df['released_day'].astype(str), format='%Y-%m-%d')

    df.drop(['released_year', 'released_month', 'released_day'], axis=1, inplace=True)

    return df

def main():
    file_path = '../DB/spotify.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    X = df.drop(['streams', 'Popularidade', 'track_name', 'artist(s)_name', 'release_date'], axis=1)
    y = df['Popularidade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test
    confusion_matrix(y_test, y_pred)
    cm = ConfusionMatrix(model)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Porcentagem de Acerto do modelo: {accuracy * 100: .2f}%")

    plt.figure(figsize=(20, 10))
    feature_names = list(X.columns)
    plot_tree(model, filled=True, feature_names=feature_names, class_names=['0', '1'], max_depth=3)

    plt.show()


if __name__ == "__main__":
    main()
