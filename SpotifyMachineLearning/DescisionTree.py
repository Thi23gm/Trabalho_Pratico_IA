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

# Função para carregar os dados de um arquivo CSV
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', encoding='ISO-8859-1')
    return df

def handle_missing_data(df, columns_to_fill=None):
    if columns_to_fill is None:
        columns_to_fill = ['streams', 'key', 'mode', 'season', 'track_name', 'artist(s)_name']
    
    for column in columns_to_fill:
        if column in df.columns:
            # Calcular a moda (valor mais frequente) da coluna
            mode_value = df[column].mode().iloc[0]  # Obter o primeiro valor da moda (pode haver múltiplos valores)
            
            # Preencher os valores ausentes com a moda
            df[column].fillna(mode_value, inplace=True)
    
    return df


# Função para tratar dados faltantes em colunas numéricas
def handle_missing_data_numeric(df):
    numeric_columns = ['streams']

    # Preencher os valores ausentes nas colunas numéricas com a média da coluna
    for column in numeric_columns:
        df[column].fillna(df[column].mean(), inplace=True)

    return df

# Função para normalizar nomes de artistas
def normalize_artist_names(df):
    def normalize_artist_name(artist_name):
        matches = re.search(r'([A-Z][a-z]+)\s([A-Z][a-z]+)', artist_name)
        if matches:
            return f"{matches.group(1)} {matches.group(2)}"
        else:
            return artist_name

    df['artist(s)_name'] = df['artist(s)_name'].apply(normalize_artist_name)
    return df

# Função para codificar variáveis categóricas
def encode_categorical_variables(df):
    df = pd.get_dummies(df, columns=['key', 'mode'])
    return df

# Função para padronizar valores numéricos
def standardize_numeric_values(df, numeric_columns):
    for column in numeric_columns:
        if df[column].dtype == 'O':
            df[column] = df[column].str.replace(',', '.').astype(float)

    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Função para detectar e remover outliers
def detect_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    outlier_indices = np.where(z_scores > threshold)[0]
    df = df.drop(outlier_indices, axis=0).reset_index(drop=True)
    return df

# Função para criar uma coluna de "Popularidade" com base em streams
def create_popularity_column(df):
    # Adicionar 1 para Popular e 0 para Não Popular
    df['Popularidade'] = np.where(df['streams'] > 0, '1', '0')
    return df

# Função para porecessar os dados e chamar todos os outros
def preprocess_data(df):
    df = handle_missing_data(df)
    df = handle_missing_data_numeric(df)
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

# Função principal
def main():
    file_path = '../DB/spotify.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    X = df.drop(['streams', 'Popularidade', 'track_name', 'artist(s)_name', 'release_date'], axis=1) # Dados de texto
    y = df['Popularidade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X.to_csv('spotify_preprocessed.csv', index=False)
    y.to_csv('spotify_preprocessed.csv', index=False)

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
