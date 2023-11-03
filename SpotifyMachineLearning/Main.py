# pip install numpy pandas matplotlib scipy scikit-learn imbalanced-learn yellowbrick keras

import re
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

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

def balance_data(data, target_column_name):
    X = data.drop(target_column_name, axis=1)
    y = data[target_column_name]

    # Inicializar o RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)

    # Realizar oversampling
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Criar um novo DataFrame com dados balanceados
    balanced_data = X_resampled.copy()
    balanced_data[target_column_name] = y_resampled

    return balanced_data

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
    # Calcular a média dos valores na coluna 'streams'
    mean_value = df['streams'].mean()
    print(mean_value)

    # Usar np.where para atribuir '1' se maior que a média, '0' caso contrário
    df['Popularidade'] = np.where(df['streams'] > mean_value, '1', '0')
    
    return df


# Função para porecessar os dados e chamar todos os outros
def preprocess_data(df):
    df = handle_missing_data(df)
    df = handle_missing_data_numeric(df)
    df = normalize_artist_names(df)
    df = encode_categorical_variables(df)
    #df = data_balanced(df)

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
    file_path = '../dados/spotify.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    
    # Balanceamento dos dados
    df = balance_data(df, 'Popularidade')  # 'Popularidade' é a coluna alvo

    X = df.drop(['streams', 'Popularidade', 'track_name', 'artist(s)_name', 'release_date'], axis=1) # Dados de texto
    y = df['Popularidade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X.to_csv('spotify_preprocessed.csv', index=False)
    y.to_csv('spotify_preprocessed.csv', index=False)
    
    #------------------------ Arvore de Decisão ------------------------

    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test
    confusion_matrix(y_test, y_pred)
    cm = ConfusionMatrix(model)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    print("Relatório de Classificação (Arvore de Decisão):\n");
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Porcentagem de Acerto do modelo Arvore de Decisão: {accuracy * 100: .2f}%")

    plt.figure(figsize=(20, 10))
    feature_names = list(X.columns)
    plot_tree(model, filled=True, feature_names=feature_names, class_names=['0', '1'], max_depth=3)

    plt.show()

    # ------------------------ Random Forest ------------------------
    
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=3)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)


    cm_rf = ConfusionMatrix(rf_model)
    cm_rf.fit(X_train, y_train)
    cm_rf.score(X_test, y_test)
    print("Relatório de Classificação (Random Forest):\n", classification_report(y_test, y_pred_rf))

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Porcentagem de Acerto do modelo Random Forest: {accuracy_rf * 100: .2f}%")

    plt.figure(figsize=(20, 10))
    feature_names = list(X.columns)
    for estimator in rf_model.estimators_:
        plot_tree(estimator, filled=True, feature_names=feature_names, class_names=['0', '1'], max_depth=3)
 
    plt.show()

    # ------------------------ MPLs ------------------------ 

    # Tratando valores ausentes usando SimpleImputer (substituindo NaN pela média)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Treinando o modelo MLP
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)

    # Fazendo previsões com o modelo MLP
    y_pred_mlp = mlp_model.predict(X_test)

    # Mostrando resultados
    print("Relatório de Classificação (MLP):")
    print(classification_report(y_test, y_pred_mlp))

    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"Porcentagem de Acerto do modelo MLP: {accuracy_mlp * 100: .2f}%")

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(mlp_model.coefs_) - 1):
        ax.matshow(mlp_model.coefs_[i], cmap='viridis')

    ax.set_title('Estrutura da MLP')
    plt.show()

    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)

    plt.plot(mlp_model.loss_curve_)
    plt.title('Curva de Perda durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()

    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)

    # Supondo que 'mlp_model' seja o modelo treinado
    fig, axes = plt.subplots(1, len(mlp_model.coefs_), figsize=(15, 5))
    for i, coef in enumerate(mlp_model.coefs_):
        axes[i].imshow(coef, cmap='viridis', aspect='auto')
        axes[i].set_title(f'Camada {i+1} Pesos')
    plt.show()

    # ------------------------ Naive Bayes ------------------------
    
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, y_train)
    y_pred_nb = naive_bayes_model.predict(X_test)

    print("Relatório de Classificação (Naive Bayes):")
    print(classification_report(y_test, y_pred_nb))

    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Porcentagem de Acerto do modelo Naive Bayes: {accuracy_nb * 100: .2f}%")

    plt.figure(figsize=(20, 10))

    feature_names = list(X.columns)
    plt.figure(figsize=(20, 10))  # Definir o tamanho individual da figura para cada árvore
    plot_tree(estimator, filled=True, feature_names=feature_names, class_names=['0', '1'], max_depth=3)
    plt.show()  # Mostrar cada árvore separadamente

    conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

    # Exibição da matriz de confusão usando seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_nb, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matriz de Confusão - Naive Bayes')
    plt.show()

if __name__ == "__main__":
    main()
