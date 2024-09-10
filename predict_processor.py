import os
import logging
from sklearn.exceptions import NotFittedError
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Wczytywanie i przygotowanie danych
def load_and_prepare_data(data_filepath, feature_columns, rookies=False):
    # Wczytywanie danych
    logging.info(f"Wczytywanie danych z pliku: {data_filepath}")
    nba_data = pd.read_csv(data_filepath)

    # Sprawdzenie brakujących cech
    if not set(feature_columns).issubset(nba_data.columns):
        missing_features = list(set(feature_columns) - set(nba_data.columns))
        logging.error(f"Brakuje następujących cech w danych: {missing_features}")
        raise ValueError(f"Brak wymaganych cech: {missing_features}")

    # Filtrowanie graczy
    target_column = "ANBARK" if rookies else "ANBA"
    filtered_players = nba_data.query('RK == @rookies')

    # Sprawdzenie pustego zestawu
    if filtered_players.empty:
        logging.warning("Nie znaleziono graczy spełniających kryteria.")

    labels = filtered_players[target_column]
    logging.info(f"Liczba graczy: {len(filtered_players)}")

    # Skalowanie cech
    features_matrix = filtered_players[feature_columns]
    scaler = StandardScaler()

    try:
        # Skalowanie danych
        scaled_features = scaler.fit_transform(features_matrix)
    except NotFittedError as e:
        logging.error(f"Nie udało się dopasować skalera: {e}")
        raise

    logging.info(f"Dane zostały znormalizowane. Liczba cech: {len(feature_columns)}")
    return scaled_features, labels


# Trenowanie modelu Random Forest
def train_random_forest(X_data, y_labels):

    logging.info("Rozpoczynanie trenowania modelu Random Forest...")

    # Tworzenie modelu
    rf_model = RandomForestClassifier(random_state=21)

    # Siatka hiperparametrów
    hyperparameters = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 5, 15]
    }

    # Optymalizacja hiperparametrów
    hyperparameter_search = GridSearchCV(rf_model, hyperparameters, cv=5, scoring='accuracy')
    hyperparameter_search.fit(X_data, y_labels)

    # Wybór najlepszego modelu
    optimized_model = hyperparameter_search.best_estimator_

    # Zapis modelu
    model_path = "./model.pkl"
    with open(model_path, "wb") as output_model:
        pickle.dump(optimized_model, output_model)

    logging.info(f"Model został wytrenowany i zapisany jako {model_path}")
    return optimized_model


# Predykcja graczy
def predict_top_players(rookies=False, algorithm="random_forest"):

    features = ["G", "MP", "PER", "ORB%", "DRB%", "AST%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "BPM", "VORP"]

    # Przygotowanie danych treningowych
    start_time = time.time()
    logging.info("Rozpoczynanie przygotowania danych treningowych...")
    X_train, y_train = load_and_prepare_data("./data/nba_data.csv", features, rookies)

    # Wybór algorytmu
    logging.info(f"Wybór algorytmu: {algorithm}")
    if algorithm == "random_forest":
        model = train_random_forest(X_train, y_train)
    elif algorithm == "gradient_boosting":
        model = train_gradient_boosting(X_train, y_train)
    else:
        raise ValueError("Nieobsługiwany algorytm: wybierz 'random_forest' lub 'gradient_boosting'.")

    # Czas trenowania
    training_duration = time.time() - start_time
    logging.info(f"Czas trenowania modelu: {training_duration:.2f} sekund")

    # Wczytywanie danych 2024
    logging.info("Wczytywanie danych z sezonu 2024...")
    X_2024, _ = load_and_prepare_data("./data/nba_data_2024.csv", features, rookies)

    # Dokonywanie predykcji
    logging.info("Dokonywanie predykcji dla sezonu 2024...")
    start_prediction_time = time.time()
    predictions = model.predict(X_2024)
    prediction_duration = time.time() - start_prediction_time

    logging.info(f"Czas dokonania predykcji: {prediction_duration:.2f} sekund")

    # Pobieranie danych o graczach
    nba_data_2024 = pd.read_csv("./data/nba_data_2024.csv")
    if rookies:
        players_2024 = nba_data_2024[nba_data_2024["RK"] == 1]
    else:
        players_2024 = nba_data_2024[nba_data_2024["RK"] == 0]

    # Gracze wybrani do zespołów
    predicted_players = players_2024[predictions == 1]["Player"].tolist()

    # Wyświetlanie liczby graczy
    logging.info(f"Liczba wczytanych graczy w 2024 roku: {len(nba_data_2024)}")
    logging.info(f"Liczba przewidzianych graczy: {len(predicted_players)}")

    # Uzupełnianie braków rookies
    if rookies and len(predicted_players) < 10:
        logging.warning("Zbyt mała liczba debiutantów, uzupełnianie braków...")
        missing_count = 10 - len(predicted_players)
        predicted_players += ["Brak"] * missing_count

    return predicted_players


# Trenowanie Gradient Boosting
def train_gradient_boosting(X_train, y_train):
    logging.info("Trenowanie modelu Gradient Boosting...")
    model = GradientBoostingClassifier(random_state=21)

    # Siatka parametrów
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10]
    }

    # Optymalizacja parametrów
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    # Zapis najlepszego modelu
    best_model = grid_search.best_estimator_
    with open("./model_gb.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)

    logging.info("Model Gradient Boosting został wytrenowany i zapisany jako ./model_gb.pkl")
    return best_model
