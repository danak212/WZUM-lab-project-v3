#### ======= ENGLISH BELOW =======

# Projekt - Przewidywania nagród NBA 2023/2024

## Opis

Projekt NBA Player Awards prediction służy do przewidywania najlepszych graczy w NBA na podstawie danych statystycznych. Projekt obejmuje predykcje dla graczy All-NBA oraz All-Rookie za pomocą algorytmów uczenia maszynowego.

## Struktura Projektu

- `main.py`: Główny skrypt uruchamiający program. Wykonuje predykcje dla graczy All-NBA i All-Rookie, zapisuje wyniki do pliku JSON.
- `predict_processor.py`: Moduł odpowiedzialny trenowanie modeli oraz wykonywanie predykcji. Obsługuje modele Random Forest i Gradient Boosting.
- `data_collector.py`: Skrypt do pobierania danych z serwisu Basketball Reference dla określonego sezonu NBA.
- `requirements.txt`: Lista wymaganych bibliotek Python.

## Zależności

- `matplotlib`
- `numpy`
- `scikit-learn~=1.5.1`
- `pandas~=2.2.2`
- `requests`
- `bs4~=0.0.2`
- `httpx~=0.27.0`
- `beautifulsoup4~=4.12.3`

## Instalacja

1. **Klonowanie repozytorium**:
   ```bash
   git clone <https://github.com/danak212/WZUM-lab-project-v3>
   cd <WZUM-lab-project-v3>
   ```

2. **Instalacja zależności**:
   Użyj poniższego polecenia, aby zainstalować wszystkie wymagane pakiety Python.
   ```bash
   pip install -r requirements.txt
   ```

## Uruchomienie

**Trenowanie modeli i wykonywanie predykcji**:
   Uruchom `main.py`, podając ścieżkę do pliku JSON, w którym mają zostać zapisane wyniki predykcji.
   ```bash
   python main.py results.json
   ```

## Składniki projektu

- **`predict_processor.py`**:
  - `load_and_prepare_data(data_filepath, feature_columns, rookies=False)`: Wczytuje dane, przygotowuje cechy i etykiety.
  - `train_random_forest(X_data, y_labels)`: Trenuje model Random Forest.
  - `train_gradient_boosting(X_train, y_train)`: Trenuje model Gradient Boosting.
  - `predict_top_players(rookies=False, algorithm="random_forest")`: Wykonuje predykcję graczy przy użyciu wybranego algorytmu.

- **`data_collector.py`**:
  - `NBADataSelection`: Klasa do pobierania i przetwarzania danych statystycznych z serwisu Basketball Reference.

## Opis szczegółowy wybranych funkcji

#### 1. **predict_processor.py** - `train_random_forest()`

```python
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
```

**Opis funkcji**:  
Funkcja ta trenuje model Random Forest przy użyciu zestawu danych wejściowych (`X_data`) oraz etykiet (`y_labels`). Najpierw definiuje model Random Forest, a następnie optymalizuje jego parametry (np. liczbę drzew i maksymalną głębokość) za pomocą siatki hiperparametrów (GridSearchCV). Po znalezieniu najlepszego modelu, jest on zapisany do pliku `model.pkl`, co umożliwia późniejsze wykorzystanie modelu bez konieczności ponownego trenowania.

---

#### 2. **predict_processor.py** - `predict_top_players()`

```python
def predict_top_players(rookies=False, algorithm="random_forest"):
    features = ["G", "MP", "PER", "ORB%", "DRB%", "AST%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "BPM", "VORP"]

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

    # Uzupełnianie braków rookies
    if rookies and len(predicted_players) < 10:
        logging.warning("Zbyt mała liczba debiutantów, uzupełnianie braków...")
        missing_count = 10 - len(predicted_players)
        predicted_players += ["Brak"] * missing_count

    return predicted_players
```

**Opis funkcji**:  
Funkcja ta dokonuje predykcji graczy All-NBA lub All-Rookie. Najpierw przygotowuje dane, wczytując je z pliku, a następnie trenuje wybrany model (Random Forest lub Gradient Boosting). Po trenowaniu, funkcja używa modelu do przewidywania najlepszych graczy na podstawie danych z sezonu 2024. Wyniki predykcji (listę wybranych graczy) zwraca jako listę. W przypadku, gdy jest zbyt mało debiutantów, funkcja uzupełnia listę brakami.

---

#### 3. **data_collector.py** - `fetch_data()`

```python
async def fetch_data(self):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.full_url)

        if response.status_code == 200:
            logging.info(f"Pobrano dane dla sezonu {self.season_year}.")
            return self._process_table(response.content, "advanced_stats")
        else:
            logging.error(f"Błąd podczas pobierania danych: HTTP {response.status_code}")

    except httpx.RequestError as err:
        logging.error(f"Błąd HTTP: {err}")
    except Exception as e:
        logging.error(f"Nieoczekiwany błąd: {e}")

    return pd.DataFrame()
```

**Opis funkcji**:  
Funkcja ta asynchronicznie pobiera dane statystyczne z serwisu Basketball Reference za pomocą HTTPx. W przypadku udanej odpowiedzi HTTP, przetwarza pobrane dane, wyciągając je z tabeli HTML (za pomocą `BeautifulSoup`) i zwraca je jako ramkę danych `pandas`. W przypadku problemów z połączeniem lub błędów HTTP, loguje odpowiednie komunikaty, a w razie niepowodzenia zwraca pustą ramkę danych.

## Podsumowanie

Projekt przewidywania nagród NBA na sezon 2023/2024 wykorzystuje algorytmy uczenia maszynowego, takie jak Random Forest i Gradient Boosting, do analizowania zaawansowanych statystyk graczy NBA. Dzięki temu możliwe jest przewidzenie, którzy zawodnicy zostaną wyróżnieni w zespołach All-NBA i All-Rookie. Projekt łączy w sobie zarówno elementy przetwarzania danych, jak i optymalizacji modeli, co pozwala na uzyskanie dokładniejszych wyników. Głównym celem jest automatyzacja procesu wyboru najlepszych graczy na podstawie ich osiągnięć statystycznych, a w przyszłości, dalsze udoskonalanie modelu może prowadzić do jeszcze bardziej precyzyjnych predykcji.

#
#

#### ======= ENGLISH VERSION =======

# Project - 2023/2024 NBA Awards prediction

## Description

The NBA Player Awards prediction project is designed to predict the top players in the NBA based on statistical data. The project includes predictions for All-NBA and All-Rookie players using machine learning algorithms.

## Project Structure

- `main.py`: The main script that runs the program. It generates predictions for All-NBA and All-Rookie players and saves the results to a JSON file.
- `predict_processor.py`: A module responsible for training models and making predictions. It supports Random Forest and Gradient Boosting models.
- `data_collector.py`: A script for fetching data from the Basketball Reference website for a specific NBA season.
- `requirements.txt`: A list of required Python libraries.

## Dependencies

- `matplotlib`
- `numpy`
- `scikit-learn~=1.5.1`
- `pandas~=2.2.2`
- `requests`
- `bs4~=0.0.2`
- `httpx~=0.27.0`
- `beautifulsoup4~=4.12.3`

## Installation

1. **Cloning the repository**:
   ```bash
   git clone <https://github.com/danak212/WZUM-lab-project-v3>
   cd <WZUM-lab-project-v3>
   ```

2. **Installing dependencies**:
   Use the following command to install all required Python packages.
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

**Training models and making predictions**:
   Run `main.py` by providing the path to the JSON file where prediction results will be saved.
   ```bash
   python main.py results.json
   ```

## Project Components

- **`predict_processor.py`**:
  - `load_and_prepare_data(data_filepath, feature_columns, rookies=False)`: Loads data, prepares features, and labels.
  - `train_random_forest(X_data, y_labels)`: Trains the Random Forest model.
  - `train_gradient_boosting(X_train, y_train)`: Trains the Gradient Boosting model.
  - `predict_top_players(rookies=False, algorithm="random_forest")`: Makes player predictions using the selected algorithm.

- **`data_collector.py`**:
  - `NBADataSelection`: A class for retrieving and processing statistical data from the Basketball Reference website.

## Detailed Description of Selected Functions

#### 1. **predict_processor.py** - `train_random_forest()`

```python
def train_random_forest(X_data, y_labels):
    logging.info("Starting training of the Random Forest model...")

    # Model creation
    rf_model = RandomForestClassifier(random_state=21)

    # Hyperparameter grid
    hyperparameters = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 5, 15]
    }

    # Hyperparameter optimization
    hyperparameter_search = GridSearchCV(rf_model, hyperparameters, cv=5, scoring='accuracy')
    hyperparameter_search.fit(X_data, y_labels)

    # Selecting the best model
    optimized_model = hyperparameter_search.best_estimator_

    # Saving the model
    model_path = "./model.pkl"
    with open(model_path, "wb") as output_model:
        pickle.dump(optimized_model, output_model)

    logging.info(f"The model has been trained and saved as {model_path}")
    return optimized_model
```

**Function Description**:  
This function trains a Random Forest model using input data (`X_data`) and labels (`y_labels`). It first defines a Random Forest model and then optimizes its parameters (e.g., the number of trees and maximum depth) using a hyperparameter grid (GridSearchCV). Once the best model is found, it is saved to a file `model.pkl` so that the model can be reused without retraining.

---

#### 2. **predict_processor.py** - `predict_top_players()`

```python
def predict_top_players(rookies=False, algorithm="random_forest"):
    features = ["G", "MP", "PER", "ORB%", "DRB%", "AST%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "BPM", "VORP"]

    # Loading 2024 data
    logging.info("Loading 2024 season data...")
    X_2024, _ = load_and_prepare_data("./data/nba_data_2024.csv", features, rookies)

    # Making predictions
    logging.info("Making predictions for the 2024 season...")
    start_prediction_time = time.time()
    predictions = model.predict(X_2024)
    prediction_duration = time.time() - start_prediction_time

    logging.info(f"Prediction time: {prediction_duration:.2f} seconds")

    # Fetching player data
    nba_data_2024 = pd.read_csv("./data/nba_data_2024.csv")
    if rookies:
        players_2024 = nba_data_2024[nba_data_2024["RK"] == 1]
    else:
        players_2024 = nba_data_2024[nba_data_2024["RK"] == 0]

    # Selected players
    predicted_players = players_2024[predictions == 1]["Player"].tolist()

    # Filling in missing rookies
    if rookies and len(predicted_players) < 10:
        logging.warning("Not enough rookies, filling in the gaps...")
        missing_count = 10 - len(predicted_players)
        predicted_players += ["None"] * missing_count

    return predicted_players
```

**Function Description**:  
This function predicts All-NBA or All-Rookie players. It first prepares data by loading it from a file, then trains the selected model (Random Forest or Gradient Boosting). After training, the function uses the model to predict the top players based on the 2024 season data. The prediction results (a list of selected players) are returned. If there are too few rookies, the function fills the list with placeholders.

---

#### 3. **data_collector.py** - `fetch_data()`

```python
async def fetch_data(self):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.full_url)

        if response.status_code == 200:
            logging.info(f"Data for the {self.season_year} season has been successfully fetched.")
            return self._process_table(response.content, "advanced_stats")
        else:
            logging.error(f"Error fetching data: HTTP {response.status_code}")

    except httpx.RequestError as err:
        logging.error(f"HTTP error: {err}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return pd.DataFrame()
```

**Function Description**:  
This function asynchronously fetches statistical data from the Basketball Reference website using HTTPx. In case of a successful HTTP response, it processes the data by extracting it from an HTML table (using `BeautifulSoup`) and returns it as a pandas DataFrame. If there are connection issues or HTTP errors, it logs the appropriate messages, and in case of failure, returns an empty DataFrame.

## Summary

The NBA Awards prediction project for the 2023/2024 season uses machine learning algorithms like Random Forest and Gradient Boosting to analyze advanced NBA player statistics. This makes it possible to predict which players will be selected for the All-NBA and All-Rookie teams. The project integrates both data processing and model optimization, allowing for more accurate results. The main goal is to automate the process of selecting top players based on their statistical achievements, with future improvements aimed at making predictions even more precise.