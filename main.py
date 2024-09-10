import argparse
import json
import logging
from pathlib import Path
from predict_processor import predict_top_players

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Argumenty wejściowe
    parser = argparse.ArgumentParser(description="Generowanie wyników predykcji All-NBA i All-Rookie.")
    parser.add_argument("results_file", type=str, help="Ścieżka do pliku wynikowego .json")
    args = parser.parse_args()

    # Ścieżka pliku wynikowego
    results_file_path = Path(args.results_file)

    # Predykcja All-NBA
    logging.info("Rozpoczynanie predykcji graczy All-NBA...")
    all_nba_players = predict_top_players(rookies=False)

    # Predykcja All-Rookie
    logging.info("Rozpoczynanie predykcji graczy All-Rookie...")
    all_rookie_players = predict_top_players(rookies=True)

    # Dane wynikowe
    json_data = {
        "first all-nba team": all_nba_players[:5],
        "second all-nba team": all_nba_players[5:10],
        "third all-nba team": all_nba_players[10:15],
        "first rookie all-nba team": all_rookie_players[:5],
        "second rookie all-nba team": all_rookie_players[5:10],
    }

    try:
        # Sprawdzenie istnienia pliku
        if results_file_path.exists():
            logging.warning(f"Plik {results_file_path} istnieje, nadpisywanie...")
        else:
            logging.info(f"Tworzenie nowego pliku wynikowego: {results_file_path}")

        # Zapis pliku JSON
        with results_file_path.open("w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)

        logging.info(f"Wyniki zostały zapisane do pliku: {results_file_path}")

    # Obsługa błędów
    except Exception as e:
        logging.error(f"Błąd podczas zapisywania wyników: {e}")


if __name__ == "__main__":
    main()
