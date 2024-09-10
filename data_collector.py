import pandas as pd
import httpx
from bs4 import BeautifulSoup
import logging
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NBADataSelection:
    # Inicjalizacja klasy
    def __init__(self, season_year, stat_category="advanced"):
        self.season_year = season_year
        self.stat_category = stat_category
        self.base_url = "https://www.basketball-reference.com/leagues/"
        self.full_url = self._build_url()

    # Tworzenie URL
    def _build_url(self):
        return f"{self.base_url}NBA_{self.season_year}_{self.stat_category}.html"

    # Przetwarzanie tabeli HTML
    def _process_table(self, html_content, table_id):
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table", {"id": table_id})

        if not table:
            logging.error(f"Tabela o ID '{table_id}' nie została znaleziona!")
            return pd.DataFrame()

        # Wyciąganie nagłówków i wierszy
        headers = [header.text.strip() for header in table.find_all("th")][1:]
        rows = [
            [cell.text.strip() for cell in row.find_all("td")]
            for row in table.find_all("tr") if row.find_all("td")
        ]

        # Logowanie liczby wierszy i kolumn
        if rows:
            logging.info(f"Pobrano {len(rows)} wierszy i {len(headers)} kolumn.")
        else:
            logging.error("Nie udało się znaleźć żadnych danych w tabeli!")

        return pd.DataFrame(rows, columns=headers)

    # Pobieranie danych
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


# Inicjalizacja selektora danych dla sezonu 2024
selector = NBADataSelection(2024)
data_frame = asyncio.run(selector.fetch_data())

# Wyświetlenie wyników
if not data_frame.empty:
    print(data_frame.head())
else:
    logging.error("Nie znaleziono danych dla wybranego sezonu!")
