import requests
import csv

# URLs
currencies_url = "https://gist.githubusercontent.com/nhalstead/4c1652563dd13357ab936fc97703c019/raw/d5de097ef68f37501fb4d06030ca49f10f5f963a/currency-symbols.json"
historical_data_url_template = "https://bank.gov.ua/NBU_Exchange/exchange_site?start=20100101&end=20241122&valcode={symbol}&sort=exchangedate&order=desc&json"

# Load all currency symbols
response = requests.get(currencies_url)
if response.status_code != 200:
    raise Exception("Failed to fetch currency symbols")

currencies = response.json()

# List to store historical data
all_historical_data = []

# Fetch historical data for each currency
for currency in currencies:
    symbol = currency.get("abbreviation")
    if not symbol:
        continue

    print(f"Fetching data for: {symbol}")
    response = requests.get(historical_data_url_template.format(symbol=symbol))
    if response.status_code != 200:
        print(f"Failed to fetch data for {symbol}")
        continue

    historical_data = response.json()
    for record in historical_data:
        record["currency"] = currency.get("currency", "Unknown")
        record["abbreviation"] = symbol
        all_historical_data.append(record)

# Write to CSV
csv_file = "currency_historical_data.csv"
csv_columns = [
    "currency", "abbreviation", "exchangedate", "r030", "cc", "txt",
    "enname", "rate", "units", "rate_per_unit", "group", "calcdate"
]

with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=csv_columns)
    writer.writeheader()
    for data in all_historical_data:
        writer.writerow(data)

print(f"Data has been saved to {csv_file}.")
