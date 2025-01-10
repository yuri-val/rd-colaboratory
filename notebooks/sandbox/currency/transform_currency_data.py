import pandas as pd

# Read the CSV file
df = pd.read_csv("data/clear_currency_historical_data.csv", delimiter=";")

# Convert exchangedate to datetime
df["exchangedate"] = pd.to_datetime(df["exchangedate"], format="%d.%m.%Y")

# Aggregate duplicate entries by taking the mean of rate_per_unit
df_aggregated = df.groupby(["exchangedate", "abbreviation"], as_index=False).mean()

# Pivot the DataFrame
pivot_df = df_aggregated.pivot(
    index="exchangedate", columns="abbreviation", values="rate_per_unit"
)

# Sort the DataFrame by exchangedate
pivot_df = pivot_df.sort_index()

# Save the transformed DataFrame to a new CSV file
pivot_df.to_csv("data/transformed_currency_data.csv")

print(
    "Data transformation complete. The new file is saved as 'transformed_currency_data.csv'."
)
