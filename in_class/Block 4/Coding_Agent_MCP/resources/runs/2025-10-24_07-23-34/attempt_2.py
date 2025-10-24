import pandas as pd
import matplotlib.pyplot as plt

# Load the data robustly
file_path = "/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/passenger_frequencies.csv"
try:
    df = pd.read_csv(file_path)
except Exception:
    df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")
df.columns = [str(c).strip() for c in df.columns]

# Calculate the average passenger frequency per station
df['Average_Passengers'] = df[['DTV_TJM_TGM', 'DWV_TMJO_TFM', 'DNWV_TMJNO_TMGNL']].apply(pd.to_numeric, errors='coerce').mean(axis=1)
average_passengers_per_station = df.groupby('Bahnhof_Gare_Stazione')['Average_Passengers'].mean().sort_values(ascending=False).reset_index()

# Rename columns to match required output
average_passengers_per_station.columns = ['station', 'busiest_avg_per_year']

# Save the result to a CSV
result_df = average_passengers_per_station
result_df.to_csv("result.csv", index=False)

# Plot the top 10 busiest stations
top_10_stations = average_passengers_per_station.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10_stations['station'], top_10_stations['busiest_avg_per_year'], color='skyblue')
plt.title('Top 10 Busiest Train Stations in Switzerland (Average Passengers per Year)')
plt.xlabel('Train Station')
plt.ylabel('Average Passengers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plot.png")

# Save the answer text
busiest_station = top_10_stations.iloc[0]['station']
busiest_station_avg = top_10_stations.iloc[0]['busiest_avg_per_year']
answer_text = f"The busiest train station in Switzerland on average per year is {busiest_station} with an average of {busiest_station_avg:.2f} passengers."

['stdout', 'result.csv', 'plot.png', answer_text]