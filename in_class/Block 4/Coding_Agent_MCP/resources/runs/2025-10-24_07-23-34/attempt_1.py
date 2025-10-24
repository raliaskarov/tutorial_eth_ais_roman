import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/passenger_frequencies.csv"
data = pd.read_csv(file_path)

# Calculate the average passenger frequency per station
data['Average_Passengers'] = data[['DTV_TJM_TGM', 'DWV_TMJO_TFM', 'DNWV_TMJNO_TMGNL']].mean(axis=1)
average_passengers_per_station = data.groupby('Bahnhof_Gare_Stazione')['Average_Passengers'].mean().sort_values(ascending=False)

# Find the busiest station
busiest_station = average_passengers_per_station.idxmax()
busiest_station_avg = average_passengers_per_station.max()

# Save the result to a CSV
result_csv_path = "result.csv"
average_passengers_per_station.to_csv(result_csv_path, header=['Average_Passengers'])

# Plot the top 10 busiest stations
top_10_stations = average_passengers_per_station.head(10)
plt.figure(figsize=(10, 6))
top_10_stations.plot(kind='bar', color='skyblue')
plt.title('Top 10 Busiest Train Stations in Switzerland (Average Passengers per Year)')
plt.xlabel('Train Station')
plt.ylabel('Average Passengers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plot_path = "plot.png"
plt.savefig(plot_path)

# Save the answer text
answer_text = f"The busiest train station in Switzerland on average per year is {busiest_station} with an average of {busiest_station_avg:.2f} passengers."

['stdout', result_csv_path, plot_path, answer_text]