import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data robustly
file_path = "/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/passenger_frequencies.csv"
if not os.path.exists(file_path):
    file_path = os.environ.get('CSV_PATH', file_path)

try:
    data = pd.read_csv(file_path)
except:
    data = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

# Normalize column names
data.columns = [str(c).strip() for c in data.columns]

# Filter data for Zurich HB station
zurich_hb_data = data[data["Bahnhof_Gare_Stazione"] == "Zürich HB"]

# Sort data by year
zurich_hb_data = zurich_hb_data.sort_values(by="Jahr_Annee_Anno")

# Calculate the yearly change in passenger flow (DTV_TJM_TGM)
zurich_hb_data["Yearly_Change"] = zurich_hb_data["DTV_TJM_TGM"].diff()

# Rename columns to match required names
zurich_hb_data.rename(columns={"Yearly_Change": "forecast_trend"}, inplace=True)

# Save the result to a CSV file
result_df = zurich_hb_data
result_df.to_csv("result.csv", index=False)

# Plot the trend of passenger flow change
plt.figure(figsize=(10, 6))
plt.plot(zurich_hb_data["Jahr_Annee_Anno"], zurich_hb_data["DTV_TJM_TGM"], marker='o', label="Passenger Flow (DTV_TJM_TGM)")
plt.title("Passenger Flow Trend at Zürich HB Station")
plt.xlabel("Year")
plt.ylabel("Passenger Flow (DTV_TJM_TGM)")
plt.grid()
plt.legend()
plt.savefig("plot.png")
plt.close()