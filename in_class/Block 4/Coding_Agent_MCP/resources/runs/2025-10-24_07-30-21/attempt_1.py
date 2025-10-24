import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/passenger_frequencies.csv"
data = pd.read_csv(file_path)

# Filter data for Zurich HB station
zurich_hb_data = data[data["Bahnhof_Gare_Stazione"] == "Zürich HB"]

# Sort data by year
zurich_hb_data = zurich_hb_data.sort_values(by="Jahr_Annee_Anno")

# Calculate the yearly change in passenger flow (DTV_TJM_TGM)
zurich_hb_data["Yearly_Change"] = zurich_hb_data["DTV_TJM_TGM"].diff()

# Save the result to a CSV file
zurich_hb_data.to_csv("result.csv", index=False)

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