import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/fridge_inventory.csv'
data = pd.read_csv(file_path)

# Plot the frequency of fridge consumption
plt.figure(figsize=(10, 6))
data.groupby('Item')['Quantity'].sum().sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Frequency of Fridge Consumption')
plt.xlabel('Item')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('plot.png')