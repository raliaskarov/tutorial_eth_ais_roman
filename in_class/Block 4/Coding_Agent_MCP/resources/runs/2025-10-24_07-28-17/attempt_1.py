import pandas as pd

# Load the CSV file
file_path = '/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/fridge_inventory.csv'
df = pd.read_csv(file_path)

# Save the result to 'result.csv'
df.to_csv('result.csv', index=False)

# Print the dataframe to stdout
print(df)