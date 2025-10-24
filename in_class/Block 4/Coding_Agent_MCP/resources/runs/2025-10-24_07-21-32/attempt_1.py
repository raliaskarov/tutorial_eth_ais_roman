import pandas as pd

# Load the CSV file
file_path = '/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/supermarket_sales.csv'
data = pd.read_csv(file_path)

# Find the most bought product
most_bought_product = data.groupby('Product line')['Quantity'].sum().idxmax()

# Save the result to a CSV file
result = pd.DataFrame({'Most Bought Product': [most_bought_product]})
result.to_csv('result.csv', index=False)