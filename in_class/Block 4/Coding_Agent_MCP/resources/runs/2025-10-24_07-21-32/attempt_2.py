import pandas as pd

# Load the CSV file robustly
file_path = '/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/supermarket_sales.csv'
try:
    df = pd.read_csv(file_path)
except:
    df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

# Normalize column names
df.columns = [str(c).strip() for c in df.columns]

# Ensure numeric conversion for Quantity
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

# Group by Product line and find the most bought product
most_bought_product = df.groupby('Product line')['Quantity'].sum().idxmax()
most_bought_quantity = df.groupby('Product line')['Quantity'].sum().max()

# Prepare the result DataFrame
result_df = pd.DataFrame({'Product': [most_bought_product], 'Quantity': [most_bought_quantity]})

# Save the result to a CSV file
result_df.to_csv('result.csv', index=False)