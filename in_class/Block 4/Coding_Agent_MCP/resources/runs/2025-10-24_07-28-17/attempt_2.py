import pandas as pd

# Load the CSV file robustly
file_path = '/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/fridge_inventory.csv'
try:
    df = pd.read_csv(file_path)
except Exception:
    df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

# Normalize column names
df.columns = [str(c).strip() for c in df.columns]

# Rename columns to match the required names
df.rename(columns={"Item": "item", "Quantity": "quantity", "Expiry Date": "expiry_date"}, inplace=True)

# Assign to result_df
result_df = df

# Save the result to 'result.csv'
result_df.to_csv('result.csv', index=False)

# Print the dataframe to stdout
print(result_df)