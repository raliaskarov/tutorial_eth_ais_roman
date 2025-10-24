import pandas as pd

# Load the fridge inventory CSV
file_path = '/home/jovyan/work/tutorial_eth_ais_roman/in_class/Block 4/Coding_Agent_MCP/resources/data/fridge_inventory.csv'
fridge_data = pd.read_csv(file_path)

# Add two Tomatoes to the fridge
if 'Tomatoes' in fridge_data['Item'].values:
    fridge_data.loc[fridge_data['Item'] == 'Tomatoes', 'Quantity'] += 2
else:
    new_row = {'Item': 'Tomatoes', 'Quantity': 2, 'Expiry Date': ''}
    fridge_data = pd.concat([fridge_data, pd.DataFrame([new_row])], ignore_index=True)

# Save the updated fridge data to result.csv
fridge_data.to_csv('result.csv', index=False)