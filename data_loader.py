import pandas as pd
import os

def load_all_data():
    """
    Load all CSV files from the data folder
    Returns a dictionary with all dataframes
    """
    
    # Define the data folder path
    data_folder = 'data'
    
    # Dictionary to store all dataframes
    data = {}
    
    # List of all CSV files we need to load
    csv_files = {
        'orders': 'orders.csv',
        'delivery': 'delivery_performance.csv',
        'routes': 'routes_distance.csv',
        'fleet': 'vehicle_fleet.csv',
        'inventory': 'warehouse_inventory.csv',
        'feedback': 'customer_feedback.csv',
        'costs': 'cost_breakdown.csv'
    }
    
    # Load each CSV file
    print("Loading data files...")
    for key, filename in csv_files.items():
        filepath = os.path.join(data_folder, filename)
        try:
            data[key] = pd.read_csv(filepath)
            print(f"✓ Loaded {filename}: {len(data[key])} records")
        except FileNotFoundError:
            print(f"✗ Error: {filename} not found in data folder!")
            data[key] = None
        except Exception as e:
            print(f"✗ Error loading {filename}: {str(e)}")
            data[key] = None
    
    return data

def get_data_summary(data):
    """
    Print summary information about all loaded datasets
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for name, df in data.items():
        if df is not None:
            print(f"\n{name.upper()} Dataset:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Column names: {', '.join(df.columns.tolist())}")
        else:
            print(f"\n{name.upper()} Dataset: NOT LOADED")
    
    print("\n" + "="*60)

# Test function - only runs when this file is executed directly
if __name__ == "__main__":
    # Load all data
    all_data = load_all_data()
    
    # Show summary
    get_data_summary(all_data)
    
    # Show first few rows of orders as example
    if all_data['orders'] is not None:
        print("\nFirst 5 rows of ORDERS data:")
        print(all_data['orders'].head())