# diagnostic.py
import pandas as pd

file_path = r"D:\Code wala scene\techsphere-analytics\data\raw\layoffs.xlsx"

# Load and show the actual structure
print("=" * 60)
print("Domain Summary Sheet - First 5 rows")
print("=" * 60)
df = pd.read_excel(file_path, sheet_name='Domain Summary', header=1)
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nData types:", df.dtypes)

print("\n" + "=" * 60)
print("Company Detail Sheet - First 5 rows")
print("=" * 60)
df2 = pd.read_excel(file_path, sheet_name='Company Detail', header=1)
print(df2.head())
print("\nColumns:", df2.columns.tolist())