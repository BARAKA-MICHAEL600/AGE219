import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os

# Create an images folder automatically if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# 1. Programmatically locate and merge all 10 CSV files
data_folder = 'DATA'
all_files = glob.glob(os.path.join(data_folder, "*.csv"))

print(f"Found {len(all_files)} files to process.")

df_list = []
for file in all_files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

# Combine into a single unified Pandas DataFrame
df = pd.concat(df_list, axis=0, ignore_index=True)
print("Data successfully merged!")

# 2. Data Cleaning
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df.dropna(subset=['Year', 'Value'], inplace=True)

# Group by year to find the annual average temperature change
annual_df = df.groupby('Year')['Value'].mean().reset_index()

# 3. SciPy Scientific Computing (Linear Regression)
slope, intercept, r_value, p_value, std_err = stats.linregress(annual_df['Year'], annual_df['Value'])
print(f"Analysis Complete: Temp is changing at {slope:.4f} °C per year.")

# 4. Scientific Visualizations (Matplotlib)
# Plot 1: Trend Analysis Plot (Line Chart)
plt.figure(figsize=(8, 5))
plt.plot(annual_df['Year'], annual_df['Value'], marker='o', color='green', label='Annual Change')
plt.title('Temperature Change Over Time (Kenya)')
plt.xlabel('Year')
plt.ylabel('Temperature Change [°C]')
plt.grid(True)
plt.legend()
plt.savefig('images/trend_analysis.png')
plt.close()

# Plot 2: Categorical Comparison (Bar Chart)
plt.figure(figsize=(8, 5))
plt.bar(annual_df['Year'].tail(5).astype(str), annual_df['Value'].tail(5), color='teal')
plt.title('Recent Temperature Changes Comparison')
plt.xlabel('Year')
plt.ylabel('Temperature Change [°C]')
plt.grid(True)
plt.savefig('images/categorical_comparison.png')
plt.close()

# Plot 3: Correlation Plot with SciPy Trend Line
plt.figure(figsize=(8, 5))
plt.scatter(annual_df['Year'], annual_df['Value'], color='blue', label='Actual Data')
# Calculate the trend line points
trend_line = slope * annual_df['Year'] + intercept
plt.plot(annual_df['Year'], trend_line, color='red', linestyle='--', label=f'Trend Line (R²={r_value**2:.2f})')
plt.title('Correlation: Year vs Temperature Shift')
plt.xlabel('Year')
plt.ylabel('Temperature Change [°C]')
plt.grid(True)
plt.legend()
plt.savefig('images/correlation_plot.png')
plt.close()

print("All 3 engineering standard plots saved inside the 'images/' folder!")