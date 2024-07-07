#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:33:16 2024

@author: hxkhkh
"""

# reading csc files

import pandas as pd
csv_file = "../../data/csv_examples/data.csv"

# Reading the CSV file
df = pd.read_csv(csv_file)
df_no_header = pd.read_csv(csv_file, header=None)
(n_rows, n_cols) = df.shape

# Get the header (column names)
header = df.columns

# Displaying the first five rows of the DataFrame. 
# This helps to understand the structure and contents of the file.
print(df.head())

# A concise summary of the DataFrame
print(df.info())


# Check if there are any NaN values in the DataFrame
# In pandas, null is a broader term that encompasses both NaN and None.
has_nan = df.isnull().values.any()

# Count NaN values in each column
if has_nan:
    nan_count_per_column = df.isnull().sum()
    total_nan_count = df.isnull().sum().sum()
    rows_with_nan = df[df.isnull().any(axis=1)]
    print("Rows with any NaN values:\n", rows_with_nan)


# Get a specific column (e.g., column named 'column_name')
specific_column = df[header[10]]
column_list = specific_column.tolist()

# Get a specific row by its index (e.g., the 3rd row, index 2)
specific_row = df.iloc[2]

# Calculate the max and mean values directly from the Series
max_value = specific_column.max()
mean_value = specific_column.mean()

# Isolate feature columns (e.g., columns 2 to the end)
features = df.iloc[:, 2:]
labels = df.iloc[:, 0:2]

all_zeros = (features == 0).all(axis=1)
zero_rows = df[all_zeros]
print("\nRows where all feature columns are zero:")
print(zero_rows)
zero_row_indices = df[all_zeros].index


# Create a new DataFrame without rows where all feature columns are zero
# Use ~ to negate the boolean mask to select non-zero rows
nonzero_rows_df = df[~all_zeros].copy()  

# Create another DataFrame with only the rows where all feature columns are zero
zero_rows_df = df[all_zeros].copy()


        

