# Jupyter Notebook Analysis Files

🔹 Overview
This notebook automates the process of creating data dictionaries from raw Excel data files. It reads all sheets from RAW_DATA_IMPORT.xlsx, analyzes their structure, detects anomalies (missing values, high cardinality, etc.), and generates detailed metadata for each table.

The goal is to ensure data consistency, documentation completeness, and readiness for downstream data transformation or modeling.

🔹 Objective
Extract all data sheets from the input Excel file.
Create a data dictionary for each sheet describing column properties.
Identify anomalies such as missing values, inconsistent formats, or invalid ranges.
Suggest corrective handling for data cleaning.

🔹 Tools & Libraries
Library	Purpose
pandas	- Data extraction and analysis
matplotlib / seaborn - visualization of anomalies
openpyxl	- Excel sheet parsing

🔹 Technical Documentation
Step 1 – Import Libraries and Load Data
import pandas as pd

file_path = "RAW_DATA_IMPORT.xlsx"
sheets_dict = pd.read_excel(file_path, sheet_name=None)

Purpose: Load the entire Excel file into a Python dictionary where each sheet name maps to a DataFrame.
sheet_name=None ensures all sheets are read.
sheets_dict.keys() → shows available tables like “Customers”, “Orders”, etc.

Step 2 – Select Specific Sheet (e.g., Customers)
customers_df = sheets_dict["Customers"]

Extracts the “Customers” table from the workbook.
Used as input for dictionary generation.

Step 3 – Build Data Dictionary for Customers
customers_dict = {
    "customer_ID": {
        "dtype": "int64",
        "description": "Unique identifier for each customer",
        "nulls": customers_df["customer_ID"].isnull().sum(),
        "observations": "Primary key, no duplicates expected",
        "handling": "Keep as is, ensure uniqueness"
    },
    ...
}

Purpose: Programmatically generate a data dictionary for every column.
Includes:
dtype – inferred data type
description – business meaning
nulls – missing count
observations – key insights (e.g., duplicates, anomalies)
handling – recommended action

Step 4 – Repeat for Other Tables
You will see similar code for other sheets like Orders, Products, Sales, etc., each with its own dictionary.
Each dictionary ensures documentation consistency across tables.

Step 5 – Generate Metadata Table (Optional)
Markdown defines structure for documentation:

Column Name	          Description
table_name	          Name of the table
column_name	          Column in that table
dtype	                Data type
unique_count	        Count of unique values
missing_count	        Count of nulls

This structure standardizes reporting across all tables.

Step 6 – Detect Data Anomalies
The notebook includes markdown guidance to identify:
Missing values → incomplete data
Constant columns → no variance
High cardinality → unique identifiers
Outliers or invalid data → based on simple range checks
Visualizations (bar charts, summary tables) are used to highlight affected columns.


The notebook recommends:

Missing Value Chart: Bar chart of columns vs missing counts
Constant Columns Table: Columns with same value throughout
High Cardinality Columns: Identify columns with >90% unique values
These help prioritize data cleaning tasks.
