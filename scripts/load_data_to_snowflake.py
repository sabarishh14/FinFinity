import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ======================
# 1. Snowflake connection settings
# ======================
SNOWFLAKE_USER = "DEV"
SNOWFLAKE_PASSWORD = "*"
SNOWFLAKE_ACCOUNT = "zlqznbs-ii93813"  # Just the account identifier, no https:// or .snowflakecomputing.com
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_DATABASE = "CASE_STUDY"
SNOWFLAKE_SCHEMA = "CASE_STUDY_SCHEMA"

# ======================
# 2. Connect to Snowflake
# ======================
conn = snowflake.connector.connect(
    user=SNOWFLAKE_USER,
    password=SNOWFLAKE_PASSWORD,
    account=SNOWFLAKE_ACCOUNT,
    warehouse=SNOWFLAKE_WAREHOUSE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_SCHEMA
)

# ======================
# 3. Load Excel worksheets
# ======================
file_path = "RAW_DATA_V2.xlsx"  # update path if needed
excel_file = pd.ExcelFile(file_path)
sheets_dict = {sheet: excel_file.parse(sheet) for sheet in excel_file.sheet_names}

# ======================
# 4. Upload each worksheet as table
# ======================
cursor = conn.cursor()

for sheet_name, df in sheets_dict.items():
    # Clean column names for Snowflake compatibility
    df.columns = [col.upper().replace(" ", "_") for col in df.columns]
    
    table_name = sheet_name.upper()
    
    try:
        # Upload dataframe to Snowflake
        success, nchunks, nrows, _ = write_pandas(
            conn,
            df,
            table_name,
            auto_create_table=True,  # Automatically create table if it doesn't exist
            overwrite=True  # Overwrite existing table
        )

        print(f"{sheet_name}: uploaded {nrows} rows ({'success' if success else 'failed'})")
        
    except Exception as e:
        print(f"{sheet_name}: failed to upload - {str(e)}")

cursor.close()

# ======================
# 5. Close connection
# ======================
conn.close()
