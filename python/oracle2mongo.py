import decimal
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pymongo import MongoClient

# -------------------------------
# Step 1: Create Spark Session
# -------------------------------
spark = SparkSession.builder \
    .appName("OracleToMongo") \
    .config("spark.jars", "C:\Installers\sqldeveloper\jdbc\lib\ojdbc11.jar") \
    .getOrCreate()

# -------------------------------
# Step 2: Oracle connection props
# -------------------------------
oracle_url = "jdbc:oracle:thin:@//localhost:1521/xepdb1"
oracle_props = {
    "user": "HR",
    "password": "admin",
    "driver": "oracle.jdbc.OracleDriver"
}


# -------------------------------
# Step 3: Load Oracle Tables
# -------------------------------
cust_eng_df = spark.read.jdbc(url=oracle_url, table="CUSTOMER_ENGAGEMENT_PREFERENCES", properties=oracle_props)
freq_df     = spark.read.jdbc(url=oracle_url, table="ENGAGEMENT_FREQUENCY", properties=oracle_props)
etype_df    = spark.read.jdbc(url=oracle_url, table="ENGAGEMENT_TYPE", properties=oracle_props)

# -------------------------------
# Step 4: Replace IDs with Names
# -------------------------------
cust_with_freq = cust_eng_df.join(
    freq_df,
    cust_eng_df.FREQUENCY_ID == freq_df.FREQUENCY_ID,
    "left"
).drop(freq_df.FREQUENCY_ID)

cust_full = cust_with_freq.join(
    etype_df,
    cust_with_freq.ENGAGEMENT_TYPE_ID == etype_df.ENGAGEMENT_TYPE_ID,
    "left"
).drop(etype_df.ENGAGEMENT_TYPE_ID)

final_df = cust_full.select(
    "CUSTOMER_ID",
    "ENGAGEMENT_TYPE_NAME",
    "FREQUENCY_NAME"
)

# -------------------------------
# Step 5: Collect + Fix Decimals
# -------------------------------
customer_docs = []
for row in final_df.collect():
    doc = row.asDict()
    # Convert CUSTOMER_ID from Decimal to int
    if isinstance(doc["CUSTOMER_ID"], decimal.Decimal):
        doc["cust_id"] = int(doc.pop("CUSTOMER_ID"))
    else:
        doc["cust_id"] = doc.pop("CUSTOMER_ID")
    # Rename columns
    doc["engagement_type"] = doc.pop("ENGAGEMENT_TYPE_NAME")
    doc["engagement_frequency"] = doc.pop("FREQUENCY_NAME")
    customer_docs.append(doc)

# -------------------------------
# Step 6: Insert into MongoDB
# -------------------------------
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["mydb"]
collection = db["customer_engagements"]

collection.insert_many(customer_docs)

print(f"Inserted {len(customer_docs)} documents into MongoDB.")