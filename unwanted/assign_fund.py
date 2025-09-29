import pandas as pd

# Load your CSV
df1 = pd.read_csv("RAW_DATA_V2 - Customer_Risk_Final.csv")

def assign_fund_and_flag(row):
    age_group = row["Age Group"]
    risk = row["risk_score"]
    marital = row["marital_status"]

    flag = None  # default

    # --- 30-55 group ---
    if age_group == "30-55":
        if 1.12 <= risk < 2.95 and marital == "Married":
            return "Conservative Investors", flag
        elif 2.95 <= risk < 3.56 and marital == "Married":
            return "Balanced Investors", flag
        elif 3.56 <= risk < 5 and marital in ["Single", "Divorced", "Single & Divorced"]:
            return "Aggressive Investors", flag
        else:
            # flag1 case
            flag = "Flag1"
            if 1.12 <= risk < 2.95:
                return "Conservative Investors", flag
            elif 2.95 <= risk < 3.56:
                return "Balanced Investors", flag
            elif 3.56 <= risk < 5:
                return "Aggressive Investors", flag

    # --- 55-75 group ---
    elif age_group in ["55-75", "55+"]:
        if 1.12 <= risk < 2.95 and marital == "Married":
            return "Pre-Retirees", flag
        elif 2.95 <= risk <= 5 and marital in ["Single", "Divorced", "Single & Divorced"]:
            return "Second Chance Retirees", flag
        else:
            # flag2 case
            flag = "Flag2"
            if 1.12 <= risk < 2.95:
                return "Pre-Retirees", flag
            elif 2.95 <= risk <= 5:
                return "Second Chance Retirees", flag

    return None, flag  # default if no match

# Apply the function
df1[["Fund", "Flag"]] = df1.apply(assign_fund_and_flag, axis=1, result_type="expand")

# Save to CSV
df1.to_csv("hello.csv", index=False)

print(df1.head())
