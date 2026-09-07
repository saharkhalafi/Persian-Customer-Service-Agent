import pandas as pd
from sqlalchemy import create_engine


engine = create_engine(
    "postgresql+psycopg://postgres:12345678@localhost:5432/modiseh_ai"
)


COLUMN_MAPPING = {
    "CustomerCode": "customer_code",
    "Customer Group": "customer_group",
    "Customer_Type": "customer_type",
    "City": "city",
    "Personal_Code": "personal_code",
    "Customer Name": "customer_name",
    "Gender": "gender",
    "Mobile": "mobile",
    "Email": "email",
    "'Common DimCustomer'[CreatedDate_Persian]": "created_date_persian",
    "FirstPurchase": "first_purchase",
    "LastPurchase": "last_purchase",
    "Success Ordered Cnt": "success_ordered_cnt",
    "Success Order Price": "success_order_price",
    "Item Cnt": "item_cnt",
}


files = [
    "data/1401.xlsx",
    "data/1402.xlsx",
    "data/1403.xlsx",
]


dfs = []

for file in files:

    print(f"\nReading: {file}")

    # اول بدون header می‌خوانیم تا ساختار واقعی را ببینیم
    temp = pd.read_excel(
        file,
        sheet_name="Sheet1",
        header=None
    )

    print("Raw shape:", temp.shape)
    print("First row:")
    print(temp.iloc[0].tolist())

    # ردیف اول را header می‌کنیم
    temp.columns = temp.iloc[0]

    # ردیف header را از data حذف می‌کنیم
    temp = temp.iloc[1:].reset_index(drop=True)

    print("Columns:")
    print(temp.columns.tolist())

    dfs.append(temp)


# =========================
# Combine
# =========================

df = pd.concat(
    dfs,
    ignore_index=True
)


print("\nCombined shape:")
print(df.shape)


# =========================
# Rename
# =========================

df = df.rename(
    columns=COLUMN_MAPPING
)


print("\nColumns after normalization:")
print(df.columns.tolist())


# =========================
# Validate columns
# =========================

expected_columns = list(COLUMN_MAPPING.values())

missing_columns = [
    col for col in expected_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# =========================
# Convert everything to TEXT
# =========================

df = df.astype("string")


# =========================
# Remove duplicates
# =========================

before = len(df)

df = df.drop_duplicates(
    subset=["customer_code"]
)

after = len(df)

print("\nDuplicate check:")
print("Before:", before)
print("After:", after)
print("Removed:", before - after)


# =========================
# PostgreSQL
# =========================

df.to_sql(
    "users_raw",
    engine,
    schema="staging",
    if_exists="append",
    index=False
)


print("\nUsers imported successfully!")
print("Final rows:", len(df))