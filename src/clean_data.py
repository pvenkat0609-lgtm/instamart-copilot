from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"


def audit_data():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")

    datasets = {
        "CUSTOMERS": customers,
        "PRODUCTS": products,
        "ORDERS": orders,
    }

    for name, df in datasets.items():
        print(f"\n{'=' * 50}")
        print(name)
        print("=" * 50)
        print("Shape:", df.shape)
        print("\nData types:")
        print(df.dtypes)

        missing = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_percentage": (df.isna().mean() * 100).round(2)
        })
        print("\nMissing values:")
        print(missing)

    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

    print("\nORDER AUDIT CHECKPOINTS")
    print("Raw rows:", len(orders))
    print("Exact duplicate rows:", orders.duplicated().sum())
    print("Pending rows:", orders["status"].str.strip().str.lower().eq("pending").sum())
    print("Distinct order IDs:", orders["order_id"].nunique())
    print("Non-numeric amounts:", amount.isna().sum())
    print("Negative delivery times:", delivery.lt(0).sum())
    print("Blank delivery times:", delivery.isna().sum())

    age = pd.to_numeric(customers["age"], errors="coerce")
    gender = customers["gender"].astype("string").str.strip()
    city = customers["city"]

    print("\nCUSTOMER AUDIT CHECKPOINTS")
    print("Blank ages:", age.isna().sum())
    print("Ages outside 10-100:", ((age < 10) | (age > 100)).sum())
    print("Distinct raw city spellings:", city.nunique())
    print(
        "Distinct non-blank raw gender spellings:",
        gender[gender.notna() & gender.ne("")].nunique()
    )
    print("Blank genders:", (gender.isna() | gender.eq("")).sum())


def clean_customers():
    customers = pd.read_csv(RAW / "customers.csv")

    customers["city"] = customers["city"].str.strip().str.title()

    gender_map = {
        "f": "female",
        "female": "female",
        "m": "male",
        "male": "male"
    }

    customers["gender"] = (
        customers["gender"]
        .str.strip()
        .str.lower()
        .map(gender_map)
        .fillna("unknown")
    )

    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )

    customers.loc[
        ~customers["age"].between(10, 100), "age"
    ] = pd.NA

    print("\nAges missing before fill:", customers["age"].isna().sum())

    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    )

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    CLEAN.mkdir(parents=True, exist_ok=True)
    customers.to_csv(CLEAN / "customers.csv", index=False)

    print("\nCUSTOMER CLEANING CHECK")
    print("City counts:")
    print(customers["city"].value_counts().sort_index())

    print("\nGender counts:")
    print(customers["gender"].value_counts())

    print("\nMissing ages:", customers["age"].isna().sum())
    print("Age range:", customers["age"].min(), "to", customers["age"].max())


def clean_orders():
    orders = pd.read_csv(RAW / "orders.csv")
    products = pd.read_csv(RAW / "products.csv")
    print("\nORDER CLEANING STEPS")
    print("Step 1 raw rows:", len(orders))

    # Convert order timestamp
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # Remove exact duplicate rows
    orders = orders.drop_duplicates()
    print("Step 2 after removing exact duplicates:", len(orders))

    # Keep latest record for each order ID
    orders = orders.sort_values("order_ts")
    orders = orders.drop_duplicates(
        subset="order_id", keep="last"
    ).copy()
    print("Step 3 after keeping latest record per order_id:", len(orders))

    # Standardize status
    orders["status"] = orders["status"].str.strip().str.lower()

    # Convert numeric columns
    orders["quantity"] = pd.to_numeric(
        orders["quantity"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Add product category and price
    orders = orders.merge(
        products[["product_id", "category", "price_inr"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Fill missing order amounts using product price
    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    print("Step 4 amounts filled from product price:", orders["order_amount"].isna().sum())
    orders["order_amount"] = orders["order_amount"].fillna(
        calculated_amount
    )

    # Replace negative delivery times with missing values
    print("Step 5 negative delivery times set to missing:", (orders["delivery_minutes"] < 0).sum())
    orders.loc[
        orders["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = pd.NA

    # Fill missing delivery times for completed orders
    completed = orders["status"].eq("completed")
    print("Step 6 completed delivery times filled:", orders.loc[completed, "delivery_minutes"].isna().sum())

    completed_category_medians = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    orders.loc[completed, "delivery_minutes"] = (
        orders.loc[completed, "delivery_minutes"]
        .fillna(completed_category_medians)
    )

    # Save cleaned orders and products
    CLEAN.mkdir(parents=True, exist_ok=True)

    orders_to_save = orders.drop(
        columns=["category", "price_inr"]
    ).copy()

    orders_to_save["order_ts"] = (
        orders_to_save["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    orders_to_save.to_csv(
        CLEAN / "orders.csv", index=False
    )

    products.to_csv(
        CLEAN / "products.csv", index=False
    )

    print("\nORDER CLEANING CHECK")
    print("Rows:", len(orders))
    print("Status counts:")
    print(orders["status"].value_counts())
    print("Missing order amounts:", orders["order_amount"].isna().sum())

    print(
        "Missing completed delivery times:",
        orders.loc[completed, "delivery_minutes"].isna().sum()
    )

    print(
        "Missing cancelled delivery times:",
        orders.loc[
            orders["status"].eq("cancelled"),
            "delivery_minutes"
        ].isna().sum()
    )

    print(
        "Completed revenue:",
        round(
            orders.loc[completed, "order_amount"].sum(), 2
        )
    )


if __name__ == "__main__":
    audit_data()
    clean_customers()
    clean_orders()