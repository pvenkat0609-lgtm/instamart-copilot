
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"


def load_database():
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)

        print("SQLite database created successfully.")

        for table in ["customers", "products", "orders"]:
            count = pd.read_sql_query(
                f"SELECT COUNT(*) AS row_count FROM {table}",
                conn
            )["row_count"].iloc[0]

            print(f"{table}: {count} rows")


if __name__ == "__main__":
    load_database()
