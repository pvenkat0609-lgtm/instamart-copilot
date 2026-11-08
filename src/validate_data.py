from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
CLEAN = ROOT / "data" / "clean"


def validate_data():
    checks = []

    # Check 1: Clean customers row count
    customers = pd.read_csv(CLEAN / "customers.csv")
    checks.append(len(customers) == 1200)

    # Check 2: Clean products row count
    products = pd.read_csv(CLEAN / "products.csv")
    checks.append(len(products) == 40)

    # Check 3: Clean orders row count
    orders = pd.read_csv(CLEAN / "orders.csv")
    checks.append(len(orders) == 12309)

    with sqlite3.connect(DB) as conn:

        # Check 4: Completed order count
        completed = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders WHERE status = 'completed'",
            conn
        )["n"].iloc[0]
        checks.append(completed == 11110)

        # Check 5: Q1 result count
        q1 = pd.read_sql_query("""
            SELECT customer_id
            FROM orders
            WHERE status = 'completed' AND order_amount > 500
            GROUP BY customer_id
            HAVING SUM(order_amount) > 8000
        """, conn)
        checks.append(len(q1) == 30)

        # Check 6: Q2 total revenue
        q2 = pd.read_sql_query("""
            SELECT SUM(order_amount) AS revenue
            FROM orders
            WHERE status = 'completed'
        """, conn)["revenue"].iloc[0]
        checks.append(round(q2, 2) == 2998668.80)

        # Check 7: Q4 customers without completed orders
        q4 = pd.read_sql_query("""
            SELECT c.customer_id
            FROM customers AS c
            LEFT JOIN orders AS o
                ON c.customer_id = o.customer_id
                AND o.status = 'completed'
            WHERE o.order_id IS NULL
        """, conn)
        checks.append(len(q4) == 25)

        # Check 8: Q5 above-average customers
        q5 = pd.read_sql_query("""
            WITH customer_totals AS (
                SELECT customer_id, SUM(order_amount) AS total_spend
                FROM orders
                WHERE status = 'completed'
                GROUP BY customer_id
            )
            SELECT *
            FROM customer_totals
            WHERE total_spend > (
                SELECT AVG(total_spend)
                FROM customer_totals
            )
        """, conn)
        checks.append(len(q5) == 550)

        # Check 9: Q7 daily revenue row count
        q7 = pd.read_sql_query("""
            SELECT SUBSTR(order_ts, 1, 10) AS day,
                   SUM(order_amount) AS revenue
            FROM orders
            WHERE status = 'completed'
            GROUP BY SUBSTR(order_ts, 1, 10)
        """, conn)
        checks.append(len(q7) == 183)

    if all(checks):
        print("ALL CHECKPOINTS PASSED")
    else:
        failed = [i + 1 for i, passed in enumerate(checks) if not passed]
        raise AssertionError(f"Validation failed at checkpoints: {failed}")


if __name__ == "__main__":
    validate_data()