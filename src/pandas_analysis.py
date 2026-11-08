from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"


def pandas_analysis():
    orders = pd.read_csv(CLEAN / "orders.csv")
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")

    # Keep completed orders for revenue analysis
    completed = orders[orders["status"] == "completed"].copy()

    # Join completed orders with customers and products
    analysis = completed.merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left"
    )

    analysis = analysis.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    # 1. Revenue pivot table by city and category
    print("\n1. REVENUE PIVOT TABLE")
    pivot = pd.pivot_table(
        analysis,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        fill_value=0
    )
    print(pivot.round(2))

    # 2. City-wise revenue, average and order count
    print("\n2. CITY-WISE SUMMARY")
    city_summary = analysis.groupby("city").agg(
        total_revenue=("order_amount", "sum"),
        average_order_amount=("order_amount", "mean"),
        order_count=("order_id", "count")
    )
    print(city_summary.round(2))

    # 3. Distribution statistics
    print("\n3. ORDER AMOUNT DISTRIBUTION")
    amount = completed["order_amount"]

    print("Mean:", round(amount.mean(), 2))
    print("Median:", round(amount.median(), 2))
    print("Skewness:", round(amount.skew(), 2))
    print("Q1:", round(amount.quantile(0.25), 2))
    print("Q3:", round(amount.quantile(0.75), 2))

    q1 = amount.quantile(0.25)
    q3 = amount.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = amount[
        (amount < lower_fence) | (amount > upper_fence)
    ]

    print("Lower fence:", round(lower_fence, 2))
    print("Upper fence:", round(upper_fence, 2))
    print("Outlier count:", len(outliers))

    # 4. Correlation analysis
    print("\n4. NUMERICAL CORRELATIONS")
    numeric = completed[
        ["quantity", "discount_pct", "order_amount", "delivery_minutes"]
    ]

    correlation = numeric.corr()
    print(correlation.round(3))

    # Find the strongest absolute correlation, excluding self-correlation
    corr_pairs = correlation.abs().where(
        ~pd.DataFrame(
            [[i == j for j in correlation.columns]
             for i in correlation.columns],
            index=correlation.index,
            columns=correlation.columns
        )
    )

    strongest = corr_pairs.stack().idxmax()
    strongest_value = correlation.loc[strongest[0], strongest[1]]

    print(
        "Strongest absolute correlation:",
        strongest[0],
        "and",
        strongest[1],
        "=",
        round(strongest_value, 3)
    )


if __name__ == "__main__":
    pandas_analysis()