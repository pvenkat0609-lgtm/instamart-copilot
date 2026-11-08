from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT / "sql" / "queries.sql"
DB = ROOT / "data" / "instamart.db"


def run_queries():
    sql_text = SQL_FILE.read_text(encoding="utf-8")

    queries = [
        query.strip()
        for query in sql_text.split(";")
        if query.strip()
    ]

    if len(queries) != 7:
        raise ValueError(f"Expected 7 SQL queries, found {len(queries)}")

    with sqlite3.connect(DB) as conn:
        for index, query in enumerate(queries, start=1):
            result = pd.read_sql_query(query, conn)

            print(f"\n{'=' * 50}")
            print(f"Q{index} RESULTS")
            print("=" * 50)
            print("Rows:", len(result))

            if index == 2:
                print(result.to_string(index=False))
                print("Revenue total:", round(result["revenue"].sum(), 2))
                print("Largest revenue row:")
                print(result.loc[result["revenue"].idxmax()].to_dict())

            elif index == 5:
                print("Average customer spend:",
                      round(result["avg_spend"].iloc[0], 2))
                print(result.head(5).to_string(index=False))

            elif index == 6:
                print(
                    "Rows where RANK differs from ROW_NUMBER:",
                    int(
                        (
                            result["rank_position"]
                            != result["row_number_position"]
                        ).sum()
                    )
                )
                top_customers = result[result["rank_position"] == 1]
                print(top_customers.to_string(index=False))

            elif index == 7:
                print("First 5 rows:")
                print(result.head().to_string(index=False))
                print("Last row:")
                print(result.tail(1).to_string(index=False))

            else:
                print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    run_queries()