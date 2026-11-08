
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"


def build_rfm():
    orders = pd.read_csv(CLEAN / "orders.csv")
    customers = pd.read_csv(CLEAN / "customers.csv")

    orders = orders[orders["status"] == "completed"].copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    snapshot = pd.Timestamp("2025-10-01 00:00:00")

    rfm = orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ]

    # Standardize RFM features
    features = ["recency", "frequency", "monetary"]
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[features])

    # Evaluate K-means for K = 2 to 8
    results = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )
        labels = model.fit_predict(scaled_rfm)

        results.append({
            "K": k,
            "inertia": round(model.inertia_, 2),
            "silhouette": round(
                silhouette_score(scaled_rfm, labels, working_memory=32), 3
            )
        })

    metrics = pd.DataFrame(results)
    metrics.to_csv(OUTPUTS / "kmeans_evaluation.csv", index=False)

    print("\nK-MEANS EVALUATION")
    print(metrics.to_string(index=False))

    # Final K=4 clustering
    final_model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42
    )
    rfm["segment_id"] = final_model.fit_predict(scaled_rfm)

    # Profile the clusters
    profile = rfm.groupby("segment_id").agg(
        customers=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).round(2)

    print("\nK=4 SEGMENT PROFILES")
    print(profile.to_string())

    # Assign names based on the verified cluster profiles
    segment_names = {
        0: "Regular Customers",
        1: "Premium Customers",
        2: "Loyal Customers",
        3: "At-Risk Customers"
    }

    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"] = (rfm["segment_id"] == 1).astype(int)

    # PCA on standardized RFM features
    pca = PCA(n_components=2)
    pca.fit_transform(scaled_rfm)

    print("\nPCA EXPLAINED VARIANCE")
    print("Component 1:", round(pca.explained_variance_ratio_[0], 3))
    print("Component 2:", round(pca.explained_variance_ratio_[1], 3))
    print("Total:", round(pca.explained_variance_ratio_.sum(), 3))

    # Save the final customer segments
    final_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]

    final_segments = rfm[final_columns]
    final_segments.to_csv(
        OUTPUTS / "customer_segments.csv",
        index=False
    )

    print("\nFINAL CUSTOMER SEGMENTS SAVED")
    print("Customers:", len(final_segments))
    print("\nSegment names:")
    print(final_segments["segment_name"].value_counts().to_string())
    print("\nPremium customers:", final_segments["is_premium"].sum())
    print("Output: outputs/customer_segments.csv")


if __name__ == "__main__":
    build_rfm()
