
"""Part B: Customer lapse prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.base import clone


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01")
SEPTEMBER_END = pd.Timestamp("2025-10-01")

NUMERIC = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

CATEGORICAL = [
    "city",
    "gender",
    "favourite_category"
]


def build_dataset():
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    products = pd.read_csv(CLEAN / "products.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Cohort and features use orders before September 1.
    pre = orders[orders["order_ts"] < CUTOFF].copy()
    completed = pre[pre["status"] == "completed"].copy()

    cohort_ids = completed["customer_id"].unique()

    # September orders are used only for target labels.
    september = orders[
        (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < SEPTEMBER_END)
        & (orders["status"] == "completed")
    ]

    active_ids = set(september["customer_id"])

    cohort = pd.DataFrame({"customer_id": cohort_ids})
    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(active_ids)
    ).astype(int)

    # Add product categories.
    completed = completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    # Completed-order features.
    features = completed.groupby("customer_id").agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean")
    )

    features["recency"] = (
        CUTOFF - features["last_order_date"]
    ).dt.days

    favourite = (
        completed.groupby("customer_id")["category"]
        .agg(lambda x: x.mode().sort_values().iloc[0])
        .rename("favourite_category")
    )

    features = features.join(favourite)

    # Cancellation rate from all final-status pre-cutoff orders.
    final_orders = pre[
        pre["status"].isin(["completed", "cancelled"])
    ].copy()

    cancellation_rate = (
        final_orders.assign(
            cancelled=(
                final_orders["status"] == "cancelled"
            ).astype(int)
        )
        .groupby("customer_id")["cancelled"]
        .mean()
        .rename("cancellation_rate")
    )

    features = features.join(cancellation_rate)

    # Customer information and tenure.
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"]
    )

    details = customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ].copy()

    details["tenure_days"] = (
        CUTOFF - details["signup_date"]
    ).dt.days

    features = features.reset_index().merge(
        details,
        on="customer_id",
        how="left"
    )

    features = features.merge(
        cohort,
        on="customer_id",
        how="inner"
    )

    features = features.drop(
        columns=["last_order_date", "signup_date"]
    )

    assert (
        features[NUMERIC + CATEGORICAL]
        .isnull()
        .sum()
        .sum()
        == 0
    )

    print("\nCOHORT")
    print("Customers:", len(features))
    print("Lapsed:", features["lapsed"].sum())
    print(
        "Lapse rate:",
        round(features["lapsed"].mean() * 100, 2),
        "%"
    )
    print("Feature dataset shape:", features.shape)

    return features


def evaluate(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )
    print(
        "Precision:",
        round(
            precision_score(
                y_test, predictions, zero_division=0
            ),
            4
        )
    )
    print(
        "Recall:",
        round(
            recall_score(
                y_test, predictions, zero_division=0
            ),
            4
        )
    )
    print(
        "F1:",
        round(
            f1_score(
                y_test, predictions, zero_division=0
            ),
            4
        )
    )

    return f1_score(
        y_test, predictions, zero_division=0
    )


def main():
    features = build_dataset()

    X = features[NUMERIC + CATEGORICAL]
    y = features["lapsed"]
    customer_ids = features["customer_id"]

    X_train, X_test, y_train, y_test, id_train, id_test = (
        train_test_split(
            X,
            y,
            customer_ids,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    )

    print("\nTRAIN/TEST SPLIT")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    # Baseline: always predict not lapsed.
    baseline = np.zeros(len(y_test), dtype=int)

    print("\nBASELINE")
    print(
        "Accuracy:",
        round(accuracy_score(y_test, baseline), 4)
    )
    print(
        "Lapsed F1:",
        round(
            f1_score(y_test, baseline, zero_division=0),
            4
        )
    )

    # Preprocessing fitted only on training data.
    preprocessor = ColumnTransformer([
        (
            "num",
            StandardScaler(),
            NUMERIC
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            CATEGORICAL
        )
    ])

    # Logistic Regression grid search.
    logistic_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    logistic_grid = GridSearchCV(
        logistic_pipeline,
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=1
        
    )

    logistic_grid.fit(X_train, y_train)

    print("\nLOGISTIC REGRESSION")
    print("Best parameters:", logistic_grid.best_params_)
    print(
        "Best CV F1:",
        round(logistic_grid.best_score_, 4)
    )

    # Random Forest grid search.
    forest_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    forest_grid = GridSearchCV(
        forest_pipeline,
        {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20]
        },
        cv=5,
        scoring="f1",
        n_jobs=1
    )

    forest_grid.fit(X_train, y_train)

    print("\nRANDOM FOREST")
    print("Best parameters:", forest_grid.best_params_)
    print(
        "Best CV F1:",
        round(forest_grid.best_score_, 4)
    )

    # Evaluate both models on the untouched test set.
    log_f1 = evaluate(
        "LOGISTIC REGRESSION TEST RESULTS",
        logistic_grid.best_estimator_,
        X_test,
        y_test
    )

    rf_f1 = evaluate(
        "RANDOM FOREST TEST RESULTS",
        forest_grid.best_estimator_,
        X_test,
        y_test
    )

    # Confirm that no feature column has a missing value.
    missing_total = int(X.isna().sum().sum())
    print("\nMISSING VALUE CHECK")
    print("Missing values in feature columns:", missing_total)
    assert missing_total == 0
    # Compare scaler training mean with full-cohort frequency mean.
    scaler = (
        logistic_grid.best_estimator_
        .named_steps["preprocessor"]
        .named_transformers_["num"]
    )

    frequency_index = NUMERIC.index("frequency")

    print("\nFREQUENCY MEAN COMPARISON")
    print(
        "Training scaler mean:",
        round(scaler.mean_[frequency_index], 4)
    )
    print(
        "Full cohort mean:",
        round(X["frequency"].mean(), 4)
    )

    # Oversample lapsed training customers only.
    X_balanced = X_train.copy()
    y_balanced = y_train.copy()

    lapsed_rows = X_train[y_train == 1]

    extra_count = int(
        (y_train == 0).sum() - (y_train == 1).sum()
    )

    rng = np.random.RandomState(42)

    extra_indices = rng.choice(
        lapsed_rows.index,
        size=extra_count,
        replace=True
    )

    X_balanced = pd.concat([
        X_balanced,
        X_train.loc[extra_indices]
    ])

    y_balanced = pd.concat([
        y_balanced,
        y_train.loc[extra_indices]
    ])

    print("\nOVERSAMPLING")
    print("Balanced training counts:")
    print(y_balanced.value_counts().sort_index())

    balanced_logistic = clone(
        logistic_grid.best_estimator_
    )

    balanced_logistic.fit(
        X_balanced,
        y_balanced
    )

    evaluate(
        "OVERSAMPLED LOGISTIC REGRESSION",
        balanced_logistic,
        X_test,
        y_test
    )

    # Random Forest feature importance.
    best_rf = forest_grid.best_estimator_

    feature_names = (
        best_rf.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        best_rf.named_steps["model"].feature_importances_
    )

    importance_table = pd.Series(
        importances,
        index=feature_names
    ).sort_values(ascending=False)

    print("\nTOP 5 RANDOM FOREST FEATURES")
    print(importance_table.head(5))

    # Logistic Regression coefficient signs.
    best_log = logistic_grid.best_estimator_

    coefficients = pd.Series(
        best_log.named_steps["model"].coef_[0],
        index=feature_names
    )

    print("\nLOGISTIC REGRESSION COEFFICIENTS")
    print("Three largest positive:")
    print(
        coefficients.sort_values(
            ascending=False
        ).head(3)
    )

    print("Three largest negative:")
    print(coefficients.sort_values().head(3))

    # Select using cross-validation F1, not test-set F1.
    if forest_grid.best_score_ >= logistic_grid.best_score_:
        selected_model = best_rf
        selected_name = "Random Forest"
    else:
        selected_model = best_log
        selected_name = "Logistic Regression"

    print("\nSELECTED MODEL:", selected_name)
    print("Selection based on cross-validation F1.")

    # Save and reload selected model.
    model_path = OUTPUTS / "lapse_model.joblib"

    joblib.dump(selected_model, model_path)

    loaded_model = joblib.load(model_path)

    original_predictions = selected_model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)

    assert np.array_equal(
        original_predictions,
        loaded_predictions
    )

    print("Model saved and reloaded successfully.")
    print(
        "Predictions match:",
        np.array_equal(
            original_predictions,
            loaded_predictions
        )
    )

    # Premium retention list: post-hoc segment filter,
    # not a model feature.
    segments = pd.read_csv(
        OUTPUTS / "customer_segments.csv"
    )

    test_results = pd.DataFrame({
        "customer_id": id_test.values,
        "actually_lapsed": y_test.values,
        "lapse_probability": (
            selected_model.predict_proba(X_test)[:, 1]
        )
    })

    premium_ids = set(
        segments.loc[
            segments["segment_name"] == "Premium Customers",
            "customer_id"
        ]
    )

    retention = test_results[
        test_results["customer_id"].isin(premium_ids)
    ].sort_values(
        "lapse_probability",
        ascending=False
    ).head(10)

    retention = retention[
        [
            "customer_id",
            "lapse_probability",
            "actually_lapsed"
        ]
    ]

    retention.to_csv(
        OUTPUTS / "retention_list.csv",
        index=False
    )

    print("\nRETENTION LIST")
    print(retention.to_string(index=False))
    print(
        "Actually lapsed in retention list:",
        retention["actually_lapsed"].sum()
    )

    print("\nAll Part B model outputs saved.")


if __name__ == "__main__":
    main()




