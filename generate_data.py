import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000

data = pd.DataFrame({
    "transaction_id": range(1, N + 1),
    "amount": np.random.lognormal(mean=7, sigma=1, size=N),
    "failed": np.random.binomial(1, 0.08, N),
    "new_device": np.random.binomial(1, 0.12, N),
    "international": np.random.binomial(1, 0.10, N),
    "rapid_frequency": np.random.poisson(2, N),
})

# Normal transactions
data["is_fraud"] = 0

# Inject several fraud-spike periods
spike_ranges = [
    (2000, 2200),
    (5000, 5250),
    (8000, 8300),
]

for start, end in spike_ranges:
    data.loc[start:end, "failed"] = np.random.binomial(
        1, 0.35, end - start + 1
    )
    data.loc[start:end, "new_device"] = np.random.binomial(
        1, 0.45, end - start + 1
    )
    data.loc[start:end, "international"] = np.random.binomial(
        1, 0.40, end - start + 1
    )
    data.loc[start:end, "rapid_frequency"] = np.random.poisson(
        7, end - start + 1
    )
    data.loc[start:end, "amount"] *= np.random.uniform(
        1.5, 3.0, end - start + 1
    )
    data.loc[start:end, "is_fraud"] = 1

data.to_csv("data/transactions.csv", index=False)

print("Dataset created successfully!")
print(f"Total transactions: {len(data)}")
print(f"Fraud transactions: {data['is_fraud'].sum()}")
print("Saved to: data/transactions.csv")