import os
import re
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dfs = []

for fname in sorted(os.listdir(folder)):
    if not fname.endswith('.csv'):
        continue
    m = re.match(r'kappa([\d.]+)_tau([\de+-]+)\.csv', fname)
    if not m:
        continue
    kappa = float(m.group(1))
    tau = float(m.group(2))
    df = pd.read_csv(os.path.join(folder, fname))
    df['kappa'] = kappa
    df['tau'] = tau
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
out_path = os.path.join(folder, 'combined_results.csv')
combined.to_csv(out_path, index=False)
print(f"Saved {len(combined)} rows to {out_path}")
