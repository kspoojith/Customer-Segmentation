"""
Starter segmentation pipeline script (smoke test).
- Prints dataset columns and first 5 rows using Python's csv module (no external deps).
- Contains skeleton functions and instructions for the next steps (use pandas/sklearn).

How to use:
    python segmentation_pipeline.py

Later steps (in notebook):
- Use pandas to load and analyze data
- Compute RFM-like features (Recency requires date; if absent compute proxies)
- Preprocess (missing values, outliers, encode, scale)
- Run clustering experiments (KMeans, Agglomerative, DBSCAN)
- Visualize (PCA / UMAP + cluster colors)
- Summarize top-3 features per cluster and provide marketing recommendations
"""

import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data.csv')

def preview_csv(path, n=5):
    """Read CSV using stdlib csv and print header + first n rows."""
    with open(path, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        rows = []
        for i, r in enumerate(reader):
            rows.append(r)
            if i >= n:
                break
    # Clean header quirks (leading tabs/whitespace)
    header = [h.strip() for h in rows[0]]
    print('\nColumns (cleaned):')
    for i, h in enumerate(header):
        print(f"  {i+1}. {h}")
    print('\nFirst %d data rows (raw):' % min(n, max(0, len(rows)-1)))
    for r in rows[1:]:
        print('  ', r)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print(f"data.csv not found at {DATA_PATH}. Place your data.csv in the project root.")
    else:
        print('Running smoke preview of data.csv...')
        preview_csv(DATA_PATH, n=5)
        print('\nNext: open `notebooks/Customer_Segmentation.ipynb` (or run the notebook provided) to continue with EDA and clustering.')
