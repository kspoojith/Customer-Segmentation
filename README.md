# Customer Segmentation (E-commerce)

This repository contains a starter pipeline for customer segmentation to support personalized marketing.

Files added:
- `segmentation_pipeline.py` — smoke-test script that previews `data.csv` using built-in csv (no external deps).
- `requirements.txt` — recommended Python packages for the full pipeline and notebooks.

Next steps (recommended):
1. Create a Jupyter notebook `notebooks/Customer_Segmentation.ipynb` to perform EDA and iterate on models.
2. Preprocessing:
   - Clean/normalize column names (e.g., remove stray leading tabs in the header).
   - Handle missing values and outliers.
   - Convert/encode categorical features (e.g., Avatar color if useful).
   - Feature engineering: RFM-like features (Recency requires transaction date; otherwise use proxies like Length of Membership, Time on App/Website, and Yearly Amount Spent).
   - Standardize or RobustScale numeric features before clustering.
3. Clustering experiments:
   - Try KMeans (grid search over k), Agglomerative, and DBSCAN.
   - Evaluate with silhouette score, Davies-Bouldin index, and stability.
4. Dimensionality reduction & visualization:
   - Use PCA/UMAP/t-SNE to visualize clusters.
   - Plot cluster-level summaries and the top-3 defining features per cluster.
5. Deliverables:
   - Notebook with EDA, preprocessing, model selection and visualizations.
   - Short report with segment descriptions and marketing recommendations.

How to run the smoke test (Windows PowerShell):

```powershell
python segmentation_pipeline.py
```

If you want to run the full notebook, install dependencies:

```powershell
python -m pip install -r requirements.txt
jupyter lab
```
