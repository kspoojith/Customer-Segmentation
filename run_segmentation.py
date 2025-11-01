"""End-to-end segmentation runner.
Loads data.csv, preprocesses, runs clustering experiments (KMeans/GMM/Agglomerative/DBSCAN),
profiles clusters, saves artifacts, and writes a short markdown report and recommendation JSON.
"""

import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, 'data.csv')
OUT_DIR = ROOT

RANDOM_STATE = 42

def load_and_clean(path):
    df = pd.read_csv(path, engine='python', encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    # basic string trim
    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
    # drop address
    if 'Address' in df.columns:
        df = df.drop(columns=['Address'])
    return df


def feature_engineer(df):
    feat = df.copy()
    for c in ['Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']:
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors='coerce')
    if 'Time on App' in feat.columns and 'Time on Website' in feat.columns:
        feat['App_vs_Web_ratio'] = feat['Time on App'] / (feat['Time on Website'] + 1e-6)
    if 'Yearly Amount Spent' in feat.columns and 'Length of Membership' in feat.columns:
        feat['Spend_per_membership_year'] = feat['Yearly Amount Spent'] / (feat['Length of Membership'] + 1e-6)
    # Avatar frequency
    if 'Avatar' in feat.columns:
        freq = feat['Avatar'].value_counts(normalize=True)
        feat['Avatar_freq'] = feat['Avatar'].map(freq)
    # choose features
    candidate = [c for c in feat.columns if c in ['Time on App','Time on Website','Length of Membership','Yearly Amount Spent','App_vs_Web_ratio','Spend_per_membership_year','Avatar_freq']]
    X = feat[candidate].copy()
    return feat, X, candidate


def handle_outliers_winsorize(df, cols, lower_q=0.01, upper_q=0.99):
    df2 = df.copy()
    lower = df2[cols].quantile(lower_q)
    upper = df2[cols].quantile(upper_q)
    df2[cols] = df2[cols].clip(lower=lower, upper=upper, axis=1)
    return df2


def build_numeric_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])


def run_kmeans(X, k_range=range(2,9)):
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels))>1 else np.nan
        db = davies_bouldin_score(X, labels) if len(set(labels))>1 else np.nan
        ch = calinski_harabasz_score(X, labels) if len(set(labels))>1 else np.nan
        results.append({'k':k, 'silhouette':sil, 'davies_bouldin':db, 'calinski_harabasz':ch, 'model': km})
    return results


def choose_best_kmeans(results):
    df = pd.DataFrame([{'k':r['k'], 'silhouette': r['silhouette']} for r in results])
    best_row = df.loc[df['silhouette'].idxmax()]
    for r in results:
        if r['k'] == int(best_row['k']):
            return r
    return results[0]


def profile_clusters(feat_df, features, label_col='cluster'):
    profiles = {}
    global_means = feat_df[features].mean()
    for label in sorted(feat_df[label_col].unique()):
        member = feat_df[feat_df[label_col]==label]
        means = member[features].mean()
        diff = (means - global_means) / (global_means.replace(0, 1))
        top3 = diff.abs().sort_values(ascending=False).head(3).index.tolist()
        profiles[int(label)] = {
            'size': int(len(member)),
            'means': {k: float(v) for k,v in means.to_dict().items()},
            'top3_features': top3
        }
    return profiles


def generate_marketing_recs(profiles):
    baseline_conversion = {}
    for label, p in profiles.items():
        mean_spend = p['means'].get('Yearly Amount Spent', 0)
        if mean_spend >= 600:
            baseline_conversion[label] = 0.06
        elif mean_spend >= 450:
            baseline_conversion[label] = 0.035
        elif mean_spend >= 300:
            baseline_conversion[label] = 0.02
        else:
            baseline_conversion[label] = 0.01
    recs = {}
    for label, p in profiles.items():
        top = p['top3_features']
        rec = {}
        if 'Time on App' in top or 'App_vs_Web_ratio' in top:
            rec['channel'] = 'Push + in-app'
            rec['message'] = 'App-exclusive offers and seamless checkout'
        elif 'Time on Website' in top:
            rec['channel'] = 'Email + website'
            rec['message'] = 'Personalized web carousels and flash discounts'
        elif 'Yearly Amount Spent' in top or 'Spend_per_membership_year' in top:
            rec['channel'] = 'Personalized email + VIP offers'
            rec['message'] = 'VIP bundles and loyalty incentives'
        else:
            rec['channel'] = 'Email + ads'
            rec['message'] = 'Discovery discounts and recommendations'
        if p['means'].get('Yearly Amount Spent',0) >= 600:
            rec['offer'] = 'Premium bundles and loyalty invites'
        elif p['means'].get('Yearly Amount Spent',0) >= 400:
            rec['offer'] = 'Free shipping threshold / % discount'
        else:
            rec['offer'] = 'Intro 10-15% discount'
        uplift_pct = 0.15
        rec['baseline_conversion'] = baseline_conversion[label]
        rec['expected_conversion'] = rec['baseline_conversion'] * (1 + uplift_pct)
        rec['expected_absolute_lift'] = rec['expected_conversion'] - rec['baseline_conversion']
        rec['size'] = p['size']
        rec['top_features'] = p['top3_features']
        recs[label] = rec
    return recs


def save_artifacts(feat_df, profiles, recs, pipeline, model):
    out_path = os.path.join(OUT_DIR, 'customer_segments.csv')
    feat_df.to_csv(out_path, index=False)
    joblib.dump(pipeline, os.path.join(OUT_DIR, 'preprocessing_pipeline.joblib'))
    joblib.dump(model, os.path.join(OUT_DIR, 'kmeans_model.joblib'))
    manifest = {'created_at': datetime.utcnow().isoformat(), 'rows': int(feat_df.shape[0]), 'features': features, 'random_state': RANDOM_STATE}
    with open(os.path.join(OUT_DIR, 'manifest.json'),'w',encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT_DIR, 'marketing_recommendations.json'),'w',encoding='utf-8') as f:
        json.dump(recs, f, indent=2)
    with open(os.path.join(OUT_DIR, 'cluster_profiles.json'),'w',encoding='utf-8') as f:
        json.dump(profiles, f, indent=2)
    # short md report
    lines = ['# Customer Segmentation Report', f'Generated: {datetime.utcnow().isoformat()} UTC', '\n## Segment summaries']
    for label, p in profiles.items():
        r = recs[label]
        lines.append(f"\n### Segment {label} — size: {p['size']}")
        lines.append(f"Top-3 features: {', '.join(p['top3_features'])}")
        lines.append('Mean metrics:')
        for feat_name, val in p['means'].items():
            lines.append(f"- {feat_name}: {val:.2f}")
        lines.append(f"Recommended channel: {r['channel']}")
        lines.append(f"Suggested offer: {r['offer']}")
        lines.append(f"Baseline conv: {r['baseline_conversion']:.3f}, expected conv: {r['expected_conversion']:.3f} (abs lift {r['expected_absolute_lift']:.3f})")
    with open(os.path.join(OUT_DIR, 'segment_report.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Artifacts saved to', OUT_DIR)


if __name__ == '__main__':
    print('Loading data...')
    df = load_and_clean(DATA_PATH)
    print('Rows:', len(df), 'Columns:', df.shape[1])
    feat_df, X, features = feature_engineer(df)
    print('Candidate features:', features)
    # handle outliers
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X2_df = handle_outliers_winsorize(X, numeric_cols)
    pipeline = build_numeric_pipeline()
    X_prepared = pipeline.fit_transform(X2_df)
    print('Prepared feature matrix shape:', X_prepared.shape)
    # PCA for diagnostics
    pca = PCA(n_components=min(10, X_prepared.shape[1]))
    try:
        X_pca = pca.fit_transform(X_prepared)
    except Exception:
        X_pca = None
    # run KMeans
    print('Running KMeans experiments...')
    km_results = run_kmeans(X_prepared, k_range=range(2,9))
    best = choose_best_kmeans(km_results)
    best_k = best['k']
    best_model = best['model']
    labels = best_model.labels_
    feat_df['cluster'] = labels
    print('Chosen KMeans k=', best_k, 'silhouette=', best['silhouette'])
    # alternative clustering
    print('Running alternative clustering (GMM, Agglomerative, DBSCAN)')
    gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE).fit(X_prepared)
    gmm_labels = gmm.predict(X_prepared)
    agg = AgglomerativeClustering(n_clusters=best_k).fit_predict(X_prepared)
    dbscan = DBSCAN(eps=0.8, min_samples=5).fit_predict(X_prepared)
    # profile
    profiles = profile_clusters(feat_df, features, label_col='cluster')
    recs = generate_marketing_recs(profiles)
    # save
    save_artifacts(feat_df, profiles, recs, pipeline, best_model)
    print('Done.')
