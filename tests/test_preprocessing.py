import os
import json
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data.csv')
MANIFEST = os.path.join(os.path.dirname(__file__), '..', 'manifest.json')


def test_data_loads():
    assert os.path.exists(DATA_PATH), f"data.csv not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH, engine='python')
    assert df.shape[0] > 0, 'data.csv is empty'


def test_manifest_exists():
    # Manifest created by notebook when run
    assert os.path.exists(MANIFEST), 'manifest.json is missing; run the notebook to generate artifacts.'
    with open(MANIFEST,'r',encoding='utf-8') as f:
        m = json.load(f)
    assert 'rows' in m and m['rows'] > 0
