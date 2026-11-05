import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as client:
        yield client

def test_health_and_info(client):
    health=client.get('/health');info=client.get('/model-info')
    assert health.status_code==200 and health.json()['status']=='ok'
    assert health.json()['model_version']==info.json()['model_version']
    assert info.json()['train_rows']==80668

def test_golden_predictions(client):
    file=Path('artifacts')/client.get('/health').json()['model_version']/'etalon.json'
    rows=json.loads(file.read_text());assert len(rows)==5
    for row in rows:
        response=client.post('/predict',json=row['request'])
        assert response.status_code==200
        assert response.json()['predicted_rating']==pytest.approx(row['predicted_rating'],abs=1e-10)
        assert response.json()['class']==row['class']
        assert response.json()['probability']==pytest.approx(row['probability'],abs=1e-10)

@pytest.mark.parametrize('payload',[{}, {'user_id':0,'movie_id':1}, {'user_id':'1','movie_id':1}, {'user_id':1,'movie_id':-1}])
def test_invalid(client,payload):
    assert client.post('/predict',json=payload).status_code==422

def test_unknown_movie(client):
    assert client.post('/predict',json={'user_id':1,'movie_id':99999999}).status_code==404

def test_recommendations_exclude_seen(client):
    response=client.get('/recommend/1?limit=10');assert response.status_code==200
    movies=response.json()['recommendations'];assert len(movies)==10
    model=joblib.load('artifacts/1.0.0/pipeline.joblib')
    assert not ({x['movie_id'] for x in movies} & model.named_steps['features'].seen_[1])
    assert all(x['smoothed_rating']>=4 and x['ratings_count']>=5 for x in movies)

def test_new_user_and_limit(client):
    assert client.get('/recommend/999999?limit=5').status_code==200
    assert client.get('/recommend/1?limit=0').status_code==422
