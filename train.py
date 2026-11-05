import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from app.model import RatingFeatures

BASE = Path(__file__).resolve().parent

def train(version='1.0.0'):
    ratings_path = BASE/'data/ratings.csv'
    df = pd.read_csv(ratings_path).drop_duplicates().sort_values(['timestamp','userId','movieId'], kind='stable').reset_index(drop=True)
    if df.isna().any().any() or df.duplicated(['userId','movieId']).any() or not df.rating.between(.5,5).all():
        raise ValueError('Неверные или повторяющиеся данные')
    split, end = int(len(df)*.8), int(len(df)*.9)
    training, validation = df.iloc[:split], df.iloc[split:end]
    assert training.timestamp.max() < validation.timestamp.min() and validation.timestamp.max() < df.iloc[end:].timestamp.min()
    alpha = 100.0 if version == '0.9.0' else 10.0
    pipe = Pipeline([('features', RatingFeatures()), ('scale', StandardScaler()), ('model', Ridge(alpha=alpha))])
    pipe.fit(training[['userId','movieId']], training.rating)
    # Отдельная классификационная голова оценивает вероятность rating >= 4.
    classifier = Pipeline([('features', RatingFeatures()), ('scale', StandardScaler()), ('model', LogisticRegression(max_iter=1000))])
    # Признаки строятся на исходных оценках, а не на бинарных метках.
    feature_builder = RatingFeatures()
    train_features = feature_builder.fit_transform(training[['userId','movieId']], training.rating)
    classifier.steps[0] = ('features', feature_builder)
    classifier.named_steps['scale'].fit(train_features)
    classifier.named_steps['model'].fit(classifier.named_steps['scale'].transform(train_features), (training.rating >= 4).astype(int))
    prediction = np.clip(pipe.predict(validation[['userId','movieId']]), .5, 5)
    rmse = float(np.sqrt(np.mean((prediction-validation.rating.to_numpy())**2)))
    dest = BASE/'artifacts'/version
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, dest/'pipeline.joblib')
    joblib.dump(classifier, dest/'classifier.joblib')
    metadata = {'model_version':version,'model':'Ridge regression','alpha':alpha,'dataset':'MovieLens ml-latest-small',
                'ratings_sha256':hashlib.sha256(ratings_path.read_bytes()).hexdigest(),
                'train_rows':len(training),'validation_rows':len(validation),'test_rows_held_out':len(df)-end,
                'split':'global chronological 80/10/10','validation_rmse':rmse,
                'target':'rating 0.5–5.0','probability':'Logistic Regression estimate of rating >= 4; not calibrated',
                'practice3_selected_baseline':'popularity with smoothed rating >= 4'}
    (dest/'meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    golden=[]
    for row in validation.iloc[:5].itertuples():
        x=pd.DataFrame([{'userId':row.userId,'movieId':row.movieId}])
        score=float(np.clip(pipe.predict(x)[0],.5,5))
        probability=float(classifier.predict_proba(x)[0,1])
        golden.append({'request':{'user_id':int(row.userId),'movie_id':int(row.movieId)},'predicted_rating':score,'probability':probability,'class':int(probability>=.5)})
    (dest/'etalon.json').write_text(json.dumps(golden,ensure_ascii=False,indent=2))
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='1.0.0',choices=['1.0.0','0.9.0'])
    train(parser.parse_args().version)
