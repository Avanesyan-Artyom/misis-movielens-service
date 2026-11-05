import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
VERSION = os.environ.get('MODEL_VERSION','1.0.0')
if VERSION not in {'1.0.0','0.9.0'}:
    raise RuntimeError('Неизвестная версия модели')
ARTIFACT = BASE/'artifacts'/VERSION

@asynccontextmanager
async def lifespan(app):
    app.state.model=joblib.load(ARTIFACT/'pipeline.joblib')
    app.state.classifier=joblib.load(ARTIFACT/'classifier.joblib')
    app.state.meta=json.loads((ARTIFACT/'meta.json').read_text())
    app.state.movies=pd.read_csv(BASE/'data/movies.csv').set_index('movieId')
    yield

app=FastAPI(title='Рекомендации фильмов MovieLens',version='1.0.0',lifespan=lifespan)

class PredictionRequest(BaseModel):
    user_id:int=Field(gt=0, strict=True)
    movie_id:int=Field(gt=0, strict=True)

@app.get('/health')
def health():
    return {'status':'ok','model_version':VERSION}

@app.get('/model-info')
def model_info():
    return app.state.meta

@app.post('/predict')
def predict(request:PredictionRequest):
    if request.movie_id not in app.state.movies.index:
        raise HTTPException(status_code=404,detail='Фильм не найден в каталоге')
    features=app.state.model.named_steps['features']
    if request.movie_id not in features.movies_.index:
        raise HTTPException(status_code=404,detail='Для фильма нет оценок в обучающей выборке')
    x=pd.DataFrame([{'userId':request.user_id,'movieId':request.movie_id}])
    score=float(np.clip(app.state.model.predict(x)[0],.5,5))
    probability=float(app.state.classifier.predict_proba(x)[0,1])
    return {'user_id':request.user_id,'movie_id':request.movie_id,'predicted_rating':score,
            'class':int(probability>=.5),'probability':probability,
            'explanation':'Вероятность оценки >= 4 по логистической регрессии; класс 1 при вероятности >= 0,5. Калибровка не проверена.',
            'known_user':request.user_id in features.users_.index,'model_version':VERSION}

@app.get('/recommend/{user_id}')
def recommend(user_id:int,limit:int=10):
    if user_id<=0 or not 1<=limit<=20:
        raise HTTPException(status_code=422,detail='user_id > 0, limit от 1 до 20')
    f=app.state.model.named_steps['features']; stats=f.movies_.copy()
    stats=stats[stats['count']>=5].copy();stats['mean']=stats['sum']/stats['count']
    stats['smoothed']=(stats['sum']+10*f.mean_)/(stats['count']+10)
    stats['movie_id']=stats.index
    stats=stats[(stats.smoothed>=4)&(~stats.movie_id.isin(f.seen_.get(user_id,set())))]
    stats=stats.sort_values(['count','mean','movie_id'],ascending=[False,False,True]).head(limit)
    return {'user_id':user_id,'method':'popularity_with_rating_filter','recommendations':[
        {'movie_id':int(row.movie_id),'title':str(app.state.movies.loc[row.movie_id,'title']),
         'ratings_count':int(row['count']),'smoothed_rating':float(row.smoothed)} for _,row in stats.iterrows()]}
