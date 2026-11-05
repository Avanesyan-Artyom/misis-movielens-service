import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RatingFeatures(BaseEstimator, TransformerMixin):
    """Сглаженные сведения о пользователях и фильмах, изученные на train."""

    def fit(self, X, y):
        frame = X.copy()
        frame['rating'] = np.asarray(y)
        self.mean_ = float(frame.rating.mean())
        self.users_ = frame.groupby('userId').rating.agg(['sum', 'count'])
        self.movies_ = frame.groupby('movieId').rating.agg(['sum', 'count'])
        self.seen_ = {int(u): set(g.movieId.astype(int)) for u, g in frame.groupby('userId')}
        return self

    def transform(self, X):
        users = self.users_.reindex(X.userId).fillna(0)
        movies = self.movies_.reindex(X.movieId).fillna(0)
        return np.column_stack([
            (users['sum'] + 5*self.mean_) / (users['count'] + 5),
            (movies['sum'] + 10*self.mean_) / (movies['count'] + 10),
            np.log1p(users['count']), np.log1p(movies['count'])])

    def fit_transform(self, X, y=None, **kwargs):
        # Leave-one-out: целевая оценка не входит в собственные признаки.
        self.fit(X, y)
        users = self.users_.reindex(X.userId)
        movies = self.movies_.reindex(X.movieId)
        y = np.asarray(y)
        prior = (len(y)*self.mean_ - y)/(len(y)-1)
        return np.column_stack([
            (users['sum'].to_numpy()-y+5*prior)/(users['count'].to_numpy()-1+5),
            (movies['sum'].to_numpy()-y+10*prior)/(movies['count'].to_numpy()-1+10),
            np.log1p(users['count'].to_numpy()-1), np.log1p(movies['count'].to_numpy()-1)])
