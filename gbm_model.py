import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from constants import POVERTY_MAPPING

BASE_DIR = os.path.dirname(__file__)
# Custom scorer for cross validation
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')

class Model:
    def __init__(self, dataset):
        # Labels for training
        self.train_labels = np.array(list(dataset[dataset['Target'].notnull()]['Target'].astype(np.uint8)))

        self.model_path = os.path.join(BASE_DIR, 'data/best_model.txt')

        # Extract the training data
        self.train_raw = dataset[dataset['Target'].notnull()]
        self.test_raw = dataset[dataset['Target'].isnull()]
        self.features_raw = list(self.test_raw.columns)
        self.train_set = self.train_raw.drop(columns = ['Id', 'idhogar', 'Target'])
        self.test_set = self.test_raw.drop(columns = ['Id', 'idhogar', 'Target'])
        self.features = list(self.train_set.columns)
        self.new_features = []
        self.pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 
                                  ('scaler', MinMaxScaler())])         
        self._fit_transform_data()
        self.model = self.load_model()       

    def load_model(self):
        best = lgb.Booster(model_file=self.model_path)
        return best

    def _fit_transform_data(self, predict_data=None, predict=False):
        if predict == False:
            # Fit and transform training data
            self.train_set = self.pipeline.fit_transform(self.train_set)
            self.test_set = self.pipeline.transform(self.test_set)

            # convert to pd
            self.train_set = pd.DataFrame(self.train_set, columns = self.features)
            self.test_set = pd.DataFrame(self.test_set, columns = self.features)
            self.train_set, self.test_set = self.train_set.align(self.test_set, axis = 1, join = 'inner')
            self.new_features = list(self.train_set.columns)

            self.test_raw = pd.DataFrame(self.test_raw, columns = self.features_raw)

            return None
        else:
            predict_data = predict_data.drop(columns = ['Id', 'idhogar', 'Target'])
            predict_data = self.pipeline.transform(predict_data)
            predict_data = pd.DataFrame(self.test_set, columns = self.features)
            _, predict_data = self.train_set.align(predict_data, axis = 1, join = 'inner')
            return predict_data

    def predict(self, X):
        # X = self._fit_transform_data(X, True)
        pred = self.model.predict(X)
        pred_df = pd.DataFrame(pred, columns = [1, 2, 3, 4])
        result = pred_df[[1, 2, 3, 4]].idxmax(axis = 1)
        return result.map(POVERTY_MAPPING)