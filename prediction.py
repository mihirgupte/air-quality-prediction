# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
import joblib
import scipy.stats as stats

# Initialising global variables
COLUMNS_TO_BINARIZE = ['Benzene', 'Toluene', 'Xylene']
COLUMNS_TO_YEO = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
SEED = 100
BINARIZE_THRESHOLD = {
    'Benzene' : 30,
    'Xylene' : 30,
    'Toluene' : 30,
}
YEO_TRANSFORM_PARAMS = {
    'CO': -1.0633958776243513,
    'NH3': 0.15070997294840976,
    'NO': -0.1203892296507965,
    'NO2': 0.16068251821046106,
    'NOx': 0.19306461216459575,
    'O3': 0.41101531529167773,
    'PM10': 0.23448117542164032,
    'PM2.5': -0.026893075959141775,
    'SO2': -0.249627316956199
}
COLUMNS_TO_DROP = COLUMNS_TO_BINARIZE+['City','Date','NOx','AQI_Bucket']
COLUMNS_TO_LOG = ['AQI']
MODEL_FILE = './Regression model.pkl'
SCALER_FILE = './scaler.pkl'

# Defining functions for transformers

class BinarizeTransformer(BaseEstimator,TransformerMixin):

    def __init__(self,variables,mappings):
        if not isinstance(variables,list):
            raise ValueError('variables should be in the form of a list')
        self.variables = variables
        self.mappings = mappings

    def fit(self,X,y=None):
        return self

    def binarize(self, value, threshold):
        if value<=threshold:
            return 0
        return 1

    def transform(self,X):
        for col in self.variables:
            new_col = col+"_binarized"
            X[new_col] = X[col].apply(lambda x: self.binarize(x, self.mappings[col]))
        return X

class YeoTransformer(BaseEstimator,TransformerMixin):

    def __init__(self,variables,mappings):
        if not isinstance(variables,list):
            raise ValueError('variables should be in the form of a list')
        self.variables = variables
        self.mappings = mappings

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        for col in self.variables:
            yeo_t = stats.yeojohnson(X[col],self.mappings[col])
            X[col] = yeo_t
        return X

class ColumnTransformer(BaseEstimator,TransformerMixin):

    def __init__(self,variables):
        if not isinstance(variables,list):
            raise ValueError('variables should be in the form of a list')
        self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        for col in self.variables:
            try:
                X = X.drop(columns=[col])
            except:
                continue
        return X

# Creating a pipeline

model_lr = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

pipeline = Pipeline([
    ('binarize', BinarizeTransformer(variables=COLUMNS_TO_BINARIZE, mappings=BINARIZE_THRESHOLD)),
    ('yeo_transform',YeoTransformer(variables=COLUMNS_TO_YEO, mappings=YEO_TRANSFORM_PARAMS)),
    ('col_transform', ColumnTransformer(variables=COLUMNS_TO_DROP)),
    ('scaler', scaler),
    ('regression', model_lr)
])

# Sample prediction

l = {
    "PM2.5" :   83.13,
    "PM10"  :   118.180852,
    "NO"    :   6.93,
    "NO2"   :   28.71,
    "NOx"   :   33.72,
    "NH3"   :   24.96941,
    "CO"    :   6.93,
    "SO2"   :   49.52,
    "O3"    :   59.76,
    "Benzene"   :   0.020000,
    "Toluene"   :   0.000000,
    "Xylene"    :   3.140000,
}

data = pd.DataFrame(l,index=[0])

def predict(data):
    return np.exp(pipeline.predict(data))[0]