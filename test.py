#Bring in lightweight dependencies

import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
#from ydata_profiling import ProfileReport
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import ensemble
#from typing import Tuplefrom # cai nay khong nhan duoc? 
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()


with open("logistic_regression_clf.pkl", "rb") as file:
    clf = pkl.load(file)

class request_body(BaseModel):
    age: int
    gender: int
    height: int
    weight: int
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active:int
    
@app.post('/predict')
async def predict(data: request_body):
    test_data = [[
            data.age, 
            data.gender,
            data.height, 
            data.weight,
            data.ap_hi,
            data.ap_lo,
            data.cholesterol,
            data.gluc,
            data.smoke,
            data.alco,
            data.active
    ]]
    class_idx = clf.predict(test_data)[0]
    return {'class': clf.predict(test_data)}