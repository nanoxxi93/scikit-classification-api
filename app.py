import os
import io
import copy
import json
from enum import Enum
import uvicorn
from fastapi import FastAPI, Query, Form, File, UploadFile
import logging
import datetime
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import pickle
import joblib

class OperationName(str, Enum):
    select = 'select'
    insert = 'insert'
    update = 'update'
    delete = 'delete'

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

app = FastAPI()

logging.basicConfig(
    handlers=[
        logging.FileHandler(filename='log.log', encoding='utf-8', mode='a+')
    ],
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y%m%d.%H%M%S'
)

# stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
def setstopwords_fn():
    prepositions =['a','ante','bajo','cabe','con','contra','de','desde','en','entre','hacia','hasta','para','por','según','sin','so','sobre','tras']
    prep_alike = ['durante','mediante','excepto','salvo','incluso','más','menos']
    adverbs = ['no','si','sí']
    articles = ['el','la','los','las','un','una','unos','unas','este','esta','estos','estas','aquel','aquella','aquellos','aquellas']
    aux_verbs = ['he','has','ha','hemos','habéis','han','había','habías','habíamos','habíais','habían']
    return set(stopwords.words('spanish')+prepositions+prep_alike+adverbs+articles+aux_verbs+list(punctuation))
spanish_stop_words = setstopwords_fn()

def preprocessing_df(df, colx, coly):
    ### Preprocesamiento ###
    # Pasando a minúsculas el texto
    df[colx] = df[colx].str.lower()
    # Eliminando los signos de puntuacion
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for i in punctuation:
        df[colx] = df[colx].str.replace('\\' + i, ' ', regex=True)
    # for i in spanish_stop_words:
    #   df[colx] = df[colx].replace(i, ' ')
    # Eliminando tildes
    df[colx] = df[colx].str.replace('[á]', 'a', regex=True)
    df[colx] = df[colx].str.replace('[é]', 'e', regex=True)
    df[colx] = df[colx].str.replace('[í]', 'i', regex=True)
    df[colx] = df[colx].str.replace('[ó]', 'o', regex=True)
    df[colx] = df[colx].str.replace('[ú]', 'u', regex=True)
    df[colx] = df[colx].str.replace('[à]', 'a', regex=True)
    df[colx] = df[colx].str.replace('[è]', 'e', regex=True)
    df[colx] = df[colx].str.replace('[ì]', 'i', regex=True)
    df[colx] = df[colx].str.replace('[ò]', 'o', regex=True)
    df[colx] = df[colx].str.replace('[ù]', 'u', regex=True)
    df[colx] = df[colx].str.replace('[â]', 'a', regex=True)
    df[colx] = df[colx].str.replace('[ê]', 'e', regex=True)
    df[colx] = df[colx].str.replace('[î]', 'i', regex=True)
    df[colx] = df[colx].str.replace('[ô]', 'o', regex=True)
    df[colx] = df[colx].str.replace('[û]', 'u', regex=True)
    # Eliminando espacios dobles
    df[colx] = df[colx].replace('\s+', ' ', regex=True)
    # Eliminando espacios iniciales y finales
    df[colx] = df[colx].str.strip()
    df[coly] = df[coly].str.replace('\"', '', regex=True)
    df[coly] = df[coly].str.replace('__label__', ' __label__')
    df[coly] = df[coly].str.strip()
    return df

def open_file(filename, fileextension, formfile, columns):
    if fileextension == 'csv':
        return pd.read_csv(formfile, usecols=columns)
    elif fileextension == 'txt':
        return pd.read_csv(formfile, usecols=columns)
    elif fileextension == 'xlsx':
        return pd.read_excel(formfile, usecols=columns, engine='openpyxl')
    else:
        raise ValueError('Archivo inválido.')

def preprocess_labels(df_data):
    df_data[df_data.keys()[1]] = df_data[df_data.keys()[1]].str.lower()
    df_data[df_data.keys()[1]] = df_data[df_data.keys()[1]].replace('\s+', '_', regex=True)
    if not str.__contains__(df_data[df_data.keys()[1]][0],'label'):
        df_data[df_data.keys()[1]] = '__label__' + df_data[df_data.keys()[1]]
    return df_data

def fn_autodefine_columns(df):
    x = df.iloc[:, :-1].values # independent variable matrix
    y = df.iloc[:, 3].values # dependent variable matrix
    return x, y

def fn_clean_null(df):
    return df.dropna()

def fn_clean_missing_data(x):
    from sklearn.preprocessing.impute import SimpleImputer
    imputer = SimpleImputer(missing_values='NaN', strategy='mean')
    imputer = SimpleImputer.fit(x[:,1:3])
    x[:,1:3] = SimpleImputer.transform(x[:,1:3])
    return x

def fn_encode_categorical_data(x, y):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_x = LabelEncoder()
    x[:,0] = labelencoder_x.fit_transform(X[:,0])

    onehotenconder = OneHotEncoder(categorical_features=[0])
    x = onehotenconder.fit_transform(x).toarray()

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    return x, labelencoder_x, y, labelencoder_y

def fn_train_test_split(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    return x_train, x_test, y_train, y_test

# Split Train Test data
def set_train_test_split(df, col_x, col_y, filename):
    texts = df[col_x]
    labels = df[col_y]
    from sklearn.feature_extraction.text import TfidfVectorizer
    # min_df min quantity of texts for a word
    # max_df max percentage of texts for a word
    min_df = 5
    max_df = 0.7
    if (len(df) < 5):
        min_df = 1
        max_df = 1.0
    tfid = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=spanish_stop_words)
    # X is the numeric representation of every word in texts
    X = tfid.fit_transform(texts)
    # Saving tfidf
    pickle.dump(X, open('{}_tfidf.pickle'.format(filename), "wb"))
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
    return ({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test})

def fn_standard_scaler(x):
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    return x_train, x_test

# LinearSVC
from sklearn.svm import LinearSVC
def svcclassifier_fn(x,y,c=1.0,i=1000,t=1e-3,d=False):
    model = LinearSVC(C=c, max_iter=i, tol=t, dual=d)
    model.fit(x, y)
    return model

# GaussianNB
from sklearn.naive_bayes import GaussianNB
def gnbcclassifier_fn(x,y,p=None):
    model = GaussianNB(priors=None)
    model.fit(x, y)
    return model

# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
def knclassifier_fn(x,y,n=3):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(x, y)
    return model

# RandomForestClassifier
from sklearn.ensemble  import RandomForestClassifier
def rfclassifier_fn(x,y,n=1000,r=0):
    model = RandomForestClassifier(n_estimators=n, random_state=r)
    model.fit(x, y)
    return model

# Scikit Scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def score_fn(score, real, pred, modelname, avg='weighted'):
    acc = accuracy_score(real, pred, normalize = True)
    num_acc = accuracy_score(real, pred, normalize = False)
    prec = precision_score(real, pred, average=avg, zero_division=0)
    recall = recall_score(real, pred, average=avg, zero_division=0)
    score[modelname] = [acc, num_acc, prec, recall]

def train_model_fn(model_data, score):
    if (model_data['y_train'].nunique() > 1):
        model1 = knclassifier_fn(model_data['x_train'], model_data['y_train'])
        y_pred = model1.predict(model_data['x_train'])
        score_fn(score, model_data['y_train'], y_pred, 'knclassifier_train')
        y_pred = model1.predict(model_data['x_test'])
        score_fn(score, model_data['y_test'], y_pred, 'knclassifier_test')

        # model2 = gnbcclassifier_fn(model_data['x_train'].toarray(), model_data['y_train'])
        # y_pred = model2.predict(model_data['x_train'].toarray())
        # score_fn(score, model_data['y_train'], y_pred, 'gnbcclassifier_train')
        # y_pred = model2.predict(model_data['x_test'].toarray())
        # score_fn(score, model_data['y_test'], y_pred, 'gnbcclassifier_test')

        # model3 = rfclassifier_fn(model_data['x_train'], model_data['y_train'])
        # y_pred = model3.predict(model_data['x_train'])
        # score_fn(score, model_data['y_train'], y_pred, 'rfclassifier_train')
        # y_pred = model3.predict(model_data['x_test'])
        # score_fn(score, model_data['y_test'], y_pred, 'rfclassifier_test')

        # model4 = svcclassifier_fn(model_data['x_train'], model_data['y_train'])
        # y_pred = model4.predict(model_data['x_train'])
        # score_fn(score, model_data['y_train'], y_pred, 'svcclassifier_train')
        # y_pred = model4.predict(model_data['x_test'])
        # score_fn(score, model_data['y_test'], y_pred, 'svcclassifier_test')
        return model1

@app.post('/model')
async def model_controller(
    file: UploadFile = File(...),
    columns: str = Form(...),
    objective: str = Form(...)
):
    # Loading file
    filename = file.filename.rsplit(".",1)[0] # nombre del archivo
    fileextension = file.filename.rsplit(".",1)[1] # extensión del archivo
    content = file.file.read()
    # Setting columns
    col_x = columns.split(',')
    col_x.remove(objective)
    col_y = objective
    # Loading data frame
    df = open_file(filename, fileextension, io.BytesIO(content), columns.split(','))
    # df = preprocess_labels(df)
    # df = preprocessing_df(df, col_x[0], col_y)
    # Cleaning nulls
    df = fn_clean_null(df)
    # Setting model_data
    model_data = set_train_test_split(df, col_x[0], col_y, filename)
    # Setting model training and results
    score = pd.DataFrame(index=['Acurracy','#Acurracy','Precision','Recall'])
    model = train_model_fn(model_data, score)
    # Save the trained model as a pickle string.
    joblib.dump(model, '{}.pkl'.format(filename))
    return {
        'col_x': col_x,
        'col_y': col_y,
        'shape': df.shape,
        'labels': df[col_y].value_counts().to_dict(),
        'scores': score.to_dict()
    }

@app.post('/classify')
async def model_controller(
    file: UploadFile = File(...),
    column: str = Form(...),
    model_name: str = Form(...)
):
    # Loading file
    filename = file.filename.rsplit(".",1)[0] # nombre del archivo
    fileextension = file.filename.rsplit(".",1)[1] # extensión del archivo
    content = file.file.read()
    # Loading data frame
    df = open_file(filename, fileextension, io.BytesIO(content), [column])
    # Cleaning nulls
    df = fn_clean_null(df)
    # Loading model
    model = joblib.load('{}.pkl'.format(model_name))
    # Loading tfidf
    from sklearn.feature_extraction.text import TfidfVectorizer
    # min_df min quantity of texts for a word
    # max_df max percentage of texts for a word
    min_df = 5
    max_df = 0.7
    if (len(df) < 5):
        min_df = 1
        max_df = 1.0
    tfid = TfidfVectorizer(input='{}_tfidf.pickle'.format(model_name), min_df=min_df, max_df=max_df, stop_words=spanish_stop_words)
    # X is the numeric representation of every word in texts
    X = tfid.fit_transform(df[column])
    # Predict
    pred = model.predict(X)
    df['label'] = pred
    return df.to_dict('records')

@app.get('/values')
async def values_controller():
    return 'Api is running'

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8085, reload=True) # uvicorn app:app --port=8083 --reload
    # app.run(host='0.0.0.0') # flask run