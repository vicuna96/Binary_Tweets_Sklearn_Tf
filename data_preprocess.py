import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from helper_functions import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import spacy

def clean_df(dataset,testset):
    for column in dataset.columns:
        if column not in testset.columns:
            dataset.drop(column,axis=1, inplace=True)


    dataset['http']=dataset['text'].apply(count_http)
    dataset['text']=dataset['text'].apply(remove_http)
    dataset['ats']=dataset['text'].apply(lambda x : x.count('@'))
    dataset['tags']=dataset['text'].apply(get_ats)
    dataset['text']=dataset['text'].apply(remove_ats)
    dataset['text'] = dataset['text'].apply(strip_punctuation)

    for column in dataset.columns:
        if dataset[column].dtype == np.bool:
            dataset[column]=dataset[column].apply(bool_to_int)

    # Days and seconds are relative '12/14/15' just because of the dataset
    dataset['days'] = dataset.created.apply(to_days)
    dataset['seconds'] = dataset.created.apply(to_seconds)
    dataset['replyToSN'] = dataset.replyToSN.apply(lambda x : 1 if x else 0)
    dataset['latitude'] = dataset.latitude.apply(nan_to_0)
    dataset['longitude'] = dataset.longitude.apply(nan_to_0)

    dataset.drop('created',axis=1,inplace=True)
    dataset.drop('screenName',axis=1,inplace=True)
    dataset.drop('replyToUID',axis=1,inplace=True)
    dataset.drop('days',axis=1,inplace=True)
    # Features to be used by our model
    features = []

    for column in dataset.columns:
        if column not in ['id','id.1','days','seconds','tags','text']:
            lst = dataset[column].unique()
            if len(lst)>1:
                features.append(column)

    #features = features + ['days','seconds']
    features = features + ['seconds']

    return features, dataset

def normalize_df(dataset, testset):
    y_train = np.array(dataset.label.apply(lambda x: 0 if x == -1 else 1).tolist())

    features_train, train_df = clean_df(dataset,testset)
    features_test, test_df = clean_df(testset,testset)

    _, f_t = train_df.shape
    _, f_te = test_df.shape

    assert(f_t == f_te)

    cv=CountVectorizer(max_df=0.8) # max_features=10000
    word_count_vector=cv.fit_transform(train_df['text'].tolist() + test_df['text'].tolist())

    tfidf_transformer = TfidfTransformer(smooth_idf=True)
    tfidf_transformer.fit(word_count_vector)

    train_text_matrix = tfidf_transformer.transform(cv.transform(train_df['text']))
    test_text_matrix = tfidf_transformer.transform(cv.transform(test_df['text']))

    train_df, test_df = train_df[features_train], test_df[features_train]

    # Get range of df_train to normalize data
    df_norm = (train_df.max() - train_df.min())

    print("we are using the following features: "+str(features_train))
    train_feature_matrix = np.array(train_df/df_norm)
    test_feature_matrix = np.array(test_df/df_norm)


    x_train = hstack([train_feature_matrix,train_text_matrix]).toarray()
    x_test = hstack([test_feature_matrix,test_text_matrix]).toarray()

    print("The size of our training data is "+str(x_train.shape))
    print("The size of our testing data is "+str(x_test.shape))

    return x_train, x_test, y_train

def text_clean(dataset, testset):
    y_train = np.array(dataset.label.apply(lambda x: 0 if x == -1 else 1).tolist())

    features_train, train_df = clean_df(dataset,testset)
    features_test, test_df = clean_df(testset,testset)

    _, f_t = train_df.shape
    _, f_te = test_df.shape

    assert(f_t == f_te)

    nlp = spacy.load('en_core_web_lg')

    text_train = train_df.text.tolist()
    text_test = test_df.text.tolist()

    text_vectors_train = np.array([nlp(tweet).vector for tweet in text_train])
    text_vectors_test = np.array([nlp(tweet).vector for tweet in text_test])

    train_df, test_df = train_df[features_train], test_df[features_train]

    # Get range of df_train to normalize data
    df_norm = (train_df.max() - train_df.min())

    print("we are using the following features: "+str(features_train))
    train_feature_matrix = np.array(train_df/df_norm)
    test_feature_matrix = np.array(test_df/df_norm)

    print(train_feature_matrix.shape)
    print(text_vectors_train.shape)

    x_train = np.hstack([train_feature_matrix,text_vectors_train])
    x_test = np.hstack([test_feature_matrix,text_vectors_test])

    print("The size of our training data is "+str(x_train.shape))
    print("The size of our testing data is "+str(x_test.shape))

    return x_train, x_test, y_train

def get_data(data_fun):
    dataset = read_csv('train.csv')
    testset = read_csv('test.csv')
    return data_fun(dataset, testset)