from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

@api_view(["POST"])
def spamFilter(value):
    try:
        response = None
        text=json.loads(value.body)
        convertString=str(text)

        df = pd.read_csv('spam.csv', encoding="latin-1")
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
        X = df['v2']
        y = df['label']
        cv = CountVectorizer()
        X = cv.fit_transform(X) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        clf = MultinomialNB()
        clf.fit(X_train,y_train)
        clf.score(X_test,y_test)
        message = convertString
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        if(my_prediction== 1):
            print("Spam")
            response = "Spam"
        else:
            print("Ham")
            response = "Ham"

        return JsonResponse("The Text Message is:"+response,safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)