from django.shortcuts import render

from django.http import HttpResponse
from .forms import SearchForm
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def Home(request):
    form = SearchForm(request.POST or None)
    response = None
    if form.is_valid():
        value = form.cleaned_data.get("q")

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
        y_pred = clf.predict(X_test)
        message = value
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        if(my_prediction== 1):
            print("Spam")
            response = "Spam"
        else:
            print("Ham")
            response = "Ham" 

        return render(request, 'result.html', {"response": response})
    return render(request, 'form.html', {"form": form})




