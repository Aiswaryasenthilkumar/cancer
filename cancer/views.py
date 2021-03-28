from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request,'index.html',{"predicted":""})



def predict(request):
    exp1 = float(request.GET['exp1'])
    exp2 = float(request.GET['exp2'])
    exp3 = float(request.GET['exp3'])
    exp4 = float(request.GET['exp4'])
    exp5 = float(request.GET['exp5'])
    exp6 = float(request.GET['exp6'])
    exp7 = float(request.GET['exp7'])
    exp8 = float(request.GET['exp8'])
    exp9 = float(request.GET['exp9'])
    exp10 = float(request.GET['exp10'])
    exp11 = float(request.GET['exp11'])
    exp12 = float(request.GET['exp12'])
    exp13 = float(request.GET['exp13'])
    exp14 = float(request.GET['exp14'])
    exp15 = float(request.GET['exp15'])
    exp16 = float(request.GET['exp16'])
    exp17 = float(request.GET['exp17'])
    exp18 = float(request.GET['exp18'])
    exp19 = float(request.GET['exp19'])
    exp20 = float(request.GET['exp20'])
    exp21 = float(request.GET['exp21'])
    exp22 = float(request.GET['exp22'])
    exp23 = float(request.GET['exp23'])
    exp24 = float(request.GET['exp24'])
    exp25 = float(request.GET['exp25'])
    exp26 = float(request.GET['exp26'])
    exp27 = float(request.GET['exp27'])
    exp28 = float(request.GET['exp28'])
    exp29 = float(request.GET['exp29'])
    exp30 = float(request.GET['exp30'])


    rawdata = staticfiles_storage.path('cancer_dataset.csv')
    dataset = pd.read_csv(rawdata)
    X = dataset.iloc[0:, 2:32].values
    y = dataset.iloc[0:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    yet_to_predict = np.array([[exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10, exp11, exp12, exp13, exp14, exp15, exp16, exp17, exp18, exp19, exp20, exp21, exp22, exp23, exp24, exp25, exp26, exp27, exp28, exp29, exp30]])
    y_pred = model.predict(yet_to_predict)
    accuracy = model.score(X_test, y_test)
    accuracy = accuracy*100
    accuracy = int(accuracy)
    return render(request,'index.html',{"predicted":y_pred[0],"exp1":exp1,"exp2":exp2,"exp3":exp3,"exp4":exp4,"exp5":exp5,"exp6":exp6,"exp7":exp7,"exp8":exp8,"exp9":exp9,"exp10":exp10,"exp11":exp11,"exp12":exp12,"exp13":exp13,"exp14":exp14,"exp15":exp15,"exp16":exp16,"exp17":exp17,"exp18":exp18,"exp19":exp19,"exp20":exp20,"exp21":exp21,"exp22":exp22,"exp23":exp23,"exp24":exp24,"exp25":exp25,"exp26":exp26,"exp27":exp27,"exp28":exp28,"exp29":exp29,"exp30":exp30})




def courses(request):
    data = "this is courses page"
    return HttpResponse(data)