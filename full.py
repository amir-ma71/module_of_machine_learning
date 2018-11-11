import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from tkinter.filedialog import askopenfilename
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
import pickle

# ask user for choose the model that need
# to be continue to complete...
def load_model():
    num_model = int(input("لطفا عدد مدل را انتخاب کنید: \n 1. درخت تصمیم \n 2. شبکه عصبی\n"))
    if num_model == 1:
        num_algh = int(input("لطفا نوع درخت تصمیم را انتخاب کنید: \n 1. طبقه بندی \n 2. رگرسیون\n"))
        if num_algh == 1:
            num_algh2 = int(input("لطفا نوع الگوریتم را انتخاب کنید: \n 1. جینی \n 2. انتروپی\n"))
            DecisionTree.learn_model(load_data(),num_algh2)
        else:
            DecisionTree.learn_model(load_data(),3)
    return

# load Data
def load_data():
    # select separator
    separator = input("لطفا جداکننده را وارد کرده و سپس فایل دیتاست را انتخاب کنید: \n")
    # reading file path
    filepath = askopenfilename()
    dataset = pd.read_csv(filepath, sep=separator, encoding="utf-8")
    featurelist = list(dataset)

    # delete features that don't use
    print("لیست ویژگی های دیتاست به شرح ذیل است:")
    for i in range(len(featurelist)):
        print(i, featurelist[i])
    deleteAtr = input("آیا تمایل به حذف تعدادی از ویژگی ها را دارید؟)Y/N) \n")
    delete_Atr = []
    if deleteAtr == "Y" or deleteAtr == "y":
        print(
            "لطفا اندیس ویژگی ای که قصد حذف آن را دارید وارد کرده و بعد از هر اندیس اینتر بزنید و در انتها * را وارد کنید")
        while True:
            c = input()
            if c == "*":
                break
            else:
                delete_Atr.append(featurelist[int(c)])
    dataset.drop(delete_Atr, axis=1, inplace=True)

    # select Label and put it at the end of list
    label = featurelist[int(input("لطفا اندیس برچسب را وارد کنید\n"))]
    label2 = dataset.pop(label)
    dataset[label] = label2

    return dataset


class DecisionTree:
    def __init__(self, dataset,num_model):
        self.balance_data = dataset
        self.numb_model = num_model

    def learn_model(dataset,num_model):

        def gini():
            # Decision Tree Classifier with criterion gini index
            clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=15, min_samples_leaf=7)
            clf_gini.fit(X_train, y_train)
            print("train gini accuracy: ", clf_gini.score(X_train, y_train) * 100)

            # save model
            pickle.dump(clf_gini, open("DTmodel_gini.pkl", "wb"))

        def entropy():
            clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=15,
                                                 min_samples_leaf=7)
            clf_entropy.fit(X_train, y_train)
            print("train entropy accurcy: ", clf_entropy.score(X_train, y_train) * 100)

            # save model
            pickle.dump(clf_entropy, open("DTmodel_entropy.pkl", "wb"))

        def regression():
            forest = RandomForestRegressor()
            forest.fit(X_train, y_train)
            # save model
            pickle.dump(forest, open("DTmodel_regressor.pkl", "wb"))
            print("train regressor accurcy: ", forest.score(X_train, y_train) * 100)

        balance_data = dataset
        numb_model = num_model

        # add id to each record of feature for gini or entropy
        column = list(balance_data)
        if numb_model != 3:
            for i in column:
                newIdName = 'id' + "_" + i
                balance_data[newIdName] = (balance_data[i]).astype('category').cat.codes
                balance_data.drop(i, axis=1, inplace=True)

        X = balance_data.values[:, 0:len(column) - 2]
        Y = balance_data.values[:, len(column) - 1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

        if numb_model == 1:
            gini()
        elif numb_model == 2:
            entropy()
        else:
            regression()

    def predict_model(adress_pck, data_pred):
        model = pickle.load(open(adress, "rb"))
        y_pred = model.predict(data_pred)

        return y_pred


# Main

load_model()
DecisionTree.predict_model()