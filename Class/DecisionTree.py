import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from tkinter.filedialog import askopenfilename
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
import pickle

class DecisionTree:
    def __init__(self, dataset,num_model):
        self.balance_data = dataset
        self.numb_model = num_model

    def learn_model(dataset,num_model):

        def gini():
            # Decision Tree Classifier with criterion gini index
            clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=15, min_samples_leaf=7)
            clf_gini.fit(X_train, y_train)
            #Accuracy
            print("train gini accuracy: ", clf_gini.score(X_train, y_train) * 100)

            # save model
            pickle.dump(clf_gini, open("DTmodel_gini.pkl", "wb"))

        # Decision Tree Classifier with criterion entropy index
        def entropy():
            clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=15,
                                                 min_samples_leaf=7)
            clf_entropy.fit(X_train, y_train)

            # Accuracy
            print("train entropy accurcy: ", clf_entropy.score(X_train, y_train) * 100)

            # save model
            pickle.dump(clf_entropy, open("DTmodel_entropy.pkl", "wb"))

        # Decision Tree Classifier for regression data
        def regression():
            forest = RandomForestRegressor()
            forest.fit(X_train, y_train)
            # Accuracy
            print("train regressor accurcy: ", forest.score(X_train, y_train) * 100)

            # save model
            pickle.dump(forest, open("DTmodel_regressor.pkl", "wb"))

        balance_data = dataset
        numb_model = num_model

        # add id to each record of feature for gini or entropy
        column = list(balance_data)
        if numb_model != 3: #if not regression
            for i in column:
                newIdName = 'id' + "_" + i
                balance_data[newIdName] = (balance_data[i]).astype('category').cat.codes
                balance_data.drop(i, axis=1, inplace=True)
        # choose feature and label of dataset
        X = balance_data.values[:, 0:len(column) - 2]
        Y = balance_data.values[:, len(column) - 1]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

        if numb_model == 1:
            gini()
        elif numb_model == 2:
            entropy()
        else:
            regression()
    # load new data and predict it from model
    def predict_model(adress_pck, data_pred):
        model = pickle.load(open(adress, "rb"))
        y_pred = model.predict(data_pred)
        return y_pred

