import pandas as pd
import d6tflow as d6t
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from features import GetFeatures

class TrainXgModel(d6t.tasks.TaskPickle):
    def requires(self):
        return GetFeatures()

    def run(self):
        shots = self.inputLoad()

        # select features
        feature_cols = ['angle', 'distance', 'distance2', 'angle2', 'dist_angle','prev_action','bodypart_name','type_name']
        X = shots[feature_cols]
        y = shots['result']



        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # train model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # predict test data
        y_pred = model.predict(X_test)

        # evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy: {:.2f}'.format(accuracy))

        self.save(model)