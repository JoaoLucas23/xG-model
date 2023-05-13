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
        feature_cols = ['angle', 'distance', 'distance2', 'angle2', 'dist_angle']
        X = shots[feature_cols]
        cat_vars = ['prev_action', 'bodypart_name', 'type_name']
        X = pd.concat([X, pd.get_dummies(shots[cat_vars])], axis=1)
        y = shots['goal']

        # Create a logistic regression model and fit it to the training data
        lr_model = LogisticRegression(max_iter=500)
        lr_model.fit(X, y)

        # Evaluate the model on the test data
        train_acc = lr_model.score(X, y)
        print("Accuracy on train set: {:.2f}%".format(train_acc * 100))

        coef_df = pd.DataFrame({'feature': X.columns, 'coeficientes': lr_model.coef_[0]})
        print(coef_df)

        self.save(lr_model)
