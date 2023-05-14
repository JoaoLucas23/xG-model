import d6tflow.tasks
import pandas as pd
import d6tflow as d6t
import numpy as np
from tqdm import tqdm

from features import GetFeatures
from training import TrainXgModel


class PredictXg(d6tflow.tasks.TaskCSVPandas):
    competition = d6t.Parameter()
    train_comps = d6t.ListParameter()

    def requires(self):
        return GetFeatures(competition=self.competition), TrainXgModel(train_comps=self.train_comps)

    def run(self):
        shots = self.input()[0].load()
        model = self.input()[1].load()

        feature_cols = ['angle', 'distance', 'distance2', 'angle2', 'dist_angle']
        X = shots[feature_cols]
        cat_vars = ['prev_action', 'bodypart_name', 'type_name']
        X = pd.concat([X, pd.get_dummies(shots[cat_vars])], axis=1)

        shots['xg'] = model.predict_proba(X)[:, 1]

        self.save(shots)
