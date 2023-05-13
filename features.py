import pandas as pd
import d6tflow as d6t
import numpy as np
from tqdm import tqdm

from loader import LoadWyscoutToSPADL


class GetFeatures(d6t.tasks.TaskCSVPandas):

    def requires(self):
        return LoadWyscoutToSPADL()

    def run(self):
        actions = self.inputLoad()
        shots_mask = (actions.type_name.isin(['shot', 'shot_freekick', 'shot_penalty']))
        shots = actions.loc[shots_mask]

        goal_x = 105
        goal_y = 34
        # get shot distance to center of goal
        shots['distance'] = np.sqrt((goal_x - shots['start_x']) ** 2 + (goal_y - shots['start_y']) ** 2)
        # get shot angle
        shots['angle'] = np.arctan(7.32 * (goal_x - shots['start_x']) / (
                    (goal_x - shots['start_x']) ** 2 + (goal_y - shots['start_y']) ** 2 - (7.32 / 2) ** 2))
        # get angle squared
        shots['angle2'] = shots['angle'] ** 2
        # get distance squared
        shots['distance2'] = shots['distance'] ** 2
        # get distance times angle
        shots['dist_angle'] = shots['distance'] * shots['angle']
        # get previous event
        shots['previous_event'] = shots['type_name'].shift(1)

        self.save(shots)
