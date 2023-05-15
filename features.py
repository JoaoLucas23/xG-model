import pandas as pd
import d6tflow as d6t
import numpy as np
from tqdm import tqdm

from loader import LoadWyscoutToSPADL


class GetFeatures(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def requires(self):
        return LoadWyscoutToSPADL(competition=self.competition)

    def run(self):
        actions = self.inputLoad()
        shots_mask = ['shot_freekick', 'shot']

        shots = actions.loc[actions['type_name'].isin(shots_mask)].reset_index(drop=False)
        shots = shots.rename(columns={'index': 'original_action_id'})

        goal_x = 105
        goal_y = 34
        # get shot distance to center of goal
        shots['distance'] = np.sqrt((goal_x - shots['start_x']) ** 2 + (goal_y - shots['start_y']) ** 2)
        # get shot angle
        shots['angle'] = np.arctan(7.32 * (goal_x - shots['start_x']) / ((goal_x - shots['start_x']) ** 2 + (goal_y - shots['start_y']) ** 2 - (7.32 / 2) ** 2))
        shots['angle'] = shots['angle'].apply(lambda a: a * (180 / np.pi) if a > 0 else (a + np.pi) * (180 / np.pi))
        # get angle squared
        shots['angle2'] = shots['angle'] ** 2
        # get distance squared
        shots['distance2'] = shots['distance'] ** 2
        # get distance times angle
        shots['dist_angle'] = shots['distance'] * shots['angle']
        # get previous event
        shots['prev_action'] = shots['original_action_id'].apply(lambda x: actions.at[x - 1, 'type_name'])
        # get goal or not
        shots['goal'] = shots['result_name'].apply(lambda r: 1 if r == 'success' else 0)

        self.save(shots)


class PenaltyFeatures(d6t.tasks.TaskCSVPandas):
        competition = d6t.Parameter()

        def requires(self):
            return LoadWyscoutToSPADL(competition=self.competition)
        
        def run(self):
            actions = self.inputLoad()
            shots_mask = ['shot_penalty']

            penalties = actions.loc[actions['type_name'].isin(shots_mask)].reset_index(drop=False)
            penalties = penalties.rename(columns={'index': 'original_action_id'})
            penalties['goal'] = penalties['result_name'].apply(lambda r: 1 if r == 'success' else 0)
            penalties_taken = len(penalties)
            penalty_goals = len(penalties.loc[penalties.goal==1])
            xg = penalty_goals / penalties_taken
            penalties['xg'] = xg
            
            self.save(penalties)
            