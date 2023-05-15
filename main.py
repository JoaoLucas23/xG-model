from tqdm import tqdm
import d6tflow as d6t
import pandas as pd

from predictions import PredictXg
from wyDataLoader import wyLoadData
from wyDataLoader import wyLoadTimePlayed
from features import PenaltyFeatures
from features import GetFeatures

DATA_DIR = "H:/Documentos/SaLab/Soccermatics/Wyscout Data"

ENGLAND = 'English first division'
SPAIN = 'Spanish first division'
TRAIN_COMPS = ['French first division', 'German first division', 'Italian first division']

wyFlow = d6t.Workflow(wyLoadData, params={'data_dir':DATA_DIR})
wyFlow.run()
wyData = wyFlow.outputLoad()

playedFlow = d6t.Workflow(wyLoadTimePlayed, params={'data_dir':DATA_DIR})
playedFlow.run()
played_time = playedFlow.outputLoad()
minutes_table = played_time.groupby('player_name')['minutes_played'].sum().reset_index(drop=False)
minutes_table = minutes_table.rename(columns={'sum': 'minutes_played'})

englandWorkflow = d6t.Workflow(PredictXg, params={'competition': ENGLAND, 'train_comps': TRAIN_COMPS})
englandWorkflow.run()
englandXg = englandWorkflow.outputLoad()
englandPenalties = PenaltyFeatures(competition=ENGLAND)
englandPenalties.run()
englandPenalties = englandPenalties.outputLoad()
englandXg = pd.concat([englandXg, englandPenalties],axis=0)
englandXg = englandXg.merge(wyData)

spainWorkflow = d6t.Workflow(PredictXg, params={'competition': SPAIN, 'train_comps': TRAIN_COMPS})
spainWorkflow.run()
spainXg = spainWorkflow.outputLoad()
spainPenalties = PenaltyFeatures(competition=SPAIN)
spainPenalties.run()
spainPenalties = spainPenalties.outputLoad()
spainXg = pd.concat([spainXg, spainPenalties],axis=0)
spainXg = spainXg.merge(wyData)

shotsXg = pd.concat([englandXg, spainXg], axis=0).reset_index(drop=True)

xg_table = shotsXg.groupby('player_name')['xg'].agg(['count','sum']).reset_index(drop=False)
goal_table = shotsXg.groupby('player_name')['goal'].sum().reset_index(drop=False)
summ_table = xg_table.merge(goal_table).merge(minutes_table)
summ_table = summ_table.rename(columns={'sum':'xg'})
summ_table['xg_per_90'] = (summ_table['xg'] * 90) / summ_table['minutes_played'] 
summ_table['xg_per_shot'] = summ_table['xg'] / summ_table['count']
summ_table['goal_per_90'] = (summ_table['goal'] * 90) / summ_table['minutes_played'] 
summ_table['goal_per_shot'] = summ_table['goal'] / summ_table['count']
summ_table = summ_table.sort_values(['xg','goal','xg_per_90','xg_per_shot','goal_per_90','goal_per_shot'], ascending=False).reset_index(drop=False)


