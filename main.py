from tqdm import tqdm
import d6tflow as d6t

from predictions import PredictXg

COMPETITION = 'English first division'
TRAIN_COMPS = ['French first division', 'German first division', 'Italian first division']

englandWorkflow = d6t.Workflow(PredictXg, params={'competition': COMPETITION, 'train_comps': TRAIN_COMPS})
englandWorkflow.run()
englandXg = englandWorkflow.outputLoad()
