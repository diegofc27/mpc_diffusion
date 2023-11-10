import numpy as npco
import os
import numpy as np
from pathlib import Path
import pickle
import json
idx=0
results = {'rewards':10, 'cost':2}
save_results_path = os.path.join("/home/fernandi/projects/decision-diffuser/code/weights/diffuser/safe_omni_ppo_noinv_5/", f"results_{idx}.json")
with open(save_results_path, 'w') as f:
    json.dump(results, f)