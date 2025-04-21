from experiment_whole_policy_evaluation import experiment_whole_policy_evaluation
from tqdm import tqdm
settings = [0]
n_repetitions = 15

for i in tqdm(range(n_repetitions), desc = f"Experiment", leave = False):
    experiment_whole_policy_evaluation(setting=settings[0])

