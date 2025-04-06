import gymnasium as gym
import numpy as np
import collections
import pickle
import minari


# Define dataset names
minari_datasets = {
    'halfcheetah': ['mujoco/halfcheetah/medium-v0', 'mujoco/halfcheetah/simple-v0', 'mujoco/halfcheetah/expert-v0'],
    'hopper': ['mujoco/hopper/medium-v0', 'mujoco/hopper/simple-v0', 'mujoco/hopper/expert-v0'],
    'walker2d': ['mujoco/walker2d/medium-v0', 'mujoco/walker2d/simple-v0', 'mujoco/walker2d/expert-v0'],
    'reacher2d': ['mujoco/reacher/medium-v0', 'mujoco/reacher/expert-v0', 'mujoco/reacher/simple-v0']
}

REF_MIN_SCORE = {
    'walker2d': 1.6,
    'hopper': 1.6,
    'halfcheetah': -280.178953,
    'ant': 0.0,
}
REF_MAX_SCORE = {
    'walker2d': 4592.3,
    'hopper': 3234.3,
    'halfcheetah': 12135.0,
    'ant': 6444.0,
}

datasets = []

for env_name, dataset_list in minari_datasets.items():
    ref_min = REF_MIN_SCORE[env_name]
    ref_max = REF_MAX_SCORE[env_name]
    
    for dataset_id in dataset_list:
        dataset = minari.load_dataset(dataset_id, download=True) 
        paths = []
        
        for episode in dataset.iterate_episodes():
            data_ = collections.defaultdict(list)
            N = len(episode.rewards)  # ✅ Get number of steps in the episode

            for i in range(N):
                done_bool = episode.terminations[i] or episode.truncations[i]  # ✅ Use correct attributes

                for k in ['observations', 'actions', 'rewards', 'terminations', 'truncations']:
                    data_[k].append(getattr(episode, k)[i])

                if done_bool:
                    episode_data = {k: np.array(v) for k, v in data_.items()}
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)

        print(f"Collected {len(paths)} episodes from {dataset_id}")
        returns = np.array([np.sum(p['rewards']) for p in paths])
        normalised_returns = (returns - ref_min) / (ref_max - ref_min)
        print(f"Normalised returns: mean = {np.mean(normalised_returns)}, std = {np.std(normalised_returns)}, max = {np.max(normalised_returns)}, min = {np.min(normalised_returns)}")
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'Number of samples collected: {num_samples}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

        with open(f'{dataset_id.replace("/", "_")}.pkl', 'wb') as f:
            pickle.dump(paths, f)
