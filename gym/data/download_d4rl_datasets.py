import gymnasium as gym
import numpy as np
import collections
import pickle
import minari



dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0", download=True)
print(dataset)
 

# # Define dataset names
minari_datasets = {
    'halfcheetah': ['mujoco/halfcheetah/medium-v0', 'mujoco/halfcheetah/simple-v0', 'mujoco/halfcheetah/expert-v0'],
    'hopper': ['mujoco/hopper/medium-v0', 'mujoco/hopper/simple-v0', 'mujoco/hopper/expert-v0'],
    'walker2d': ['mujoco/walker2d/medium-v0', 'mujoco/walker2d/simple-v0', 'mujoco/walker2d/expert-v0']
}

datasets = []

for env_name, dataset_list in minari_datasets.items():
    for dataset_id in dataset_list:
        dataset = minari.load_dataset(dataset_id, download=True)  # ✅ Use dataset_id dynamically
        paths = []
        
        for episode in dataset.iterate_episodes():  # ✅ Iterate over episodes properly
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
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'Number of samples collected: {num_samples}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

        with open(f'{dataset_id.replace("/", "_")}.pkl', 'wb') as f:
            pickle.dump(paths, f)
            
# Output:
"""
<minari.dataset.minari_dataset.MinariDataset object at 0x77180d3f4f70>
Collected 1000 episodes from mujoco/halfcheetah/medium-v0
Number of samples collected: 1000000
Trajectory returns: mean = 12089.210357562066, std = 3008.948862141606, max = 14238.905064312376, min = 234.62614125796586
Collected 1000 episodes from mujoco/halfcheetah/simple-v0
Number of samples collected: 1000000
Trajectory returns: mean = 6926.754686471576, std = 412.9061818637937, max = 7297.430151742212, min = 1253.28180889769
Collected 1000 episodes from mujoco/halfcheetah/expert-v0
Number of samples collected: 1000000
Trajectory returns: mean = 16242.92451511629, std = 1018.3841708481308, max = 16584.931906529895, min = -79.20461737383002
Collected 1327 episodes from mujoco/hopper/medium-v0
Number of samples collected: 999404
Trajectory returns: mean = 2817.788804523498, std = 877.0932607499858, max = 3908.854598719914, min = 1350.3360739190998
Collected 1952 episodes from mujoco/hopper/simple-v0
Number of samples collected: 999206
Trajectory returns: mean = 1655.6250879864187, std = 1139.7355126341458, max = 3298.5853804160047, min = 574.2854485574121
Collected 1086 episodes from mujoco/hopper/expert-v0
Number of samples collected: 999164
Trajectory returns: mean = 3857.803603002423, std = 644.2104757416718, max = 4376.326745837254, min = 395.6389765468187
Collected 1044 episodes from mujoco/walker2d/medium-v0
Number of samples collected: 999613
Trajectory returns: mean = 5701.642653750667, std = 1179.2482799615473, max = 6198.89836798548, min = 0.2035283034852593
Collected 1017 episodes from mujoco/walker2d/simple-v0
Number of samples collected: 999942
Trajectory returns: mean = 4075.5686730537645, std = 492.95606400918473, max = 4561.616126406589, min = 193.0315345078412
Collected 1000 episodes from mujoco/walker2d/expert-v0
Number of samples collected: 999190
Trajectory returns: mean = 6847.792741361276, std = 190.46582953184353, max = 6972.797332575603, min = 2664.171691972849
"""
