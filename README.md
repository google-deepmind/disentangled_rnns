# Cognitive Model Discovery via Disentangled RNNs

This is the code associated with the following paper:

[Cognitive Model Discovery via Disentangled RNNs](https://www.biorxiv.org/content/10.1101/2023.06.23.546250v1).
Kevin Miller, Maria Eckstein, Matthew Botvinick, and Zeb Kurth-Nelson. _bioRxiv_ 2023

Disentangled RNNs are a recurrent neural network architecture including several architectural features that are designed to encourage networks to learn simple, human-interpretable model fits. In this paper, we apply these networks to synthetic and laboratory behavioral datasets from two tasks that are representative of those often used in behavioral neuroscience.

This code allows generating synthetic datasets, packaging laboratory datasets, training disRNNs with different hyperparameters, and inspecting the fit networks.

## Installation

To install the necessary requirements (run these commands from the directory
that you wish to clone `disrnn_paper` into):

```shell
git clone https://github.com/deepmind/disrnn_paper.git
python3 -m venv disrnn_venv
source disrnn_venv/bin/activate
pip install --upgrade pip
pip install -r ./requirements.txt
```


## Usage

### Selecting a Dataset

Datasets are generated or packaged using the functions in `get_datasets.py`. Five options are available:

* `get_q_learning_dataset`: Generates a synthetic dataset from a Q-Learning agent with standard parameters performing the drifting two-armed bandit task.
* `get_actor_critic_dataset`: Generates a synthetic dataset from an Actor-Critic agent with standard parameters performing the drifting two-armed bandit task.
* `get_bounded_accumulator_dataset`: Generates a synthetic dataset from a Bounded Accumulator agent with standard parameters performing the click accumulation task.
* `get_rat_bandit_datasets`: Packages laboratory datasets from rats performing the drifting two-armed bandit dataset from [Miller, Botvinick, and Brody, 2018](https://www.biorxiv.org/content/10.1101/461129v3). Requires a path to the file `tab_dataset.json`, which is available on Figshare from the authors at [this link](https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356
).
* `get_pclicks_datasets`: Packages laboratory datasets from rats performing the click accumulation task from [Brunton, Botvinick, and Brody, 2013](https://pubmed.ncbi.nlm.nih.gov/23559254/). Requires a path to the file `brunton-dataset.json`, which is available at [somewhere?](). We generated this file using the open dataset released by the authors at [this link](https://github.com/Brody-Lab/brunton_dataset), converting the precise spiketimes into binned spike counts to facilitate training discrete-time neural networks.

### Fitting a disRNN

Define a dataset, network architecture, and optimizer

```python
from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import two_armed_bandits
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import disrnn
import optax

# Synthetic dataset from a q-learning agent. See other options above.
agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
dataset = two_armed_bandits.create_dataset(
    agent,
    environment,
    n_steps_per_session=200,
    n_sessions=1000,
    batch_size=1000,
)

update_mlp_shape = (5,5,5)
choice_mlp_shape = (2,2)
latent_size = 5

def make_network():
  return disrnn.HkDisRNN(update_mlp_shape=update_mlp_shape,
                        choice_mlp_shape=choice_mlp_shape,
                        latent_size=latent_size,
                        obs_size=2, target_size=2)

learning_rate = 1e-3
opt = optax.adam(learning_rate)
```

Optimizing network parameters

```python
# Train one step to initialize
params, opt_state, losses = rnn_utils.train_network(
   make_network,
    dataset,
    dataset,
    opt = optax.adam(1e-2),
    loss="penalized_categorical",
    n_steps=0)

# Train additional steps
n_steps = 1000
params, opt_state, losses = rnn_utils.train_network(
make_network,
    dataset,
    dataset,
    loss="penalized_categorical",
    params=params,
    opt_state=opt_state,
    opt = optax.adam(1e-3),
    penalty_scale = 1e-3,
    n_steps=n_steps,
    do_plot = True)
```

### Inspecting a fit disRNN

Plot bottlenecks and update rules

```python
# Eval mode runs the network with no noise
def make_network_eval():
  return disrnn.HkDisRNN(update_mlp_shape=update_mlp_shape,
                        choice_mlp_shape=choice_mlp_shape,
                        latent_size=latent_size,
                        obs_size=2, target_size=2,
                        eval_mode=True)


disrnn.plot_bottlenecks(params, make_network_eval)
disrnn.plot_update_rules(params, make_network_eval)
```

Run network and output internal states

```python
xs, ys = next(dataset)
_ , network_states = rnn_utils.eval_network(make_network_eval, params, xs)
```


## Citing this work

If you use this code, please cite the following paper:

```
@misc{miller_disRNN_2023,
  title = {Cognitive Model Discovery via Disentangled RNNs},
  author = {Miller, Kevin J and Eckstein, Maria and Botvinick, Matthew and Kurth-Nelson, Zeb},
  journal = {Neural Information Processing Systems},
  year = {2023},
}
```
## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
