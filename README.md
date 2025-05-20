# Cognitive Model Discovery via Disentangled RNNs

Disentangled RNN (DisRNN) is a recurrent neural network architecture designed
for discovering interpretable dynamical systems consistent with a dataset. It
includes several architectural features that encourage simplicity, in the sense
of having a small number of latent variables carrying independent information
and updated in a sparse way.

We have explored fitting these to behavioral data from humans and other animals
performing simple learning and decision-making tasks, and found that the
resulting systems perform well as cognitive models and can readily be
interpreted. You can read more about this work in our paper [Cognitive Model Discovery via Disentangled RNNs](https://proceedings.neurips.cc/paper_files/paper/2023/file/c194ced51c857ec2c1928b02250e0ac8-Paper-Conference.pdf).

The code here allows generating synthetic datasets, packaging laboratory
datasets, training disRNNs with different hyperparameters as well as standard
RNNs, and inspecting the fit networks.

## Exploring DisRNN in Colab

We provide several ipynb notebooks you can use to explore DisRNN. The links
below will open these notebooks in Google Colab. We recommend creating a copy so
that you will be able to edit the notebook (File -> Save a copy in Drive), and
connecting your notebook to a GPU or TPU backend (Connect button in the top
right -> Connect to a hosted runtime)

*   The
    [Train GRU](https://colab.research.google.com/github/google-deepmind/disentangled_rnns/blob/main/disentangled_rnns/notebooks/train_single_gru.ipynb)
    notebook demonstrates fitting a synthetic using a gated recurrent unit (GRU)
    network. The GRU is a popular network architecture and, with the correct
    hyperparameters and a sufficiently large dataset, is expected to provide
    very good quality-of-fit in most situations.
*   The
    [Train DisRNN](https://colab.research.google.com/github/google-deepmind/disentangled_rnns/blob/main/disentangled_rnns/notebooks/train_single_disrnn.ipynb)
    notebook demonstrates fitting a syntheic dataset with a DisRNN network. It
    also demonstrates some of the tools available for inspecting the fit DisRNN
    and interpreting the resulting model.
*   The
    [Train Multisubject DisRNN](https://colab.research.google.com/github/google-deepmind/disentangled_rnns/blob/main/disentangled_rnns/notebooks/train_multisubject_disrnn.ipynb])
    notebook demonstrates fitting a synthetic dataset containing data from
    multiple "individuals" which vary parametrically in their cognitive
    strategy. We use a "Multisubject DisRNN" to fit both similarities and
    differences using a single network. This combines ideas from DisRNN with
    prior ideas from the literature about disentangled subject embeddings
    ([Dezfouli et al., 2019](https://papers.nips.cc/paper_files/paper/2019/file/e077e1a544eec4f0307cf5c3c721d944-Paper.pdf),
    [Song et al., 2021](https://escholarship.org/uc/item/3wj7w4x7))

## Installing and running locally

These instruction assume you will be using a virtual environment created with
[conda](https://anaconda.org/anaconda/conda).

1. **Create and activate the virtual environment**
  ```shell
  conda create --name disrnn_venv python=3.11
  conda activate disrnn_venv
  ```

2. **Install the version of JAX suitable for your hardware**
    *   **For CPU only:** `pip install -U "jax[cpu]"`
    *   **For NVIDIA GPU:** `pip install -U "jax[cuda12]"`
    *   **For other architectures:** Consult the
        [Official JAX Installation Guide](https://github.com/jax-ml/jax#installation).

3. **Clone the github repo and install remaining requirements**
  ```shell
  git clone https://github.com/google-deepmind/disentangled_rnns.git
  !git clone https://github.com/google-deepmind/disentangled_rnns
  %cd disentangled_rnns
  !pip install .
  !pip install -r requirements.txt
  %cd ..
  ```

## Usage

### Selecting a Dataset

Datasets are generated or packaged using the functions in `get_datasets.py`.
Five options are available:

* `get_q_learning_dataset`: Generates a synthetic dataset from a Q-Learning agent with standard parameters performing the drifting two-armed bandit task.
* `get_actor_critic_dataset`: Generates a synthetic dataset from an Actor-Critic agent with standard parameters performing the drifting two-armed bandit task.
* `get_bounded_accumulator_dataset`: Generates a synthetic dataset from a Bounded Accumulator agent with standard parameters performing the click accumulation task.
* `get_rat_bandit_datasets`: Packages laboratory datasets from rats performing the drifting two-armed bandit dataset from [Miller, Botvinick, and Brody, 2018](https://www.biorxiv.org/content/10.1101/461129v3). Requires internet access to download [the dataset](https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356).
* `get_pclicks_datasets`: Packages laboratory datasets from rats performing the click accumulation task from [Brunton, Botvinick, and Brody, 2013](https://pubmed.ncbi.nlm.nih.gov/23559254/). Requires internet access to download [the dataset](https://github.com/Brody-Lab/brunton_dataset).

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
[Cognitive Model Discovery via Disentangled RNNs](https://proceedings.neurips.cc/paper_files/paper/2023/file/c194ced51c857ec2c1928b02250e0ac8-Paper-Conference.pdf)

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
