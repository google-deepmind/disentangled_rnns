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
    notebook demonstrates fitting a synthetic dataset with a DisRNN network. It
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
  %cd ..
  ```

4. **Test your setup using the example script**
```shell
python example.py
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
