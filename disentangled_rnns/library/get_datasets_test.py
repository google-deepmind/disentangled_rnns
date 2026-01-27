# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke tests for get_datasets.py."""

from absl.testing import absltest
from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import rnn_utils


class GetDatasetsTest(absltest.TestCase):

  def test_q_learning_dataset(self):
    """Test that synthetic Q-Learning datasets generate correctly."""
    dataset = get_datasets.get_q_learning_dataset(n_trials=10, n_sessions=10)
    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)

  def test_actor_critic_dataset(self):
    """Test that synthetic Actor-Critic datasets generate correctly."""
    dataset = get_datasets.get_actor_critic_dataset(n_trials=10, n_sessions=10)
    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)

  def test_bounded_accumulator_dataset(self):
    """Test that synthetic Bounded Accumulator datasets generate correctly."""
    dataset = get_datasets.get_bounded_accumulator_dataset(n_trials=10)
    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)

  def test_q_learning_multisubject_dataset(self):
    """Test that synthetic Q-Learning multisubject datasets generate correctly."""
    dataset = get_datasets.get_q_learning_multisubject_dataset(
        n_trials=10, n_sessions=10, alphas=[0.1, 0.2]
    )
    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)

  def test_dataset_list_to_multisubject(self):
    """Test that dataset_list_to_multisubject generates correctly."""
    dataset1 = get_datasets.get_q_learning_dataset(n_trials=10, n_sessions=10)
    dataset2 = get_datasets.get_q_learning_dataset(n_trials=12, n_sessions=10)
    multisubject_dataset = get_datasets.dataset_list_to_multisubject(
        [dataset1, dataset2]
    )
    self.assertIsInstance(multisubject_dataset, rnn_utils.DatasetRNN)


if __name__ == "__main__":
  absltest.main()
