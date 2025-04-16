# Copyright 2024 DeepMind Technologies Limited.
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


if __name__ == "__main__":
  absltest.main()
