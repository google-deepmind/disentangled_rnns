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

"""Test that example.py is able to run correctly."""

from absl import flags
from absl.testing import absltest
from disentangled_rnns import example

FLAGS = flags.FLAGS


class ExampleTest(absltest.TestCase):

  def test_example_script(self):
    try:
      FLAGS.n_steps_per_session = 10
      FLAGS.n_sessions = 10
      FLAGS.n_training_steps = 10
      FLAGS.n_warmup_steps = 10
      example.main(None)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.fail(f"Example script failed: {e}")


if __name__ == "__main__":
  absltest.main()
