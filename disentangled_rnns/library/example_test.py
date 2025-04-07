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

"""Test that example.py is able to run correctly."""

from disentangled_rnns import example
from google3.testing.pybase import googletest


class ExampleTest(googletest.TestCase):

  def test_example_script(self):
    try:
      example.main(None)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.fail(f"Example script failed: {e}")


if __name__ == "__main__":
  googletest.main()
