# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for Poisson clicks generation and drift-diffusion."""

from absl.testing import absltest
from disentangled_rnns.library import pclicks


class PclicksTest(absltest.TestCase):

  def test_generate_clicktrains(self):
    """Smoke test generating clicktrains."""
    xs, stim_durations = pclicks.generate_clicktrains(
        n_trials=10,
        base_click_rate=20,
        click_rate_diffs=(-38, -34, -27, -12, 12, 27, 34, 38),
        stim_duration_max=100,
        stim_duration_min=20,
    )

    self.assertEqual(xs.shape, (101, 10, 2))
    self.assertEqual(stim_durations.shape, (10,))

  def test_drift_diffusion_model(self):
    """Smoke test running DDM on clicktrains."""
    xs, _ = pclicks.generate_clicktrains(
        n_trials=10,
        base_click_rate=20,
        click_rate_diffs=(-38, -34, -27, -12, 12, 27, 34, 38),
        stim_duration_max=100,
        stim_duration_min=20,
    )
    decisions, _ = pclicks.drift_diffusion_model(xs)
    self.assertEqual(decisions.shape, (10,))
    self.assertTrue(set(decisions).issubset({0, 1}))

  def test_drift_diffusion_model_lapse(self):
    """Verify lapse flips both left (0 -> 1) and right (1 -> 0) choices."""
    xs, _ = pclicks.generate_clicktrains(n_trials=5)

    # Exclusively right clicks (negative evidence -> unlapsed decision == 0).
    xs[:, :, :] = 0.0
    xs[:5, :, 1] = 1.0
    decisions_no_lapse, _ = pclicks.drift_diffusion_model(
        xs, noise_per_click=0.0, noise_per_timestep=0.0, lapse=0.0
    )
    decisions_full_lapse, _ = pclicks.drift_diffusion_model(
        xs, noise_per_click=0.0, noise_per_timestep=0.0, lapse=1.0
    )
    self.assertEqual(list(decisions_no_lapse), [0] * 5)
    self.assertEqual(list(decisions_full_lapse), [1] * 5)

    # Exclusively left clicks (positive evidence -> unlapsed decision == 1).
    xs[:, :, :] = 0.0
    xs[:5, :, 0] = 1.0
    decisions_no_lapse, _ = pclicks.drift_diffusion_model(
        xs, noise_per_click=0.0, noise_per_timestep=0.0, lapse=0.0
    )
    decisions_full_lapse, _ = pclicks.drift_diffusion_model(
        xs, noise_per_click=0.0, noise_per_timestep=0.0, lapse=1.0
    )
    self.assertEqual(list(decisions_no_lapse), [1] * 5)
    self.assertEqual(list(decisions_full_lapse), [0] * 5)


if __name__ == "__main__":
  absltest.main()
