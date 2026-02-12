# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-deepmind/disentangled_rnns/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`
* Update the version in __init__.py

-->

## [Unreleased]

- Refactor DatasetRNN to be subclassable for different target types and
  introduce subclasses for the existing target types. DatasetRNN can no longer
  be used directly.

## [0.1.4] - 2026-01-22

- Modify default behavior of DatasetRNN to use random batching, and allow
  passing a random number generator for reproducibility.
- Modify DatasetRNN to return a dict instead of a tuple. This paves the way for
  generalizations with fields other than just inputs and targets.

## [0.1.3] - 2026-01-15

- Fix a bug which prevented some two_armed_bandit agents and environments from working correctly to generate synthetic datasets
- Add a new loss for jointly training on continuous and categorical targets. The intended use-case is NeuroDisRNN.
- Improve typing for rnn params
- Fix plotting functions for NeuroDisRNN

## [0.1.2] - 2025-11-17

- Validate input types to DatasetRNN
- Improve docstrings in rnn_utils
- Update code for plotting two-armed bandit session data and add examples to
  the notebook

## [0.1.1] - 2025-11-10

* Initial Release on PyPi

Previous version was not released on PyPI, so bumping up the version to trigger
a release and act as a baseline for future releases.

## [0.1.0] - 2022-01-01

* Initial release

[Unreleased]: https://github.com/google-deepmind/disentangled_rnns/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/google-deepmind/disentangled_rnns/releases/tag/v0.1.1
[0.1.2]: https://github.com/google-deepmind/disentangled_rnns/releases/tag/v0.1.2
[0.1.3]: https://github.com/google-deepmind/disentangled_rnns/releases/tag/v0.1.3
[0.1.4]: https://github.com/google-deepmind/disentangled_rnns/releases/tag/v0.1.4
