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

#!/usr/bin/env python3
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

"""Fails only on type errors that a pull request newly introduces.

The repository has a substantial number of pre-existing pyrefly errors that are
suppressed inline. Running `pyrefly check` and failing on any error would
therefore require a large suppression sweep before the check could be turned on
at all. Instead this compares two reports -- one from the base branch and one
from the pull request -- and fails only on the difference.

Errors are fingerprinted on (path, error name, description) and deliberately
*not* on line or column, so that inserting or deleting lines does not make
untouched errors look new. Occurrences are counted rather than deduplicated, so
adding a second instance of an error that already appears once in the same file
is still reported.

Usage:
  pyrefly_new_errors.py <baseline.json> <pr.json>

Both inputs are `pyrefly check --output-format json` reports.
"""

import collections
import json
import sys


def _load(path):
  """Reads a pyrefly JSON report into a fingerprint counter and an index."""
  try:
    with open(path, 'rt') as f:
      report = json.load(f)
  except (OSError, json.JSONDecodeError) as e:
    # A missing or malformed report means the check did not run properly. Fail
    # loudly rather than silently reporting "no new errors".
    sys.exit(f'Could not read pyrefly report {path!r}: {e}')

  counts = collections.Counter()
  by_fingerprint = collections.defaultdict(list)
  for error in report.get('errors', []):
    fingerprint = (
        error.get('path', ''),
        error.get('name', ''),
        error.get('concise_description', ''),
    )
    counts[fingerprint] += 1
    by_fingerprint[fingerprint].append(error)
  return counts, by_fingerprint


def main(argv):
  if len(argv) != 3:
    sys.exit(f'Usage: {argv[0]} <baseline.json> <pr.json>')

  baseline_counts, _ = _load(argv[1])
  pr_counts, pr_errors = _load(argv[2])

  new_errors = []
  for fingerprint, pr_count in pr_counts.items():
    extra = pr_count - baseline_counts.get(fingerprint, 0)
    if extra > 0:
      # Report the last `extra` occurrences; which specific ones we pick is
      # arbitrary, but the count is what matters.
      new_errors.extend(pr_errors[fingerprint][-extra:])

  total_baseline = sum(baseline_counts.values())
  total_pr = sum(pr_counts.values())
  print(f'Base branch: {total_baseline} error(s).')
  print(f'Pull request: {total_pr} error(s).')

  if not new_errors:
    print('No new type errors introduced.')
    # Note that pre-existing errors are not reported as fixed here; a pull
    # request that removes errors simply passes.
    return 0

  new_errors.sort(key=lambda e: (e.get('path', ''), e.get('line', 0)))
  print(
      f'\n{len(new_errors)} new type error(s) introduced by this pull'
      ' request:\n'
  )
  for error in new_errors:
    path = error.get('path', '?')
    line = error.get('line', 0)
    col = error.get('column', 0)
    name = error.get('name', '?')
    description = error.get('concise_description', '')
    # `::error` annotations attach the message to the line in the PR diff view.
    print(f'::error file={path},line={line},col={col}::[{name}] {description}')
    print(f'  {path}:{line}:{col} [{name}] {description}')

  print(
      '\nIf an error is a false positive, add a suppression comment on the line'
      '\n*above* the offending line, for example:'
      '\n    # pyrefly: ignore[bad-argument-type]'
      '\nPlacing it above rather than at the end of the line keeps it attached'
      '\nwhen a formatter rewraps the code.'
  )
  return 1


if __name__ == '__main__':
  sys.exit(main(sys.argv))
