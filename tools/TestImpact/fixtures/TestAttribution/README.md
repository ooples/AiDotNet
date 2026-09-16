# Test-level impact attribution: feasibility checkpoint

Status: **incomplete; production test selection is not enabled**. This fixture is
the first gate of the test-level selection work, separate from PR #2213. It does
not change workflows, filters, certificates, baseline maps, or production code.

Run from the repository root:

```powershell
pwsh -NoProfile -File tools/TestImpact/Test-TestAttributionFeasibility.ps1
```

The script builds only two small fixture projects, runs six test invocations,
and retains TRX, OpenCover reports, and `evidence.json` in the printed temporary
directory. Its successful exit means the counterexample assertions passed, not
that the repository can safely select individual tests.

## Measured result

Local Windows run on 2026-09-16, SDK 10.0.401; package versions match master's
test SDK, xUnit, adapter, and collector at baseline
`2a878c52d9dcbfcc5e2376cd3556b8575f988bb9`.

| Invocation | Passed cases | Wall time (seconds, rounded) |
| --- | ---: | ---: |
| Normal execution, no coverage | 3 | 2.15 |
| Combined coverage | 3 | 2.78 |
| Combined coverage, ownership swapped | 3 | 2.80 |
| First method only, both theory rows | 2 | 2.83 |
| Second method only | 1 | 2.84 |
| First method only, ownership swapped | 2 | 2.80 |

Build: zero warnings/errors. All six invocations and probe assertions passed.
These are single-sample fixture measurements, not a production benchmark or a
projection of repository-wide cost.

The combined runs have identical visited production sequence points and no
`TrackedMethodRef` attribution. The isolated controls prove that First changes
from Left to Right when ownership is swapped. Each isolated report contains
the expected dependency, excludes the other dependency, and retains shared
fixture setup/cleanup. Async continuations and Task.Run execute in the fixture.
The theory filter retains both rows; observed test methods/counts are checked
against independently specified expectations.

Two isolated method invocations took 5.669 seconds versus 2.78 seconds combined.
Isolation also repeats class-fixture lifetime. It is a control, not an approved
production collection strategy. Existing aggregate reports cannot recover the
lost test ownership after the fact.

Adversarial review caught duplicate coverage attachments created by the TRX
logger: the probe checks all discovered reports are byte-identical rather than
silently selecting one conflicting report. Review also aligned the adapter
version with the production project before the final run.

## Remaining gates (not verified)

- Choose and validate a per-test collection mechanism. Parallel, worker-process,
  detached/background execution, and shared-state attribution remain unproven.
- Combine coverage with old/new dependency analysis and conservative fallbacks
  for new branches, dispatch, reflection, resources, and unresolved dependencies.
- Bind discovery/execution to the same binaries and reconcile all theory cases,
  profiles, retries, skips, failures, cancellations, and missing results.
- Protect complete baselines from partial results and superseded publishers;
  version evidence and validate run/tree/environment/provenance at import time.
- Implement PR and changed-base post-merge selection/reuse, including workloads
  with no case-level evidence and an explicit full-validation override.
- Exercise invalid diffs, stale/missing/conflicting evidence, incomplete
  inventories, filter limits, artifact races, and failing-test gate propagation.
- Verify safeguards with controlled mutations and independent expected sets.
- Run live production-change, indirect-dependency, identical-tree reuse,
  changed-base reuse, and failed/missing-evidence canaries before readiness.

No live canary, production per-test selection, or overall performance benefit
has been established by this checkpoint. The next instrumentation design needs
review before replacing the approved shard-level pipeline.
