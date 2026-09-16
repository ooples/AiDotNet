# Custom attribution collector prototype

Status: **fixture prototype verified locally; production selection disabled**.
This is the next checkpoint on draft PR #2215, not completion of its production
selection/reuse acceptance criteria. PR #2213 is untouched.

## Run the proof

```powershell
pwsh -NoProfile -File tools/TestImpact/Test-AttributionPrototype.ps1
```

The script builds only the fixture projects. `-NoBuild` reuses already built,
unchanged fixture binaries. It instruments private temporary copies, never the
original build output. It retains raw TRX, method maps, per-process attribution
JSON, mutated evidence and `proof.json` in the printed temporary directory.

The deliberate failing test in the log is expected: the harness must reject its
evidence. A successful harness exit means all positive assertions and expected
rejections passed. It is not a production CI validation certificate.

## Implementation

- Mono.Cecil inserts tracking calls at source-backed method entries, including
  compiler-generated async methods. Method identity includes the original
  assembly SHA-256 and metadata token. Maps record binary/PDB hashes and source
  spans. These are conservative method dependencies, **not executed-line or
  branch coverage**. Signed and already instrumented assemblies are rejected.
- xUnit v2 Before/After boundaries carry ownership through AsyncLocal. Theory
  rows share a method identity, but the harness independently checks that both
  rows executed. No state is stored on reusable attribute instances.
- Shared fixture and suppressed-context hits belong to the entire execution
  group. They are never guessed to belong to a neighboring active test.
- Explicit worker registration passes a unique token/run/owner through that
  child's environment. Parent completion and a matching completed child report
  are both required. Nested workers are unsupported by the prototype validator.
- Closed-scope late hits, unclosed scopes and incomplete workers poison the
  report. Reports publish with a pending-file/rename protocol at process exit;
  pending, missing, conflicting or invalid evidence cannot satisfy the harness.

## Actual verification

Windows, .NET SDK **10.0.401**, 2026-09-16:

- Build: **zero warnings and errors**.
- Positive execution: **7/7 cases passed**, both plain and instrumented.
- **9 source-backed methods** instrumented; **4 simultaneous test scopes**
  observed. A rendezvous requires real overlap between the two parallel tests.
- Async/Task.Run and overlapping Left/Right tests received their expected
  dependencies without receiving the other test's exclusive dependency.
- Shared setup/cleanup and suppressed-context work appeared in the execution
  group. Child-only `WorkerOnly` execution appeared under its owning test.
- **14 negative checks rejected**: late work, missing worker, unclosed worker,
  unjoined worker, failing test, wrong run, stale binary, missing test, missing
  artifact, pending artifact, malformed artifact, unknown method, skipped case,
  and mismatched worker owner. Artifact cases mutate copies of real output.
- Before/after wall time: **1.509 s plain / 2.352 s instrumented**. This is a
  single small-fixture measurement including process startup and worker output,
  not a repository benchmark or evidence of net CI savings.
- Original binary SHA-256 unchanged after the run.

Adversarial review added the explicit parent/worker completion requirement;
child completion alone did not establish that its parent test joined it. It
also added corrupted real-artifact tests and retained whole-group attribution
for execution contexts that cannot safely identify an individual owner.

## Limitations / remaining production gates

- This is not a VSTest-packaged collector or automatic test inventory adapter;
  fixture test classes opt in through an attribute and workers explicitly join.
  Arbitrary unregistered workers and arbitrary background tasks that never hit
  instrumented code are not detected. Late hits are tested only when executed
  before report publication. These limitations block production enablement.
- Source spans are local PDB paths. Repository normalization, source-content
  verification, generated-source policies, dependency closure, native/GPU code,
  other target frameworks/platforms, and full assembly coverage are not proven.
- The runtime uses a lock on every recorded hit. Hot-path overhead, memory use,
  large-method inventories and report size require representative measurement
  and likely optimization before rollout. No production performance claim.
- Report checks establish local fixture consistency, not trusted GitHub
  provenance, artifact authenticity, or hardened validation of hostile JSON.
- No selective production filters, complete-baseline publishing, PR/post-merge
  reuse, compatibility migration, or live repository canary has been enabled.
- Safeguard-removal mutation testing and process-crash/cancellation testing are
  still needed; corrupt-artifact tests are not substitutes for those checks.

The aggregate-coverage counterexample and original feasibility measurements
remain in [README.md](README.md). Both proof scripts now record their effective
SDK rather than assuming the roll-forward version in the root global.json.
