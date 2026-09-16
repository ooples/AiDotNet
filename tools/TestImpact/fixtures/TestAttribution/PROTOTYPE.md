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
- The xUnit method runner carries ownership through AsyncLocal using the concrete
  test type, including inherited tests. It preserves custom case runners and wraps
  per-test construction and asynchronous cleanup. Theory rows share a method
  identity, but the harness independently checks that both rows executed.
- Shared fixture and suppressed-context hits belong to the entire execution
  group. They are never guessed to belong to a neighboring active test.
- Explicit worker registration passes a unique token/run/owner through that
  child's environment. Parent completion and a matching completed child report
  are both required. Nested workers are unsupported by the prototype validator.
- Task-returning call sites register the original task without wrapping it.
  An unfinished observed task at test closure poisons the report even if it has
  never executed covered code. Instrumented direct timer/thread/process escapes
  conservatively invalidate evidence; escapes inside uninstrumented code remain open.
- Closed-scope late hits, unclosed scopes and incomplete workers poison the
  report. Reports publish with a pending-file/rename protocol at process exit;
  pending, missing, conflicting or invalid evidence cannot satisfy the harness.
  A later observed operation revokes an already published report using an invalid
  marker. Validation occurs after process exit and preserves/rejects these markers.
- Static IL dependencies include calls on untaken branches, state-machine methods,
  shared static fields, and explicit unresolved/virtual/external boundaries. This
  graph is not yet a complete dependency certificate or production selector.

## Actual verification

Windows, .NET SDK **10.0.401**, 2026-09-16:

- Build: **zero warnings and errors**.
- Positive execution: **11/11 cases passed**, both plain and instrumented.
- **11 source-backed methods** and **45 test task call sites** instrumented; overlapping test scopes
  observed. A rendezvous requires real overlap between the two parallel tests.
- Async/Task.Run and overlapping Left/Right tests received their expected
  dependencies without receiving the other test's exclusive dependency.
- Shared setup/cleanup and suppressed-context work appeared in the execution
  group. Child-only `WorkerOnly` execution appeared under its owning test.
- **21 negative/mutation checks passed**: late work, detached task, unregistered process,
  never-fired timer, missing worker, unclosed worker,
  unjoined worker, failing test, wrong run, stale binary, missing test, missing
  artifact, pending artifact, malformed artifact, unknown method, skipped case,
  mismatched worker owner, forcibly killed worker, task-guard removal, preserved
  custom-case skipping, and covered execution after report publication.
  Artifact cases mutate copies of real output. The guard-removal control rewrites
  a private collector DLL and proves the ordinary detached-task regression
  assertion fails when the observer is removed.
- **15 execution-protocol cases passed**: exact result inventories, complete
  theory rows, full-vs-partial scope, source/build/profile/run binding, immutable
  verified case lists, safe identical-plan reuse, and malformed receipt rejection.
  The protocol does not authenticate GitHub or determine dependency impact.
- Before/after wall time: **1.53 s plain / 2.58 s instrumented**. This is a
  single small-fixture measurement including process startup and worker output,
  not a repository benchmark or evidence of net CI savings.
- Original binary SHA-256 unchanged after the run.
- Hot-call measurement (seven samples of 131,072 calls, identical independently
  checked checksum): median **1.62 ns plain / 171.31 ns serialized collector /
  74.77 ns cached collector**. Repeated published hits avoid the global lock,
  but closure is checked before cache lookup. The late-hit test primes this
  cache before closing its test. This is not a representative model benchmark.
- Opt-in real AiDotNet test-project build: **zero errors** (existing repository
  warnings remain). Five existing `CpuOffloadShardingConfigTests.ShardingConfiguration_*`
  tests passed, with five matching completed owners and zero collector faults.
  This verifies adapter integration, not production-code coverage: that assembly
  was not instrumented. CPU-only initialization is unchanged. MSBuild evaluation
  confirms opt-in applies only to net10.0, not net8.0/net471 or ordinary builds.

Adversarial review added the explicit parent/worker completion requirement;
child completion alone did not establish that its parent test joined it. It
also added corrupted real-artifact tests and retained whole-group attribution
for execution contexts that cannot safely identify an individual owner.

## Limitations / remaining production gates

- This is not a VSTest-packaged collector or automatic test inventory adapter;
  the fixture assembly opts in once and workers explicitly join.
  Direct process starts, timer construction and thread/queue starts in instrumented
  code now poison evidence unless supported/registered. Escapes inside uninstrumented
  dependencies are not generally detected. Both pre-publication and process-exit
  post-publication hits are tested; arbitrary unobserved execution is not proven.
  These limitations block production enablement.
- Source spans are local PDB paths. Repository normalization, source-content
  verification, generated-source policies, dependency closure, native/GPU code,
  other target frameworks/platforms, and full assembly coverage are not proven.
- The runtime uses a lock on first hits and unowned hits; repeated scoped hits
  use a concurrent cache. Hot-path overhead, memory use,
  large-method inventories and report size require representative measurement
  and likely optimization before rollout. No production performance claim.
- Report checks establish local fixture consistency, not trusted GitHub
  provenance, artifact authenticity, or hardened validation of hostile JSON.
- No selective production filters, complete-baseline publishing, PR/post-merge
  reuse, or compatibility migration has been enabled. The dedicated Linux
  collector canary uploads raw evidence; its result must be checked separately.
- Task-observer safeguard removal and a forcibly killed worker are now exercised.
  Broader process/runner cancellation paths are still needed; these controls do
  not establish production workflow cancellation correctness.

The aggregate-coverage counterexample and original feasibility measurements
remain in [README.md](README.md). Both proof scripts now record their effective
SDK rather than assuming the roll-forward version in the root global.json.

## Live collector verification

[Linux run 35096965934](https://github.com/ooples/AiDotNet/actions/runs/35096965934)
passed at commit `d835f76e58`. Its uploaded artifact was downloaded and checked:
7 plain and 7 collected positive results; 15 expected rejections; 3 overlapping
scopes; SDK 10.0.401; production selection disabled. The deliberately failing
test's TRX is failed, while the harness correctly rejects that result.
This run predates the two new process/timer rejection cases; it does not prove
those cases or production PR/post-merge selective execution.

[Linux run 35099019043](https://github.com/ooples/AiDotNet/actions/runs/35099019043)
passed at commit `a0a0076b50`; its artifact was downloaded and inspected: 7 positive
cases, 19 rejection controls and 15 protocol cases. It predates the lifecycle,
static-graph and post-publication changes described above, which require a new
live collector run. Neither run proves production selective execution or reuse.
