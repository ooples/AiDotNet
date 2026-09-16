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
  Calls encode one shared assembly-hash string plus an integer method token;
  canonical text keys are formatted only when publishing the report. The output
  user-string heap is checked before a method map can be issued.
- The xUnit method runner carries ownership through AsyncLocal using the concrete
  test type, including inherited tests. It preserves custom case runners and wraps
  per-test construction and asynchronous cleanup. Theory rows share a method
  identity, but the harness independently checks that both rows executed.
- Schema-3 reports record discovered xUnit case IDs before execution and observe
  actual passed/failed/skipped messages plus case completion. A deferred theory
  retains one discovery identity with all its runtime rows. The independent TRX
  must match; missing, duplicate, unfinished, or unsuccessful ledger entries are
  rejected. VSTest exit code zero alone is never success evidence.
- Shared fixture and suppressed-context hits belong to the entire execution
  group. They are never guessed to belong to a neighboring active test.
- Explicit worker registration passes a unique token/run/owner through that
  child's environment. Parent completion and a matching completed child report
  are both required. A ticket is atomically consumed on its first checked start;
  replay or completion without a start invalidates evidence. Nested workers are
  unsupported by the prototype validator.
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
- Positive execution: **13/13 cases passed**, both plain and instrumented,
  including both deferred-theory rows under one discovered case.
- Source-backed methods and test task call sites instrumented; overlapping test scopes
  observed. A rendezvous requires real overlap between the two parallel tests.
- Async/Task.Run and overlapping Left/Right tests received their expected
  dependencies without receiving the other test's exclusive dependency.
- Shared setup/cleanup and suppressed-context work appeared in the execution
  group. Child-only `WorkerOnly` execution appeared under its owning test.
- **31 negative/mutation checks passed**: late work, detached task, unregistered process,
  never-fired timer, missing worker, unclosed worker,
  unjoined worker, failing test, wrong run, stale binary, missing test, missing
  artifact, pending artifact, malformed artifact, unknown method, skipped case,
  mismatched worker owner, forcibly killed worker, task-guard removal, preserved
  custom-case skipping, and covered execution after report publication.
  Additional controls remove, duplicate, leave unfinished, or skip ledger cases,
  and remove all TRX results while leaving its success summary untouched.
  The review batch adds ticket replay with a reportless second process, completion
  without a start, and non-generic/generic/directly constructed `ValueTask` cases
  backed by an unfinished `IValueTaskSource`, not by an observable `Task`.
  Artifact cases mutate copies of real output. The guard-removal control rewrites
  a private collector DLL and proves the ordinary detached-task regression
  assertion fails when the observer is removed.
- **15 execution-protocol cases passed**: exact result inventories, complete
  theory rows, full-vs-partial scope, source/build/profile/run binding, immutable
  verified case lists, safe identical-plan reuse, and malformed receipt rejection.
  The protocol does not authenticate GitHub or determine dependency impact.
- **17 conservative dependency-selection cases passed**: old/new call closure,
  deleted calls/tests, new tests/branches, cycles, shared-state chains, fixture
  state, unknown changes, and unresolved/virtual/reflection/external/native
  boundaries. The IL extraction controls retain an untaken call, implicit type
  initialization, and external fields. These are algorithm/fixture checks, not
  authenticated production graph construction or actual narrowed execution.
- Before/after wall time: **1.54 s plain / 2.68 s instrumented**. This is a
  single small-fixture measurement including process startup and worker output,
  not a repository benchmark or evidence of net CI savings.
- Original binary SHA-256 unchanged after the run.
- Hot-call measurement (seven samples of 131,072 calls, identical independently
  checked checksum): median **1.66 ns plain / 344.32 ns serialized collector /
  134.55 ns cached collector**. Repeated published hits avoid the global lock,
  but closure is checked before cache lookup. The late-hit test primes this
  cache before closing its test. This is not a representative model benchmark.
- Opt-in real AiDotNet test-project build: **zero errors** (existing repository
  warnings remain). Five existing `CpuOffloadShardingConfigTests.ShardingConfiguration_*`
  tests passed, with five matching completed owners and zero collector faults.
  This verifies adapter integration, not production-code coverage: that assembly
  was not instrumented. CPU-only initialization is unchanged. MSBuild evaluation
  confirms opt-in applies only to net10.0, not net8.0/net471 or ordinary builds.

## Real-assembly compatibility probe

The first full-assembly rewrite exposed a defect absent from the small fixture:
175,687 unique injected key strings grew AiDotNet's user-string heap to
**33,227,820 bytes**. Startup failed with `BadImageFormatException`, and VSTest
returned exit code zero with **no tests executed**. That run is failed evidence,
not a successful test. The metadata emitter's
[user-string offset limit](https://source.dot.net/System.Reflection.Metadata/System/Reflection/Metadata/Ecma335/MetadataBuilder.Heaps.cs.html)
requires bounded encoding.

After replacing per-method strings with shared module identity plus integer tokens:

- Original heap **7,050,456 bytes**; rewritten heap **7,050,588 bytes** (+132).
- All **175,687 methods** and **2,926 production task sites** instrumented in
  **32.82 s**; the test assembly also included **32,436 task sites**.
- The same five existing `ShardingConfiguration_*` cases passed before and after,
  with identical discovered IDs, five completed owners, **83 owner/method pairs**
  (including 20 shared-group dependencies), and **zero collector faults**.
- Test wall time: **22.42 s control / 19.73 s instrumented**, including discovery.
  This single noisy sample does not demonstrate a speedup or representative model
  performance. TRX test durations were 31 ms and 32 ms respectively.
- Original AiDotNet DLL SHA-256 remained
  `1860E217DF46BFD9105F9453296E20992B0635D68BA579DDF1685D74D5FDF737`.
  Only private temporary copies were rewritten. The fixture now asserts bounded
  heap growth, and the real-run check requires five actual passing TRX results.

This is compatibility/attribution evidence only, not proof of production selective
filters, baseline authenticity, or changed-base reuse.

Adversarial review added the explicit parent/worker completion requirement;
child completion alone did not establish that its parent test joined it. It
also added corrupted real-artifact tests and retained whole-group attribution
for execution contexts that cannot safely identify an individual owner.

## Limitations / remaining production gates

### Bound runner selection (local proof)

The opt-in xUnit adapter now discovers the actual workload inventory without
executing it, then accepts a strict execution plan bound to that inventory,
workload, source identity, binary-directory contents, and effective execution
profile. It runs only the required discovery cases, preserving all rows of a
selected theory. The binary bundle is checked again after execution. Schema-3
reports record the plan; the independent CLI verifier checks both the case ledger
and TRX. This single-bundle path rejects worker-backed evidence.

`Test-PlannedAttribution.ps1` demonstrated **4 selected runtime rows versus 12
full-workload rows** (3 versus 11 discovery cases). It rejected partial results
presented as a full baseline, a stale binary identity, a removed theory case,
an unknown method, changed invocation profile/workload/source, and a revoked
report. Five additional receipt mutations cover missing/duplicate discovery
cases, skipped results, a foreign run identity, and a missing TRX row. Rejected
runner invocations must fail with the expected reason and execute zero tests.
The full fixture harness also passed its 6 runner-binding tests, 15
execution-protocol tests, 17 dependency-selection tests, and existing 31 rejection
controls. These are actual runner checks, not merely assertions on a filter string.

The selected methods in this proof are explicitly supplied. This does **not**
prove automatic source-change selection, authenticated GitHub provenance,
changed-base reuse, or production CI enablement. Those remain separate gates.

The real AiDotNet opt-in net10.0 build also passed (zero errors). With the same
five-test `CpuOffloadShardingConfigTests.ShardingConfiguration_` VSTest filter
for discovery and execution, the adapter discovered five cases and the plan
executed exactly `ShardingConfiguration_DefaultsAllOffloadFlagsToFalse`: one
passing result, independently verified against its recorded plan and TRX.
This checks real-project runner integration, not impact-map completeness; that
run did not instrument production methods or claim a performance improvement.

### Changed-base partition protocol

`ExecutionReuse` now partitions a current inventory using the conservative
old/new dependency selector and a previously verified **full** execution. It
requires exact graph-to-inventory method coverage and graph revision/build
bindings. Changed theory inventories rerun the complete current method; changed
profiles and unmapped changes force full execution. Completion requires the
exact fresh execution plan, while reused cases retain their baseline provenance.
A reuse-only partition is not represented as a successful zero-test execution,
and a mixed partition cannot become a new freshly observed full baseline.

Twelve protocol tests cover these boundaries, mutable returned plans, old edges,
open native dependencies, shared state, and missing/stale/wrong-scope execution.
The caller still must establish graph/change-set completeness and authenticate
the baseline. This is **not live changed-base CI reuse proof**; automatic graph
construction, trusted artifact import, and production dispatch remain unwired.

- This is not a VSTest-packaged collector; the xUnit assembly opts in once,
  its adapter supplies the discovery inventory, and workers explicitly join.
  Direct process starts, timer construction and thread/queue starts in instrumented
  code now poison evidence unless supported/registered. Executed unsupported
  `ValueTask` calls/constructors also poison evidence without consuming/converting
  their return values; merely having an unused boundary does not prevent rewriting.
  Escapes inside uninstrumented
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

[Linux run 35105583876](https://github.com/ooples/AiDotNet/actions/runs/35105583876)
passed at commit `13e6c578ac`. Its artifact was downloaded and the raw TRXs
checked: 13 plain and 13 instrumented positive rows, 15 execution-protocol tests,
17 dependency-selection tests, and 31 rejection controls. This run predates the
bound runner-selection changes above and does not verify them.

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
