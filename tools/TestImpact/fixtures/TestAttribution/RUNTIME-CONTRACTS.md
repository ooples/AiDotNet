# Runtime-contract trust boundary

Contracts preserve normal xUnit execution; they do not create a process per test.
An unknown effect still requires full execution. A reviewed semantic statement
is distinct from evidence that its preconditions hold at a particular call site.

The first catalog entry covers only `Assert.True(bool, string)` and
`Assert.False(bool, string)` in xunit.assert 2.9.3, SHA-256
`bcd9711b22d227ac8cd4c1568e72433a0f576d2ec8aae64ac508081892015b1d`.
Other overloads, replacement bytes, unresolved references, and linked paths do
not inherit it. The passing path copies a bool into `Nullable<bool>`, reads that
value, and returns. The analyzer checks the runtime's three transitive nullable
operations against their exact field-access shapes and records its binary hash.
Changing the package requires a new review, not an automatic hash refresh.
The assertion type also has a static initializer which constructs shared lookup
sets. Its effects are a separate mandatory precondition; the scalar leaf summary
does not discard type/module initialization or assume it already ran.

The failure path throws and may invoke process-wide first-chance handlers. A
test can swallow that failure, pass, and still affect another test. The executable
adversarial control demonstrates exactly this. Consequently an enclosing passing
test cannot by itself satisfy the successful-assertion requirement.

The caller checker distinguishes an uncaught synchronous exception from a
canonical async catch forwarding the same exception to its own task builder.
Additional handlers, callbacks, replaced exception locals, foreign/static state,
and success completion are rejected. Forwarding does **not** prove that the
builder was pending or that xUnit observed that exact task. The report retains
`RequirementsProven=false`; it cannot authorize reuse.

The bounded async kickoff check now binds the returned task to the same builder
used by the exception handler. It rejects substituted tasks, other builders,
altered interface dispatch, and unsupported kickoff shapes. A local factory
result consumed by reviewed assertions can therefore be classified as
`ConditionalOnSuccessfulOwner`, not as an unconditional noninterference proof.
The null-argument factory path is checked separately against the unchanged
guard; this does not declare exception construction or propagation pure.

Opt-in CPU initialization records a combined digest of startup inputs before
normalization. The runner profile binds both those inputs and the effective
environment. Missing and conflicting observations remain explicit; raw license
credentials are not written to evidence. These observations are not a complete
model of process state or proof that no runtime observers exist.

Local verification on 2026-09-16 passed 236 protocol/control tests and 33
rejection controls. The rebuilt real CPU sharding-config slice passed 5/5.
A plan discovered with `OMP_NUM_THREADS=2` was rejected with startup value `3`,
although initialization normalizes both to `1`. This was a local component
check with a synthetic source identity, not authenticated source-change or
production-reuse evidence. The real five-test workload is still not safely
narrowed by these checks alone.

The observed-owner proof now retains standard-case identities through local
verification and authenticated workflow import. Its bounded catalog pins
xUnit execution/core 2.9.3, the trusted attribution runner/runtime, and the
reviewed Windows .NET 10.0.12 core library. Other runtime binaries remain
unresolved. The timeout path requires the underlying task to complete; custom
case runners, overridden framework execution, substituted tasks, stale bundles,
and unverified results cannot satisfy this proof. Bundle hashing is batched
across owners rather than repeated for each test.

The profile preimage now accompanies discovery and must hash to the execution
identity. Startup records distinguish completed CPU setup from merely entering
initialization, without changing non-attribution builds. Assembly Before/After
callbacks are per-test roots; construction stays shared, and shared attribute
instance fields retain explicit state dependencies.

Verification of this integration passed 274 protocol/control tests and 33
rejection controls. A fresh real five-test execution passed 5/5, all five owners
received the reviewed task-observation proof, CPU completion was present in the
verified profile, and changed startup inputs were rejected. These remain local
component results, not a changed-source reuse certificate or live proof.

Remaining integration gates are the reviewed CPU initialization semantics,
per-owner file/AsyncLocal lifecycle and concrete numeric-provider effects, and their
composition with the normal selector and authenticated reuse evidence. Neither
these diagnostic contracts nor the 2/5 experimental candidate set close those
boundaries. Production test selection remains disabled until those gates and
the real-change/live acceptance cases pass.

Discovery also has a narrowly reviewed `SkippableFact` 1.5.85 contract, bound
to package SHA-256
`f8fb7e54fb771f40c0a6b773e20954545277fb0ba71286c87bfd07410c3c1160` and the
reviewed xUnit/runtime binaries. It reads exception-type metadata and creates
a case with null method arguments; it does not invoke the test. Execution of
that custom case remains unresolved. Unknown trait discoverers remain shared
discovery boundaries. The real assembly no longer acquires unused inherited
`System.Attribute.GetCustomAttribute` helpers as executable roots.

Standalone and opt-in builds previously embedded different commit metadata in
unchanged attribution dependencies. These non-packable tools now share build
settings; release/package projects are unchanged. The canary forcibly rebuilds
the three runtime components with different revision IDs and opt-in modes,
requiring identical DLL **and** PDB hashes rather than normalizing a mismatch.
The combined local harness passed 283 protocol/control tests, 33 rejection
controls, and both forced-build comparisons; the refreshed real slice passed
5/5 with owner/CPU observations and startup-input rejection intact.

The numeric assessment now binds the concrete `double` call to the reviewed
Tensors binary (`eb681ae60f23b03cf08e0bf3ab70a372673927acd87a428c74536d424846d5e7`)
and runtime. Generic arguments are matched by their owning definition, kind,
and position, not parameter display names. Float, open/foreign parameters,
malformed signatures, and missing/replaced package bytes remain unresolved.
The shared MathHelper initializer and possible external cache mutation are
still explicit requirements; this does not declare the generic provider pure.

The scope reader verifies the actual save/set/restore implementation of the
string AsyncLocal override, including its private readonly saved value and
callback-free slot construction. Changed slots, extra behavior, synchronization,
internal-call flags, and altered disposal are rejected. Correct lifetime,
execution-context flow, and file isolation remain separate requirements.
Inspection is bounded to discovered workload owners plus shared group roots.
Both assessments are included in the real-bundle experiment, which still
reports `CanAuthorizeReuse=false`; neither closes a normal selector boundary.

The combined local harness passed 321 protocol/control cases and 33 rejection
controls, including forced build-binding and real small-fixture source changes.
The existing AiDotNet before/after bundles yielded two numeric assessments and
one scope assessment across the five-owner workload. This is component evidence;
normal AiDotNet changed-source selection/reuse and live verification are pending.

The opt-in shared trial hook now reports owner-bound scope boundaries. Verification
requires absent main/tombstone paths, distinct canonical path identities, and exact
restoration of the previous string value. Linked/foreign paths, duplicate owners or
paths, incomplete lifetimes, custom runners, and aggregated theory rows cannot
supply this observation. Cleanup runs after the final observation, so deleting a
file cannot conceal its presence at the end of the test. Raw paths are not emitted.
These observations survive local verification and authenticated workflow import;
they do not prove absence of intermediate I/O or authorize reuse independently.

The scope initializer reader separately checks private readonly slot/lock allocations
against the reviewed runtime. Callbacks, other calls, mutable or foreign fields,
extra control flow, and unsupported constructors retain an unresolved initializer
requirement. The actual ModelPersistenceGuard initializer matched this check. Its
scope contract still requires exclusive lifetime, no external slot mutation, and
correct owner execution-context flow.

Local verification passed the 359-case harness and 33 rejection controls, then
240 focused cases after adding the 15 initializer controls (182 runtime-effect,
58 runner-protocol cases). The updated real five-test assembly passed 5/5 and
reported five distinct completed scopes; changed startup inputs were rejected.
The first harness attempt ran out of disk space; its replacement completed.
These remain component results, not changed-source reuse or new live CI proof.

The CPU review also found that the pinned Tensors `ResetToCpu` path reads
`AIDOTNET_QUIET` and may invoke a configured logger (or console output). The
environment digest now binds that input; the callback/output effects remain
unresolved, not implicitly pure because CPU completion was recorded. The
updated tools build passed with no warnings/errors and the focused runtime /
runner set passed 241/241. Real-assembly revalidation is pending: at about
21:12 Eastern the active worktree's production and test bin/obj directories
disappeared between consecutive reads, after the earlier real-slice proof.
Those earlier results are retained, but do not validate the newer profile bytes.

The recovered real assembly now verifies five completed owner/scope bindings
against the actual Before/After hook. The hook reader checks construction,
save/restore and observation ordering, scope clearing, and the exact cleanup
paths/filter. Altered cleanup regions, foreign generic-parameter owners, missing
or custom owner observations, and replaced bundles retain unresolved contracts.
Observation inspection visits only requested classes; full source mapping still
includes the complete lifecycle graph. Body file/slot isolation and owner-context
flow remain requirements, not conclusions drawn from boundary samples.

CPU reset entry is now recorded separately from completion. The real AMD host
reported a derived CPU entry: the pinned Tensors GPU engine inherits CpuEngine,
and its module initializer can auto-detect GPU before the test initializer.
Consequently neither `is CpuEngine` nor a CPU result establishes a callback-free
reset. The opt-in completion observation now requires the exact CPU type. Missing
and derived entry states remain ineligible. GPU opt-out, verbose initialization,
and the reviewed CPU initializer inputs join the environment digest; production
GPU behavior is unchanged.

With explicit CPU-only and quiet host inputs, the refreshed real slice passed
5/5 and verified five completed trial hooks. Changing pre-normalization thread
settings or logging mode rejected the original plan. The unconfigured host's
derived entry did not satisfy the CPU precondition. These are local component
controls, not normal changed-source selection, a reuse certificate, or live CI
proof. Shared initializer/cache effects, body isolation, normal-selector
composition, and real changed-source/live acceptance remain unfinished.

Final verification of this batch passed 441 protocol/control cases and all 33
rejection controls in `attribution-hook-reset-final-20260916`. Reusing the valid
real execution outputs, the final hook reader verified all five bindings in
9.34 seconds (`attribution-real-runtime-inputs-20260916-v9/verified-final.json`).
The real test-only build completed with zero errors and 4,013 existing warnings.

### Runtime-effect boundary checks (September 17)

The CPU profile now distinguishes GPU opt-out from diagnostics-dump opt-out.
The pinned Tensors module initializes diagnostics before honoring GPU opt-out;
a configured dump can still create a timer, exit callbacks and a file. The real
CPU-only negative control produced that file and rejected the previous plan.

Verified lifecycle output now retains the canonical single-yield body window,
fixture-free nonparallel collection metadata, and first-chance-observer samples
at both trial boundaries. Unsupported runtime layouts remain unknown. Samples
do not establish absence throughout the body or exclude arbitrary host workers.

The locked-initialization reader recognizes a private constant insert-if-absent
tail. The prefix/callsite readers derive rank, world size and nonempty-key facts
from actual IL, record the base-constructor obligation, and retain map lifetime,
initialization, observer and external-access requirements. GUID formatting proves
nonempty string shape, not uniqueness or RNG purity. All four real backend call
sites matched; the null-backend test correctly supplies no constructor proof.

Local evidence: `attribution-contracts-final-20260917` passed the 556-case harness
and 33 rejection controls. The subsequent constructor-prefix/callsite batch passed
445 focused runtime/runner cases. `attribution-real-runtime-inputs-20260917-v11`
executed 5/5 real tests, verified all five body/collection/hook bindings and sampled
no first-chance handlers at either boundary. Changed startup, logging and diagnostics
inputs rejected the old plan. These are component results: normal real-workload
selective reuse, complete body/shared-initializer closure and updated live CI
acceptance remain unfinished; production selection is still disabled.

The backend base constructor now binds its generic argument through the actual
inheritance edge to the reviewed `double` provider. Its structural check allows
only the Object constructor and receiver-field initialization; provider/cache
lifetime requirements remain explicit. Shared-field matching also preserves
generic storage identity: `Backend<float>.Map` cannot borrow a contract for the
current `Backend<T>.Map` just because both resolve to the same field definition.
Foreign parameter owners, wrong inheritance edges, added effects and substituted
closed field owners are rejection controls. The follow-up focused set passed
467/467; all four real constructor sites verified the numeric-base contract in
`attribution-real-constructor-inputs-final-20260917.jsonl`.

The combined no-build harness then passed 601 cases and all 33 rejection controls
in `attribution-all-contracts-final-20260917`. A subsequent pinned-xUnit IL review
found case-insensitive grouping of collection definitions followed by a
case-sensitive lookup. The concurrency reader rejects case-only definition
collisions rather than choosing the apparently matching spelling. The runtime
experiment's final bundle recheck also now follows every contract reader,
including constructor inputs and async-body windows.
The final focused set passed 468/468 and the corrected verifier retained all
five real collection bindings (`v11/verified-final.json`). The composed real
mutation experiment reported four constructor/numeric-base bindings and 2/5
diagnostic candidates, while correctly retaining `CanAuthorizeReuse=false` and
`RequiresFullControl=true`; this is not a normal-selector acceptance result.

### Default startup branch (2026-09-17)

The signing reader checks the entire fresh-HMAC method, including branch targets,
runtime identity and disposal. The key-override reader checks the locked copy and
its gate initializer. The startup-flow reader evaluates the observed default
license/default-MDOP branch, but deliberately returns targets needing independent
authentication, not a reuse certificate. An overridden MDOP is not treated as the
default: parsing it can reach a culture provider. CPU-reset completion is recorded
immediately after the call; a swallowed reset exception cannot borrow that marker.

`attribution-startup-batch-final-20260917/contracts.trx`: 690/690 focused cases.
The actual net10.0 project built with zero errors. A fresh five-case execution in
`attribution-real-startup-20260917-v13/verified.json` verified all five standard
owners, collection and hook bindings, with `ResetOutcome=Completed` and both
default-input policies. Neither this run nor shape recognition proves selective
reuse. An earlier local discovery attempt used an invalid hit-mode label and was
rejected; it is not evidence of test execution.

The complete `attribution-startup-full-harness-20260917` harness also passed,
including build identity, planned execution, rejection controls and both source
selection fixtures (1/3 selected, 3/3 configuration fallback). Those fixture
results are not a substitute for changed-code selective reuse on real AiDotNet.

The actual module entry has a second call to `TestAssemblyDeterminismInit.Init`.
It reaches `BlasProvider.SetDeterministicMode(true)`, whose type initializer starts
the background BLAS thread and whose native-availability probe can dispatch a
native GEMM. Therefore completing `TestModuleInitializer.InitializeCpuMode` does
not establish absence of startup workers. Native-library provenance, dispatch
completion and later worker access must be covered before this path can authorize
reuse; the five passing tests do not discharge those obligations.
