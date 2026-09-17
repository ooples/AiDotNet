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
