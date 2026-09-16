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

Remaining integration gates are the observed-owner completion proof, reviewed
CPU initialization and per-owner file/AsyncLocal lifecycle effects, and their
composition with the normal selector and authenticated reuse evidence. Neither
these diagnostic contracts nor the 2/5 experimental candidate set close those
boundaries. Production test selection remains disabled until those gates and
the real-change/live acceptance cases pass.
