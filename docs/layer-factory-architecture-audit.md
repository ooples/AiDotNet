# Architecture parameter audit (#2147)

Run from the repository root:

```sh
dotnet run --project tools/LayerFactoryArchitectureAudit -c Release -- --self-test
dotnet run --project tools/LayerFactoryArchitectureAudit -c Release -- --fail-on-active-unused > architecture-inventory.json
```

The tool parses every `src/Helpers/LayerHelper*.cs` with Roslyn and reports each
public static method whose parameter type is `NeuralNetworkArchitecture<T>`.
It counts identifier references inside the exact block or expression body.
Comments, strings, parameter declarations and adjacent documentation cannot
count as uses. The self-test checks those boundaries, forwarding, qualified
type names and expression bodies. The gate exits unsuccessfully when an active
factory accepts an architecture without referencing it.

The `Layer factory architecture contracts` workflow runs these commands when
the helpers, audit tool or architecture regression tests change, and retains
the complete JSON inventory as an artifact.

On 2026-11-02 the combined #2294 branch has 184 such overloads. Of those, 89
never reference the parameter, and **all 89 are marked obsolete**. Of these,
88 forward to overloads without the unused architecture argument. The remaining
historical FEDformer factory constructs a generic encoder and is explicitly
retired in favour of `FEDformer<T>`'s native network. Zero active factories
accept an architecture and never reference it. Obsolete factories remain in
the inventory so their API migration is visible rather than hidden.

These are current syntax counts, not the issue's historical 96-method estimate.
An identifier reference does not prove correct sizing: validation, forwarding,
and dimensional reconciliation still need per-model review. In particular,
time-series horizons and feature axes cannot be replaced mechanically with a
flat architecture width. `LayerFactoryArchitectureTests` covers explicit and
unset output sizes for PatchTST, iTransformer and ETSformer. This audit does
not claim regression coverage for every historical model in the issue.
