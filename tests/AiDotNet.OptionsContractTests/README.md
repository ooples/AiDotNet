# Sequence options review proof (#2128)

This focused project compiles the **actual 17 sequence options classes and their complete
base chain**, plus `DocumentNeuralNetworkOptions`. It source-links the same behavioral test
fixture included in the main AiDotNet test project. There are no replacement bases, model
stubs, tensor packages, or generator changes in this runner. Its six additional documentation
cases inspect the compiler-emitted XML, not copies of the source comments.

## Reproduce

From the repository root, run each target with a unique results file:

```powershell
dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj -c Release -f net10.0 --logger "trx;LogFileName=options-net10.trx" --results-directory artifacts/options-contracts
dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj -c Release -f net8.0 --logger "trx;LogFileName=options-net8.trx" --results-directory artifacts/options-contracts
dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj -c Release -f net471 --logger "trx;LogFileName=options-net471.trx" --results-directory artifacts/options-contracts
```

The .NET Framework target executes on Windows. Linux uses the first command in the
official `mcr.microsoft.com/dotnet/sdk:10.0` container. Copy source into an isolated build
directory, or redirect all build outputs; do not let Linux overwrite Windows `bin`/`obj`.

## Before and after (2026-09-11)

The full 347-case fixture was run against source archived from PR head
`671836a347436f4b023ebe3f02a86ea333ad0686`, then against the fixes, in separate Linux build
directories. Both used the same final test files and package versions.

| Proof | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Old source, Linux .NET 10 | 174 | 173 | 0 |
| Fixed source, Linux .NET 10 | 347 | 0 | 0 |
| Fixed source, Windows .NET 10 | 347 | 0 | 0 |
| Fixed source, Windows .NET 8 | 347 | 0 | 0 |
| Fixed source, Windows .NET Framework 4.7.1 | 347 | 0 | 0 |

The old-source failures are 19 complete-copy cases, 13 missing-copy/default cases,
13 missing-copy/null cases, 113 numeric-validation cases, eight hybrid-interval cases,
one Finch default regression, five class-documentation cases, and one missing property
`value` documentation case. Every old run executed its tests; none is a build-error proxy.
Local TRXs are retained under `artifacts/options-contracts/linux` and
`artifacts/options-contracts/final` (ignored build evidence, not checked-in generated data).

### What the cases establish

- Every public writable property, including inherited `Seed`, `EncoderLayerCount`,
  `MaxGradNorm`, and all nine sequence properties, receives a non-default sentinel before
  copying. The assertion reports **all** lost values. Copies remain independent when the
  original is changed, preserve nullable defaults, and reject null as `other`.
- Each invalid-value case first validates an untouched default instance, changes exactly
  one property selected by a typed expression, and requires an exception identifying both
  the `options` parameter and that exact type/property. An unrelated dimension error cannot
  satisfy the test.
- `Validate()` now checks the stored optimizer configuration even if a model caller supplies
  a custom optimizer. This is a stricter configuration boundary than the former default-only
  checks; a custom optimizer still controls actual training, but does not make an invalid
  stored rate, beta, decay, epsilon, or enabled clipping threshold valid. The GLA/GatedDelta
  rate documentation explicitly states this distinction.
- Required rates/ratios reject zero, negatives, NaN, and both infinities. AdamW decay permits
  zero; betas permit zero and the greatest representable double below one but reject one.
  A disabled gradient clip does not require an otherwise unused threshold. Pure recurrent
  models do not suddenly require attention-only properties.
- All 17 shipped sequence-size configurations and the declared optimizer defaults are
  guarded explicitly. Finch is the single intentional restoration: its options initializer
  was `3e-4` at `671836a347^`; the migration added an overriding `0.001` assignment copied
  from a model scalar that was stored only in an unread field.
- The five requested class descriptions cite the original papers, explain the architecture
  to beginners, and distinguish published model sizes from this library's shipped settings.
  Emitted XML assertions ensure these comments belong to the classes, not their constructors.

This is proof of the options copy/validation/documentation contracts, **not** proof of full
model-family training, GPU execution, model checkpoint equivalence, or the CI shard selector.
The new document-base constructor enables derived classes to chain to it; it does not claim
that every existing document leaf has already been migrated to that copy pattern.

The first draft of the documentation fixture used `Assembly.Location`; .NET Framework's
shadow copying exposed that it could not find the adjacent XML (six explicit test failures).
Using `AppContext.BaseDirectory` fixes that test precondition. Final runs above use unique
TRX filenames and the corrected fixture; the earlier setup failures are not counted as green.
