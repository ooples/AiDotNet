# PR #2128 review evidence — 2026-09-11

Reviewed starting head: `671836a347436f4b023ebe3f02a86ea333ad0686`.
This record distinguishes focused local checks from hosted CI and actual model tests.

## Options copy and validation

The frozen options correction is `80e7540d69806c5e26de7a38b12c98932604d979`.
The identical 347 tests against the original source on Linux reported **173 failed,
174 passed**; after correction, **347 passed, zero failed, zero skipped**. Windows
`net10.0`, `net8.0` and `net471` also passed all 347, independently repeated by
the primary reviewer. See [the focused runner](../tests/AiDotNet.OptionsContractTests/README.md)
for the exact source-linked build, commands and test scope.

Two additional contracts verify all ten audio-base properties have value and
beginner documentation and that the newly introduced vision-language base uses
the same tower property names as the document base. With these included, the
primary reviewer's three-framework run passed **349 tests per framework**, no skips:

```powershell
dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj -c Release --logger 'trx;LogFilePrefix=pr2128-options-root-shared-docs' --results-directory artifacts/options-contracts/independent
```

These are executable scalar-options/metadata contracts, not model training or
GPU performance evidence. Existing default dimensions are pinned; Finch's
pre-PR public options rate of `3e-4` is preserved rather than overwritten by a
new constructor assignment. Its unread model `_learningRate` field was removed.

## Generator and diagnostic controls

[The generator runner](../tools/SequenceFixtureReview/README.md) reports **4 failed,
9 passed before; 13 passed afterward**. All six effective constructor-fixture
checks pass on both sources. The deleted XLSTM/Hawk/Griffin rules were unreachable;
their removal does not substitute a different live fixture. This is source-generator
syntax/selection evidence, not semantic/runtime validation of synthetic models.
The final generator runner also passes all 13 on `net8.0` and `net471`, with no
skips (39 final passing executions across the three target frameworks).

`CI_SHARD_INVENTORY.md` now records the actual cancelled run: one resolver job and
**zero executed shards**, not the obsolete June pass list. The PowerShell example
was parsed and exercised with ten controlled boundary cases: three invalid filters;
executed, empty, skipped-only and missing reports; and changed test, AiDotNet or
Tensors assemblies. It rejects invalid comparisons and restores the arena variable
in all cases. These helper controls did not execute model tests. Arena-off success
is explicitly not a root-cause diagnosis.

Reproduce these documentation-only boundary controls with
`pwsh -NoProfile -File .github/scripts/Test-ArenaComparisonExample.ps1`.

## Adversarial and compatibility checks

An independent review checked the generator cleanup, Finch field removal, new
vision-base naming, documentation, and diagnostic example. It identified that
hashing only the test DLL could compare changed production code; the example now
checks the test, AiDotNet and Tensors assemblies, with a negative control for each.

The request to rename `EagleOptions` was rejected as a breaking compatibility change:
both public `EagleOptions` types already exist at merge-base
`2a53ff3d4e27845773b4c9ad0a52b81f089613f7`. The language-model type was introduced
in `e966541dd941ea7095157e434683dcff95c3711d` (#1231), not this PR. Existing callers
can continue using their namespace qualification/alias. No existing public type
is removed to address that pre-existing naming ambiguity.

The core `src/AiDotNet.csproj` Release/`net10.0` build completed with **zero errors**
and 2,775 existing warnings in 4m48s. This is not a full test-project/compatibility
build or evidence that all CI shards pass. Actual sequence-model behavior and
the final hosted/review readiness disposition need their own completed results.
