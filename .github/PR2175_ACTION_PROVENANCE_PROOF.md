# PR #2175 action provenance follow-up

This addresses [review 3994356321](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3994356321),
thread `PRRT_kwDOKSXUF86hrPEC`, against pushed head
`0eb3f0834a2b3d7d5268f2abaf55d422dbc86b36`. The fresh review query had one unresolved
thread and `hasNextPage=false` for both thread and comment pagination. Earlier
[115 + 39 case evidence](PR2175_REVIEW_PROOF.md) remains historical evidence for that head;
it is not substituted for this follow-up's verification.

The production/test change is `77fbd427f7981a6aee818927a1123c41f9dd7f69`.
The final runner-initialization correction is
`c46947e20be6e98e9c16b6ffa75ed5386273f12c`.

## Root cause and contract

The old `TryGetValue(...) && ...` guard admitted any exact one-hot vector without
a selection stamp. Merely flipping that condition would break supervised
`Train(state, target)`: its shared implementation discards the sampled action and
constructs a different one-hot vector from the label.

Public A2C collection now requires the original, unconsumed action object sampled
by this agent in training mode, in the current policy epoch, for the same state
values. Caller-created, cloned, foreign-agent, greedy, changed and stale actions
are rejected. This deliberately tightens the previously accepted caller-created
action behavior; callers must retain the original `SelectAction` result.

The state copy moves from storage to selection. Successful storage transfers that
owned snapshot into the transition without copying it a second time. Action and
next-state copies remain defensive. State comparison stays in `T`, not a rounded
`double` representation. Invalid input does not consume the stamp or enqueue a
transition; a successful enqueue consumes it once. Enqueue precedes capacity
eviction so an allocation failure cannot first drop an existing transition.

The shared protected `StoreSupervisedExperience` hook defaults to the original
public `StoreExperience` dispatch for other agents. A2C instead validates and
isolates the labelled transition, invalidates pending rollout/action provenance,
and uses the existing one-shot update. No public flag bypasses the collection
guard. The actual DQN subclass control proves the default dispatch and update.

A caller sweep found that `FinRLAgent` forwarded collection and parameterless
training but inherited the wrapper's supervised implementation. It therefore
lost the inner agent's supervised readiness flag, and A2C's stricter boundary
would reject the wrapper-created target. The wrapper now forwards the complete
supervised call to its inner owner. Five actual controls retain A2C, DQN, PPO and
SAC coverage, including A2C's previously working batch-size-one path.

The existing limitation for opaque custom parameter storage, retained writable
spans and tensor inference-mode writes is unchanged: such callers must use the
agent's explicit update boundary, and cannot mutate policy concurrently with
collection. This does not claim that identity stamps detect otherwise unobservable
external writes or provide a thread-safe actor-training API.

## Failure-first proof and preserved assertions

All runs use the real AiDotNet library on Windows CPU, with the existing serial
xUnit configuration. The old core hash is
`4C5C6F988C22D27039D58EB15477D91E57E64D802A04A3331BC41F847D0F8C93`.

| Actual old-core cohort | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| New action-provenance cases | 3 | 9 | 0 |
| Original 115 plus new 12 | 118 | 9 | 0 |
| Existing shared-boundary cohort | 39 | 0 | 0 |
| Expanded original 115 plus new 17 | 120 | 12 | 0 |

The expanded 132-case negative control reproduced **120 passed / 12 failed / 0
skipped** on net10.0, net8.0 and net471. The unchanged net8.0 old-core hash is
`47A4404F459D3BCDB2E39D076C402E6314CDE628294ECDAE677E638BBBA1BE91`;
the unchanged net471 old-core hash is
`57ED244696C9E2ACD315A8973DC7795446CCDB96C22946D29174EFCF7D7E89F6`.

The first expanded net471 run had 119 passes and 13 failures: the additional
existing DQN epsilon-serialization test reached the trial persistence limit.
Unlike modern targets, net471 has no automatic module-initializer attribute, and
the small financial runner had not registered the existing shared xUnit framework.
Linking `FinancialSharedReviewTestFramework.cs` initializes the existing CPU and
test-license policy before discovery. No licensing implementation or per-test
bypass changed. A test-only rebuild against the exact same core restored the
expected 120/12 result. The initial failure TRX and pre-initializer runner are
retained, rather than treating that first run as a clean compatibility baseline.

The five wrapper controls on the old core had two passes and three failures:
A2C at batch size one and PPO passed; A2C, DQN and SAC with warmup failed to update.
On the first strict core, before forwarding was fixed, the same five had one pass
and four failures, proving the new A2C batch-size-one regression as well. The full
intermediate cohort was **128 passed / 4 failed / 0 skipped**: all original 115,
all 12 provenance controls and the PPO control passed. Its core hash is
`7C7D85C9EB502ECCA2005B020E8A531DD8E192C56CBE3C2C535F7EF2BB67B12C`.

The nine failures reproduce missing/clone/foreign provenance, reuse, state
mismatch, supervised rollout mixing and public collection during a supervised
update. Passing controls cover failed next-state/action validation followed by a
valid store and another agent's existing supervised dispatch.

Existing numeric fixtures still request the same action indices and retain their
seeds and assertions. They now obtain those actions from bounded genuine sampling
instead of constructing one-hot vectors. The helper fails if the desired action
is not sampled in 512 draws; it cannot silently return a different action. The
owned-vector comparison explicitly checks both agents sampled identical actions
before retaining its exact final-parameter comparison. The running-stat mutation
test now uses a still-unconsumed action, so single-use rejection cannot make the
old policy-version control vacuous. All adapted original 115 cases passed on the
unchanged core before the production fix was built.

The separate shared cohort retains its existing filter excluding two
foundation-scale EMMDiT `HeavyTimeout` cases. Those two are not claimed as run;
no new skip, timeout increase or tolerance change was introduced.

Retained baseline TRXs:

- `artifacts/pr2175-provenance-before/results/provenance-before.trx`
- `artifacts/pr2175-provenance-before/results/provenance-full-before.trx`
- `artifacts/pr2175-provenance-before/results/provenance-shared-before.trx`
- `artifacts/pr2175-provenance-before/results/provenance-expanded-before.trx`
- `artifacts/pr2175-provenance-before/results/provenance-expanded-before-net8.0.trx`
- `artifacts/pr2175-provenance-before/results/provenance-expanded-before-net471.trx` (initial runner gap)
- `artifacts/pr2175-provenance-before/results/provenance-expanded-before-initialized-net471.trx`
- `artifacts/pr2175-provenance-before/results/AiDotNetTests-before-framework-net471.dll`
- `artifacts/pr2175-provenance-before/results/finrl-before.trx`
- `artifacts/pr2175-provenance-final/results/finrl-intermediate-before.trx`
- `artifacts/pr2175-provenance-final/results/provenance-intermediate132.trx`

The retained expanded full-baseline runner hash is
`C0F2A9BCB8E80CFE0FB594CAA9AAFCCCEC17E4767B43E9AA39FF56801FB66D2B`;
the baseline shared runner hash is
`BFD37A2E5F9BA97979F50C316708557DBCE2CB441C1CEA6F8D9ED4EF9EBF8C9B`.
The retained expanded intermediate runner hash is
`D40493F2AF2766270B6D5109994D6F23877C1C25E53C7B57AA9D3BC38F60CAAE`.
The corrected net471 baseline runner hash is
`C1B2515F8CB79C3B5DB04B655B00A8E37A898F1EEFC7DF40AA81B84BE7FA8985`.
The initial 12/127-case TRXs remain historical runs; their runner was later
rebuilt test-only for the expanded 132-case controls above.

One intermediate test-only build failed with CS8104/CS0016 because the disk was
full. No test ran from that failed build. After non-destructive artifact
compression and an independent recovery of free space, the test-only rebuild
succeeded and produced the intermediate evidence above. Both retained core
hashes remained unchanged; no baseline or final core/TRX was deleted.

## Corrected-library verification

The corrected build uses source/test revision
`c46947e20be6e98e9c16b6ffa75ed5386273f12c`, including the FinRL forwarding fix.
All three actual-library builds completed with zero errors:

| Target | Build time | Warnings |
| --- | ---: | ---: |
| net10.0 | 4m24s | 2,829 |
| net8.0 | 4m11.84s | 2,761 |
| net471 | 4m00.39s | 2,763 |

The separate shared runner test-only builds had zero errors and zero warnings
on all three targets.

| Corrected target | Financial cohort | Existing shared cohort | Failures / skips |
| --- | ---: | ---: | ---: |
| net10.0 | 132 / 132 (13s) | 39 / 39 (12s) | 0 / 0 |
| net8.0 | 132 / 132 (8s) | 39 / 39 (10s) | 0 / 0 |
| net471 | 132 / 132 (14s) | 39 / 39 (18s) | 0 / 0 |

Times are VSTest-reported execution durations, excluding process startup. These
are actual Windows CPU results, not GPU numeric or full-repository CI claims.
After independently reviewing the source, the root reviewer reran the exact
net10.0 financial library/runner hashes below: **132 passed / 0 failed / 0 skipped
in 10s**. That replay did not rebuild or change the test inputs.

Frozen artifacts under `artifacts/pr2175-provenance-corrected`:

| Target | Artifact | SHA-256 |
| --- | --- | --- |
| net10.0 | Actual AiDotNet library | `08A9460A2E9123083A11CE27248AD0E6AE82C0FC6520077E559B8C756AE53F48` |
| net10.0 | Financial 132-case runner | `58484E50A041466406642B018260F36DE2426F722718D09005BD3DB1D7AFDE64` |
| net10.0 | Shared 39-case runner | `D72AF5DE6651C763F00B223779A16D0FAD7DAD891BC35236B00353A2DFF42077` |
| net8.0 | Actual AiDotNet library | `62DF35E04A24B64F487EA7A63EF4C5053B2E62760B3EAB1622632E7BE49CE929` |
| net8.0 | Financial 132-case runner | `E042346DA33AE615241D9045BDBE6F803E457B2031F9D7362BE37EB45B0B187E` |
| net8.0 | Shared 39-case runner | `9623C4CAAEAA4CE0DCC1287DB6034A99966B20BE2C319DCAEE488D98662B4040` |
| net471 | Actual AiDotNet library | `C2B854BD8E372DE5B24C9591DFCCC9C79246DB929790C7D6D9D827FCF18C77B0` |
| net471 | Financial 132-case runner | `312EED8947D0273818A3A026C76D7253D0F19BA97E04CD9DA18095432F15EFEF` |
| net471 | Shared 39-case runner | `0312E428F4EA40CC469072F6D00B516B3A9522EC3609ADB2B73F02A41E30A73E` |

Result files:

- `results/provenance-corrected-net10.0.trx`
- `results/provenance-shared-corrected-net10.0.trx`
- `results/provenance-corrected-net8.0.trx`
- `results/provenance-shared-corrected-net8.0.trx`
- `results/provenance-corrected-net471.trx`
- `results/provenance-shared-corrected-net471.trx`
- `results/build-net10.0.log`
- `results/build-net8.0.log`
- `results/build-net471.log`
- `results/provenance-root-independent-net10.trx` (independent replay)
- `root-independent-net10.log` (independent replay log)

## Reproduction

Use a fresh artifact directory from the follow-up revision. Commands fail closed
on build or test errors; no stale assembly is accepted after a failed build.
Framework identifiers below are SDK target names, not runtime policy choices.

```powershell
$ErrorActionPreference = 'Stop'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$artifactRoot = 'artifacts/pr2175-provenance-reproduction'
if (Test-Path -LiteralPath $artifactRoot) { throw 'Choose a fresh artifact directory.' }
$project = 'tests/AiDotNet.FinancialAgentReview/AiDotNet.FinancialAgentReview.csproj'
$sharedProject = 'tests/AiDotNet.FinancialSharedReview/AiDotNet.FinancialSharedReview.csproj'
foreach ($tfm in @('net10.0', 'net8.0', 'net471')) {
    dotnet build $project -c Release -f $tfm --artifacts-path $artifactRoot -m:1 `
        -p:UseSharedCompilation=false -p:CopyLocalRuntimeTargetAssets=false `
        -p:_GetChildProjectCopyToOutputDirectoryItems=false -p:GeneratePackageOnBuild=false -v:quiet --nologo
    if ($LASTEXITCODE -ne 0) { throw "Actual $tfm build failed." }
    $runner = Join-Path $artifactRoot ('bin/AiDotNet.FinancialAgentReview/release_' + $tfm)
    pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 -RunnerJson (Join-Path $runner 'xunit.runner.json')
    if ($LASTEXITCODE -ne 0) { throw 'Runner hardening failed.' }
    dotnet vstest (Join-Path $runner 'AiDotNetTests.dll') `
        ('/Logger:trx;LogFileName=provenance-' + $tfm + '.trx') `
        ('/ResultsDirectory:' + (Join-Path $artifactRoot 'results'))
    if ($LASTEXITCODE -ne 0) { throw "Actual $tfm financial cohort failed." }

    dotnet build $sharedProject -c Release -f $tfm --artifacts-path $artifactRoot -m:1 `
        -p:BuildProjectReferences=false -p:UseSharedCompilation=false `
        -p:CopyLocalRuntimeTargetAssets=false -p:_GetChildProjectCopyToOutputDirectoryItems=false `
        -p:GeneratePackageOnBuild=false -v:quiet --nologo
    if ($LASTEXITCODE -ne 0) { throw "Shared $tfm runner build failed." }
    $sharedRunner = Join-Path $artifactRoot ('bin/AiDotNet.FinancialSharedReview/release_' + $tfm)
    pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 -RunnerJson (Join-Path $sharedRunner 'xunit.runner.json')
    if ($LASTEXITCODE -ne 0) { throw 'Shared runner hardening failed.' }
    dotnet vstest (Join-Path $sharedRunner 'AiDotNetTests.dll') '/TestCaseFilter:Category!=HeavyTimeout' `
        ('/Logger:trx;LogFileName=provenance-shared-' + $tfm + '.trx') `
        ('/ResultsDirectory:' + (Join-Path $artifactRoot 'results'))
    if ($LASTEXITCODE -ne 0) { throw "Actual $tfm shared cohort failed." }
}
```

To reproduce the negative control independently, create an unused detached
worktree at `0eb3f0834a2b3d7d5268f2abaf55d422dbc86b36`, apply only the test changes
from `c46947e20be6e98e9c16b6ffa75ed5386273f12c`, and build the real old library.
The original core used for the recorded runs was retained from the earlier
source-equivalent build; rebuilding can change embedded Git metadata and hashes.

```powershell
$ErrorActionPreference = 'Stop'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$baseline = '0eb3f0834a2b3d7d5268f2abaf55d422dbc86b36'
$followup = 'c46947e20be6e98e9c16b6ffa75ed5386273f12c'
$baselinePath = Join-Path (Split-Path (Get-Location) -Parent) 'pr2175-provenance-baseline'
if (Test-Path -LiteralPath $baselinePath) { throw 'Choose an unused baseline worktree path.' }
git worktree add --detach $baselinePath $baseline
if ($LASTEXITCODE -ne 0) { throw 'Baseline checkout failed.' }
$testPatch = git diff $baseline $followup -- tests/AiDotNet.FinancialAgentReview tests/AiDotNet.Tests/UnitTests/Finance
if ($LASTEXITCODE -ne 0) { throw 'Baseline test-only diff failed.' }
$testPatch | git -C $baselinePath apply -
if ($LASTEXITCODE -ne 0) { throw 'Baseline test-only patch failed.' }
Push-Location $baselinePath
try {
    dotnet build tests/AiDotNet.FinancialAgentReview/AiDotNet.FinancialAgentReview.csproj `
        -c Release -f net10.0 --artifacts-path artifacts/provenance-before -m:1 `
        -p:UseSharedCompilation=false -p:CopyLocalRuntimeTargetAssets=false `
        -p:_GetChildProjectCopyToOutputDirectoryItems=false -p:GeneratePackageOnBuild=false -v:quiet --nologo
    if ($LASTEXITCODE -ne 0) { throw 'Old-source build failed.' }
    pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 `
        -RunnerJson artifacts/provenance-before/bin/AiDotNet.FinancialAgentReview/release_net10.0/xunit.runner.json
    if ($LASTEXITCODE -ne 0) { throw 'Baseline runner hardening failed.' }
    dotnet vstest artifacts/provenance-before/bin/AiDotNet.FinancialAgentReview/release_net10.0/AiDotNetTests.dll `
        '/Logger:trx;LogFileName=before.trx' '/ResultsDirectory:artifacts/provenance-before/results'
    $beforeExit = $LASTEXITCODE
    [xml]$beforeTrx = Get-Content -LiteralPath artifacts/provenance-before/results/before.trx
    $counts = $beforeTrx.TestRun.ResultSummary.Counters
    if ($beforeExit -ne 1 -or $counts.passed -ne '120' -or $counts.failed -ne '12' -or $counts.notExecuted -ne '0') {
        throw 'Expected exactly 120 passes, 12 reproduced failures and zero skips.'
    }
} finally {
    Pop-Location
}
```
