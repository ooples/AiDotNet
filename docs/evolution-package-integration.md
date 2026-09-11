# Standalone evolution dependency

This companion feature branch incorporates the existing package migration from PR #2092 at
`104a2a41a47ebc8143cac230069462f8c14c08c3`; the migration PR and `master` remain unchanged. The merge is
`f317ed0442822abde290487d3b55e08cf234e0da`. It removes the embedded generic engine and its duplicated tests,
retaining consumer-specific program, facade, AutoML and result integration. Removed files remain in Git history.

The compiler/model roadmap needs the resource ledger and costed-proposal interfaces in the standalone engine.
Reusing those contracts avoids a second, divergent accounting implementation in AiDotNet. The new exact-source
identity and protected-edit checks remain in the consumer; owned snapshots copy their exact source and identity.

## Migration

- Reference the standalone `AiDotNet.Evolution` package. Generic engine types now belong to its assembly and
  `AiDotNet.Evolution` namespace, including engine options, enums and interfaces previously spread across
  `AiDotNet.Configuration`, `AiDotNet.Enums` and `AiDotNet.Interfaces`. Update imports and recompile consumers;
  old binary references are not redirected by type forwarders.
- Implement the owned-genome contract when supplying custom genomes. `ProgramGenome` already supplies an
  owned immutable snapshot; exact-source identity is unchanged by that copy.
- External `IAiModelBuilder` implementations must adopt the migrated `ConfigureEvolution` signatures, including
  `archiveFactory` and `winnerModelFactory`, in addition to `ConfigureProgramCorrectness`. The optional parameters
  also change the compiled method signatures, so existing binary callers must recompile.
- Old embedded-engine checkpoints are not silently migrated. Start a fresh run or explicitly validate and
  re-evaluate a migration; do not relabel old evidence.

## Local validation path

Normal builds consume the centrally pinned NuGet package. For local companion development:

```powershell
dotnet restore src/AiDotNet.csproj -p:UseLocalEvolution=true
dotnet build src/AiDotNet.csproj -c Release -f net10.0 -p:UseLocalEvolution=true
```

`EvolutionProjectPath` can override the default sibling path to
`../AiDotNet.Evolution/src/AiDotNet.Evolution/AiDotNet.Evolution.csproj`. Record the exact core commit and assembly
hashes: a project-reference build does not prove that a same-named published package contains that code.
Local validation of new core APIs must not be presented as validation of an older NuGet artifact. Hosted checks,
package-path validation, and current-head review remain separate requirements.

The public NuGet flat-container index was checked on September 10, 2026 and lists `0.1.0-preview.1`.
Its restored nuspec identifies source commit `f0f282cf2e9027ab5278a1953493c805d6ab52ad`, and its .NET 10 DLL
SHA-256 is `86ba7fb12e4bdac1676c7ea02635e6297d3471fa1bb8da866f27868cbcb13fcc`.
The migration PR's old publication-waiting note is therefore stale. That published artifact predates the new
resource ledger and costed-proposal APIs; consumers of those APIs require a subsequently published version,
not merely this already-available preview.

The compiler-guided source validation workflow uses sibling `consumer/` and `evolution/` checkouts. Nesting core
under the consumer inherited `Directory.Packages.props` and failed NU1008 before tests in hosted run
[`34548224747`](https://github.com/ooples/AiDotNet/actions/runs/34548224747). The normal package-path wiki check
also failed to resolve `EvolutionResourceLedger` in run
[`34548224654`](https://github.com/ooples/AiDotNet/actions/runs/34548224654), confirming the dependency gap.
Fixing source-checkout layout does not repair or replace that release requirement.

## Migration-stage evidence (historical)

With core revision `6d9aeb2dd44f5020469c81ad2cb4046f9f3495bb`, the merged consumer library builds on .NET 10
and .NET 8, and 857 selected authored tests pass on each. The selection includes program evolution, facade,
real MAP-Elites AutoML and evolution YAML integration. The previous 1,082-test count included duplicated generic
engine tests removed by the migration; counts are not directly comparable. One old reference-identity assertion
failed after adopting owned snapshots and was corrected to require the exact engine-owned parent, a distinct
caller object, and unchanged source/identity on repair exhaustion.

Production and test-copy assembly hashes matched before testing. Consumer DLL SHA-256:
.NET 10 `5113a7eb8faa5ab5cf33758e6dbaaaaae1a79cb58a1a11d994660b4a8eaeb417`;
.NET 8 `4228692e71147efc9f6c03f72abf39eef7fb84e4d26636b054f5cea4efed588d`.
The .NET 10 standalone core DLL was `a5c62d892576390741dbb4199b72f3b89b4a1476a406e7cb5065fc14c5487e69`.
The local harness compiled existing tests without unrelated generated fixtures; diagnostic analyzers were
disabled. These results do not establish whole-repository, .NET Framework, hosted, or published-package-path
validation of the merged consumer.
