# Generator incremental-caching refactor — state

Branch: refactor/generator-incremental-caching (worktree AiDotNet-genrefactor)
Pushed: 688aa13ef

## The defect being fixed
Generators returned INamedTypeSymbol from CreateSyntaxProvider `transform:` and then
Combine()d with CompilationProvider. ISymbol is not value-equatable (nothing caches)
and each retained symbol roots the whole Compilation (cache pins compilations in RAM).
That is the VBCSCompiler multi-GB growth.

## Recipe (proven 4x)
1. transform returns a value-equatable model, not a symbol; semantic work moves into Analyze().
2. Compilation read transiently via ctx.SemanticModel.Compilation; never escapes pipeline.
3. Collect().Combine(CompilationProvider) -> plain Collect(); Execute -> Emit(ImmutableArray<Model>).
4. Model: immutable, ImmutableArray<>, explicit IEquatable. Equality MUST reach through
   nested collections (nested types need IEquatable too).

## Known traps
- `.Count` on ImmutableArray binds to the LINQ extension METHOD GROUP -> use `.Length`.
- Text anchors can collide with the generator's own emitted string literals (e.g. "return entry;").
- Write conversion scripts to files; bash heredocs break on embedded C# quoting.

## Verification
Build src, diff src/Generated against snapshot at
scratchpad/genbaseline (1245 files). Must be ZERO differences.
NOTE: src/Generated is UNTRACKED and survives branch switches — re-snapshot from a clean build.

## Status
DONE+verified+committed: DiscoveryApi(24ac7a489), ComponentDiscoveryApi(749aff908), ComponentRegistry(688aa13ef)
CONVERTED, compiles, verification pending: Documentation
REMAINING (8): CompatibilityMatrix, ComponentMetadataValidation, ModelMetadataValidation,
ModelParameter, ModelRegistry, TrainableParameter, YamlConfigSource, TestScaffold(18k lines - own PR)

## Environment blockers
- Builds die in the GENERATION phase when free RAM is low. Fix: kill VBCSCompiler
  (it reached 10 GB), which restores several GB. Machine has 15.4 GB total.
- Disk chronically near-full; NuGet global-packages already cleared once (35 GB).
- Another session works in AiDotNet-pr2034 (PR #2035) — do NOT use that worktree.

## Deferred, not done
SOM/RBF split: fix SelfOrganizingMap all-zero output as a product bug; exempt
RadialBasisFunctionNetwork from ScaledInput_ShouldChangeOutput with rationale
(Gaussian saturation far from centres is correct).
