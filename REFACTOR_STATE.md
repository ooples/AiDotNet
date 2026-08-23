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

## Status: COMPLETE

All fifteen generators converted and verified. Nothing here is outstanding; this file is a record
of the approach, not a work queue.

Converted: DiscoveryApi, ComponentDiscoveryApi, ComponentRegistry, Documentation,
CompatibilityMatrix, ModelRegistry, YamlConfigSource, ModelParameter, TrainableParameter,
ModelMetadataValidation, ComponentMetadataValidation, TestScaffold, and then AgentToolSchema,
ShapeContract, TensorPortContract.

The last three were NOT in the original list of twelve. That list was scoped by "uses
CompilationProvider", which is the wrong symptom -- they leak symbols without using it. Conversely
YamlConfigSource legitimately needs CompilationProvider and keeps it. Scope by "symbols in cached
pipeline state", never by the API used to reach them.

## What is NOT fixed

Retention only, for ModelParameter, TrainableParameter, both validation generators and the final
three: they no longer hold compilations alive, but they still re-run every compilation because
their symbol-walking bodies were left outside the transform. Making them genuinely cacheable is
follow-up work.

No memory win was demonstrated. Three successive builds against one compiler server measured
192/165/165 MB before and 200/180/179 MB after -- neither arm grows, and the two are within noise.
That harness does not reproduce the 12.6 GB VBCSCompiler seen during development, so a better
experiment (design-time builds, or many more incremental compilations) is still owed.

## Environment notes

- Builds die in the GENERATION phase when free RAM is low. Fix: kill VBCSCompiler
  (it reached 10 GB), which restores several GB. Machine has 15.4 GB total.
- Disk chronically near-full; NuGet global-packages already cleared once (35 GB).

## SOM/RBF changes

Completed in this PR: `SelfOrganizingMap.Predict` now returns a one-hot BMU vector, its
contract has direct regression coverage, and the SOM/RBF scaled-input exemptions document
why global rescaling is not a valid sensitivity probe for these local/competitive models.
