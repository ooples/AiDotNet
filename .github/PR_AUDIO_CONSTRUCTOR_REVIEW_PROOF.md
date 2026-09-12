# Audio constructor review proof — PRs #2119, #2120 and #2124

The existing stack is preserved: #2119 → #2120 → #2124. Eleven unsealed
ASR/TTS models called an overridable initialization hook from their constructors.
A derived instance could therefore execute before its constructor body was ready.

Each constructor now calls a private initialization core. The protected override
remains available and delegates to that same core after construction. Independent
whole-file comparison verified that all eleven production changes consist only
of those call substitutions and method extraction: the original layer-factory,
custom-layer, options, optimizer and ONNX-mode bodies are unchanged. No model or
hook was sealed, and no eager initialization was removed.

## Executed before/after proof

- Unchanged final-stack baseline `fc12d85b062c5a415772dea21d1adb6355445fd6`:
  **11 failed, 0 passed/skipped**, all with the exact derived-constructor-readiness
  error and stacks through the real model constructors.
- Fixed combined stack: **11 passed, 0 failed/skipped**, repeated twice by the
  implementing reviewer and once independently by the main reviewer.
- The twelve changed source/test files from the earlier two PRs are byte-identical
  on the tested final stack; independent Git comparisons confirmed this. This is
  combined-stack runtime evidence, not a claim of full test runs on each branch.

The tests construct real derived models with a supplied Dense layer. They assert
no derived initialization during construction, an eagerly present layer and
nonempty parameters, finite correctly shaped predictions, an accessible later
virtual hook after clearing/reinitializing layers, and unchanged predictions with
no duplicate layers during that lifecycle. These focused probes do not claim
full default-model, ONNX, speech-quality or GPU execution coverage. Generated
model-scaffold files were not manually edited.

The preserved baseline AiDotNet.dll SHA-256 is
`9052A22CB04845925B0BBCD71721F9848F49EB08DF4B371E80B5D4F106C13EC9`;
the initial net10 verification binary is
`C70179FE727BB1EECF8FB31F1B0AC10AC1A30F000040D8D39D4A008A9DC33DFE`.
The actual source/generator build succeeded with existing analyzer warnings.
A temporary source-linked test project compiled the three checked-in
`PaperOptimizerBatch*ConstructorTests` classes and their shared base, plus the
repository deterministic CPU initializer and global imports. Reports include
`before-constructor-fixes.trx`, `after-constructor-fixes.trx`,
`after-constructor-fixes-repeat.trx` and `root-constructor-review.trx`.

In a normal checkout, the same checked-in cases can be selected with:

```powershell
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --filter "FullyQualifiedName~PaperOptimizerBatch4ConstructorTests|FullyQualifiedName~PaperOptimizerBatch5ConstructorTests|FullyQualifiedName~PaperOptimizerBatch6ConstructorTests" --logger "trx;LogFileName=audio-constructor-review.trx"
```

Earlier branches contain only their corresponding test classes. Pushed-head
CodeQL still needs to confirm the alerts are fixed; local runtime proof is not
represented as a completed hosted scan. No suppression or warning threshold was
weakened.

## Compatibility follow-up: reproduced failure, corrected shared helper

Hosted compatibility jobs `103277146532` and `103277765505` exposed a missing
local target check: `Assert.Equal(int[], TensorShape)` used an assertion overload
available on modern .NET but not `net471`. The independent source-linked `net471`
build reproduced the exact **CS1503 at ConstructorInitializationTestBase.cs:40**,
before running tests. The original net10-only evidence did not prove compatibility.

Commit `d45d93ebbe` changes only that shared test base to compare shape rank and
dimension through the typed shape API, preserving the full assertion without
changing production initialization or TensorShape. No generated model file was edited.

The source-linked project was expanded to every declared target and passed
**11/11 on net10.0, 11/11 on net8.0, and 11/11 on net471**, zero skips.
The primary reviewer then built the **actual complete compatibility test project**:

```powershell
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -p:CompatBuildOnly=true
```

Result: **build succeeded, zero errors**, with 13,488 existing warnings reported.
The eleven regressions were then independently executed from those actual
`AiDotNetTests.dll` outputs, not the focused harness: **11/11 net8.0 and 11/11
net471, zero skips**. The commands used `dotnet test` on the same project with
`-c Release -f <framework> --no-build --no-restore` and the filter above.

Preserved evidence in the local `recipe-constructor-review-harness-20260911` directory:
`compat-constructor-before.log`, `constructor-all-frameworks-after.log`,
`actual-compat-project-build.log`, the three `constructor-all-frameworks-after_*.trx`
reports, and `audio-actual-net8.0-after.trx` / `audio-actual-net471-after.trx`.
This closes the reproduced compile regression; it is not a claim that every
model shard or pending hosted CodeQL/CI check has completed successfully.
