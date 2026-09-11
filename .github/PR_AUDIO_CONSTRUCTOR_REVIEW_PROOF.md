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
the tested final binary is
`C70179FE727BB1EECF8FB31F1B0AC10AC1A30F000040D8D39D4A008A9DC33DFE`.
The actual source/generator build succeeded with existing analyzer warnings.
A temporary source-linked test project compiled the three checked-in
`PaperOptimizerBatch*ConstructorTests` classes and their shared base, plus the
repository deterministic CPU initializer and global imports. Reports include
`before-constructor-fixes.trx`, `after-constructor-fixes.trx`,
`after-constructor-fixes-repeat.trx` and `root-constructor-review.trx`.

In a normal checkout, the same checked-in cases can be selected with:

```powershell
dotnet test tests/AiDotNet.Tests/AiDotNet.Tests.csproj -c Release -f net10.0 --filter "FullyQualifiedName~PaperOptimizerBatch4ConstructorTests|FullyQualifiedName~PaperOptimizerBatch5ConstructorTests|FullyQualifiedName~PaperOptimizerBatch6ConstructorTests" --logger "trx;LogFileName=audio-constructor-review.trx"
```

Earlier branches contain only their corresponding test classes. Pushed-head
CodeQL still needs to confirm the alerts are fixed; local runtime proof is not
represented as a completed hosted scan. No suppression or warning threshold was
weakened.
