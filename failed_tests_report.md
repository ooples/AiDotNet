# Failed Tests Report for PR #719 - Memory<T> Migration

## Summary of Failed Tests

### 1. GemmKernelValidationTests (InferenceOptimization)
**Test:** `Execute_MatchesNaiveGemm`, `GemmTransposeB_MatchesNaive`
**Error:** `Assert.Equal() Failure: Collections differ`
- Expected: `[58, 64, 139, 154]`
- Actual: `[0, 0, 0, 0]`
**Root Cause:** Data not being copied to tensor properly - likely `Array.Copy` to `Memory<T>` issue

### 2. AudioProcessingTests (Diffusion)
**Tests:** `AudioProcessor_PadOrTruncate_TruncatesCorrectly`, `AudioProcessor_PadOrTruncate_PadsCorrectly`
**Error:** `Assert.Equal() Failure: Values differ`
- Expected: `1`
- Actual: `0`
**Root Cause:** Likely foreach over `Memory<T>` or indexing issue

### 3. LoRALayerTests / VBLoRAAdapterTests (NeuralNetworks)
**Tests:** `UpdateParameters_UpdatesParametersCorrectly`, `GetParameterGradients_AfterBackward_ReturnsValidGradients`, `DenseLayerForward_WithNonZeroInput_ProducesNonZeroOutput`
**Error:** `Assert.True() Failure` and `Assert.Contains() Failure: Filter not matched in collection`
**Root Cause:** Parameter/gradient access issues with Memory<T>

### 4. CpuAdvancedAlgebraEngineTests (Engines)
**Tests:**
- `So3ComposeBatch_IdentityComposition`
- `So3ExpLogBatch_RoundTrip`
- `So3AdjointBatch_ReturnsRotationMatrices`
- `So3ExpBatch_ZeroVectors_ReturnsIdentities`
- `Se3ExpBatch_ZeroVector_ReturnsIdentity`
- `Se3ExpLogBatch_RoundTrip`
**Error:** Stack trace only (likely exception thrown)
**Root Cause:** Memory<T> indexing in Lie algebra operations

### 5. LieGroupTests (Groups)
**Tests:** `So3_ExpLog_IdentityRoundTrip`, `Se3_ExpLog_IdentityRoundTrip`
**Error:** Stack trace only (likely exception thrown)
**Root Cause:** Same as CpuAdvancedAlgebraEngineTests

### 6. GradientCorrectnessTests (Autodiff)
**Tests:** `DenseLayer_AutodiffGradients_MatchManualGradients`, `ResidualLayer_AutodiffGradients_MatchManualGradients`
**Error:** Stack trace only (likely exception thrown)
**Root Cause:** Gradient computation with Memory<T>

### 7. ScientificMLTests / NeuralOperatorTrainingTests (PhysicsInformed)
**Tests:** `HamiltonianNeuralNetwork_TrainUpdatesParameters`, `DeepOperatorNetwork_TrainUpdatesParameters`
**Error:** `Assert.False() Failure`
- Expected: `False`
- Actual: `True`
**Root Cause:** Training parameter update check failing

### 8. InferenceOptimizerTests (Inference)
**Tests:** `InferenceOptimizer_WeightOnlyQuantization_RewritesDenseLayer_OnClonedModel_AndPreservesOutputs`, `Debug_WeightOnlyQuantization`
**Error:** Stack trace only
**Root Cause:** Model cloning/quantization with Memory<T>

## Common Patterns to Fix

1. **Array.Copy → Span.CopyTo**: Tests using `Array.Copy(src, tensor.Data, len)` need `src.AsSpan().CopyTo(tensor.Data.Span)`

2. **Direct indexing on Memory<T>**: `tensor.Data[i]` must be `tensor.Data.Span[i]`

3. **foreach over Memory<T>**: `foreach (var x in tensor.Data)` must use `.ToArray()` or iterate over `.Span`

4. **Assert.Equal on Memory<T>**: `Assert.Equal(expected, tensor.Data)` comparing Memory objects, not values

## Files to Check

1. `tests/AiDotNet.Tests/InferenceOptimization/GemmKernelValidationTests.cs`
2. `tests/AiDotNet.Tests/UnitTests/Diffusion/AudioProcessingTests.cs`
3. `tests/AiDotNet.Tests/UnitTests/NeuralNetworks/LoRALayerTests.cs`
4. `tests/AiDotNet.Tests/UnitTests/NeuralNetworks/VBLoRAAdapterTests.cs`
5. `tests/AiDotNet.Tests/UnitTests/Engines/CpuAdvancedAlgebraEngineTests.cs`
6. `tests/AiDotNet.Tests/UnitTests/Groups/LieGroupTests.cs`
7. `tests/AiDotNet.Tests/UnitTests/Autodiff/GradientCorrectnessTests.cs`
8. `tests/AiDotNet.Tests/UnitTests/PhysicsInformed/ScientificML/ScientificMLTests.cs`
9. `tests/AiDotNet.Tests/UnitTests/PhysicsInformed/NeuralOperators/NeuralOperatorTrainingTests.cs`
10. `tests/AiDotNet.Tests/UnitTests/Inference/InferenceOptimizerTests.cs`
11. `src/AiDotNet.Tensors/Engines/CpuEngine.cs` (Lie group implementations)
