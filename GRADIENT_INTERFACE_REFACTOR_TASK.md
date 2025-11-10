# TASK: Refactor Gradient Computation Interfaces for ZeRO-2 Distributed Training

## Context

AiDotNet is a .NET ML framework (targets: net462, net471, net8.0, net9.0) with distributed training support (DDP, ZeRO-1, ZeRO-2, FSDP).

**CRITICAL ISSUE**: ZeRO-2 distributed training is currently broken because `IFullModel.Train()` couples gradient computation with parameter updates, making it impossible to:
1. Compute true gradients without updating parameters
2. Synchronize gradients before parameter updates (required for ZeRO-2)
3. Use adaptive optimizers correctly (Adam, RMSprop) in distributed settings

**Current broken flow:**
```
ZeRO2Model.Train() → WrappedModel.Train() → parameters updated
  → compute "gradients" as params_after - params_before (WRONG! These are parameter deltas, not gradients)
```

**Correct ZeRO-2 flow:**
```
1. ComputeGradients() → true gradients (∂Loss/∂params)
2. ReduceScatter gradients → each rank gets shard
3. UpdateParameters(shard_params, shard_grads) → update local shard only
4. AllGather shards → reconstruct full parameters
```

## Current Architecture

### Existing Interfaces

**src/Interfaces/IGradientComputable.cs** (MAML-focused, needs splitting):
- `ComputeGradients(input, target, ILossFunction<T> lossFunction)` - requires loss function
- `ApplyGradients(gradients, learningRate)` - applies gradients
- `ComputeSecondOrderGradients(...)` - MAML-specific second-order derivatives

**src/Interfaces/IFullModel.cs** (needs updating):
- Does NOT extend IGradientComputable currently
- Models implement Train() which couples gradient+update

**src/Models/Inputs/OptimizationInputData.cs**:
- Already has `InitialSolution` property for distributed optimizers

## Your Task: Implement Interface Refactor

### Step 1: Split IGradientComputable into Base + Extended

**Rename existing file to:** `src/Interfaces/ISecondOrderGradientComputable.cs`

**Create new file:** `src/Interfaces/IGradientComputable.cs` with ONLY basic gradient methods:

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that can compute gradients for optimization and distributed training.
/// </summary>
public interface IGradientComputable<T, TInput, TOutput>
{
    /// <summary>
    /// Computes gradients WITHOUT updating parameters.
    /// </summary>
    /// <param name="input">Input data</param>
    /// <param name="target">Target output</param>
    /// <param name="lossFunction">Loss function (null = use model default)</param>
    /// <returns>Gradient vector</returns>
    Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null);

    /// <summary>
    /// Applies pre-computed gradients.
    /// </summary>
    void ApplyGradients(Vector<T> gradients, T learningRate);
}
```

**Update:** `src/Interfaces/ISecondOrderGradientComputable.cs` to extend base:

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Extended gradient computation for MAML meta-learning.
/// </summary>
public interface ISecondOrderGradientComputable<T, TInput, TOutput> : IGradientComputable<T, TInput, TOutput>
{
    /// <summary>
    /// Computes second-order gradients for MAML.
    /// </summary>
    Vector<T> ComputeSecondOrderGradients(
        List<(TInput input, TOutput target)> adaptationSteps,
        TInput queryInput,
        TOutput queryTarget,
        ILossFunction<T> lossFunction,
        T innerLearningRate);
}
```

### Step 2: Update IFullModel

**Modify:** `src/Interfaces/IFullModel.cs`

Add to interface inheritance list:
```csharp
public interface IFullModel<T, TInput, TOutput> :
    IModel<TInput, TOutput, ModelMetadata<T>>,
    IGradientComputable<T, TInput, TOutput>,  // ✅ ADD THIS
    IModelSerializer,
    // ... rest
{
    /// <summary>
    /// Default loss function for gradient computation.
    /// </summary>
    ILossFunction<T> DefaultLossFunction { get; }
}
```

### Step 3: Update Existing Models

For each model class implementing IFullModel:

1. **Add DefaultLossFunction property**:
```csharp
private ILossFunction<T> _lossFunction;

public ILossFunction<T> DefaultLossFunction => _lossFunction
    ?? throw new InvalidOperationException("Loss function not configured");
```

2. **Implement ComputeGradients**:
```csharp
public Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
{
    var loss = lossFunction ?? DefaultLossFunction;

    // Forward pass
    var predictions = Predict(input);

    // Backpropagate to compute gradients
    // (Implementation varies by model type)

    return gradients;
}
```

3. **Implement ApplyGradients**:
```csharp
public void ApplyGradients(Vector<T> gradients, T learningRate)
{
    var currentParams = GetParameters();
    var newParams = new Vector<T>(currentParams.Length);

    for (int i = 0; i < currentParams.Length; i++)
    {
        newParams[i] = NumOps.Subtract(currentParams[i],
            NumOps.Multiply(learningRate, gradients[i]));
    }

    SetParameters(newParams);
}
```

4. **Update Train() to use new methods**:
```csharp
public void Train(TInput input, TOutput expectedOutput)
{
    var gradients = ComputeGradients(input, expectedOutput);
    ApplyGradients(gradients, _learningRate);
}
```

### Step 4: Find Models to Update

Run these commands to find all models:

```bash
# Find model classes
grep -r "class.*IFullModel" src/ --include="*.cs" | grep -v interface

# Find classes that might need loss functions
grep -r "ILossFunction" src/Models/ --include="*.cs"
```

Common model classes to update:
- `NeuralNetworkModel`
- `LinearRegressionModel`
- `LogisticRegressionModel`
- Any custom model implementations

### Step 5: Testing and Verification

1. **Build project**:
```bash
dotnet build
```

2. **Verify backward compatibility**:
   - Existing Train() calls should still work
   - Models can now call ComputeGradients separately

3. **Test ZeRO-2 integration**:
   - ZeRO2Model should be able to call WrappedModel.ComputeGradients()
   - Gradients should be true gradients, not parameter deltas

### Step 6: Commit Changes

```bash
git add src/Interfaces/IGradientComputable.cs
git add src/Interfaces/ISecondOrderGradientComputable.cs
git add src/Interfaces/IFullModel.cs
# Add updated model files
git commit -m "refactor: split gradient interfaces for distributed training support

- Split IGradientComputable into base (basic) and ISecondOrderGradientComputable (MAML)
- Make IFullModel extend IGradientComputable
- Add DefaultLossFunction property to IFullModel
- Implement ComputeGradients in base model classes

Enables ZeRO-2 to compute true gradients instead of parameter deltas.
Fixes gradient synchronization with adaptive optimizers (Adam, RMSprop).

Backward compatible: Train() continues to work as before."
```

## Critical Constraints

- Target frameworks: net462, net471, net8.0, net9.0
- NEVER use null-forgiving operator (!)
- ALWAYS use Newtonsoft.Json (NOT System.Text.Json)
- Use `is not null` pattern for null checks
- Document all public APIs with XML comments
- Follow existing code style

## Success Criteria

- [x] IGradientComputable has only basic gradient methods
- [x] ISecondOrderGradientComputable extends it with MAML methods
- [x] IFullModel extends IGradientComputable
- [x] All IFullModel classes implement ComputeGradients and ApplyGradients
- [x] Build succeeds with zero errors
- [x] No breaking changes to existing code
- [x] ZeRO2Model can call WrappedModel.ComputeGradients()

## Expected Files Changed

**Must change:**
1. `src/Interfaces/IGradientComputable.cs` - New base interface
2. `src/Interfaces/ISecondOrderGradientComputable.cs` - MAML extension
3. `src/Interfaces/IFullModel.cs` - Extend IGradientComputable

**Should change:**
4. `src/Models/NeuralNetwork/NeuralNetworkModel.cs` - Implement gradient methods
5. `src/Models/Regression/LinearRegressionModel.cs` - Implement gradient methods
6. Other model base classes as needed

## Questions or Issues?

If you encounter problems:
1. Check if models already have loss function configuration
2. Look for existing gradient computation code to refactor
3. Ensure Train() calls ComputeGradients internally for backward compatibility
4. Start with one model class, verify it works, then update others

## Next Steps After This Refactor

Once interfaces are updated:
1. Update ZeRO2Model to use WrappedModel.ComputeGradients()
2. Update ZeRO2Optimizer to use true gradients
3. Remove parameter delta workarounds
4. Add unit tests for gradient computation

Good luck! This refactor enables production-ready distributed training.
