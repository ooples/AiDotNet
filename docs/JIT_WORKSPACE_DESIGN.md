# JIT Compiler + TensorWorkspace: Zero-Allocation Forward Pass Design

## Problem Statement

The current CodeGenerator emits calls to `TensorOperations<T>` which wraps every intermediate result in a `ComputationNode<T>` and allocates a new tensor per operation. The "fused" IR ops I added (FusedGroupNormActivationOp, FusedConv2DBiasActivationOp, etc.) just chain the same allocating calls — they provide ZERO actual performance benefit.

For a production SD15 UNet with 50 denoising steps, the current approach allocates ~60GB of intermediate tensors, causing OOM.

## Target Architecture

```
Inputs: Tensor<T>[]
    |
    v
WorkspaceCodeGenerator.Generate<T>(IRGraph, MemoryPlan)
    |
    v
Compiled Function: Action<Tensor<T>[], TensorWorkspace<T>, IEngine>
    |
    v
At runtime:
    1. workspace = pre-allocated TensorWorkspace with all slots
    2. engine = current IEngine (CpuEngine or DirectGpuTensorEngine)
    3. compiled(inputs, workspace, engine)  // ZERO allocation
```

## Key Design Decisions

### 1. New Code Generator: WorkspaceCodeGenerator

The existing `CodeGenerator` targets `TensorOperations<T>` (autodiff-enabled, allocating). We need a SEPARATE code generator that targets `IEngine` directly (zero-allocation).

```csharp
public class WorkspaceCodeGenerator
{
    // Generates: Action<Tensor<T>[], TensorWorkspace<T>, IEngine>
    // NOT: Func<Tensor<T>[], Tensor<T>[]>
    public Action<Tensor<T>[], TensorWorkspace<T>, IEngine> Generate<T>(
        IRGraph graph,
        Dictionary<int, int> tensorToSlot)
    {
        // For each IR operation, emit the corresponding IEngine call
        // using workspace.Get(slot) for all intermediates
    }
}
```

### 2. Operation Mapping: IROp -> IEngine Method

Each IR operation maps to a specific IEngine method:

| IROp | IEngine Method | Allocation |
|------|---------------|------------|
| AddOp | TensorAddInto(dest, a, b) | Zero (dest = workspace slot) |
| Conv2DOp | Conv2DInto(dest, input, kernel, ...) | Zero |
| GroupNormOp | GroupNormInto(dest, input, ...) | Small (mean/var stats) |
| SwishOp | SwishInto(dest, input) **[MISSING]** | Zero |
| ReLUOp | ReLUInto(dest, input) | Zero |
| SigmoidOp | SigmoidInto(dest, input) | Zero |
| MatMulOp | MatMulInto(dest, a, b) **[MISSING]** | Zero |
| ConcatOp | ConcatInto(dest, tensors, axis) **[MISSING]** | Zero |
| ReshapeOp | View only (no data copy needed) | Zero |
| TransposeOp | TransposeInto(dest, input, axes) **[MISSING]** | Zero |

**Missing IEngine "Into" methods that MUST be added:**
- SwishInto / SwishInPlace (SiLU activation — used in every DiffusionResBlock)
- MishInto / MishInPlace
- GELUInto / GELUInPlace
- TanhInto / TanhInPlace
- MatMulInto (matrix multiply into pre-allocated output)
- ConcatInto (concatenate into pre-allocated output)
- TransposeInto (transpose into pre-allocated output)
- LeakyReLUInto / LeakyReLUInPlace

### 3. Fused Operations Mapping

For TRULY fused operations, we need single-pass kernels:

| Fused IROp | Implementation Strategy |
|-----------|----------------------|
| FusedGroupNormActivationOp | New IEngine method: GroupNormSwishInto(dest, input, gamma, beta, groups, eps) — single pass: normalize + SiLU |
| FusedConv2DBiasActivationOp | Existing: Engine.FusedConv2D(input, kernel, bias, ..., activation) |
| FusedGroupNormActivationConv2DOp | Two calls: GroupNormSwishInto(temp, input, ...) then Conv2DInto(dest, temp, kernel, ...) — temp is a workspace slot |
| FusedAddGroupNormOp | New IEngine method: AddGroupNormInto(dest, a, b, gamma, beta, groups, eps) — single pass: add + normalize |

**New IEngine methods needed for true fusion:**
- GroupNormSwishInto: GroupNorm + SiLU in single data pass
- GroupNormReLUInto: GroupNorm + ReLU in single data pass
- AddGroupNormInto: Add + GroupNorm in single data pass

### 4. Memory Plan Integration

The `MemoryPlanningPass` computes `tensorToSlot` mapping. The `WorkspaceCodeGenerator` uses this mapping:

```csharp
// For each operation in the IR graph:
int outputSlot = tensorToSlot[op.OutputId];
var outputTensor = workspace.Get(outputSlot);

// Emit: Engine.Conv2DInto(outputTensor, inputTensor, kernelTensor, ...)
```

The workspace is allocated ONCE at model construction time and reused for every forward pass.

### 5. Expression Tree Generation

The WorkspaceCodeGenerator builds expression trees that reference:
- `workspace.Get(slotId)` — workspace parameter
- `Engine.SomeIntoMethod(dest, src, ...)` — engine parameter
- `inputs[i]` — input array parameter

```csharp
// Generated expression tree (conceptual):
(inputs, workspace, engine) => {
    var slot0 = workspace.Get(0);
    engine.GroupNormInto(slot0, inputs[0], 32, gamma, beta, 1e-5, out _, out _);

    var slot1 = workspace.Get(1);
    engine.SwishInto(slot1, slot0);  // SiLU into separate slot (preserves slot0 for potential reuse)

    var slot2 = workspace.Get(2);
    engine.Conv2DInto(slot2, slot1, conv1_kernel, 1, 1, 1);

    engine.TensorBroadcastAddInPlace(slot2, conv1_bias_4d);

    // ... second half of ResBlock ...

    engine.TensorAddInPlace(slot4, skip_connection);  // residual add

    // slot4 is the output — copy to output array
    return slot4;
}
```

### 6. Integration Points

#### A. NeuralNetworkBase.CompileForward()
1. Export computation graph from all layers
2. Build IR graph via IRBuilder
3. Run optimization passes (fusion, constant folding, dead code elimination)
4. Run MemoryPlanningPass to get tensor-to-slot mapping
5. Create TensorWorkspace with slot shapes from memory plan
6. Generate compiled function via WorkspaceCodeGenerator
7. Store compiled function + workspace

#### B. UNetNoisePredictor.CompileForward()
Same as above but for the UNet-specific graph structure (encoder + middle + decoder with skip connections).

#### C. Predict/Forward path
```csharp
if (_compiledForward != null)
{
    _compiledForward(inputs, _workspace, Engine);
    return _workspace.Get(_outputSlot);
}
else
{
    // Interpreted fallback
    return InterpretedForward(inputs);
}
```

## Implementation Plan (Ordered by Priority)

### Phase 1: Missing IEngine "Into" Methods (AiDotNet.Tensors)
Add SwishInto/InPlace, GELUInto/InPlace, TanhInto/InPlace, MishInto/InPlace, MatMulInto, ConcatInto, TransposeInto to IEngine, CpuEngine, DirectGpuTensorEngine.

### Phase 2: WorkspaceCodeGenerator (AiDotNet)
New code generator that emits IEngine calls with workspace slots. Handles all current IR operations via their Into/InPlace equivalents.

### Phase 3: True Fused Kernels (AiDotNet.Tensors)
GroupNormSwishInto, AddGroupNormInto — single-pass kernels that eliminate intermediates.

### Phase 4: Integration
Wire CompileForward into NeuralNetworkBase and UNetNoisePredictor. Add IsCompiled flag and automatic fallback.

### Phase 5: Benchmarks
BenchmarkDotNet project comparing:
- Interpreted vs Compiled forward pass
- Our compiled vs PyTorch (via TorchSharp)
- Operation-level comparisons (Conv2D, GroupNorm, Attention)
- Memory usage comparisons

## Gaps in Current Code (What I Claimed to Do But Didn't)

1. **CodeGenerator fused ops**: Generate the SAME allocating calls as unfused — zero benefit
2. **CompileForward**: Chains ExportComputationGraph through layers but still targets TensorOperations — zero benefit
3. **UNet CompileForward**: Exports graph but doesn't handle skip connections, time embedding, or the encoder-decoder structure properly
4. **FusedGroupNormActivationConv2DOp code gen**: Calls GroupNorm then Swish then Conv2D separately — no fusion
5. **FusedAddGroupNormOp code gen**: Calls Add then GroupNorm separately — no fusion
