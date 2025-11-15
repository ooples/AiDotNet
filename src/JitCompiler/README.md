# AiDotNet JIT Compiler

Just-In-Time compilation for AiDotNet computation graphs, providing 5-10x performance improvements.

## Features

- **Automatic Optimization**: Constant folding, dead code elimination, operation fusion
- **Expression Tree Compilation**: Converts IR to optimized .NET code
- **Intelligent Caching**: Avoids recompiling identical graph structures
- **Comprehensive API**: Simple to use, powerful when needed

## Quick Example

```csharp
using AiDotNet.JitCompiler;

// Create JIT compiler
var jit = new JitCompiler();

// Compile your computation graph
var compiled = jit.Compile(outputNode, inputNodes);

// Execute (5-10x faster!)
var result = compiled(inputTensors);
```

## Architecture

```
ComputationNode Graph
        ↓
    IRBuilder (converts to IR)
        ↓
    IR Graph (intermediate representation)
        ↓
    Optimization Passes
    - Constant Folding
    - Dead Code Elimination
    - Operation Fusion
        ↓
    Optimized IR Graph
        ↓
    CodeGenerator (expression trees)
        ↓
    Compiled Function (native code)
```

## Directory Structure

```
JitCompiler/
├── IR/                          # Intermediate Representation
│   ├── IROp.cs                  # Base IR operation class
│   ├── IRGraph.cs               # IR graph structure
│   ├── IRType.cs                # Type system for IR
│   ├── TensorShapeExtensions.cs # Shape utilities
│   └── Operations/              # IR operation types (43+ ops)
│       ├── ActivationOps.cs     # ReLU, Sigmoid, Tanh, Softmax
│       ├── BasicArithmeticOps.cs # Add, Subtract, Multiply, etc.
│       ├── MathOps.cs           # Exp, Log, Sqrt
│       ├── MatrixOps.cs         # MatMul, Transpose
│       └── AllOtherOps.cs       # Conv, Pool, Norm, etc.
│
├── Optimizations/               # Optimization passes
│   ├── ConstantFoldingPass.cs   # Evaluate constants at compile time
│   ├── DeadCodeEliminationPass.cs # Remove unused operations
│   └── OperationFusionPass.cs   # Fuse operations for efficiency
│
├── CodeGen/                     # Code generation
│   └── CodeGenerator.cs         # Expression tree code generation
│
├── IRBuilder.cs                 # Converts ComputationNode → IR
├── JitCompiler.cs              # Main JIT compiler API
└── README.md                    # This file
```

## Supported Operations

The JIT compiler supports 43+ operations:

**Basic Arithmetic**: Add, Subtract, Multiply, Divide, Power, Negate

**Math Functions**: Exp, Log, Sqrt

**Activations**: ReLU, Sigmoid, Tanh, Softmax, ApplyActivation

**Matrix Operations**: MatMul, Transpose

**Reductions**: Sum, Mean, ReduceMax, ReduceMean, ReduceLogVariance

**Shape Operations**: Reshape, Concat, Pad, Crop, Upsample, PixelShuffle

**Convolution**: Conv2D, ConvTranspose2D, DepthwiseConv2D, DilatedConv2D, LocallyConnectedConv2D

**Pooling**: MaxPool2D, AvgPool2D

**Normalization**: LayerNorm, BatchNorm

**Advanced**: GraphConv, AffineGrid, GridSample, RBFKernel

## Optimization Passes

### 1. Constant Folding
Evaluates expressions with constant inputs at compile time:
```
t2 = Add(2, 3); t3 = Mul(t2, x)  →  t2 = 5; t3 = Mul(5, x)
```

### 2. Dead Code Elimination
Removes operations whose results are never used:
```
t2 = Add(a, b); t3 = Mul(a, b); Output: t2  →  t2 = Add(a, b); Output: t2
```

### 3. Operation Fusion
Combines multiple operations into fused operations:
```
t2 = MatMul(x, w); t3 = Add(t2, b); t4 = ReLU(t3)  →  t4 = LinearReLU(x, w, b)
```

## Usage

See [JIT Compiler Usage Guide](../../docs/JIT-Compiler-Usage-Guide.md) for detailed documentation.

### Basic Usage

```csharp
var jit = new JitCompiler();
var compiled = jit.Compile(graph, inputs);
var output = compiled(inputTensors);
```

### With Statistics

```csharp
var (compiled, stats) = jit.CompileWithStats(graph, inputs);
Console.WriteLine(stats);  // See optimization results
```

### Custom Options

```csharp
var options = new JitCompilerOptions
{
    EnableConstantFolding = true,
    EnableDeadCodeElimination = true,
    EnableOperationFusion = true,
    EnableCaching = true
};
var jit = new JitCompiler(options);
```

## Performance

Expected speedups for typical workloads:

| Graph Type | Speedup |
|-----------|---------|
| Small (3-5 ops) | 3-5x |
| Medium (20-50 ops) | 5-8x |
| Large (50-100 ops) | 8-12x |

Speedup comes from:
- Eliminating graph interpretation overhead
- Operation fusion reducing memory traffic
- .NET JIT optimizations (inlining, SIMD)
- Dead code elimination

## Implementation Status

✅ **Complete**:
- IR infrastructure (IROp, IRGraph, 43+ operation types)
- IRBuilder (ComputationNode → IR conversion)
- Constant folding optimization
- Dead code elimination optimization
- Operation fusion optimization
- Expression tree code generation
- JIT compiler API
- Caching system
- Comprehensive documentation

🚧 **Future Work**:
- Backward pass (gradient) compilation
- GPU code generation
- More fusion patterns
- Loop unrolling and vectorization

## Testing

```bash
# Run JIT compiler tests
dotnet test tests/JitCompiler.Tests/

# Run benchmarks
dotnet run --project benchmarks/JitCompiler.Benchmarks/
```

## Contributing

When adding new operations:
1. Add IR operation class in `IR/Operations/`
2. Add code generation in `CodeGen/CodeGenerator.cs`
3. Update fusion patterns in `Optimizations/OperationFusionPass.cs` if applicable
4. Add tests

## License

Same as AiDotNet main project.
