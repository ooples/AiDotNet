# JIT Compiler Examples

This directory contains practical examples demonstrating how to use the AiDotNet JIT compiler.

## Examples Overview

### BasicUsageExample.cs

Contains 5 complete examples showing different aspects of JIT compilation:

1. **Simple Element-wise Operation**
   - Shows basic JIT compilation of a single operation
   - Demonstrates compilation stats
   - Executes compiled function

2. **Linear Layer Example**
   - Demonstrates fusion of MatMul + Add + ReLU
   - Shows optimization statistics
   - 3 operations → 1 fused operation

3. **Performance Comparison**
   - Benchmarks JIT compiled execution
   - Measures throughput and latency
   - Demonstrates real performance gains

4. **Caching Demonstration**
   - Shows cache hit/miss behavior
   - Demonstrates compilation time savings
   - Displays cache statistics

5. **Custom Compiler Options**
   - Shows how to configure optimization passes
   - Compares default vs custom configurations
   - Demonstrates selective optimization

## Running the Examples

### Option 1: From Code

```csharp
using AiDotNet.Examples.JitCompiler;

// Run all examples
BasicUsageExample.RunAllExamples();

// Or run individual examples
BasicUsageExample.SimpleElementwiseOperation();
BasicUsageExample.LinearLayerExample();
BasicUsageExample.PerformanceComparisonExample();
BasicUsageExample.CachingExample();
BasicUsageExample.CustomOptionsExample();
```

### Option 2: Create Console App

Create a simple console application:

```csharp
using AiDotNet.Examples.JitCompiler;

class Program
{
    static void Main(string[] args)
    {
        BasicUsageExample.RunAllExamples();
    }
}
```

### Option 3: Interactive (C# Interactive / LINQPad)

```csharp
#load "BasicUsageExample.cs"

using AiDotNet.Examples.JitCompiler;

BasicUsageExample.SimpleElementwiseOperation();
```

## Expected Output

### Example 1: Simple Element-wise Operation
```
=== Example 1: Simple Element-wise Operation ===

Compilation Stats:
  Original operations: 1
  Optimized operations: 1
  Compilation time: 12.34ms

Input: 1, 2, 3, 4, 5, 6, 7, 8, 9
Output (ReLU): 1, 2, 3, 4, 5, 6, 7, 8, 9
```

### Example 2: Linear Layer
```
=== Example 2: Linear Layer (MatMul + Add + ReLU) ===

Compilation Stats:
  Original operations: 3
  Optimized operations: 1
  Operations eliminated: 2 (66.7%)
  Optimizations: Constant Folding, Dead Code Elimination, Operation Fusion
  Compilation time: 18.56ms

Input: 1, 2, 3
Output: 2.3, 3.1, 3.9, 4.7
```

### Example 3: Performance Comparison
```
=== Example 3: Performance Comparison ===

Graph compiled in 15.23ms
Optimizations applied: Constant Folding, Dead Code Elimination, Operation Fusion

JIT Compiled Execution:
  1000 iterations in 45.67ms
  Average: 0.0457ms per iteration
  Throughput: 21882 operations/second
```

### Example 4: Caching
```
=== Example 4: Caching Demonstration ===

First compilation:
  Cache hit: False
  Compilation time: 12.45ms

Second compilation (same structure):
  Cache hit: True
  Compilation time: 0.00ms

Third compilation (different structure):
  Cache hit: False
  Compilation time: 11.23ms

Cache statistics:
  Cached graphs: 2
  Estimated memory: 2.00 KB
```

### Example 5: Custom Options
```
=== Example 5: Custom Compiler Options ===

With default options:
  Optimizations: Constant Folding, Dead Code Elimination, Operation Fusion

With custom options (fusion disabled):
  Optimizations: Constant Folding, Dead Code Elimination
```

## Learning Path

1. **Start with Example 1** - Understand basic compilation workflow
2. **Move to Example 2** - See real optimization in action
3. **Study Example 3** - Understand performance benefits
4. **Explore Example 4** - Learn about caching behavior
5. **Experiment with Example 5** - Customize compiler settings

## Tips and Best Practices

### Setting Operation Metadata

For JIT compilation to work, ComputationNodes must have `OperationType` set:

```csharp
var node = new ComputationNode<float>(tensor, parents: inputs)
{
    OperationType = "Add",  // Required for JIT!
    Name = "my_addition"    // Optional, for debugging
};
```

### When to Use JIT

**Best for:**
- Inference (forward pass only)
- Repeated execution of same graph structure
- Large models with many operations
- Production deployments

**Less beneficial for:**
- Training (backward pass not yet supported)
- Graphs that change structure frequently
- Very small operations (compilation overhead)

### Performance Tips

1. **Compile once, execute many times**
   ```csharp
   var compiled = jit.Compile(graph, inputs);
   for (int i = 0; i < 1000; i++) {
       var result = compiled(batchData[i]);  // Fast!
   }
   ```

2. **Let caching work for you**
   - Same graph structure → cache hit (instant)
   - Different data → same compiled function works

3. **Enable all optimizations** (default)
   - Fusion can provide 2-5x speedup alone
   - DCE removes overhead
   - Constant folding reduces runtime work

4. **Monitor compilation stats**
   ```csharp
   var (compiled, stats) = jit.CompileWithStats(graph, inputs);
   if (stats.OptimizationPercentage > 50%) {
       Console.WriteLine("Great optimizations!");
   }
   ```

## Common Issues

### "Node does not have OperationType metadata"

**Problem:** ComputationNode missing `OperationType` property.

**Solution:** Set it when creating nodes:
```csharp
node.OperationType = "ReLU";
```

### Slow first execution

**Problem:** First call includes compilation time.

**Solution:** This is normal! Compile during initialization:
```csharp
// During setup
var compiled = jit.Compile(graph, inputs);

// In hot path (fast!)
var result = compiled(data);
```

### Cache using too much memory

**Problem:** Too many compiled graphs cached.

**Solution:** Monitor and clear cache:
```csharp
var stats = jit.GetCacheStats();
if (stats.EstimatedMemoryBytes > threshold) {
    jit.ClearCache();
}
```

## Next Steps

- Read the [JIT Compiler Usage Guide](../../docs/JIT-Compiler-Usage-Guide.md)
- Explore the [Architecture README](../../src/JitCompiler/README.md)
- Run the performance benchmarks
- Integrate into your own models

## Feedback

Found an issue or have a question? Please file an issue on GitHub!
