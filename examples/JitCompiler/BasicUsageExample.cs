using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.JitCompiler;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace AiDotNet.Examples.JitCompiler;

/// <summary>
/// Basic examples demonstrating JIT compiler usage.
/// </summary>
public class BasicUsageExample
{
    /// <summary>
    /// Example 1: Simple element-wise operation
    /// </summary>
    public static void SimpleElementwiseOperation()
    {
        Console.WriteLine("=== Example 1: Simple Element-wise Operation ===\n");

        // Create input tensors
        var inputData = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < inputData.Length; i++)
        {
            inputData[i] = i + 1;  // [1, 2, 3, 4, 5, 6, 7, 8, 9]
        }

        // Build computation graph
        var input = new ComputationNode<float>(inputData)
        {
            OperationType = OperationType.Input,
            Name = "input"
        };

        // result = ReLU(input)
        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 3, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU,
            Name = "relu_output"
        };

        // Create JIT compiler and compile
        var jit = new global::AiDotNet.JitCompiler.JitCompiler();
        var (compiled, stats) = jit.CompileWithStats(result, new List<ComputationNode<float>> { input });

        Console.WriteLine($"Compilation Stats:");
        Console.WriteLine($"  Original operations: {stats.OriginalOperationCount}");
        Console.WriteLine($"  Optimized operations: {stats.OptimizedOperationCount}");
        Console.WriteLine($"  Compilation time: {stats.CompilationTime.TotalMilliseconds:F2}ms\n");

        // Execute compiled function
        var output = compiled(new[] { inputData });

        Console.WriteLine("Input: " + string.Join(", ", GetTensorValues(inputData)));
        Console.WriteLine("Output (ReLU): " + string.Join(", ", GetTensorValues(output[0])));
        Console.WriteLine();
    }

    /// <summary>
    /// Example 2: Linear layer (MatMul + Add)
    /// </summary>
    public static void LinearLayerExample()
    {
        Console.WriteLine("=== Example 2: Linear Layer (MatMul + Add + ReLU) ===\n");

        // Create inputs
        var inputData = new Tensor<float>(new[] { 1, 3 });
        inputData[0] = 1.0f; inputData[1] = 2.0f; inputData[2] = 3.0f;

        var weightsData = new Tensor<float>(new[] { 3, 4 });
        for (int i = 0; i < weightsData.Length; i++)
        {
            weightsData[i] = 0.1f * (i + 1);
        }

        var biasData = new Tensor<float>(new[] { 1, 4 });
        for (int i = 0; i < biasData.Length; i++)
        {
            biasData[i] = 0.5f;
        }

        // Build computation graph: output = ReLU(input @ weights + bias)
        var input = new ComputationNode<float>(inputData) { OperationType = OperationType.Input };
        var weights = new ComputationNode<float>(weightsData) { OperationType = OperationType.Input };
        var bias = new ComputationNode<float>(biasData) { OperationType = OperationType.Input };

        var matmul = new ComputationNode<float>(
            new Tensor<float>(new[] { 1, 4 }),
            parents: new List<ComputationNode<float>> { input, weights })
        {
            OperationType = OperationType.MatMul
        };

        var add = new ComputationNode<float>(
            new Tensor<float>(new[] { 1, 4 }),
            parents: new List<ComputationNode<float>> { matmul, bias })
        {
            OperationType = OperationType.Add
        };

        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 1, 4 }),
            parents: new List<ComputationNode<float>> { add })
        {
            OperationType = OperationType.ReLU
        };

        // Compile
        var jit = new global::AiDotNet.JitCompiler.JitCompiler();
        var (compiled, stats) = jit.CompileWithStats(relu, new List<ComputationNode<float>> { input, weights, bias });

        Console.WriteLine($"Compilation Stats:");
        Console.WriteLine($"  Original operations: {stats.OriginalOperationCount}");
        Console.WriteLine($"  Optimized operations: {stats.OptimizedOperationCount}");
        Console.WriteLine($"  Operations eliminated: {stats.OperationsEliminated} ({stats.OptimizationPercentage:F1}%)");
        Console.WriteLine($"  Optimizations: {string.Join(", ", stats.OptimizationsApplied)}");
        Console.WriteLine($"  Compilation time: {stats.CompilationTime.TotalMilliseconds:F2}ms\n");

        // Execute
        var output = compiled(new[] { inputData, weightsData, biasData });

        Console.WriteLine("Input: " + string.Join(", ", GetTensorValues(inputData)));
        Console.WriteLine("Output: " + string.Join(", ", GetTensorValues(output[0])));
        Console.WriteLine();
    }

    /// <summary>
    /// Example 3: JIT compilation performance benchmark
    /// </summary>
    public static void PerformanceComparisonExample()
    {
        Console.WriteLine("=== Example 3: JIT Performance Benchmark ===\n");

        // Create larger tensors for meaningful benchmark
        var inputData = new Tensor<float>(new[] { 100, 100 });
        for (int i = 0; i < inputData.Length; i++)
        {
            inputData[i] = (float)Math.Sin(i * 0.01);
        }

        // Build computation graph: exp(relu(input))
        var input = new ComputationNode<float>(inputData) { OperationType = OperationType.Input };

        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 100, 100 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU
        };

        var exp = new ComputationNode<float>(
            new Tensor<float>(new[] { 100, 100 }),
            parents: new List<ComputationNode<float>> { relu })
        {
            OperationType = OperationType.Exp
        };

        // Compile
        var jit = new global::AiDotNet.JitCompiler.JitCompiler();
        var (compiled, stats) = jit.CompileWithStats(exp, new List<ComputationNode<float>> { input });

        Console.WriteLine($"Graph compiled in {stats.CompilationTime.TotalMilliseconds:F2}ms");
        Console.WriteLine($"Optimizations applied: {string.Join(", ", stats.OptimizationsApplied)}\n");

        // Warm-up
        for (int i = 0; i < 10; i++)
        {
            compiled(new[] { inputData });
        }

        // Benchmark
        const int iterations = 1000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            compiled(new[] { inputData });
        }
        sw.Stop();

        double avgTimeMs = sw.Elapsed.TotalMilliseconds / iterations;
        Console.WriteLine($"JIT Compiled Execution:");
        Console.WriteLine($"  {iterations} iterations in {sw.Elapsed.TotalMilliseconds:F2}ms");
        Console.WriteLine($"  Average: {avgTimeMs:F4}ms per iteration");
        Console.WriteLine($"  Throughput: {1000.0 / avgTimeMs:F0} operations/second\n");
    }

    /// <summary>
    /// Example 4: Caching demonstration
    /// </summary>
    public static void CachingExample()
    {
        Console.WriteLine("=== Example 4: Caching Demonstration ===\n");

        var jit = new global::AiDotNet.JitCompiler.JitCompiler();

        // First compilation
        var input1 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 })) { OperationType = OperationType.Input };
        var relu1 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input1 })
        {
            OperationType = OperationType.ReLU
        };

        var (_, stats1) = jit.CompileWithStats(relu1, new List<ComputationNode<float>> { input1 });
        Console.WriteLine($"First compilation:");
        Console.WriteLine($"  Cache hit: {stats1.CacheHit}");
        Console.WriteLine($"  Compilation time: {stats1.CompilationTime.TotalMilliseconds:F2}ms\n");

        // Second compilation with same structure (should hit cache)
        var input2 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 })) { OperationType = OperationType.Input };
        var relu2 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input2 })
        {
            OperationType = OperationType.ReLU
        };

        var (_, stats2) = jit.CompileWithStats(relu2, new List<ComputationNode<float>> { input2 });
        Console.WriteLine($"Second compilation (same structure):");
        Console.WriteLine($"  Cache hit: {stats2.CacheHit}");
        Console.WriteLine($"  Compilation time: {stats2.CompilationTime.TotalMilliseconds:F2}ms\n");

        // Different structure (won't hit cache)
        var sigmoid2 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input2 })
        {
            OperationType = OperationType.Sigmoid
        };

        var (_, stats3) = jit.CompileWithStats(sigmoid2, new List<ComputationNode<float>> { input2 });
        Console.WriteLine($"Third compilation (different structure):");
        Console.WriteLine($"  Cache hit: {stats3.CacheHit}");
        Console.WriteLine($"  Compilation time: {stats3.CompilationTime.TotalMilliseconds:F2}ms\n");

        // Cache stats
        var cacheStats = jit.GetCacheStats();
        Console.WriteLine($"Cache statistics:");
        Console.WriteLine($"  Cached graphs: {cacheStats.CachedGraphCount}");
        Console.WriteLine($"  Estimated memory: {cacheStats.EstimatedMemoryBytes / 1024.0:F2} KB\n");
    }

    /// <summary>
    /// Example 5: Custom compiler options
    /// </summary>
    public static void CustomOptionsExample()
    {
        Console.WriteLine("=== Example 5: Custom Compiler Options ===\n");

        // Default options (all optimizations enabled)
        var jitDefault = new global::AiDotNet.JitCompiler.JitCompiler();

        // Custom options (selective optimizations)
        var customOptions = new JitCompilerOptions
        {
            EnableConstantFolding = true,
            EnableDeadCodeElimination = true,
            EnableOperationFusion = false,  // Disable fusion
            EnableCaching = true
        };
        var jitCustom = new global::AiDotNet.JitCompiler.JitCompiler(customOptions);

        // Build a graph
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 })) { OperationType = OperationType.Input };
        var exp = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Exp
        };

        // Compile with default options
        var (_, statsDefault) = jitDefault.CompileWithStats(exp, new List<ComputationNode<float>> { input });
        Console.WriteLine($"With default options:");
        Console.WriteLine($"  Optimizations: {string.Join(", ", statsDefault.OptimizationsApplied)}\n");

        // Compile with custom options
        var (_, statsCustom) = jitCustom.CompileWithStats(exp, new List<ComputationNode<float>> { input });
        Console.WriteLine($"With custom options (fusion disabled):");
        Console.WriteLine($"  Optimizations: {string.Join(", ", statsCustom.OptimizationsApplied)}\n");
    }

    /// <summary>
    /// Helper to get tensor values as array
    /// </summary>
    private static float[] GetTensorValues(Tensor<float> tensor)
    {
        var values = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
        {
            values[i] = tensor[i];
        }
        return values;
    }

    /// <summary>
    /// Run all examples
    /// </summary>
    public static void RunAllExamples()
    {
        try
        {
            SimpleElementwiseOperation();
            LinearLayerExample();
            PerformanceComparisonExample();
            CachingExample();
            CustomOptionsExample();

            Console.WriteLine("=== All Examples Completed Successfully! ===");
        }
        catch (Exception ex)
        {
            // Rethrow critical exceptions that should not be caught
            if (ex is OutOfMemoryException || ex is StackOverflowException || ex is System.Threading.ThreadAbortException)
                throw;

            Console.WriteLine($"Error running examples: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
