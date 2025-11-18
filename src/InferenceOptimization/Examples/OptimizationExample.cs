using AiDotNet.InferenceOptimization.Core;
using AiDotNet.Interfaces;

namespace AiDotNet.InferenceOptimization.Examples;

/// <summary>
/// Example usage of the inference optimization system.
/// </summary>
public class OptimizationExample
{
    /// <summary>
    /// Example 1: Basic optimization of a simple CNN
    /// </summary>
    public static void BasicCNNOptimization()
    {
        Console.WriteLine("=== Example 1: Basic CNN Optimization ===\n");

        // Create a simple CNN (pseudo-code, adapt to your model structure)
        var layers = new List<ILayer<double>>
        {
            // Convolutional layer + BatchNorm + ReLU (will be fused)
            // MaxPooling
            // Another Conv + BatchNorm + ReLU (will be fused)
            // Flatten
            // Dense + Bias + ReLU (will be fused)
            // Output Dense
        };

        // Build computation graph
        var graphBuilder = new GraphBuilder<double>();
        var graph = graphBuilder.BuildFromLayers(layers);

        Console.WriteLine($"Original Graph: {graph.GetStatistics()}\n");

        // Optimize with Standard level
        var options = OptimizationOptions.FromLevel(OptimizationLevel.Standard);
        options.PrintStatistics = true;

        var optimizer = new GraphOptimizer<double>(options);
        var optimizedGraph = optimizer.Optimize(graph);

        Console.WriteLine("\nOptimization complete!");
    }

    /// <summary>
    /// Example 2: Aggressive optimization for production deployment
    /// </summary>
    public static void ProductionOptimization()
    {
        Console.WriteLine("=== Example 2: Production Optimization ===\n");

        // Create your model layers
        var layers = new List<ILayer<double>>(); // Your layers here

        var graphBuilder = new GraphBuilder<double>();
        var graph = graphBuilder.BuildFromLayers(layers);

        // Use Aggressive optimization for production
        var options = new OptimizationOptions
        {
            Level = OptimizationLevel.Aggressive,
            EnableOperatorFusion = true,
            EnableMemoryReuse = true,
            EnableCSE = true,
            EnableInPlaceOptimization = true,
            TargetLayout = "NCHW", // Optimize for GPU
            PrintStatistics = true,
            ValidateAfterEachPass = true
        };

        var optimizer = new GraphOptimizer<double>(options);
        var optimizedGraph = optimizer.Optimize(graph);

        Console.WriteLine("\nProduction-ready optimized graph created!");
    }

    /// <summary>
    /// Example 3: Custom optimization pass
    /// </summary>
    public static void CustomPassExample()
    {
        Console.WriteLine("=== Example 3: Custom Optimization Pass ===\n");

        var graphBuilder = new GraphBuilder<double>();
        var layers = new List<ILayer<double>>(); // Your layers
        var graph = graphBuilder.BuildFromLayers(layers);

        // Create optimizer
        var optimizer = new GraphOptimizer<double>();

        // Add custom pass (implement your own IOptimizationPass<T>)
        // optimizer.AddPass(new MyCustomPass<double>());

        var optimizedGraph = optimizer.Optimize(graph);

        Console.WriteLine("Custom optimization applied!");
    }

    /// <summary>
    /// Example 4: Comparing different optimization levels
    /// </summary>
    public static void CompareOptimizationLevels()
    {
        Console.WriteLine("=== Example 4: Comparing Optimization Levels ===\n");

        var graphBuilder = new GraphBuilder<double>();
        var layers = new List<ILayer<double>>(); // Your layers
        var originalGraph = graphBuilder.BuildFromLayers(layers);

        var levels = new[]
        {
            OptimizationLevel.None,
            OptimizationLevel.Basic,
            OptimizationLevel.Standard,
            OptimizationLevel.Aggressive,
            OptimizationLevel.Maximum
        };

        foreach (var level in levels)
        {
            Console.WriteLine($"\n--- Testing {level} Level ---");

            var options = OptimizationOptions.FromLevel(level);
            options.PrintStatistics = true;

            var optimizer = new GraphOptimizer<double>(options);
            var optimizedGraph = optimizer.Optimize(originalGraph.Clone());

            Console.WriteLine($"Level {level} complete\n");
        }
    }

    /// <summary>
    /// Example 5: Transformer model optimization
    /// </summary>
    public static void TransformerOptimization()
    {
        Console.WriteLine("=== Example 5: Transformer Optimization ===\n");

        // Build transformer graph
        var graphBuilder = new GraphBuilder<double>();
        var layers = new List<ILayer<double>>
        {
            // Multi-head attention (will be fused)
            // Layer normalization
            // Feed-forward: Dense + Bias + GELU (will be fused)
            // Dense + Bias (will be fused)
            // Layer normalization
            // etc.
        };

        var graph = graphBuilder.BuildFromLayers(layers);

        // Optimize for transformer
        var options = new OptimizationOptions
        {
            Level = OptimizationLevel.Aggressive,
            EnableOperatorFusion = true,
            EnableMemoryReuse = true,
            PrintStatistics = true
        };

        var optimizer = new GraphOptimizer<double>(options);
        var optimizedGraph = optimizer.Optimize(graph);

        Console.WriteLine("\nTransformer optimized!");
        Console.WriteLine("Expected speedup: 2-3x");
    }

    /// <summary>
    /// Example 6: Memory-constrained optimization
    /// </summary>
    public static void MemoryConstrainedOptimization()
    {
        Console.WriteLine("=== Example 6: Memory-Constrained Optimization ===\n");

        var graphBuilder = new GraphBuilder<double>();
        var layers = new List<ILayer<double>>(); // Your layers
        var graph = graphBuilder.BuildFromLayers(layers);

        // Prioritize memory optimizations
        var options = new OptimizationOptions
        {
            Level = OptimizationLevel.Aggressive,
            EnableMemoryReuse = true,
            EnableInPlaceOptimization = true,
            EnableOperatorFusion = true, // Also reduces memory
            PrintStatistics = true
        };

        var optimizer = new GraphOptimizer<double>(options);
        var optimizedGraph = optimizer.Optimize(graph);

        Console.WriteLine("\nMemory-optimized graph created!");
        Console.WriteLine("Expected memory reduction: 30-50%");
    }

    /// <summary>
    /// Example 7: Inspect optimization passes
    /// </summary>
    public static void InspectPasses()
    {
        Console.WriteLine("=== Example 7: Inspect Optimization Passes ===\n");

        var optimizer = new GraphOptimizer<double>(
            OptimizationOptions.FromLevel(OptimizationLevel.Aggressive)
        );

        var passes = optimizer.GetPasses();

        Console.WriteLine($"Total passes: {passes.Count}\n");

        foreach (var pass in passes)
        {
            Console.WriteLine($"- {pass.Name} ({pass.PassType})");
        }
    }

    public static void Main(string[] args)
    {
        // Run all examples
        BasicCNNOptimization();
        Console.WriteLine("\n" + new string('=', 60) + "\n");

        ProductionOptimization();
        Console.WriteLine("\n" + new string('=', 60) + "\n");

        CustomPassExample();
        Console.WriteLine("\n" + new string('=', 60) + "\n");

        CompareOptimizationLevels();
        Console.WriteLine("\n" + new string('=', 60) + "\n");

        TransformerOptimization();
        Console.WriteLine("\n" + new string('=', 60) + "\n");

        MemoryConstrainedOptimization();
        Console.WriteLine("\n" + new string('=', 60) + "\n");

        InspectPasses();
    }
}
