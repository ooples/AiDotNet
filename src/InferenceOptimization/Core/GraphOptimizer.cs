using System.Diagnostics;
using AiDotNet.InferenceOptimization.Passes;

namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Main optimizer engine that orchestrates all optimization passes.
/// Applies transformations to computation graphs to improve inference performance.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class GraphOptimizer<T> where T : struct
{
    private readonly OptimizationOptions _options;
    private readonly List<IOptimizationPass<T>> _passes;

    public GraphOptimizer(OptimizationOptions? options = null)
    {
        _options = options ?? OptimizationOptions.FromLevel(OptimizationLevel.Standard);
        _passes = new List<IOptimizationPass<T>>();

        InitializePasses();
    }

    /// <summary>
    /// Optimizes the given computation graph.
    /// </summary>
    public IOptimizationGraph<T> Optimize(IOptimizationGraph<T> graph)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        if (_options.Level == OptimizationLevel.None)
        {
            return graph;
        }

        var optimizedGraph = graph.Clone();
        var stopwatch = Stopwatch.StartNew();

        if (_options.PrintStatistics)
        {
            Console.WriteLine("=== Graph Optimization Started ===");
            Console.WriteLine($"Optimization Level: {_options.Level}");
            Console.WriteLine($"Initial Graph: {optimizedGraph.GetStatistics()}");
            Console.WriteLine();
        }

        int iteration = 0;
        bool graphChanged = true;

        while (graphChanged && iteration < _options.MaxIterations)
        {
            graphChanged = false;

            foreach (var pass in _passes)
            {
                if (!pass.CanApply(optimizedGraph))
                {
                    continue;
                }

                var passStopwatch = Stopwatch.StartNew();
                bool passModified = pass.Apply(optimizedGraph);
                passStopwatch.Stop();

                if (passModified)
                {
                    graphChanged = true;

                    if (_options.PrintStatistics)
                    {
                        Console.WriteLine($"[Iteration {iteration}] {pass.Name}: Modified graph in {passStopwatch.ElapsedMilliseconds}ms");
                    }

                    if (_options.ValidateAfterEachPass && !optimizedGraph.Validate())
                    {
                        throw new InvalidOperationException($"Graph validation failed after {pass.Name}");
                    }
                }
            }

            iteration++;
        }

        stopwatch.Stop();

        if (_options.PrintStatistics)
        {
            Console.WriteLine();
            Console.WriteLine($"Final Graph: {optimizedGraph.GetStatistics()}");
            Console.WriteLine($"Total Iterations: {iteration}");
            Console.WriteLine($"Total Time: {stopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine("=== Graph Optimization Completed ===");
        }

        return optimizedGraph;
    }

    /// <summary>
    /// Adds a custom optimization pass.
    /// </summary>
    public void AddPass(IOptimizationPass<T> pass)
    {
        if (pass == null)
            throw new ArgumentNullException(nameof(pass));

        _passes.Add(pass);
    }

    /// <summary>
    /// Removes all passes of a specific type.
    /// </summary>
    public void RemovePass(Type passType)
    {
        _passes.RemoveAll(p => p.GetType() == passType);
    }

    private void InitializePasses()
    {
        // Phase 1: Algebraic simplification (reduces graph complexity early)
        if (_options.EnableAlgebraicSimplification)
        {
            _passes.Add(new AlgebraicSimplificationPass<T>());
        }

        // Phase 2: Constant folding (evaluate constants early)
        if (_options.EnableConstantFolding)
        {
            _passes.Add(new ConstantFoldingPass<T>());
        }

        // Phase 3: Operator fusion (critical for performance)
        if (_options.EnableOperatorFusion)
        {
            // Order matters: try more specific fusions first
            _passes.Add(new ConvBatchNormReLUFusionPass<T>());
            _passes.Add(new ConvBatchNormFusionPass<T>());
            _passes.Add(new MatMulBiasActivationFusionPass<T>());
            _passes.Add(new MatMulBiasFusionPass<T>());
            _passes.Add(new MultiHeadAttentionFusionPass<T>());
            _passes.Add(new ElementwiseFusionPass<T>());
        }

        // Phase 4: Common subexpression elimination
        if (_options.EnableCSE)
        {
            _passes.Add(new CommonSubexpressionEliminationPass<T>());
        }

        // Phase 5: Strength reduction
        if (_options.EnableStrengthReduction)
        {
            _passes.Add(new StrengthReductionPass<T>());
        }

        // Phase 6: Layout optimization
        if (_options.EnableLayoutOptimization)
        {
            _passes.Add(new LayoutOptimizationPass<T>(_options.TargetLayout));
        }

        // Phase 7: Memory optimizations
        if (_options.EnableInPlaceOptimization)
        {
            _passes.Add(new InPlaceOptimizationPass<T>());
        }

        if (_options.EnableMemoryReuse)
        {
            _passes.Add(new MemoryReuseOptimizationPass<T>());
        }

        // Phase 8: Dead code elimination (should be last to clean up)
        if (_options.EnableDeadCodeElimination)
        {
            _passes.Add(new DeadCodeEliminationPass<T>());
        }
    }

    /// <summary>
    /// Gets the list of active optimization passes.
    /// </summary>
    public IReadOnlyList<IOptimizationPass<T>> GetPasses()
    {
        return _passes.AsReadOnly();
    }
}
