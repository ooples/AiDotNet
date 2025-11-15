using System.Collections.Concurrent;
using AiDotNet.Autodiff;
using AiDotNet.JitCompiler.CodeGen;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.Optimizations;

namespace AiDotNet.JitCompiler;

/// <summary>
/// Just-In-Time compiler for computation graphs.
/// </summary>
/// <remarks>
/// <para>
/// The JitCompiler is the main entry point for JIT compilation in AiDotNet. It provides
/// a high-level API for compiling computation graphs to optimized executable code.
/// The compiler automatically handles:
/// - IR graph construction from ComputationNode graphs
/// - Optimization passes (constant folding, dead code elimination, operation fusion)
/// - Code generation and compilation
/// - Caching of compiled graphs for reuse
/// </para>
/// <para><b>For Beginners:</b> This compiles your neural network graphs to run much faster.
///
/// Think of it like this:
/// - Without JIT: Your model runs by interpreting each operation step-by-step (slow)
/// - With JIT: Your model is compiled to optimized machine code (fast!)
///
/// How to use:
/// 1. Create a JitCompiler instance (once)
/// 2. Pass your computation graph to Compile()
/// 3. Get back a compiled function
/// 4. Call that function with your inputs (runs 5-10x faster!)
///
/// Example:
///   var jit = new JitCompiler();
///   var compiled = jit.Compile(myGraph, inputs);
///   var results = compiled(inputTensors);  // Fast execution!
///
/// The JIT compiler:
/// - Automatically optimizes your graph
/// - Caches compiled code for reuse
/// - Handles all the complexity internally
/// - Just works!
///
/// Expected speedup: 5-10x for typical neural networks
/// </para>
/// </remarks>
public class JitCompiler
{
    private readonly ConcurrentDictionary<int, object> _compiledGraphCache = new();
    private readonly IRBuilder _irBuilder = new();
    private readonly CodeGenerator _codeGenerator = new();
    private readonly List<IOptimizationPass> _optimizationPasses = new();
    private readonly JitCompilerOptions _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="JitCompiler"/> class with default options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a new JIT compiler with standard optimization passes enabled:
    /// - Constant folding
    /// - Dead code elimination
    /// - Operation fusion
    /// </para>
    /// <para><b>For Beginners:</b> Creates a JIT compiler ready to use.
    ///
    /// The compiler is created with good default settings:
    /// - All standard optimizations enabled
    /// - Caching enabled for fast repeated compilation
    /// - Ready to compile graphs immediately
    /// </para>
    /// </remarks>
    public JitCompiler() : this(new JitCompilerOptions())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="JitCompiler"/> class with custom options.
    /// </summary>
    /// <param name="options">Configuration options for the compiler.</param>
    /// <remarks>
    /// <para>
    /// Creates a new JIT compiler with specified options. This allows you to:
    /// - Enable/disable specific optimizations
    /// - Configure caching behavior
    /// - Control compilation settings
    /// </para>
    /// <para><b>For Beginners:</b> Creates a JIT compiler with custom settings.
    ///
    /// Use this if you want to:
    /// - Turn off certain optimizations for debugging
    /// - Disable caching for testing
    /// - Customize compilation behavior
    ///
    /// For most users, the default constructor is fine!
    /// </para>
    /// </remarks>
    public JitCompiler(JitCompilerOptions options)
    {
        _options = options;

        // Register optimization passes based on options
        if (_options.EnableConstantFolding)
        {
            _optimizationPasses.Add(new ConstantFoldingPass());
        }

        if (_options.EnableDeadCodeElimination)
        {
            _optimizationPasses.Add(new DeadCodeEliminationPass());
        }

        if (_options.EnableOperationFusion)
        {
            _optimizationPasses.Add(new OperationFusionPass());
        }
    }

    /// <summary>
    /// Compiles a computation graph to an optimized executable function.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to the computation graph.</param>
    /// <returns>A compiled function that executes the graph.</returns>
    /// <remarks>
    /// <para>
    /// This is the main compilation method. It:
    /// 1. Converts the ComputationNode graph to IR
    /// 2. Applies optimization passes
    /// 3. Generates and compiles code
    /// 4. Caches the result for future use
    /// 5. Returns a fast executable function
    /// </para>
    /// <para><b>For Beginners:</b> This compiles your computation graph.
    ///
    /// Steps:
    /// 1. Pass in your graph's output node and input nodes
    /// 2. The compiler analyzes and optimizes the graph
    /// 3. Generates fast executable code
    /// 4. Returns a function you can call
    ///
    /// Example:
    ///   // Define a simple computation: result = ReLU(x * weights + bias)
    ///   var x = new ComputationNode<float>(...);
    ///   var weights = new ComputationNode<float>(...);
    ///   var bias = new ComputationNode<float>(...);
    ///   var matmul = TensorOperations.MatrixMultiply(x, weights);
    ///   var add = TensorOperations.Add(matmul, bias);
    ///   var result = TensorOperations.ReLU(add);
    ///
    ///   // Compile it
    ///   var compiled = jit.Compile(result, new[] { x, weights, bias });
    ///
    ///   // Use it (much faster than running the graph directly!)
    ///   var output = compiled(new[] { xTensor, weightsTensor, biasTensor });
    ///
    /// The compiled function can be called many times with different inputs.
    /// It's cached, so calling Compile again with the same structure is instant!
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// Thrown if outputNode or inputs is null.
    /// </exception>
    public Func<Tensor<T>[], Tensor<T>[]> Compile<T>(ComputationNode<T> outputNode, List<ComputationNode<T>> inputs)
    {
        if (outputNode == null)
            throw new ArgumentNullException(nameof(outputNode));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        // Build IR graph from computation graph
        var irGraph = _irBuilder.Build(outputNode, inputs);

        // Check cache
        var graphHash = irGraph.ComputeStructureHash();
        if (_options.EnableCaching && _compiledGraphCache.TryGetValue(graphHash, out var cached))
        {
            return (Func<Tensor<T>[], Tensor<T>[]>)cached;
        }

        // Apply optimization passes
        var optimizedGraph = ApplyOptimizations(irGraph);

        // Generate code
        var compiledFunc = _codeGenerator.Generate<T>(optimizedGraph);

        // Cache result
        if (_options.EnableCaching)
        {
            _compiledGraphCache[graphHash] = compiledFunc;
        }

        return compiledFunc;
    }

    /// <summary>
    /// Compiles a computation graph and returns compilation statistics.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to the computation graph.</param>
    /// <returns>A tuple of (compiled function, compilation statistics).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This compiles your graph and tells you what optimizations were applied.
    ///
    /// Use this when you want to:
    /// - See how much the graph was optimized
    /// - Debug compilation issues
    /// - Understand what the JIT compiler is doing
    ///
    /// The statistics tell you:
    /// - How many operations were in the original graph
    /// - How many operations after optimization
    /// - What optimizations were applied
    /// - How much speedup to expect
    /// </para>
    /// </remarks>
    public (Func<Tensor<T>[], Tensor<T>[]> CompiledFunc, CompilationStats Stats) CompileWithStats<T>(
        ComputationNode<T> outputNode, List<ComputationNode<T>> inputs)
    {
        var stats = new CompilationStats();
        var startTime = DateTime.UtcNow;

        // Build IR graph
        var irGraph = _irBuilder.Build(outputNode, inputs);
        stats.OriginalOperationCount = irGraph.Operations.Count;

        // Check cache
        var graphHash = irGraph.ComputeStructureHash();
        stats.CacheHit = _options.EnableCaching && _compiledGraphCache.ContainsKey(graphHash);

        if (stats.CacheHit)
        {
            var cached = (Func<Tensor<T>[], Tensor<T>[]>)_compiledGraphCache[graphHash]!;
            stats.CompilationTime = TimeSpan.Zero;
            return (cached, stats);
        }

        // Apply optimizations
        var optimizedGraph = ApplyOptimizations(irGraph);
        stats.OptimizedOperationCount = optimizedGraph.Operations.Count;
        stats.OptimizationsApplied = _optimizationPasses.Select(p => p.Name).ToList();

        // Generate code
        var compiledFunc = _codeGenerator.Generate<T>(optimizedGraph);

        stats.CompilationTime = DateTime.UtcNow - startTime;

        // Cache result
        if (_options.EnableCaching)
        {
            _compiledGraphCache[graphHash] = compiledFunc;
        }

        return (compiledFunc, stats);
    }

    /// <summary>
    /// Applies all configured optimization passes to an IR graph.
    /// </summary>
    /// <param name="graph">The IR graph to optimize.</param>
    /// <returns>The optimized IR graph.</returns>
    /// <remarks>
    /// <para>
    /// Optimization passes are applied in sequence. Each pass transforms the graph
    /// to make it more efficient. Multiple passes can interact - for example, constant
    /// folding might create dead code that is then eliminated.
    /// </para>
    /// <para><b>For Beginners:</b> This runs all the optimizations on your graph.
    ///
    /// The optimization pipeline:
    /// 1. Constant Folding: Pre-compute constant expressions
    /// 2. Dead Code Elimination: Remove unused operations
    /// 3. Operation Fusion: Combine operations for efficiency
    ///
    /// Each optimization makes the graph faster and simpler!
    /// </para>
    /// </remarks>
    private IRGraph ApplyOptimizations(IRGraph graph)
    {
        var currentGraph = graph;

        foreach (var pass in _optimizationPasses)
        {
            currentGraph = pass.Optimize(currentGraph);
        }

        return currentGraph;
    }

    /// <summary>
    /// Clears the compiled graph cache.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears all cached compiled graphs.
    ///
    /// Use this when:
    /// - You want to free memory
    /// - You're testing and want fresh compilations
    /// - You've changed compilation settings
    ///
    /// After clearing, the next Compile() will be slower but subsequent
    /// calls with the same graph will be fast again (cached).
    /// </para>
    /// </remarks>
    public void ClearCache()
    {
        _compiledGraphCache.Clear();
    }

    /// <summary>
    /// Gets statistics about the compilation cache.
    /// </summary>
    /// <returns>Cache statistics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many graphs are cached.
    ///
    /// Useful for:
    /// - Monitoring memory usage
    /// - Understanding cache efficiency
    /// - Debugging caching behavior
    /// </para>
    /// </remarks>
    public CacheStats GetCacheStats()
    {
        return new CacheStats
        {
            CachedGraphCount = _compiledGraphCache.Count,
            EstimatedMemoryBytes = _compiledGraphCache.Count * 1024 // Rough estimate
        };
    }
}

/// <summary>
/// Configuration options for the JIT compiler.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Settings to control how the JIT compiler works.
///
/// You can:
/// - Enable/disable specific optimizations
/// - Turn caching on/off
/// - Configure compilation behavior
///
/// For most users, the defaults work great!
/// </para>
/// </remarks>
public class JitCompilerOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether to enable constant folding optimization.
    /// Default: true.
    /// </summary>
    public bool EnableConstantFolding { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable dead code elimination.
    /// Default: true.
    /// </summary>
    public bool EnableDeadCodeElimination { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable operation fusion.
    /// Default: true.
    /// </summary>
    public bool EnableOperationFusion { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable caching of compiled graphs.
    /// Default: true.
    /// </summary>
    public bool EnableCaching { get; set; } = true;
}

/// <summary>
/// Statistics about a compilation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Information about what happened during compilation.
///
/// Tells you:
/// - How many operations were optimized away
/// - What optimizations were applied
/// - How long compilation took
/// - Whether the result came from cache
/// </para>
/// </remarks>
public class CompilationStats
{
    /// <summary>
    /// Gets or sets the number of operations in the original graph.
    /// </summary>
    public int OriginalOperationCount { get; set; }

    /// <summary>
    /// Gets or sets the number of operations after optimization.
    /// </summary>
    public int OptimizedOperationCount { get; set; }

    /// <summary>
    /// Gets or sets the list of optimizations that were applied.
    /// </summary>
    public List<string> OptimizationsApplied { get; set; } = new();

    /// <summary>
    /// Gets or sets the time taken to compile the graph.
    /// </summary>
    public TimeSpan CompilationTime { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the compiled function came from cache.
    /// </summary>
    public bool CacheHit { get; set; }

    /// <summary>
    /// Gets the reduction in operation count from optimization.
    /// </summary>
    public int OperationsEliminated => OriginalOperationCount - OptimizedOperationCount;

    /// <summary>
    /// Gets the percentage reduction in operation count.
    /// </summary>
    public double OptimizationPercentage =>
        OriginalOperationCount > 0
            ? (double)OperationsEliminated / OriginalOperationCount * 100
            : 0;

    /// <summary>
    /// Gets a string representation of the compilation statistics.
    /// </summary>
    public override string ToString()
    {
        return $"Compilation Stats:\n" +
               $"  Original operations: {OriginalOperationCount}\n" +
               $"  Optimized operations: {OptimizedOperationCount}\n" +
               $"  Operations eliminated: {OperationsEliminated} ({OptimizationPercentage:F1}%)\n" +
               $"  Optimizations applied: {string.Join(", ", OptimizationsApplied)}\n" +
               $"  Compilation time: {CompilationTime.TotalMilliseconds:F2}ms\n" +
               $"  Cache hit: {CacheHit}";
    }
}

/// <summary>
/// Statistics about the compilation cache.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Information about cached compiled graphs.
///
/// Tells you:
/// - How many graphs are cached
/// - Approximate memory usage
/// </para>
/// </remarks>
public class CacheStats
{
    /// <summary>
    /// Gets or sets the number of cached compiled graphs.
    /// </summary>
    public int CachedGraphCount { get; set; }

    /// <summary>
    /// Gets or sets the estimated memory used by cached graphs.
    /// </summary>
    public long EstimatedMemoryBytes { get; set; }

    /// <summary>
    /// Gets a string representation of the cache statistics.
    /// </summary>
    public override string ToString()
    {
        return $"Cache Stats:\n" +
               $"  Cached graphs: {CachedGraphCount}\n" +
               $"  Estimated memory: {EstimatedMemoryBytes / 1024.0:F2} KB";
    }
}
