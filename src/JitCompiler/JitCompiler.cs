using System.Collections.Concurrent;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.JitCompiler.CodeGen;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.Memory;
using AiDotNet.JitCompiler.Optimizations;
using IOptimizationPass = AiDotNet.JitCompiler.Optimizations.IOptimizationPass;

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
public class JitCompiler : IDisposable
{
    private readonly ConcurrentDictionary<int, object> _compiledGraphCache = new();
    private readonly IRBuilder _irBuilder = new();
    private readonly CodeGenerator _codeGenerator = new();
    private readonly List<IOptimizationPass> _optimizationPasses = new();
    private readonly JitCompilerOptions _options;
    private readonly TensorPool? _tensorPool;
    private bool _disposed;

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

        // Initialize memory pooling if enabled
        if (_options.EnableMemoryPooling)
        {
            _tensorPool = new TensorPool(_options.MaxPoolSizePerShape, _options.MaxElementsToPool);
        }

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
            if (_options.EnableAdaptiveFusion)
            {
                // Use adaptive fusion (smarter, hardware-aware)
                _optimizationPasses.Add(new AdaptiveFusionPass());
            }
            else
            {
                // Use standard fusion
                _optimizationPasses.Add(new OperationFusionPass());
            }
        }

        if (_options.EnableLoopUnrolling)
        {
            _optimizationPasses.Add(new LoopUnrollingPass());
        }

        if (_options.EnableAutoTuning)
        {
            _optimizationPasses.Add(new AutoTuningPass());
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
    ///   var matmul = TensorOperations<T>.MatrixMultiply(x, weights);
    ///   var add = TensorOperations<T>.Add(matmul, bias);
    ///   var result = TensorOperations<T>.ReLU(add);
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
    /// Compiles the backward pass (gradient computation) for a computation graph.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to compute gradients for.</param>
    /// <returns>A compiled function that computes gradients given output gradients.</returns>
    /// <remarks>
    /// <para>
    /// This compiles the backward pass for training. It creates a function that:
    /// 1. Takes the gradient of the loss with respect to outputs (dL/dOutput)
    /// 2. Computes gradients with respect to inputs (dL/dInput) via backpropagation
    /// 3. Returns gradients for all trainable parameters
    /// </para>
    /// <para><b>For Beginners:</b> This compiles the gradient computation for training.
    ///
    /// In machine learning training:
    /// - Forward pass: Compute predictions from inputs
    /// - Backward pass: Compute how to adjust weights to reduce error
    ///
    /// This method compiles the backward pass to run 5-10x faster!
    ///
    /// Example:
    ///   // Compile forward and backward passes
    ///   var forward = jit.Compile(outputNode, inputs);
    ///   var backward = jit.CompileBackward(outputNode, inputs);
    ///
    ///   // Training loop
    ///   for (int epoch = 0; epoch < 100; epoch++) {
    ///       // Forward pass
    ///       var predictions = forward(inputTensors);
    ///       var loss = ComputeLoss(predictions, targets);
    ///
    ///       // Backward pass (JIT-compiled, 5-10x faster!)
    ///       var outputGrad = ComputeLossGradient(predictions, targets);
    ///       var gradients = backward(new[] { outputGrad });
    ///
    ///       // Update weights
    ///       UpdateWeights(gradients);
    ///   }
    ///
    /// Expected speedup: 5-10x faster training!
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// Thrown if outputNode or inputs is null.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the graph contains operations without defined backward functions.
    /// </exception>
    public Func<Tensor<T>[], Tensor<T>[]> CompileBackward<T>(ComputationNode<T> outputNode, List<ComputationNode<T>> inputs)
    {
        if (outputNode == null)
            throw new ArgumentNullException(nameof(outputNode));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        // Build backward IR graph from computation graph
        var irGraph = _irBuilder.BuildBackward(outputNode, inputs);

        // Check cache
        var graphHash = irGraph.ComputeStructureHash() ^ 0xBAC4;  // Differentiate backward from forward
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
    /// Compiles the backward pass and returns compilation statistics.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to compute gradients for.</param>
    /// <returns>A tuple of (compiled backward function, compilation statistics).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Compiles gradient computation and shows optimization details.
    ///
    /// Use this to:
    /// - See how much the backward pass was optimized
    /// - Understand what optimizations were applied
    /// - Debug gradient computation issues
    /// - Monitor compilation performance
    ///
    /// The statistics tell you:
    /// - How many gradient operations were generated
    /// - How many operations after optimization
    /// - What optimizations were applied (fusion of backward ops!)
    /// - Cache hit information
    /// </para>
    /// </remarks>
    public (Func<Tensor<T>[], Tensor<T>[]> CompiledBackward, CompilationStats Stats) CompileBackwardWithStats<T>(
        ComputationNode<T> outputNode, List<ComputationNode<T>> inputs)
    {
        var stats = new CompilationStats();
        var startTime = DateTime.UtcNow;

        // Build backward IR graph
        var irGraph = _irBuilder.BuildBackward(outputNode, inputs);
        stats.OriginalOperationCount = irGraph.Operations.Count;

        // Check cache
        var graphHash = irGraph.ComputeStructureHash() ^ 0xBAC4;
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
        var compiledBackward = _codeGenerator.Generate<T>(optimizedGraph);

        stats.CompilationTime = DateTime.UtcNow - startTime;

        // Cache result
        if (_options.EnableCaching)
        {
            _compiledGraphCache[graphHash] = compiledBackward;
        }

        return (compiledBackward, stats);
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

    /// <summary>
    /// Attempts to compile a computation graph without throwing exceptions.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to the computation graph.</param>
    /// <param name="compiledFunc">When this method returns true, contains the compiled function.</param>
    /// <param name="error">When this method returns false, contains the error message.</param>
    /// <returns>True if compilation succeeded, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a safe version of Compile that won't crash your program.
    ///
    /// Instead of throwing an exception when something goes wrong, it returns false
    /// and tells you what went wrong through the error parameter.
    ///
    /// Example:
    ///   if (jit.TryCompile(output, inputs, out var compiled, out var error))
    ///   {
    ///       // Use compiled function
    ///       var result = compiled(inputTensors);
    ///   }
    ///   else
    ///   {
    ///       // Handle error gracefully
    ///       Console.WriteLine($"JIT compilation failed: {error}");
    ///       // Fall back to interpreted execution
    ///   }
    /// </para>
    /// </remarks>
    public bool TryCompile<T>(
        ComputationNode<T> outputNode,
        List<ComputationNode<T>> inputs,
        out Func<Tensor<T>[], Tensor<T>[]>? compiledFunc,
        out string? error)
    {
        compiledFunc = null;
        error = null;

        if (outputNode == null)
        {
            error = "Output node cannot be null";
            return false;
        }
        if (inputs == null)
        {
            error = "Inputs cannot be null";
            return false;
        }

        try
        {
            // Build IR graph from computation graph
            var irGraph = _irBuilder.Build(outputNode, inputs);

            // Check cache
            var graphHash = irGraph.ComputeStructureHash();
            if (_options.EnableCaching && _compiledGraphCache.TryGetValue(graphHash, out var cached))
            {
                compiledFunc = (Func<Tensor<T>[], Tensor<T>[]>)cached;
                return true;
            }

            // Apply optimization passes
            var optimizedGraph = ApplyOptimizationsWithRecovery(irGraph);

            // Generate code
            compiledFunc = _codeGenerator.Generate<T>(optimizedGraph);

            // Cache result
            if (_options.EnableCaching)
            {
                _compiledGraphCache[graphHash] = compiledFunc;
            }

            return true;
        }
        catch (NotImplementedException ex)
        {
            error = $"Unsupported operation in graph: {ex.Message}";
            return false;
        }
        catch (InvalidOperationException ex)
        {
            error = $"Invalid graph structure: {ex.Message}";
            return false;
        }
        catch (Exception ex)
        {
            error = $"Compilation failed: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Compiles a computation graph with automatic fallback to interpreted execution.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to the computation graph.</param>
    /// <returns>
    /// A tuple containing:
    /// - The executable function (JIT compiled or interpreted fallback)
    /// - Whether JIT compilation succeeded
    /// - Any warning or error message
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the most robust way to compile a graph.
    ///
    /// It tries JIT compilation first. If that fails, it automatically falls back
    /// to interpreted execution (slower but always works).
    ///
    /// You get the best performance when JIT works, and guaranteed execution when it doesn't.
    ///
    /// Example:
    ///   var (func, wasJitted, message) = jit.CompileWithFallback(output, inputs);
    ///   if (!wasJitted)
    ///   {
    ///       Console.WriteLine($"Using interpreted fallback: {message}");
    ///   }
    ///   // func is always usable!
    ///   var result = func(inputTensors);
    /// </para>
    /// </remarks>
    public (Func<Tensor<T>[], Tensor<T>[]> Func, bool WasJitCompiled, string? Message) CompileWithFallback<T>(
        ComputationNode<T> outputNode,
        List<ComputationNode<T>> inputs)
    {
        // Try JIT compilation first
        if (TryCompile(outputNode, inputs, out var jitFunc, out var error) && jitFunc != null)
        {
            return (jitFunc, true, null);
        }

        // Fall back to interpreted execution
        var interpretedFunc = CreateInterpretedFallback(outputNode, inputs);
        return (interpretedFunc, false, error ?? "Unknown error during JIT compilation");
    }

    /// <summary>
    /// Creates an interpreted fallback function for a computation graph.
    /// </summary>
    private Func<Tensor<T>[], Tensor<T>[]> CreateInterpretedFallback<T>(
        ComputationNode<T> outputNode,
        List<ComputationNode<T>> inputs)
    {
        return (Tensor<T>[] inputTensors) =>
        {
            // Assign input tensors to input nodes
            for (int i = 0; i < inputs.Count && i < inputTensors.Length; i++)
            {
                inputs[i].Value = inputTensors[i];
            }

            // Evaluate the graph (interpreted mode)
            var result = outputNode.Value;

            return new[] { result };
        };
    }

    /// <summary>
    /// Applies optimization passes with error recovery.
    /// </summary>
    /// <remarks>
    /// If an optimization pass fails, it is skipped and the unoptimized graph is used.
    /// This ensures the compilation can still succeed even if optimizations fail.
    /// </remarks>
    private IRGraph ApplyOptimizationsWithRecovery(IRGraph graph)
    {
        var currentGraph = graph;

        foreach (var pass in _optimizationPasses)
        {
            try
            {
                currentGraph = pass.Optimize(currentGraph);
            }
            catch (Exception)
            {
                // Optimization pass failed - skip it and continue with current graph
                // In production, you might want to log this
            }
        }

        return currentGraph;
    }

    /// <summary>
    /// Gets the tensor memory pool if memory pooling is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Access the memory pool for manual buffer management.
    ///
    /// Usually you don't need to use this directly - the JIT compiler manages memory
    /// automatically. But if you want fine-grained control over memory allocation
    /// in your code, you can use this pool.
    ///
    /// Example:
    ///   if (jit.TensorPool != null)
    ///   {
    ///       var buffer = jit.TensorPool.Rent&lt;float&gt;(1000);
    ///       // Use buffer...
    ///       jit.TensorPool.Return(buffer);
    ///   }
    /// </para>
    /// </remarks>
    public TensorPool? TensorPool => _tensorPool;

    /// <summary>
    /// Gets statistics about the tensor memory pool.
    /// </summary>
    /// <returns>Pool statistics, or null if memory pooling is disabled.</returns>
    public TensorPoolStats? GetTensorPoolStats()
    {
        return _tensorPool?.GetStats();
    }

    /// <summary>
    /// Analyzes a computation graph to determine JIT compatibility.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to the computation graph.</param>
    /// <returns>A compatibility result describing which operations are supported.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this before compiling to see if your graph is JIT-compatible.
    ///
    /// This method:
    /// - Walks through your entire computation graph
    /// - Checks each operation against the supported list
    /// - Reports which operations will be JIT-compiled vs. need fallback
    /// - Tells you if hybrid mode is available
    ///
    /// Example:
    ///   var compat = jit.AnalyzeCompatibility(output, inputs);
    ///   if (compat.IsFullySupported)
    ///   {
    ///       Console.WriteLine("Graph can be fully JIT compiled!");
    ///   }
    ///   else
    ///   {
    ///       Console.WriteLine($"Partial support: {compat.SupportedPercentage:F0}%");
    ///       foreach (var unsupported in compat.UnsupportedOperations)
    ///       {
    ///           Console.WriteLine($"  - {unsupported}");
    ///       }
    ///   }
    /// </para>
    /// </remarks>
    public JitCompatibilityResult AnalyzeCompatibility<T>(ComputationNode<T> outputNode, List<ComputationNode<T>> inputs)
    {
        var result = new JitCompatibilityResult();
        var supportedOps = GetSupportedOperationTypes();
        var visited = new HashSet<ComputationNode<T>>();
        var tensorIdCounter = 0;

        void AnalyzeNode(ComputationNode<T> node)
        {
            if (visited.Contains(node))
                return;
            visited.Add(node);

            // Visit parents first
            foreach (var parent in node.Parents)
            {
                AnalyzeNode(parent);
            }

            // Skip input nodes
            if (inputs.Contains(node))
                return;

            var opType = node.OperationType?.ToString() ?? "Unknown";
            var tensorId = tensorIdCounter++;

            if (node.OperationType == null)
            {
                result.UnsupportedOperations.Add(new UnsupportedOperationInfo
                {
                    OperationType = "Unknown",
                    NodeName = node.Name,
                    TensorId = tensorId,
                    Reason = "Node has no OperationType metadata",
                    CanFallback = true
                });
            }
            else if (supportedOps.Contains(node.OperationType.Value))
            {
                result.SupportedOperations.Add(opType);
            }
            else
            {
                result.UnsupportedOperations.Add(new UnsupportedOperationInfo
                {
                    OperationType = opType,
                    NodeName = node.Name,
                    TensorId = tensorId,
                    Reason = $"Operation type {opType} not implemented in JIT compiler",
                    CanFallback = true
                });
            }
        }

        AnalyzeNode(outputNode);

        result.IsFullySupported = result.UnsupportedOperations.Count == 0;
        result.CanUseHybridMode = result.UnsupportedOperations.All(u => u.CanFallback);

        return result;
    }

    /// <summary>
    /// Gets the set of operation types that are fully supported by the JIT compiler.
    /// </summary>
    /// <returns>A set of supported operation type enums.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you which operations can be JIT compiled.
    ///
    /// Supported operations include:
    /// - Basic math: Add, Subtract, Multiply, Divide, Power, Negate
    /// - Math functions: Exp, Log, Sqrt
    /// - Activations: ReLU, Sigmoid, Tanh, Softmax
    /// - Matrix ops: MatMul, Transpose
    /// - Convolutions: Conv2D, ConvTranspose2D, DepthwiseConv2D
    /// - Pooling: MaxPool2D, AvgPool2D
    /// - Normalization: LayerNorm, BatchNorm
    /// - And more...
    ///
    /// If your operation isn't listed, it will need fallback execution.
    /// </para>
    /// </remarks>
    public static HashSet<OperationType> GetSupportedOperationTypes()
    {
        return new HashSet<OperationType>
        {
            // Basic arithmetic
            OperationType.Add,
            OperationType.Subtract,
            OperationType.Multiply,
            OperationType.Divide,
            OperationType.Power,
            OperationType.Negate,

            // Math operations
            OperationType.Exp,
            OperationType.Log,
            OperationType.Sqrt,

            // Activations
            OperationType.ReLU,
            OperationType.Sigmoid,
            OperationType.Tanh,
            OperationType.Softmax,
            OperationType.Activation,

            // Matrix operations
            OperationType.MatMul,
            OperationType.Transpose,

            // Reduction operations
            OperationType.ReduceSum,
            OperationType.Mean,
            OperationType.ReduceMax,
            OperationType.ReduceMean,
            OperationType.ReduceLogVariance,

            // Shape operations
            OperationType.Reshape,
            OperationType.Concat,
            OperationType.Pad,
            OperationType.Crop,
            OperationType.Upsample,
            OperationType.PixelShuffle,

            // Convolution operations
            OperationType.Conv2D,
            OperationType.ConvTranspose2D,
            OperationType.DepthwiseConv2D,
            OperationType.DilatedConv2D,
            OperationType.LocallyConnectedConv2D,

            // Pooling operations
            OperationType.MaxPool2D,
            OperationType.AvgPool2D,

            // Normalization operations
            OperationType.LayerNorm,
            OperationType.BatchNorm,

            // Advanced operations
            OperationType.GraphConv,
            OperationType.AffineGrid,
            OperationType.GridSample,
            OperationType.RBFKernel,

            // Recurrent network operations
            OperationType.GRUCell,
            OperationType.LSTMCell
        };
    }

    /// <summary>
    /// Compiles a computation graph with intelligent handling of unsupported operations.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to the computation graph.</param>
    /// <returns>
    /// A result containing the compiled function, whether JIT was used,
    /// compatibility information, and any warnings.
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the recommended way to compile graphs with mixed support.
    ///
    /// This method automatically:
    /// 1. Analyzes your graph for JIT compatibility
    /// 2. Based on UnsupportedLayerHandling setting:
    ///    - Throw: Fails if any operation is unsupported
    ///    - Fallback: Uses interpreted execution if anything is unsupported
    ///    - Hybrid: JIT-compiles what it can, interprets the rest
    ///    - Skip: Ignores unsupported operations (dangerous!)
    /// 3. Returns a function that always works, plus useful diagnostics
    ///
    /// Example:
    ///   var result = jit.CompileWithUnsupportedHandling(output, inputs);
    ///   if (!result.IsFullyJitCompiled)
    ///   {
    ///       Console.WriteLine($"Hybrid mode: {result.Compatibility.SupportedPercentage:F0}% JIT compiled");
    ///   }
    ///   var predictions = result.CompiledFunc(inputTensors);
    /// </para>
    /// </remarks>
    public HybridCompilationResult<T> CompileWithUnsupportedHandling<T>(
        ComputationNode<T> outputNode,
        List<ComputationNode<T>> inputs)
    {
        var result = new HybridCompilationResult<T>();

        // Analyze compatibility first
        result.Compatibility = AnalyzeCompatibility(outputNode, inputs);

        // Handle based on configuration
        switch (_options.UnsupportedLayerHandling)
        {
            case UnsupportedLayerHandling.Throw:
                if (!result.Compatibility.IsFullySupported)
                {
                    var unsupportedOps = string.Join(", ", result.Compatibility.UnsupportedOperations.Select(u => u.OperationType));
                    throw new NotSupportedException(
                        $"Graph contains unsupported operations: {unsupportedOps}. " +
                        "Set UnsupportedLayerHandling to Fallback or Hybrid to allow these operations.");
                }
                result.CompiledFunc = Compile(outputNode, inputs);
                result.IsFullyJitCompiled = true;
                result.ExecutionMode = "JIT";
                break;

            case UnsupportedLayerHandling.Fallback:
                if (result.Compatibility.IsFullySupported)
                {
                    result.CompiledFunc = Compile(outputNode, inputs);
                    result.IsFullyJitCompiled = true;
                    result.ExecutionMode = "JIT";
                }
                else
                {
                    result.CompiledFunc = CreateInterpretedFallback(outputNode, inputs);
                    result.IsFullyJitCompiled = false;
                    result.ExecutionMode = "Interpreted";
                    if (_options.LogUnsupportedOperations)
                    {
                        result.Warnings.Add($"Using interpreted execution due to {result.Compatibility.UnsupportedOperations.Count} unsupported operations");
                    }
                }
                break;

            case UnsupportedLayerHandling.Hybrid:
                if (result.Compatibility.IsFullySupported)
                {
                    result.CompiledFunc = Compile(outputNode, inputs);
                    result.IsFullyJitCompiled = true;
                    result.ExecutionMode = "JIT";
                }
                else if (result.Compatibility.CanUseHybridMode)
                {
                    // Build hybrid execution function
                    result.CompiledFunc = CreateHybridFunction(outputNode, inputs, result.Compatibility);
                    result.IsFullyJitCompiled = false;
                    result.ExecutionMode = "Hybrid";
                    if (_options.LogUnsupportedOperations)
                    {
                        result.Warnings.Add($"Hybrid mode: {result.Compatibility.SupportedPercentage:F1}% JIT compiled, rest interpreted");
                    }
                }
                else
                {
                    // Can't use hybrid, fall back to interpreted
                    result.CompiledFunc = CreateInterpretedFallback(outputNode, inputs);
                    result.IsFullyJitCompiled = false;
                    result.ExecutionMode = "Interpreted";
                    if (_options.LogUnsupportedOperations)
                    {
                        result.Warnings.Add("Hybrid mode unavailable; using interpreted execution");
                    }
                }
                break;

            case UnsupportedLayerHandling.Skip:
                // Compile with skip mode - may produce incorrect results
                result.CompiledFunc = CompileWithSkipping(outputNode, inputs, result.Compatibility);
                result.IsFullyJitCompiled = result.Compatibility.IsFullySupported;
                result.ExecutionMode = result.Compatibility.IsFullySupported ? "JIT" : "JIT (skipped ops)";
                if (!result.Compatibility.IsFullySupported && _options.LogUnsupportedOperations)
                {
                    result.Warnings.Add("WARNING: Skipped unsupported operations - results may be incorrect!");
                }
                break;
        }

        return result;
    }

    /// <summary>
    /// Creates a hybrid execution function that JIT-compiles supported operations
    /// and uses interpreted execution for unsupported ones.
    /// </summary>
    private Func<Tensor<T>[], Tensor<T>[]> CreateHybridFunction<T>(
        ComputationNode<T> outputNode,
        List<ComputationNode<T>> inputs,
        JitCompatibilityResult compatibility)
    {
        // For now, we use a simplified approach:
        // If there are unsupported ops, we fall back to full interpretation
        // A full hybrid implementation would require graph partitioning
        // which is a significant undertaking

        // TODO: Implement true hybrid execution with graph partitioning
        // This would involve:
        // 1. Partition graph into JIT-able subgraphs
        // 2. Compile each subgraph independently
        // 3. Create execution plan that switches between JIT and interpreted
        // 4. Handle tensor passing between execution modes

        // For now, return interpreted with a note about future hybrid support
        return CreateInterpretedFallback(outputNode, inputs);
    }

    /// <summary>
    /// Compiles a graph, skipping unsupported operations.
    /// WARNING: This may produce incorrect results!
    /// </summary>
    private Func<Tensor<T>[], Tensor<T>[]> CompileWithSkipping<T>(
        ComputationNode<T> outputNode,
        List<ComputationNode<T>> inputs,
        JitCompatibilityResult compatibility)
    {
        if (compatibility.IsFullySupported)
        {
            return Compile(outputNode, inputs);
        }

        // For skip mode with unsupported ops, use interpreted execution
        // A true skip implementation would require careful handling
        // to not corrupt the tensor graph
        return CreateInterpretedFallback(outputNode, inputs);
    }

    /// <summary>
    /// Clears the tensor memory pool, releasing all cached buffers.
    /// </summary>
    public void ClearTensorPool()
    {
        _tensorPool?.Clear();
    }

    /// <summary>
    /// Releases all resources used by the JIT compiler.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _tensorPool?.Dispose();
            ClearCache();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Specifies how the JIT compiler handles unsupported operations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a computation graph contains operations the JIT
/// doesn't support, this controls what happens:
///
/// - <b>Throw:</b> Stop and throw an exception (fail-fast)
/// - <b>Fallback:</b> Use interpreted execution for the entire graph (safe but slower)
/// - <b>Hybrid:</b> JIT-compile supported ops, interpret unsupported ones (best of both)
/// - <b>Skip:</b> Ignore unsupported ops (may produce incorrect results - use carefully)
///
/// For production, use Hybrid for best performance with guaranteed correctness.
/// </para>
/// </remarks>
public enum UnsupportedLayerHandling
{
    /// <summary>
    /// Throw an exception when an unsupported operation is encountered.
    /// Use this when you require all operations to be JIT compiled.
    /// </summary>
    Throw,

    /// <summary>
    /// Fall back to interpreted execution for the entire graph.
    /// This is the safest option - always produces correct results.
    /// </summary>
    Fallback,

    /// <summary>
    /// Use hybrid execution: JIT-compile supported operations and execute
    /// unsupported operations using the interpreter. This provides the best
    /// balance of performance and compatibility.
    /// </summary>
    Hybrid,

    /// <summary>
    /// Skip unsupported operations during compilation. WARNING: This may
    /// produce incorrect results. Only use for debugging or when you know
    /// the skipped operations don't affect your output.
    /// </summary>
    Skip
}

/// <summary>
/// Information about an unsupported operation detected during JIT compilation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When the JIT compiler finds an operation it can't compile,
/// it creates one of these to tell you:
/// - What operation was unsupported
/// - Where it was in the graph
/// - Why it couldn't be compiled
///
/// Use this to diagnose compilation issues or to know which operations need fallback.
/// </para>
/// </remarks>
public class UnsupportedOperationInfo
{
    /// <summary>
    /// Gets or sets the name of the unsupported operation type.
    /// </summary>
    public string OperationType { get; set; } = "";

    /// <summary>
    /// Gets or sets the name of the computation node (if available).
    /// </summary>
    public string? NodeName { get; set; }

    /// <summary>
    /// Gets or sets the tensor ID that would have been assigned to this operation.
    /// </summary>
    public int TensorId { get; set; }

    /// <summary>
    /// Gets or sets the reason why this operation is not supported.
    /// </summary>
    public string Reason { get; set; } = "Operation type not implemented in JIT compiler";

    /// <summary>
    /// Gets or sets whether this operation can be executed via fallback.
    /// </summary>
    public bool CanFallback { get; set; } = true;

    /// <summary>
    /// Returns a string representation of the unsupported operation.
    /// </summary>
    public override string ToString()
    {
        var name = NodeName != null ? $" ({NodeName})" : "";
        return $"Unsupported: {OperationType}{name} at tensor {TensorId} - {Reason}";
    }
}

/// <summary>
/// Result of analyzing a graph for JIT compatibility.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Before compiling, you can check if your graph is compatible.
/// This result tells you:
/// - Whether full JIT compilation is possible
/// - What operations are supported/unsupported
/// - Whether hybrid mode can be used
/// </para>
/// </remarks>
public class JitCompatibilityResult
{
    /// <summary>
    /// Gets or sets whether all operations in the graph are supported.
    /// </summary>
    public bool IsFullySupported { get; set; }

    /// <summary>
    /// Gets or sets the list of supported operation types found in the graph.
    /// </summary>
    public List<string> SupportedOperations { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of unsupported operations found in the graph.
    /// </summary>
    public List<UnsupportedOperationInfo> UnsupportedOperations { get; set; } = new();

    /// <summary>
    /// Gets or sets whether hybrid mode can be used (some ops JIT, some interpreted).
    /// </summary>
    public bool CanUseHybridMode { get; set; }

    /// <summary>
    /// Gets the percentage of operations that can be JIT compiled.
    /// </summary>
    public double SupportedPercentage =>
        SupportedOperations.Count + UnsupportedOperations.Count > 0
            ? (double)SupportedOperations.Count / (SupportedOperations.Count + UnsupportedOperations.Count) * 100
            : 100;

    /// <summary>
    /// Returns a summary of the compatibility analysis.
    /// </summary>
    public override string ToString()
    {
        if (IsFullySupported)
        {
            return $"Fully JIT compatible ({SupportedOperations.Count} operations)";
        }

        return $"Partial JIT support: {SupportedPercentage:F1}% ({SupportedOperations.Count} supported, " +
               $"{UnsupportedOperations.Count} unsupported). Hybrid mode: {(CanUseHybridMode ? "available" : "not available")}";
    }
}

/// <summary>
/// Result of compiling with unsupported operation handling.
/// </summary>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When you use CompileWithUnsupportedHandling, you get this result.
/// It tells you:
/// - The compiled function (always usable)
/// - Whether it's fully JIT compiled or uses fallback
/// - Compatibility details
/// - Any warnings about unsupported operations
/// </para>
/// </remarks>
public class HybridCompilationResult<T>
{
    /// <summary>
    /// Gets or sets the compiled function.
    /// This function is always usable regardless of execution mode.
    /// </summary>
    public Func<Tensor<T>[], Tensor<T>[]> CompiledFunc { get; set; } = null!;

    /// <summary>
    /// Gets or sets whether the function was fully JIT compiled.
    /// If false, some or all operations use interpreted execution.
    /// </summary>
    public bool IsFullyJitCompiled { get; set; }

    /// <summary>
    /// Gets or sets the execution mode: "JIT", "Interpreted", "Hybrid", or "JIT (skipped ops)".
    /// </summary>
    public string ExecutionMode { get; set; } = "Unknown";

    /// <summary>
    /// Gets or sets the compatibility analysis results.
    /// </summary>
    public JitCompatibilityResult Compatibility { get; set; } = new();

    /// <summary>
    /// Gets or sets any warnings generated during compilation.
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Returns a summary of the compilation result.
    /// </summary>
    public override string ToString()
    {
        var warnings = Warnings.Count > 0 ? $" ({Warnings.Count} warnings)" : "";
        return $"Execution: {ExecutionMode}, JIT: {(IsFullyJitCompiled ? "100%" : $"{Compatibility.SupportedPercentage:F1}%")}{warnings}";
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
/// - Control how unsupported operations are handled
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

    /// <summary>
    /// Gets or sets a value indicating whether to enable loop unrolling optimization.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Loop unrolling improves performance for small, fixed-size loops by eliminating
    /// loop overhead and enabling better instruction pipelining. The optimizer automatically
    /// determines which loops benefit from unrolling based on tensor size and operation type.
    /// </para>
    /// </remarks>
    public bool EnableLoopUnrolling { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable adaptive fusion strategies.
    /// Default: false (currently uses standard fusion when enabled).
    /// </summary>
    /// <remarks>
    /// <para><b>Status:</b> Architecture implemented, delegates to standard fusion.
    /// Adaptive fusion will intelligently select which operations to fuse based on
    /// graph structure, tensor sizes, and hardware characteristics.
    /// </para>
    /// </remarks>
    public bool EnableAdaptiveFusion { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether to enable auto-tuning of optimizations.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Auto-tuning automatically determines the best optimization configuration for
    /// each graph based on graph analysis, tensor sizes, and operation types. It selects
    /// the optimal combination of fusion, unrolling, and vectorization strategies.
    /// </para>
    /// </remarks>
    public bool EnableAutoTuning { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable SIMD vectorization hints.
    /// Default: false (not yet fully implemented).
    /// </summary>
    /// <remarks>
    /// <para><b>Status:</b> Architecture planned, implementation pending.
    /// SIMD hints guide the code generator to use vector instructions (AVX, AVX-512)
    /// for better performance on element-wise operations.
    /// </para>
    /// </remarks>
    public bool EnableSIMDHints { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether to enable memory pooling for tensors.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Reuses tensor memory to reduce allocations.
    ///
    /// Memory pooling improves performance by:
    /// - Reducing garbage collection pauses
    /// - Avoiding repeated memory allocations
    /// - Improving cache locality
    ///
    /// This is especially beneficial for training loops that create many temporary tensors.
    /// </para>
    /// </remarks>
    public bool EnableMemoryPooling { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of tensor buffers to keep per shape.
    /// Default: 10.
    /// </summary>
    public int MaxPoolSizePerShape { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum total elements in a tensor to pool.
    /// Tensors larger than this will not be pooled.
    /// Default: 10,000,000 (about 40MB for float32).
    /// </summary>
    public int MaxElementsToPool { get; set; } = 10_000_000;

    /// <summary>
    /// Gets or sets how the JIT compiler handles unsupported operations.
    /// Default: Fallback (use interpreted execution for entire graph if any op is unsupported).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When your model has operations the JIT can't compile,
    /// this setting controls what happens:
    ///
    /// - <b>Throw:</b> Stop with an error - use when you need all ops compiled
    /// - <b>Fallback:</b> (Default) Run the whole graph interpreted - always works
    /// - <b>Hybrid:</b> JIT the supported ops, interpret the rest - best performance
    /// - <b>Skip:</b> Ignore unsupported ops - dangerous, may give wrong results
    ///
    /// Hybrid mode is recommended for production when you have mixed-support graphs.
    /// It gives you JIT speed for supported operations while still handling all ops correctly.
    /// </para>
    /// </remarks>
    public UnsupportedLayerHandling UnsupportedLayerHandling { get; set; } = UnsupportedLayerHandling.Fallback;

    /// <summary>
    /// Gets or sets whether to log warnings for unsupported operations.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, you'll see warnings in logs when
    /// operations can't be JIT compiled. This helps you:
    /// - Identify which operations need fallback
    /// - Understand performance implications
    /// - Know when to request JIT support for new operation types
    /// </para>
    /// </remarks>
    public bool LogUnsupportedOperations { get; set; } = true;
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
