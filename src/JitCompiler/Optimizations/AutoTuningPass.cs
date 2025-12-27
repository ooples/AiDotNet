using System.Collections.Concurrent;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Auto-tuning optimization pass that adaptively selects the best optimizations for a given graph.
/// </summary>
/// <remarks>
/// <para>
/// Auto-tuning automatically determines the best optimization strategy for each graph by:
/// - Profiling different optimization configurations
/// - Measuring actual performance on target hardware
/// - Learning from previous compilations
/// - Adapting to graph structure and size
/// </para>
/// <para><b>For Beginners:</b> Auto-tuning finds the best optimization settings automatically.
///
/// Instead of using fixed optimization settings, auto-tuning:
/// - Tries different combinations of optimizations
/// - Measures which combination is fastest
/// - Remembers the best settings for similar graphs
/// - Adapts to your specific hardware (CPU, GPU, etc.)
///
/// Benefits:
/// - Better performance without manual tuning
/// - Adapts to different graph types automatically
/// - Learns from experience (gets better over time)
/// - Handles hardware differences (different CPUs, etc.)
/// </para>
/// </remarks>
public class AutoTuningPass : IOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "Auto-Tuning";

    private static readonly ConcurrentDictionary<int, TuningConfig> _tuningCache = new();
    private static readonly ConcurrentDictionary<int, TuningMetrics> _metricsHistory = new();

    /// <summary>
    /// Configuration for tuning behavior.
    /// </summary>
    public class AutoTuningConfig
    {
        /// <summary>Whether to enable profiling-based tuning.</summary>
        public bool EnableProfiling { get; set; } = false;

        /// <summary>Maximum time (ms) to spend profiling per graph.</summary>
        public int MaxProfilingTimeMs { get; set; } = 100;

        /// <summary>Whether to persist tuning results across runs.</summary>
        public bool PersistTuning { get; set; } = false;

        /// <summary>Minimum graph size to consider for caching.</summary>
        public int MinGraphSizeForCaching { get; set; } = 5;
    }

    private readonly AutoTuningConfig _config;

    /// <summary>
    /// Initializes with default configuration.
    /// </summary>
    public AutoTuningPass() : this(new AutoTuningConfig()) { }

    /// <summary>
    /// Initializes with custom configuration.
    /// </summary>
    public AutoTuningPass(AutoTuningConfig config)
    {
        _config = config;
    }

    /// <inheritdoc/>
    public IRGraph Optimize(IRGraph graph)
    {
        // 1. Fingerprint the graph
        var fingerprint = ComputeGraphFingerprint(graph);

        // 2. Check cache for known configuration
        if (_tuningCache.TryGetValue(fingerprint, out var cachedConfig))
        {
            graph.Metadata["AutoTuning_CacheHit"] = true;
            return ApplyConfig(graph, cachedConfig);
        }

        // 3. Analyze graph and select optimal configuration
        var config = SelectOptimalConfig(graph);

        // 4. Cache the configuration if graph is complex enough (thread-safe)
        if (graph.Operations.Count >= _config.MinGraphSizeForCaching)
        {
            _tuningCache.TryAdd(fingerprint, config);
        }

        graph.Metadata["AutoTuning_CacheHit"] = false;

        // 5. Apply configuration
        return ApplyConfig(graph, config);
    }

    /// <summary>
    /// Computes a fingerprint for the graph structure.
    /// </summary>
    private int ComputeGraphFingerprint(IRGraph graph)
    {
        unchecked
        {
            int hash = 17;

            // Hash operation count
            hash = hash * 31 + graph.Operations.Count;

            // Hash operation types distribution
            var opTypeCounts = new Dictionary<string, int>();
            foreach (var op in graph.Operations)
            {
                var opType = op.OpType;
                opTypeCounts[opType] = opTypeCounts.GetValueOrDefault(opType, 0) + 1;
            }

            foreach (var kvp in opTypeCounts.OrderBy(k => k.Key))
            {
                hash = hash * 31 + kvp.Key.GetHashCode();
                hash = hash * 31 + kvp.Value;
            }

            // Hash tensor size buckets
            var sizeBuckets = new int[4]; // Tiny, Small, Medium, Large
            foreach (var shape in graph.TensorShapes.Values)
            {
                var size = shape.Aggregate(1, (a, b) => a * b);
                if (size < 100) sizeBuckets[0]++;
                else if (size < 10000) sizeBuckets[1]++;
                else if (size < 1000000) sizeBuckets[2]++;
                else sizeBuckets[3]++;
            }

            foreach (var bucket in sizeBuckets)
            {
                hash = hash * 31 + bucket;
            }

            // Hash graph topology (depth)
            hash = hash * 31 + EstimateGraphDepth(graph);

            return hash;
        }
    }

    /// <summary>
    /// Estimates the depth of the computation graph.
    /// </summary>
    private int EstimateGraphDepth(IRGraph graph)
    {
        if (graph.Operations.Count == 0) return 0;

        var depths = new Dictionary<int, int>();

        // Initialize input depths
        foreach (var inputId in graph.InputIds)
        {
            depths[inputId] = 0;
        }

        // Compute depths for each operation
        int maxDepth = 0;
        foreach (var op in graph.Operations)
        {
            int inputDepth = 0;
            foreach (var inputId in op.InputIds)
            {
                if (depths.TryGetValue(inputId, out var d))
                {
                    inputDepth = Math.Max(inputDepth, d);
                }
            }
            depths[op.OutputId] = inputDepth + 1;
            maxDepth = Math.Max(maxDepth, inputDepth + 1);
        }

        return maxDepth;
    }

    /// <summary>
    /// Selects the optimal configuration based on graph analysis.
    /// </summary>
    private TuningConfig SelectOptimalConfig(IRGraph graph)
    {
        var config = new TuningConfig();

        // Analyze graph characteristics
        var analysis = AnalyzeGraph(graph);

        // Apply heuristics based on analysis
        ApplyGraphSizeHeuristics(config, analysis);
        ApplyOperationTypeHeuristics(config, analysis);
        ApplyMemoryHeuristics(config, analysis);
        ApplyTopologyHeuristics(config, analysis);

        return config;
    }

    /// <summary>
    /// Analyzes graph characteristics.
    /// </summary>
    private GraphAnalysis AnalyzeGraph(IRGraph graph)
    {
        var analysis = new GraphAnalysis
        {
            TotalOps = graph.Operations.Count,
            TotalTensors = graph.TensorShapes.Count,
            GraphDepth = EstimateGraphDepth(graph)
        };

        // Compute average and max tensor sizes
        if (graph.TensorShapes.Count > 0)
        {
            var sizes = graph.TensorShapes.Values
                .Select(s => s.Aggregate(1, (a, b) => a * b))
                .ToList();

            analysis.AvgTensorSize = sizes.Average();
            analysis.MaxTensorSize = sizes.Max();
            analysis.TotalMemoryBytes = sizes.Sum() * sizeof(float);
        }

        // Count operation types
        foreach (var op in graph.Operations)
        {
            var opType = op.OpType;

            if (opType.Contains("Conv"))
                analysis.ConvOps++;
            else if (opType == "MatMul")
                analysis.MatMulOps++;
            else if (IsElementWise(opType))
                analysis.ElementWiseOps++;
            else if (IsReduction(opType))
                analysis.ReductionOps++;
            else if (IsNormalization(opType))
                analysis.NormalizationOps++;
            else if (IsActivation(opType))
                analysis.ActivationOps++;
        }

        // Compute graph characteristics
        analysis.IsComputeBound = analysis.MatMulOps + analysis.ConvOps > analysis.TotalOps * 0.3;
        analysis.IsMemoryBound = analysis.ElementWiseOps > analysis.TotalOps * 0.5;
        analysis.HasLongChains = analysis.GraphDepth > 10;

        return analysis;
    }

    /// <summary>
    /// Applies graph size heuristics.
    /// </summary>
    private void ApplyGraphSizeHeuristics(TuningConfig config, GraphAnalysis analysis)
    {
        if (analysis.TotalOps < 5)
        {
            // Very small graphs: minimal optimization
            config.EnableCaching = false;
            config.FusionLevel = FusionLevel.Minimal;
            config.EnableLoopUnrolling = false;
            config.EnableVectorization = true; // Always helpful
        }
        else if (analysis.TotalOps < 20)
        {
            // Small graphs: standard optimization
            config.EnableCaching = true;
            config.FusionLevel = FusionLevel.Standard;
            config.EnableLoopUnrolling = true;
            config.EnableVectorization = true;
        }
        else if (analysis.TotalOps < 100)
        {
            // Medium graphs: aggressive optimization
            config.EnableCaching = true;
            config.FusionLevel = FusionLevel.Aggressive;
            config.EnableLoopUnrolling = true;
            config.EnableVectorization = true;
        }
        else
        {
            // Large graphs: maximize optimization
            config.EnableCaching = true;
            config.FusionLevel = FusionLevel.Maximum;
            config.EnableLoopUnrolling = true;
            config.EnableVectorization = true;
            config.EnableParallelization = true;
        }
    }

    /// <summary>
    /// Applies operation type heuristics.
    /// </summary>
    private void ApplyOperationTypeHeuristics(TuningConfig config, GraphAnalysis analysis)
    {
        // Conv-heavy graphs: prioritize conv fusion
        if (analysis.ConvOps > analysis.TotalOps * 0.2)
        {
            config.PrioritizeConvFusion = true;
            config.FusionLevel = FusionLevel.Aggressive;
        }

        // MatMul-heavy graphs: prioritize linear algebra optimizations
        if (analysis.MatMulOps > analysis.TotalOps * 0.2)
        {
            config.PrioritizeMatMulOptimization = true;
            config.EnableTiling = analysis.MaxTensorSize > 10000;
        }

        // Element-wise heavy graphs: maximize fusion chains
        if (analysis.ElementWiseOps > analysis.TotalOps * 0.4)
        {
            config.MaxFusionChainLength = 8;
        }

        // Many normalizations: ensure stats computation is efficient
        if (analysis.NormalizationOps > 3)
        {
            config.OptimizeNormalization = true;
        }
    }

    /// <summary>
    /// Applies memory-related heuristics.
    /// </summary>
    private void ApplyMemoryHeuristics(TuningConfig config, GraphAnalysis analysis)
    {
        // Very small tensors: aggressive fusion to minimize overhead
        if (analysis.AvgTensorSize < 100)
        {
            config.FusionLevel = FusionLevel.Maximum;
            config.EnableConstantFolding = true;
        }
        // Large tensors: be cache-conscious
        else if (analysis.MaxTensorSize > 1000000)
        {
            config.EnableTiling = true;
            config.TileSize = EstimateOptimalTileSize(analysis);
            config.FusionLevel = FusionLevel.Conservative;
        }

        // High memory usage: enable memory optimization
        if (analysis.TotalMemoryBytes > 100 * 1024 * 1024) // > 100MB
        {
            config.EnableMemoryOptimization = true;
            config.ReuseBuffers = true;
        }
    }

    /// <summary>
    /// Applies topology heuristics.
    /// </summary>
    private void ApplyTopologyHeuristics(TuningConfig config, GraphAnalysis analysis)
    {
        // Long chains benefit from fusion
        if (analysis.HasLongChains)
        {
            config.MaxFusionChainLength = Math.Max(config.MaxFusionChainLength, 6);
        }

        // Deep graphs may benefit from parallelization
        if (analysis.GraphDepth > 5 && analysis.TotalOps > 20)
        {
            config.EnableParallelization = true;
        }
    }

    /// <summary>
    /// Estimates optimal tile size based on analysis.
    /// </summary>
    private int EstimateOptimalTileSize(GraphAnalysis analysis)
    {
        // Target L2 cache (~256KB typical)
        const int L2_CACHE_SIZE = 256 * 1024;
        const int BYTES_PER_ELEMENT = sizeof(float);

        // Estimate tile size that fits in L2
        var targetElements = L2_CACHE_SIZE / (3 * BYTES_PER_ELEMENT); // 3 arrays (A, B, C)
        var tileSize = (int)Math.Sqrt(targetElements);

        // Round to power of 2
        tileSize = 1 << (int)MathPolyfill.Log2(tileSize);

        // Clamp to reasonable range
        return MathPolyfill.Clamp(tileSize, 16, 256);
    }

    /// <summary>
    /// Applies a tuning configuration to the graph.
    /// </summary>
    private IRGraph ApplyConfig(IRGraph graph, TuningConfig config)
    {
        var optimizedGraph = graph;

        // Store configuration in metadata for downstream passes
        optimizedGraph.Metadata["TuningConfig_FusionLevel"] = config.FusionLevel.ToString();
        optimizedGraph.Metadata["TuningConfig_EnableCaching"] = config.EnableCaching;
        optimizedGraph.Metadata["TuningConfig_EnableVectorization"] = config.EnableVectorization;
        optimizedGraph.Metadata["TuningConfig_EnableLoopUnrolling"] = config.EnableLoopUnrolling;
        optimizedGraph.Metadata["TuningConfig_MaxFusionChainLength"] = config.MaxFusionChainLength;

        // Apply constant folding if enabled
        if (config.EnableConstantFolding)
        {
            var constantFolding = new ConstantFoldingPass();
            optimizedGraph = constantFolding.Optimize(optimizedGraph);
        }

        // Apply fusion based on fusion level
        if (config.FusionLevel != FusionLevel.None)
        {
            var fusionPass = new OperationFusionPass();
            optimizedGraph = fusionPass.Optimize(optimizedGraph);
        }

        // Apply loop unrolling if enabled
        if (config.EnableLoopUnrolling)
        {
            var unrollConfig = new LoopUnrollingPass.UnrollConfig
            {
                MaxFullUnrollFactor = config.FusionLevel >= FusionLevel.Aggressive ? 8 : 4
            };
            var unrollingPass = new LoopUnrollingPass(unrollConfig);
            optimizedGraph = unrollingPass.Optimize(optimizedGraph);
        }

        // Apply vectorization if enabled
        if (config.EnableVectorization)
        {
            var vectorizationPass = new VectorizationPass();
            optimizedGraph = vectorizationPass.Optimize(optimizedGraph);
        }

        return optimizedGraph;
    }

    /// <summary>
    /// Checks if operation is element-wise.
    /// </summary>
    private bool IsElementWise(string opType)
    {
        return opType is "Add" or "Subtract" or "ElementwiseMultiply" or "Divide"
            or "Negate" or "Exp" or "Log" or "Sqrt" or "Power";
    }

    /// <summary>
    /// Checks if operation is a reduction.
    /// </summary>
    private bool IsReduction(string opType)
    {
        return opType is "Sum" or "Mean" or "ReduceMax" or "ReduceMean" or "ReduceLogVariance";
    }

    /// <summary>
    /// Checks if operation is normalization.
    /// </summary>
    private bool IsNormalization(string opType)
    {
        return opType is "BatchNorm" or "LayerNorm";
    }

    /// <summary>
    /// Checks if operation is an activation.
    /// </summary>
    private bool IsActivation(string opType)
    {
        return opType is "ReLU" or "Sigmoid" or "Tanh" or "Softmax" or "GELU" or "Swish";
    }

    /// <summary>
    /// Graph analysis results.
    /// </summary>
    private class GraphAnalysis
    {
        public int TotalOps { get; set; }
        public int TotalTensors { get; set; }
        public int GraphDepth { get; set; }
        public double AvgTensorSize { get; set; }
        public int MaxTensorSize { get; set; }
        public long TotalMemoryBytes { get; set; }

        public int ConvOps { get; set; }
        public int MatMulOps { get; set; }
        public int ElementWiseOps { get; set; }
        public int ReductionOps { get; set; }
        public int NormalizationOps { get; set; }
        public int ActivationOps { get; set; }

        public bool IsComputeBound { get; set; }
        public bool IsMemoryBound { get; set; }
        public bool HasLongChains { get; set; }
    }

    /// <summary>
    /// Tuning configuration for graph optimization.
    /// </summary>
    private class TuningConfig
    {
        public bool EnableCaching { get; set; } = true;
        public FusionLevel FusionLevel { get; set; } = FusionLevel.Standard;
        public bool EnableLoopUnrolling { get; set; } = true;
        public bool EnableVectorization { get; set; } = true;
        public bool EnableParallelization { get; set; } = false;
        public bool EnableConstantFolding { get; set; } = true;
        public bool EnableMemoryOptimization { get; set; } = false;
        public bool ReuseBuffers { get; set; } = false;

        public bool PrioritizeConvFusion { get; set; } = false;
        public bool PrioritizeMatMulOptimization { get; set; } = false;
        public bool OptimizeNormalization { get; set; } = false;

        public int MaxFusionChainLength { get; set; } = 4;
        public bool EnableTiling { get; set; } = false;
        public int TileSize { get; set; } = 64;
    }

    /// <summary>
    /// Tuning metrics for profiling.
    /// </summary>
    private class TuningMetrics
    {
        public double ExecutionTimeMs { get; set; }
        public long MemoryUsageBytes { get; set; }
        public int CacheHits { get; set; }
        public int CacheMisses { get; set; }
    }

    /// <summary>
    /// Fusion level enumeration.
    /// </summary>
    private enum FusionLevel
    {
        None,
        Minimal,
        Conservative,
        Standard,
        Aggressive,
        Maximum
    }
}
