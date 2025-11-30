using System.Linq;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Adaptive fusion pass that intelligently fuses operations based on graph structure and hardware characteristics.
/// </summary>
/// <remarks>
/// <para>
/// Adaptive fusion improves upon static fusion by:
/// - Analyzing graph structure to find optimal fusion opportunities
/// - Considering hardware constraints (register pressure, cache size)
/// - Avoiding fusions that would hurt performance
/// - Dynamically adjusting fusion strategy based on tensor sizes
/// </para>
/// <para><b>For Beginners:</b> Adaptive fusion combines operations smarter.
///
/// Regular fusion: Always fuse operations when possible
/// Adaptive fusion: Fuse operations only when it helps performance
///
/// Why not always fuse?
/// - Fusing too much can increase register pressure (run out of fast memory)
/// - Large fused operations may not fit in cache
/// - Some fusion patterns are slower than separate operations
///
/// Adaptive fusion considers:
/// - Tensor sizes: Large tensors may benefit from separate passes (better cache)
/// - Operation types: Some combinations fuse well, others don't
/// - Hardware: Different CPUs have different sweet spots
/// </para>
/// </remarks>
public class AdaptiveFusionPass : IOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "Adaptive Fusion";

    /// <summary>
    /// Configuration for adaptive fusion behavior.
    /// </summary>
    public class AdaptiveFusionConfig
    {
        /// <summary>Maximum chain length for element-wise fusion.</summary>
        public int MaxElementWiseChainLength { get; set; } = 6;

        /// <summary>Maximum tensor size for aggressive fusion (elements).</summary>
        public int MaxTensorSizeForAggressiveFusion { get; set; } = 10000;

        /// <summary>Minimum tensor size for conservative fusion (elements).</summary>
        public int MinTensorSizeForConservativeFusion { get; set; } = 1000000;

        /// <summary>Whether to fuse across branches (may increase memory).</summary>
        public bool FuseAcrossBranches { get; set; } = false;

        /// <summary>Whether to consider cache size in fusion decisions.</summary>
        public bool CacheAwareFusion { get; set; } = true;

        /// <summary>Estimated L2 cache size in bytes.</summary>
        public int L2CacheSizeBytes { get; set; } = 256 * 1024;
    }

    private readonly AdaptiveFusionConfig _config;
    private int _nextTensorId;

    /// <summary>
    /// Initializes with default configuration.
    /// </summary>
    public AdaptiveFusionPass() : this(new AdaptiveFusionConfig()) { }

    /// <summary>
    /// Initializes with custom configuration.
    /// </summary>
    public AdaptiveFusionPass(AdaptiveFusionConfig config)
    {
        _config = config;
    }

    /// <inheritdoc/>
    public IRGraph Optimize(IRGraph graph)
    {
        // Initialize tensor ID counter
        _nextTensorId = graph.Operations.Any()
            ? graph.Operations.Max(op => op.OutputId) + 1
            : graph.InputIds.Any() ? graph.InputIds.Max() + 1 : 0;

        // Analyze graph and determine optimal fusion strategy
        var strategy = DetermineFusionStrategy(graph);

        // Apply fusion based on strategy
        return strategy switch
        {
            FusionStrategy.None => graph,
            FusionStrategy.Conservative => ApplyConservativeFusion(graph),
            FusionStrategy.Standard => ApplyStandardFusion(graph),
            FusionStrategy.Aggressive => ApplyAggressiveFusion(graph),
            _ => graph
        };
    }

    /// <summary>
    /// Determines the optimal fusion strategy for the graph.
    /// </summary>
    private FusionStrategy DetermineFusionStrategy(IRGraph graph)
    {
        if (graph.Operations.Count < 2)
            return FusionStrategy.None;

        // Analyze tensor sizes
        var tensorSizes = graph.TensorShapes.Values
            .Select(s => s.Aggregate(1, (a, b) => a * b))
            .ToList();

        if (tensorSizes.Count == 0)
            return FusionStrategy.Standard;

        var avgTensorSize = tensorSizes.Average();
        var maxTensorSize = tensorSizes.Max();

        // Cache-aware decision making
        if (_config.CacheAwareFusion)
        {
            var estimatedWorkingSet = tensorSizes.Sum() * sizeof(float);

            // If working set fits in L2, aggressive fusion is safe
            if (estimatedWorkingSet < _config.L2CacheSizeBytes)
            {
                return FusionStrategy.Aggressive;
            }
        }

        // Size-aware fusion strategy
        if (avgTensorSize < _config.MaxTensorSizeForAggressiveFusion)
        {
            return FusionStrategy.Aggressive;
        }
        else if (maxTensorSize > _config.MinTensorSizeForConservativeFusion)
        {
            return FusionStrategy.Conservative;
        }
        else
        {
            return FusionStrategy.Standard;
        }
    }

    /// <summary>
    /// Applies conservative fusion (only high-value patterns).
    /// </summary>
    private IRGraph ApplyConservativeFusion(IRGraph graph)
    {
        var fusedOps = new List<IROp>();
        var processed = new HashSet<int>();
        var tensorMapping = new Dictionary<int, int>();

        foreach (var op in graph.Operations.Where(o => !processed.Contains(o.OutputId)))
        {
            // Only fuse high-value patterns in conservative mode
            var pattern = FindHighValuePattern(graph, op, processed);

            if (pattern.Count > 1)
            {
                var fusedOp = CreateFusedOp(pattern, tensorMapping);
                if (fusedOp != null)
                {
                    fusedOps.Add(fusedOp);
                    foreach (var p in pattern)
                    {
                        processed.Add(p.OutputId);
                        if (p != pattern[^1])
                        {
                            tensorMapping[p.OutputId] = pattern[^1].OutputId;
                        }
                    }
                    continue;
                }
            }

            // Keep operation as-is with remapped inputs
            var remapped = RemapInputs(op, tensorMapping);
            fusedOps.Add(remapped);
            processed.Add(op.OutputId);
        }

        return CreateOptimizedGraph(graph, fusedOps, tensorMapping);
    }

    /// <summary>
    /// Applies standard fusion (balanced approach).
    /// </summary>
    private IRGraph ApplyStandardFusion(IRGraph graph)
    {
        var fusionPass = new OperationFusionPass();
        return fusionPass.Optimize(graph);
    }

    /// <summary>
    /// Applies aggressive fusion (maximize fusion).
    /// </summary>
    private IRGraph ApplyAggressiveFusion(IRGraph graph)
    {
        var fusedOps = new List<IROp>();
        var processed = new HashSet<int>();
        var tensorMapping = new Dictionary<int, int>();

        foreach (var op in graph.Operations)
        {
            if (processed.Contains(op.OutputId))
                continue;

            // Try to find any fusable pattern
            var pattern = FindFusablePattern(graph, op, processed, _config.MaxElementWiseChainLength);

            if (pattern.Count > 1)
            {
                var fusedOp = CreateFusedOp(pattern, tensorMapping);
                if (fusedOp != null)
                {
                    fusedOps.Add(fusedOp);
                    foreach (var p in pattern)
                    {
                        processed.Add(p.OutputId);
                        if (p != pattern[^1])
                        {
                            tensorMapping[p.OutputId] = pattern[^1].OutputId;
                        }
                    }
                    continue;
                }
            }

            // Keep operation as-is with remapped inputs
            var remapped = RemapInputs(op, tensorMapping);
            fusedOps.Add(remapped);
            processed.Add(op.OutputId);
        }

        return CreateOptimizedGraph(graph, fusedOps, tensorMapping);
    }

    /// <summary>
    /// Finds high-value fusion patterns (conservative).
    /// </summary>
    private List<IROp> FindHighValuePattern(IRGraph graph, IROp startOp, HashSet<int> processed)
    {
        var pattern = new List<IROp> { startOp };

        // Pattern 1: Conv + BatchNorm + Activation
        if (startOp.OpType.Contains("Conv"))
        {
            var nextOp = FindSingleConsumer(graph, startOp, processed);
            if (nextOp?.OpType == "BatchNorm")
            {
                pattern.Add(nextOp);
                var activationOp = FindSingleConsumer(graph, nextOp, processed);
                if (activationOp != null && IsActivation(activationOp))
                {
                    pattern.Add(activationOp);
                }
                return pattern;
            }
        }

        // Pattern 2: MatMul + Add (bias) + Activation
        if (startOp.OpType == "MatMul")
        {
            var nextOp = FindSingleConsumer(graph, startOp, processed);
            if (nextOp?.OpType == "Add")
            {
                pattern.Add(nextOp);
                var activationOp = FindSingleConsumer(graph, nextOp, processed);
                if (activationOp != null && IsActivation(activationOp))
                {
                    pattern.Add(activationOp);
                }
                return pattern;
            }
        }

        // Pattern 3: LayerNorm + Add (residual)
        if (startOp.OpType == "LayerNorm")
        {
            var nextOp = FindSingleConsumer(graph, startOp, processed);
            if (nextOp?.OpType == "Add")
            {
                pattern.Add(nextOp);
                return pattern;
            }
        }

        return pattern;
    }

    /// <summary>
    /// Finds any fusable pattern (aggressive).
    /// </summary>
    private List<IROp> FindFusablePattern(IRGraph graph, IROp startOp, HashSet<int> processed, int maxLength)
    {
        var pattern = new List<IROp> { startOp };

        // First try high-value patterns
        var highValue = FindHighValuePattern(graph, startOp, processed);
        if (highValue.Count > 1)
            return highValue;

        // Then try element-wise chains
        if (IsElementWise(startOp))
        {
            var currentOp = startOp;
            while (pattern.Count < maxLength)
            {
                var nextOp = FindSingleConsumer(graph, currentOp, processed);
                if (nextOp == null || !IsElementWise(nextOp) && !IsActivation(nextOp))
                    break;

                pattern.Add(nextOp);
                currentOp = nextOp;
            }
        }

        // Try activation fusion
        if (!IsActivation(startOp))
        {
            var nextOp = FindSingleConsumer(graph, startOp, processed);
            if (nextOp != null && IsActivation(nextOp) && pattern.Count == 1)
            {
                pattern.Add(nextOp);
            }
        }

        return pattern;
    }

    /// <summary>
    /// Finds the single consumer of an operation (if it has exactly one).
    /// </summary>
    private IROp? FindSingleConsumer(IRGraph graph, IROp op, HashSet<int> processed)
    {
        IROp? consumer = null;
        int consumerCount = 0;

        foreach (var candidate in graph.Operations)
        {
            if (processed.Contains(candidate.OutputId))
                continue;

            if (candidate.InputIds.Contains(op.OutputId))
            {
                consumer = candidate;
                consumerCount++;
                if (consumerCount > 1)
                    return null; // Multiple consumers - can't safely fuse
            }
        }

        return consumer;
    }

    /// <summary>
    /// Creates a fused operation from a pattern.
    /// </summary>
    private IROp? CreateFusedOp(List<IROp> pattern, Dictionary<int, int> tensorMapping)
    {
        if (pattern.Count < 2)
            return null;

        var firstOp = pattern[0];
        var lastOp = pattern[^1];

        // Determine the type of fused operation to create
        var opTypes = pattern.Select(p => p.OpType).ToList();

        // Pattern: Conv + BatchNorm (+ Activation)
        if (opTypes[0].Contains("Conv") && opTypes.Contains("BatchNorm"))
        {
            var hasActivation = pattern.Any(p => IsActivation(p));
            var activationName = hasActivation
                ? pattern.First(p => IsActivation(p)).OpType
                : "None";

            return new FusedConvBatchNormActivationOp
            {
                OutputId = lastOp.OutputId,
                InputIds = RemapInputIds(firstOp.InputIds, tensorMapping),
                OutputType = lastOp.OutputType,
                OutputShape = lastOp.OutputShape,
                ActivationName = activationName
            };
        }

        // Pattern: MatMul + Add + Activation (Linear + Bias + Activation)
        if (opTypes[0] == "MatMul" && opTypes.Contains("Add"))
        {
            var hasActivation = pattern.Any(p => IsActivation(p));
            var activationName = hasActivation
                ? pattern.First(p => IsActivation(p)).OpType
                : "None";

            // Collect all input IDs
            var allInputs = new List<int>();
            foreach (var op in pattern)
            {
                allInputs.AddRange(op.InputIds);
            }
            // Remove intermediate tensor IDs
            var intermediateIds = pattern.Select(p => p.OutputId).ToHashSet();
            var finalInputs = allInputs.Where(id => !intermediateIds.Contains(id)).Distinct().ToArray();

            return new FusedLinearActivationOp
            {
                OutputId = lastOp.OutputId,
                InputIds = RemapInputIds(finalInputs, tensorMapping),
                OutputType = lastOp.OutputType,
                OutputShape = lastOp.OutputShape,
                ActivationName = activationName,
                HasBias = true
            };
        }

        // Pattern: Element-wise chain
        if (pattern.All(p => IsElementWise(p) || IsActivation(p)))
        {
            return new FusedElementwiseChainOp
            {
                OutputId = lastOp.OutputId,
                InputIds = RemapInputIds(firstOp.InputIds, tensorMapping),
                OutputType = lastOp.OutputType,
                OutputShape = lastOp.OutputShape,
                OperationNames = opTypes
            };
        }

        // Pattern: Any operation + Activation
        if (pattern.Count == 2 && IsActivation(pattern[1]))
        {
            // Collect inputs from the first operation
            return new FusedElementwiseActivationOp
            {
                OutputId = lastOp.OutputId,
                InputIds = RemapInputIds(firstOp.InputIds, tensorMapping),
                OutputType = lastOp.OutputType,
                OutputShape = lastOp.OutputShape,
                ElementwiseOp = firstOp.OpType,
                ActivationName = pattern[1].OpType
            };
        }

        // Pattern: LayerNorm + Add
        if (opTypes[0] == "LayerNorm" && opTypes[1] == "Add")
        {
            var addOp = pattern[1];
            // Find the residual input (the one that isn't from LayerNorm)
            var residualInput = addOp.InputIds.FirstOrDefault(id => id != firstOp.OutputId);

            var allInputs = new List<int>(firstOp.InputIds);
            if (residualInput != 0)
                allInputs.Add(residualInput);

            return new FusedLayerNormAddOp
            {
                OutputId = lastOp.OutputId,
                InputIds = RemapInputIds(allInputs.ToArray(), tensorMapping),
                OutputType = lastOp.OutputType,
                OutputShape = lastOp.OutputShape
            };
        }

        // Couldn't create a specialized fused op
        return null;
    }

    /// <summary>
    /// Remaps input IDs according to tensor mapping.
    /// </summary>
    private int[] RemapInputIds(int[] inputIds, Dictionary<int, int> tensorMapping)
    {
        return inputIds.Select(id => tensorMapping.TryGetValue(id, out var mapped) ? mapped : id).ToArray();
    }

    /// <summary>
    /// Remaps inputs for an operation.
    /// </summary>
    private IROp RemapInputs(IROp op, Dictionary<int, int> tensorMapping)
    {
        var newInputIds = RemapInputIds(op.InputIds, tensorMapping);

        // Create a copy with new input IDs
        // Note: This is a simplified approach - a full implementation would clone properly
        op.InputIds = newInputIds;
        return op;
    }

    /// <summary>
    /// Creates the optimized graph with fused operations.
    /// </summary>
    private IRGraph CreateOptimizedGraph(IRGraph original, List<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        return new IRGraph
        {
            InputIds = new List<int>(original.InputIds),
            OutputIds = original.OutputIds
                .Select(id => tensorMapping.TryGetValue(id, out var mapped) ? mapped : id)
                .ToList(),
            Operations = fusedOps,
            TensorShapes = new Dictionary<int, int[]>(original.TensorShapes),
            Metadata = new Dictionary<string, object>(original.Metadata)
            {
                ["AdaptiveFusion_OriginalOps"] = original.Operations.Count,
                ["AdaptiveFusion_FusedOps"] = fusedOps.Count
            }
        };
    }

    /// <summary>
    /// Checks if an operation is element-wise.
    /// </summary>
    private bool IsElementWise(IROp op)
    {
        return op.OpType is "Add" or "Subtract" or "ElementwiseMultiply" or "Divide"
            or "Negate" or "Exp" or "Log" or "Sqrt" or "Power";
    }

    /// <summary>
    /// Checks if an operation is an activation function.
    /// </summary>
    private bool IsActivation(IROp op)
    {
        return op.OpType is "ReLU" or "Sigmoid" or "Tanh" or "Softmax" or "GELU" or "Swish" or "LeakyReLU";
    }

    /// <summary>
    /// Fusion strategies.
    /// </summary>
    private enum FusionStrategy
    {
        None,          // No fusion
        Conservative,  // Only high-value patterns
        Standard,      // Normal fusion
        Aggressive     // Maximum fusion
    }
}
