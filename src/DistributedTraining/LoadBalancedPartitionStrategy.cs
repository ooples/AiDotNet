using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Partitions model parameters across pipeline stages using estimated computational cost per layer.
/// </summary>
/// <remarks>
/// <para>
/// Instead of dividing parameters uniformly, this strategy uses a cost function to estimate
/// the computational load for each parameter group (layer). It then assigns parameters to stages
/// so that each stage has roughly equal total cost, reducing pipeline bubble overhead.
/// </para>
/// <para><b>For Beginners:</b> Imagine an assembly line where some tasks take much longer than others.
/// If you assign tasks purely by count, some workers finish early and wait while others are still busy.
/// This strategy assigns tasks by estimated time, so all workers finish at roughly the same time.
///
/// For neural networks, attention layers are much more expensive than simple normalization layers,
/// so this strategy gives fewer attention layers to each stage to balance the workload.
///
/// The cost function estimates FLOPs (floating point operations) for a block of parameters:
/// - Dense/linear layers: ~2 * inputSize * outputSize FLOPs
/// - Attention: ~4 * seqLen * d_model FLOPs
/// - LayerNorm: ~5 * d_model FLOPs
///
/// Since we don't have layer-level metadata in the parameter vector, costs are estimated from
/// parameter counts using the heuristic that computation scales quadratically with matrix dimensions.
/// </para>
/// <para><b>Reference:</b> Megatron-LM layer assignment algorithm, NVIDIA 2020.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations.</typeparam>
public class LoadBalancedPartitionStrategy<T> : IPipelinePartitionStrategy<T>
{
    private readonly Func<int, double>? _costEstimator;
    private readonly int[] _layerBoundaries;
    private readonly bool _isAutoDetect;

    /// <summary>
    /// Creates a load-balanced partition strategy with explicit layer boundaries and optional cost estimator.
    /// </summary>
    /// <param name="layerBoundaries">
    /// Array of parameter indices where each layer starts, in strictly increasing order.
    /// All values must be non-negative. For example, if a model has 3 layers
    /// with 100, 200, and 150 parameters respectively, pass [0, 100, 300].
    /// The total parameter count is inferred as layerBoundaries[last] + size of last layer.
    /// <para><b>For Beginners:</b> This tells the partitioner where each layer's parameters begin
    /// in the flat parameter vector. You can get these from your model's layer structure.</para>
    /// </param>
    /// <param name="costEstimator">
    /// Optional function that estimates the computational cost of a layer given its parameter count.
    /// If null, cost is estimated as parameterCount^(3/2) which approximates the relationship
    /// between matrix sizes and FLOP counts for dense layers.
    /// <para><b>For Beginners:</b> This function converts "number of parameters" into "how long
    /// this layer takes to compute." The default assumes dense matrix multiplication.</para>
    /// </param>
    /// <exception cref="ArgumentException">Thrown when layerBoundaries is null, empty,
    /// contains negative values, or is not strictly increasing.</exception>
    public LoadBalancedPartitionStrategy(int[] layerBoundaries, Func<int, double>? costEstimator = null)
    {
        if (layerBoundaries is null || layerBoundaries.Length == 0)
        {
            throw new ArgumentException("Layer boundaries must be provided and non-empty.", nameof(layerBoundaries));
        }

        // Validate first boundary is zero (no orphaned parameters before the first layer)
        if (layerBoundaries[0] != 0)
        {
            throw new ArgumentException(
                $"First layer boundary must be 0, but was {layerBoundaries[0]}. " +
                "Parameters before the first boundary would be unassigned to any layer.",
                nameof(layerBoundaries));
        }

        for (int i = 1; i < layerBoundaries.Length; i++)
        {
            if (layerBoundaries[i] < 0)
            {
                throw new ArgumentException(
                    $"Layer boundary at index {i} is negative ({layerBoundaries[i]}). All boundaries must be non-negative.",
                    nameof(layerBoundaries));
            }

            if (layerBoundaries[i] <= layerBoundaries[i - 1])
            {
                throw new ArgumentException(
                    $"Layer boundaries must be strictly increasing, but boundary[{i}]={layerBoundaries[i]} " +
                    $"<= boundary[{i - 1}]={layerBoundaries[i - 1]}.",
                    nameof(layerBoundaries));
            }
        }

        _layerBoundaries = layerBoundaries;
        _costEstimator = costEstimator;
        _isAutoDetect = false;
    }

    /// <summary>
    /// Creates a load-balanced partition strategy that auto-detects layer boundaries
    /// using a fixed layer size estimate.
    /// </summary>
    /// <param name="estimatedLayerSize">
    /// Estimated average number of parameters per layer.
    /// <para><b>For Beginners:</b> If you know your model has ~1000 parameters per layer,
    /// pass 1000 here and the partitioner will create synthetic layer boundaries.</para>
    /// </param>
    /// <param name="costEstimator">Optional cost estimator function.</param>
    /// <exception cref="ArgumentException">Thrown when estimatedLayerSize is not positive.</exception>
    public LoadBalancedPartitionStrategy(int estimatedLayerSize, Func<int, double>? costEstimator = null)
    {
        if (estimatedLayerSize <= 0)
        {
            throw new ArgumentException("Estimated layer size must be positive.", nameof(estimatedLayerSize));
        }

        _layerBoundaries = new[] { estimatedLayerSize };
        _costEstimator = costEstimator;
        _isAutoDetect = true;
    }

    /// <inheritdoc/>
    public (int StartIndex, int Size)[] ComputePartition(int totalParameters, int numStages)
    {
        if (totalParameters <= 0)
        {
            throw new ArgumentException("Total parameters must be positive.", nameof(totalParameters));
        }

        if (numStages <= 0)
        {
            throw new ArgumentException("Number of stages must be positive.", nameof(numStages));
        }

        // Build layer sizes from boundaries
        var layerSizes = BuildLayerSizes(totalParameters);
        var layerCosts = ComputeLayerCosts(layerSizes);

        // Use dynamic programming to find the optimal partition that minimizes
        // the maximum cost across all stages (minimize pipeline bubble)
        return OptimalPartition(layerSizes, layerCosts, numStages);
    }

    private int[] BuildLayerSizes(int totalParameters)
    {
        if (_isAutoDetect)
        {
            // Auto-detect mode: use estimated layer size to create synthetic boundaries
            int estimatedLayerSize = _layerBoundaries[0];
            int numLayers = Math.Max(1, totalParameters / estimatedLayerSize);
            var sizes = new int[numLayers];
            int baseSize = totalParameters / numLayers;
            int remainder = totalParameters % numLayers;

            for (int i = 0; i < numLayers; i++)
            {
                sizes[i] = baseSize + (i < remainder ? 1 : 0);
            }

            return sizes;
        }

        // Explicit boundaries mode: compute sizes from consecutive boundary differences
        if (_layerBoundaries[_layerBoundaries.Length - 1] > totalParameters)
        {
            throw new ArgumentException(
                $"Last layer boundary ({_layerBoundaries[_layerBoundaries.Length - 1]}) exceeds " +
                $"total parameters ({totalParameters}).",
                nameof(totalParameters));
        }

        var layerSizes = new int[_layerBoundaries.Length];
        for (int i = 0; i < _layerBoundaries.Length; i++)
        {
            int start = _layerBoundaries[i];
            int end = (i + 1 < _layerBoundaries.Length) ? _layerBoundaries[i + 1] : totalParameters;
            layerSizes[i] = end - start;
        }

        return layerSizes;
    }

    private double[] ComputeLayerCosts(int[] layerSizes)
    {
        var costs = new double[layerSizes.Length];

        for (int i = 0; i < layerSizes.Length; i++)
        {
            // Default heuristic: cost scales as paramCount^1.5
            // For a square weight matrix of dimension n: params = n^2, FLOPs = 2*n^3 = 2*(params)^1.5.
            // This is a reasonable approximation for dense/linear layers.
            costs[i] = _costEstimator is not null
                ? _costEstimator(layerSizes[i])
                : Math.Pow(layerSizes[i], 1.5);
        }

        return costs;
    }

    /// <summary>
    /// Uses dynamic programming to find the partition of layers into stages
    /// that minimizes the maximum stage cost (min-max partitioning).
    /// </summary>
    private (int StartIndex, int Size)[] OptimalPartition(int[] layerSizes, double[] layerCosts, int numStages)
    {
        int numLayers = layerSizes.Length;

        if (numStages >= numLayers)
        {
            // More stages than layers: assign one layer per stage, remaining stages get empty shards
            return AssignOneLayerPerStage(layerSizes, numStages);
        }

        // Prefix sums for parameter sizes and costs
        var paramPrefix = new long[numLayers + 1];
        var costPrefix = new double[numLayers + 1];

        for (int i = 0; i < numLayers; i++)
        {
            paramPrefix[i + 1] = paramPrefix[i] + layerSizes[i];
            costPrefix[i + 1] = costPrefix[i] + layerCosts[i];
        }

        // dp[s][l] = minimum of maximum stage cost when assigning layers 0..l-1 to stages 0..s-1
        var dp = new double[numStages + 1][];
        var splitPoint = new int[numStages + 1][];

        for (int s = 0; s <= numStages; s++)
        {
            dp[s] = new double[numLayers + 1];
            splitPoint[s] = new int[numLayers + 1];
            for (int i = 0; i < dp[s].Length; i++)
            {
                dp[s][i] = double.MaxValue;
            }
        }

        dp[0][0] = 0.0;

        // Base case: one stage gets all layers up to l
        for (int l = 1; l <= numLayers; l++)
        {
            dp[1][l] = costPrefix[l];
            splitPoint[1][l] = 0;
        }

        // Fill DP table
        for (int s = 2; s <= numStages; s++)
        {
            for (int l = s; l <= numLayers; l++)
            {
                // Try all possible split points for the last stage
                for (int k = s - 1; k < l; k++)
                {
                    double lastStageCost = costPrefix[l] - costPrefix[k];
                    double candidate = Math.Max(dp[s - 1][k], lastStageCost);

                    if (candidate < dp[s][l])
                    {
                        dp[s][l] = candidate;
                        splitPoint[s][l] = k;
                    }
                }
            }
        }

        // Backtrack to find optimal partition
        var stageEndLayers = new int[numStages];
        int currentLayer = numLayers;

        for (int s = numStages; s >= 1; s--)
        {
            stageEndLayers[s - 1] = currentLayer;
            currentLayer = splitPoint[s][currentLayer];
        }

        // Convert layer assignments to parameter partitions
        var partitions = new (int StartIndex, int Size)[numStages];
        int layerStart = 0;

        for (int s = 0; s < numStages; s++)
        {
            int layerEnd = stageEndLayers[s];
            long paramStartLong = paramPrefix[layerStart];
            long paramSizeLong = paramPrefix[layerEnd] - paramPrefix[layerStart];

            if (paramStartLong > int.MaxValue || paramSizeLong > int.MaxValue)
            {
                throw new InvalidOperationException(
                    $"Stage {s} parameter range exceeds int.MaxValue " +
                    $"(start={paramStartLong}, size={paramSizeLong}). " +
                    "Models with more than int.MaxValue parameters are not supported.");
            }

            int paramStart = (int)paramStartLong;
            int paramSize = (int)paramSizeLong;
            partitions[s] = (paramStart, paramSize);
            layerStart = layerEnd;
        }

        return partitions;
    }

    private static (int StartIndex, int Size)[] AssignOneLayerPerStage(int[] layerSizes, int numStages)
    {
        var partitions = new (int StartIndex, int Size)[numStages];
        int currentStart = 0;

        for (int i = 0; i < numStages; i++)
        {
            if (i < layerSizes.Length)
            {
                partitions[i] = (currentStart, layerSizes[i]);
                currentStart += layerSizes[i];
            }
            else
            {
                // Empty stage (more stages than layers)
                partitions[i] = (currentStart, 0);
            }
        }

        return partitions;
    }
}
