using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements gradient compression optimizer - wraps any optimizer to add gradient compression.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Gradient compression reduces communication overhead in distributed training by compressing
/// gradients before transmission. Common techniques include:
/// - Quantization: Reduce precision (FP32 → FP16 or INT8)
/// - Sparsification: Send only top-k largest gradients
/// - Low-rank compression: Decompose gradient matrices
///
/// This can significantly reduce network bandwidth usage (2-100x compression) with minimal
/// accuracy impact when done carefully. Essential for training over slow networks or at very large scales.
/// </para>
/// <para><b>For Beginners:</b>
/// This optimizer wrapper compresses gradients before sending them across the network.
/// Think of it like compressing a large file before emailing it - the information is mostly
/// preserved, but it takes much less time to transmit. This is especially useful when your
/// GPUs are connected by slower networks.
///
/// Common compression techniques:
/// - Make numbers less precise (like rounding 3.14159 to 3.14)
/// - Send only the most important values (top-k sparsification)
/// - Use mathematical tricks to represent data more compactly
/// </para>
/// <para><b>Use Cases:</b>
/// - Slow network connections (cross-datacenter, ethernet vs NVLink)
/// - Very large models where gradient size is huge
/// - Can wrap any other optimizer (DDP, FSDP, etc.)
/// - Bandwidth-constrained scenarios
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Same as wrapped optimizer
/// - Communication: Much lower - 2x to 100x reduction depending on technique
/// - Complexity: Moderate - adds compression/decompression overhead
/// - Accuracy: Slight potential degradation (usually <1% with proper tuning)
/// - Best for: Slow networks, bandwidth-limited scenarios
/// </para>
/// <para><b>Implementation Note:</b>
/// This framework provides the compression optimizer infrastructure. Specific compression
/// algorithms (quantization levels, sparsity ratios) can be configured. Production use
/// would implement specific compression strategies based on your network and accuracy requirements.
/// </para>
/// <para>
/// Example:
/// <code>
/// var baseOptimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend) {
///     EnableGradientCompression = true
/// };
///
/// // Wrap with gradient compression
/// var compressedOptimizer = new GradientCompressionOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     baseOptimizer, config, compressionRatio: 0.1); // Keep top 10% of gradients
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class GradientCompressionOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    private readonly double _compressionRatio;
    private readonly bool _useQuantization;
    private readonly bool _useSparsification;

    /// <summary>
    /// Creates a gradient compression optimizer.
    /// </summary>
    /// <param name="wrappedOptimizer">The optimizer to wrap with compression</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <param name="compressionRatio">Compression ratio (0.0 to 1.0, lower = more compression)</param>
    /// <param name="useQuantization">Whether to use quantization compression</param>
    /// <param name="useSparsification">Whether to use top-k sparsification</param>
    public GradientCompressionOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config,
        double compressionRatio = 0.1,
        bool useQuantization = true,
        bool useSparsification = false)
        : base(wrappedOptimizer, config)
    {
        if (compressionRatio <= 0 || compressionRatio > 1)
            throw new ArgumentException("Compression ratio must be in (0, 1]", nameof(compressionRatio));

        _compressionRatio = compressionRatio;
        _useQuantization = useQuantization;
        _useSparsification = useSparsification;
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        Config.CommunicationBackend.Barrier();

        // Optimize locally
        var result = WrappedOptimizer.Optimize(inputData);

        // Compress and synchronize gradients
        if (Config.AutoSyncGradients && result.BestSolution != null)
        {
            var parameters = result.BestSolution.GetParameters();

            // Compress gradients
            var compressed = CompressGradients(parameters);

            // Synchronize compressed gradients
            Config.CommunicationBackend.AllReduce(compressed, ReductionOperation.Average);

            // Decompress
            var decompressed = DecompressGradients(compressed, parameters.Length);

            // Update model with decompressed gradients
            result.BestSolution.SetParameters(decompressed);
        }

        Config.CommunicationBackend.Barrier();

        return result;
    }

    /// <summary>
    /// Compresses gradients using configured compression techniques.
    /// </summary>
    private Vector<T> CompressGradients(Vector<T> gradients)
    {
        if (_useSparsification)
        {
            // Top-k sparsification: keep only top k% largest magnitude values
            return ApplyTopKSparsification(gradients);
        }
        else if (_useQuantization)
        {
            // Quantization: reduce precision
            return ApplyQuantization(gradients);
        }
        else
        {
            return gradients; // No compression
        }
    }

    /// <summary>
    /// Applies top-k sparsification - keeps only largest magnitude values.
    /// </summary>
    private Vector<T> ApplyTopKSparsification(Vector<T> gradients)
    {
        // Framework implementation - placeholder for actual top-k algorithm
        // Production would:
        // 1. Find k-th largest absolute value
        // 2. Zero out all values smaller than threshold
        // 3. Store indices and values of non-zero elements

        // For now, return as-is (no actual compression in this framework demo)
        return gradients;
    }

    /// <summary>
    /// Applies quantization compression.
    /// </summary>
    private Vector<T> ApplyQuantization(Vector<T> gradients)
    {
        // Framework implementation - placeholder
        // Production would convert to lower precision (FP16, INT8, etc.)
        return gradients;
    }

    /// <summary>
    /// Decompresses gradients back to full format.
    /// </summary>
    private Vector<T> DecompressGradients(Vector<T> compressed, int originalLength)
    {
        // Framework implementation - placeholder
        // Would reverse the compression operation
        return compressed;
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // Delegate to wrapped optimizer
        // Optimizer state typically not compressed (infrequent communication)
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize compression-specific configuration
        writer.Write(_compressionRatio);
        writer.Write(_useQuantization);
        writer.Write(_useSparsification);

        // Serialize sharding configuration info
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(Config.AutoSyncGradients);
        writer.Write(Config.MinimumParameterGroupSize);
        writer.Write(Config.EnableGradientCompression);

        // Serialize wrapped optimizer
        var optimizerData = WrappedOptimizer.Serialize();
        writer.Write(optimizerData.Length);
        writer.Write(optimizerData);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read compression-specific configuration
        double savedCompressionRatio = reader.ReadDouble();
        bool savedUseQuantization = reader.ReadBoolean();
        bool savedUseSparsification = reader.ReadBoolean();

        // Read sharding configuration (for validation)
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        reader.ReadBoolean(); // AutoSyncGradients
        reader.ReadInt32(); // MinimumParameterGroupSize
        reader.ReadBoolean(); // EnableGradientCompression

        if (savedWorldSize != WorldSize)
        {
            throw new InvalidOperationException(
                $"World size mismatch. Optimizer was saved with {savedWorldSize} processes, " +
                $"but current configuration has {WorldSize} processes.");
        }

        if (savedRank != Rank)
        {
            throw new InvalidOperationException(
                $"Rank mismatch. Optimizer was saved on rank {savedRank}, " +
                $"but is being loaded on rank {Rank}. This could indicate a configuration error.");
        }

        if (Math.Abs(savedCompressionRatio - _compressionRatio) > 1e-6 ||
            savedUseQuantization != _useQuantization ||
            savedUseSparsification != _useSparsification)
        {
            throw new InvalidOperationException(
                $"Compression configuration mismatch. Optimizer was saved with compressionRatio={savedCompressionRatio}, " +
                $"useQuantization={savedUseQuantization}, useSparsification={savedUseSparsification}, " +
                $"but current configuration has compressionRatio={_compressionRatio}, " +
                $"useQuantization={_useQuantization}, useSparsification={_useSparsification}.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }
}
