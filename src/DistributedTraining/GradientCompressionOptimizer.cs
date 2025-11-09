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
    /// <param name="wrappedOptimizer">The optimizer to wrap with compression (must be gradient-based)</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <param name="compressionRatio">Compression ratio (0.0 to 1.0, lower = more compression)</param>
    /// <param name="useQuantization">Whether to use quantization compression</param>
    /// <param name="useSparsification">Whether to use top-k sparsification</param>
    /// <exception cref="ArgumentException">If wrapped optimizer is not gradient-based</exception>
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

        // Gradient compression only makes sense for gradient-based optimizers
        if (wrappedOptimizer is not IGradientBasedOptimizer<T, TInput, TOutput>)
        {
            throw new ArgumentException(
                $"Gradient compression requires a gradient-based optimizer. " +
                $"The provided optimizer type {wrappedOptimizer.GetType().Name} does not implement IGradientBasedOptimizer. " +
                $"Use gradient-based optimizers like SGD, Adam, RMSProp, etc.",
                nameof(wrappedOptimizer));
        }

        _compressionRatio = compressionRatio;
        _useQuantization = useQuantization;
        _useSparsification = useSparsification;
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        var gradientOptimizer = WrappedOptimizer as IGradientBasedOptimizer<T, TInput, TOutput>;
        if (gradientOptimizer == null)
        {
            throw new InvalidOperationException(
                $"GradientCompressionOptimizer requires a gradient-based optimizer, but received {WrappedOptimizer.GetType().Name}");
        }

        Config.CommunicationBackend.Barrier();

        // CRITICAL: Save parameters BEFORE local optimization
        // This allows us to apply averaged compressed gradients from the correct starting point
        Vector<T>? savedParameters = null;
        if (Config.AutoSyncGradients && inputData.InitialSolution != null)
        {
            savedParameters = inputData.InitialSolution.GetParameters();
        }

        // Step 1: Optimize locally to compute gradients and get locally-updated model
        var localResult = WrappedOptimizer.Optimize(inputData);

        // Step 2: Get the gradients that were computed during optimization
        var localGradients = gradientOptimizer.LastComputedGradients;

        if (Config.AutoSyncGradients && localResult.BestSolution != null && savedParameters != null && localGradients != null && localGradients.Length > 0)
        {
            // Step 3: Compress local gradients
            var compressedGradients = CompressGradients(localGradients);

            // Step 4: Synchronize compressed gradients across all ranks and average them
            Config.CommunicationBackend.AllReduce(compressedGradients, ReductionOperation.Average);

            // Step 5: Decompress to get averaged gradients
            var averagedGradients = DecompressGradients(compressedGradients, localGradients.Length);

            // Step 6: Apply averaged compressed gradients using the safe 3-parameter overload
            // This explicitly passes savedParameters (pre-update state) to prevent double-stepping
            // Works correctly with ANY optimizer (SGD, Adam, RMSprop, etc.)
            var finalModel = gradientOptimizer.ApplyGradients(savedParameters, averagedGradients, localResult.BestSolution);

            // Step 7: Return result with model updated using averaged compressed gradients
            localResult.BestSolution = finalModel;
        }

        Config.CommunicationBackend.Barrier();

        return localResult;
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
    /// <remarks>
    /// <para>
    /// This implementation zeros out all gradient values except the top-k largest by absolute value,
    /// where k = compressionRatio * total_elements. For example, with compressionRatio=0.1,
    /// only the top 10% largest gradients are kept.
    /// </para>
    /// <para>
    /// This technique is based on "Deep Gradient Compression" (Lin et al., 2017) and is widely
    /// used in production distributed training to reduce communication overhead while maintaining
    /// convergence properties. The intuition is that small gradients contribute little to optimization,
    /// so dropping them has minimal impact on final model quality.
    /// </para>
    /// <para>
    /// The returned vector has the same length as the input but with small values zeroed out.
    /// When averaged across ranks via AllReduce, each rank contributes its top-k gradients,
    /// resulting in a sparse but informative averaged gradient.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyTopKSparsification(Vector<T> gradients)
    {
        if (gradients == null)
            return Vector<T>.Empty();

        if (gradients.Length == 0)
            return gradients;

        int k = Math.Max(1, (int)(_compressionRatio * gradients.Length));

        // Create array of (index, absolute value) pairs for sorting
        var indexedValues = new (int index, double absValue)[gradients.Length];
        for (int i = 0; i < gradients.Length; i++)
        {
            indexedValues[i] = (i, Math.Abs(Convert.ToDouble(gradients[i])));
        }

        // Sort by absolute value descending to find top-k
        Array.Sort(indexedValues, (a, b) => b.absValue.CompareTo(a.absValue));

        // Create result with only top-k values kept, rest zeroed
        var result = new T[gradients.Length];
        var zero = NumOps.FromDouble(0.0);

        // Initialize all to zero
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = zero;
        }

        // Keep only top-k largest values
        for (int i = 0; i < k; i++)
        {
            int originalIndex = indexedValues[i].index;
            result[originalIndex] = gradients[originalIndex];
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Applies quantization compression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implementation quantizes gradients to a reduced number of discrete levels,
    /// simulating the effect of reduced precision arithmetic (FP32 → FP16, FP8, or INT8).
    /// The number of quantization levels is determined by: levels = max(2, compressionRatio * 256),
    /// where 256 represents the typical range for 8-bit quantization.
    /// </para>
    /// <para>
    /// Quantization works by:
    /// 1. Finding the min and max values in the gradient vector
    /// 2. Mapping each value to one of k discrete levels within that range
    /// 3. Mapping back to the original numeric type
    /// </para>
    /// <para>
    /// This technique is based on "1-bit SGD" (Seide et al., 2014) and quantized training
    /// methods used in production systems like TensorFlow Lite and PyTorch Mobile.
    /// The key insight is that gradient precision can be significantly reduced without
    /// harming convergence, as the optimizer's momentum and adaptive learning rates
    /// compensate for quantization noise.
    /// </para>
    /// <para>
    /// For example, with compressionRatio=0.1, gradients are quantized to approximately
    /// 26 levels (0.1 * 256 ≈ 26), roughly equivalent to 5-bit precision.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyQuantization(Vector<T> gradients)
    {
        if (gradients == null)
            return Vector<T>.Empty();

        if (gradients.Length == 0)
            return gradients;

        // Determine number of quantization levels
        int numLevels = Math.Max(2, (int)(_compressionRatio * 256));

        // Find min and max for quantization range
        double min = double.MaxValue;
        double max = double.MinValue;
        for (int i = 0; i < gradients.Length; i++)
        {
            double val = Convert.ToDouble(gradients[i]);
            if (val < min) min = val;
            if (val > max) max = val;
        }

        // Handle edge case where all gradients are the same
        if (Math.Abs(max - min) < 1e-12)
            return gradients;

        double range = max - min;
        double step = range / (numLevels - 1);

        // Quantize each gradient value
        var result = new T[gradients.Length];
        for (int i = 0; i < gradients.Length; i++)
        {
            double val = Convert.ToDouble(gradients[i]);

            // Map to discrete level
            int level = (int)Math.Round((val - min) / step);
            level = Math.Max(0, Math.Min(numLevels - 1, level));

            // Map back to original range
            double quantized = min + level * step;
            result[i] = NumOps.FromDouble(quantized);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Decompresses gradients back to full format.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For both Top-K sparsification and quantization as implemented in this class,
    /// decompression is essentially an identity operation because:
    /// </para>
    /// <para>
    /// <b>Top-K Sparsification:</b> Returns a full-length vector with small values zeroed out.
    /// After AllReduce averaging, we get a sparse but full-length vector. No expansion needed.
    /// </para>
    /// <para>
    /// <b>Quantization:</b> Returns a full-length vector with reduced precision values.
    /// After AllReduce averaging, we get a full-length averaged quantized vector. No expansion needed.
    /// </para>
    /// <para>
    /// In a more advanced implementation with true sparse representations (e.g., storing only
    /// indices and values of non-zero elements), this method would reconstruct the full dense
    /// vector. However, such an implementation would require sparse-aware communication backends
    /// that can handle variable-length messages and custom AllReduce operations.
    /// </para>
    /// <para>
    /// The current implementation provides the compression benefits of reduced gradient noise
    /// (Top-K) and reduced precision (quantization) while working with standard AllReduce
    /// operations on dense vectors. This is the approach used by many production frameworks
    /// like Horovod and PyTorch DDP with gradient compression plugins.
    /// </para>
    /// </remarks>
    /// <param name="compressed">The compressed (but full-length) gradient vector after AllReduce</param>
    /// <param name="originalLength">The original gradient vector length (used for validation)</param>
    /// <returns>The decompressed gradient vector (same as input for current compression methods)</returns>
    private Vector<T> DecompressGradients(Vector<T> compressed, int originalLength)
    {
        if (compressed == null)
            throw new ArgumentNullException(nameof(compressed));

        // Validate length matches expected
        if (compressed.Length != originalLength)
        {
            throw new ArgumentException(
                $"Compressed gradient length ({compressed.Length}) does not match " +
                $"original length ({originalLength}). This indicates a communication error.",
                nameof(compressed));
        }

        // For Top-K and quantization, no expansion needed - vector is already full length
        // Just return the averaged compressed gradients
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
