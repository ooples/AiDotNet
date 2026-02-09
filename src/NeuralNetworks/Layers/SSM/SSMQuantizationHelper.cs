using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Provides SSM-specific quantization utilities for reducing memory and accelerating inference.
/// </summary>
/// <remarks>
/// <para>
/// SSM models like Mamba have unique quantization characteristics compared to Transformers:
/// - The A, B, C projection weights are the most impactful for compression (they dominate parameter count)
/// - The D skip connection parameter is very sensitive and should usually be kept in full precision
/// - Hidden states can be quantized during inference for memory-constrained deployment
/// - The Conv1D weights are small and benefit less from quantization
/// </para>
/// <para>
/// This utility works with the existing <see cref="QuantizationConfiguration"/> and
/// <see cref="IQuantizer{T,TInput,TOutput}"/> infrastructure to provide SSM-aware quantization.
/// It applies quantization at the parameter level (via GetParameters/SetParameters) rather than
/// creating wrapped layer types, keeping the architecture clean.
/// </para>
/// <para><b>For Beginners:</b> Quantization makes the model smaller and faster by using less precise numbers.
///
/// Think of it like rounding numbers:
/// - Full precision: 3.14159265 (32-bit float, 4 bytes per number)
/// - INT8 quantized: 3.14 (8-bit integer with scale, 1 byte per number)
/// - The model gets 4x smaller and often runs 2-4x faster
///
/// For SSM models specifically:
/// - The A, B, C weight matrices benefit the most from quantization (biggest savings)
/// - The D parameter (skip connection) is sensitive - we protect it
/// - Hidden states can also be quantized during generation to save memory
///
/// This class provides tools to quantize SSM layers intelligently, knowing which parts
/// are safe to compress and which need protection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public static class SSMQuantizationHelper<T>
{
    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Quantizes an SSM layer's parameters using the provided configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extracts the layer's parameters via GetParameters(), applies min-max quantization
    /// at the configured bit width, then sets the quantized parameters back. The D parameter and
    /// any parameters below a minimum size threshold are optionally protected from quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This compresses a Mamba layer's weights by rounding them to
    /// fewer bits. The layer becomes smaller in memory but may lose a tiny bit of accuracy.</para>
    /// </remarks>
    /// <param name="layer">The SSM layer to quantize (e.g., MambaBlock).</param>
    /// <param name="config">Quantization configuration specifying bit width and strategy.</param>
    /// <param name="protectDParameter">
    /// Whether to keep the D skip connection parameter in full precision. Default: true.
    /// <para><b>For Beginners:</b> The D parameter is important for the skip connection.
    /// Quantizing it can cause more accuracy loss than quantizing larger weight matrices.</para>
    /// </param>
    public static void QuantizeSSMLayer(ILayer<T> layer, QuantizationConfiguration config, bool protectDParameter = true)
    {
        if (layer == null) throw new ArgumentNullException(nameof(layer));
        if (config == null) throw new ArgumentNullException(nameof(config));

        var parameters = layer.GetParameters();
        int bitWidth = config.EffectiveBitWidth;

        // Quantize all parameters using min-max quantization
        var quantized = QuantizeVector(parameters, bitWidth);

        // If protecting D parameter and this is a MambaBlock, restore D values
        if (protectDParameter && layer is MambaBlock<T> mambaBlock)
        {
            RestoreDParameter(quantized, mambaBlock);
        }

        layer.SetParameters(quantized);
    }

    /// <summary>
    /// Creates a new SSM state cache with precision-reduced states migrated from the source cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During autoregressive generation, the state cache can grow large for deep models.
    /// This method creates a new cache configured for quantized storage and migrates all
    /// existing SSM states and conv buffers from the source cache into it. The states are
    /// stored with reduced precision (quantize-then-dequantize), which reduces the effective
    /// precision of cached values but does not yet reduce in-memory footprint since values
    /// are still stored as Tensor&lt;T&gt;. Future implementations may use packed byte storage
    /// for true memory compression.
    /// </para>
    /// <para><b>For Beginners:</b> When generating long text, the model's memory (hidden states)
    /// can use a lot of RAM. This creates a new cache that stores states with reduced precision,
    /// which is useful on devices with limited RAM (phones, edge devices).</para>
    /// </remarks>
    /// <param name="cache">The source state cache whose states will be migrated to the new cache.</param>
    /// <param name="bitWidth">Target bit width for state values. Default: 8.</param>
    /// <returns>A new <see cref="SSMStateCache{T}"/> with precision-reduced states.</returns>
    public static SSMStateCache<T> QuantizeStateCache(SSMStateCache<T> cache, int bitWidth = 8)
    {
        if (cache == null) throw new ArgumentNullException(nameof(cache));
        if (bitWidth <= 0 || bitWidth > 32)
            throw new ArgumentException($"Bit width ({bitWidth}) must be between 1 and 32.", nameof(bitWidth));

        // Create a new cache with compression enabled and migrate existing states
        var compressed = new SSMStateCache<T>(enableCompression: true, compressionBitWidth: bitWidth);

        // Migrate all existing SSM states into the compressed cache
        // The compressed cache's CacheSSMState will apply quantization automatically
        foreach (int layerIndex in cache.GetSSMStateLayerIndices().ToList())
        {
            var state = cache.GetSSMState(layerIndex);
            if (state != null)
            {
                compressed.CacheSSMState(layerIndex, state);
            }
        }

        // Migrate all existing conv buffers
        foreach (int layerIndex in cache.GetConvBufferLayerIndices().ToList())
        {
            var buffer = cache.GetConvBuffer(layerIndex);
            if (buffer != null)
            {
                compressed.CacheConvBuffer(layerIndex, buffer);
            }
        }

        return compressed;
    }

    /// <summary>
    /// Estimates the memory savings from quantizing an SSM layer at the specified bit width.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tells you how much smaller the model will be after quantization.
    /// For example, quantizing from 32-bit to 8-bit gives roughly 4x compression.</para>
    /// </remarks>
    /// <param name="layer">The layer to analyze.</param>
    /// <param name="targetBitWidth">Target quantization bit width.</param>
    /// <returns>A tuple of (originalBytes, quantizedBytes, compressionRatio).</returns>
    public static (long OriginalBytes, long QuantizedBytes, double CompressionRatio) EstimateMemorySavings(
        ILayer<T> layer, int targetBitWidth)
    {
        if (layer == null) throw new ArgumentNullException(nameof(layer));

        int paramCount = layer.ParameterCount;
        long originalBytes = paramCount * 4L; // Assume 32-bit (4 bytes) per parameter
        long quantizedBytes = (long)Math.Ceiling(paramCount * targetBitWidth / 8.0);

        // Add overhead for scale/zero-point per group (if per-channel)
        int numGroups = Math.Max(1, paramCount / 128); // Assume group size 128
        quantizedBytes += numGroups * 8L; // 4 bytes for scale + 4 bytes for zero-point

        double ratio = originalBytes > 0 ? (double)originalBytes / quantizedBytes : 1.0;

        return (originalBytes, quantizedBytes, ratio);
    }

    /// <summary>
    /// Computes the quantization error (mean absolute error) that would result from quantizing a layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Measures how much accuracy you'd lose from quantization without
    /// actually applying it. Useful for deciding which bit width to use.</para>
    /// </remarks>
    /// <param name="layer">The layer to analyze.</param>
    /// <param name="bitWidth">Target bit width to simulate.</param>
    /// <returns>Mean absolute error between original and quantized parameters.</returns>
    public static double ComputeQuantizationError(ILayer<T> layer, int bitWidth)
    {
        if (layer == null) throw new ArgumentNullException(nameof(layer));

        var original = layer.GetParameters();
        var quantized = QuantizeVector(original, bitWidth);

        double totalError = 0;
        for (int i = 0; i < original.Length; i++)
        {
            double diff = Math.Abs(
                NumOps.ToDouble(original[i]) - NumOps.ToDouble(quantized[i]));
            totalError += diff;
        }

        return original.Length > 0 ? totalError / original.Length : 0;
    }

    private static Vector<T> QuantizeVector(Vector<T> parameters, int bitWidth)
    {
        if (parameters.Length == 0)
            return parameters;

        // Find min and max for scaling
        double minVal = NumOps.ToDouble(parameters[0]);
        double maxVal = minVal;
        for (int i = 1; i < parameters.Length; i++)
        {
            double val = NumOps.ToDouble(parameters[i]);
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

        double range = maxVal - minVal;
        if (range < 1e-10)
        {
            // All values are essentially the same, nothing to quantize
            var result = new Vector<T>(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
                result[i] = parameters[i];
            return result;
        }

        long levels = bitWidth >= 32 ? (1L << 31) - 1 : (1L << bitWidth) - 1;
        var quantized = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            double val = NumOps.ToDouble(parameters[i]);
            double normalized = (val - minVal) / range;
            double quantizedLevel = Math.Round(normalized * levels);
            double dequantized = (quantizedLevel / levels) * range + minVal;
            quantized[i] = NumOps.FromDouble(dequantized);
        }

        return quantized;
    }

    private static void RestoreDParameter(Vector<T> quantized, MambaBlock<T> block)
    {
        // The D parameter in MambaBlock is stored at a specific offset in the parameter vector.
        // We need to find it and restore the original values.
        var original = block.GetParameters();
        var dParam = block.GetDParameter();

        // Calculate the offset of D parameter in the flat parameter vector
        // Order: inputProj weights+bias, conv weights+bias, xProj weights, dtProj weights+bias, aLog, D, outProj weights+bias
        int innerDim = block.InnerDimension;
        int offset = block.ModelDimension * (innerDim * 2) + (innerDim * 2) +  // input proj
                      innerDim * block.ConvKernelSize + innerDim +               // conv
                      innerDim * (block.DtRank + block.StateDimension * 2) +    // x_proj
                      block.DtRank * innerDim + innerDim +                       // dt_proj
                      innerDim * block.StateDimension;                           // a_log

        // Restore D parameter values from original
        for (int i = 0; i < dParam.Length; i++)
        {
            quantized[offset + i] = original[offset + i];
        }
    }
}
