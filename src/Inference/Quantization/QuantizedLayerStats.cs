using System;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Internal helpers for computing storage-byte accounting on INT8-quantized layers.
/// Used by <see cref="Int8InferenceModel"/> to report compression stats without
/// exposing the internal sbyte[] storage to public callers.
/// </summary>
internal static class QuantizedLayerStats
{
    /// <summary>
    /// Returns (int8Bytes, fp32Bytes) for a quantized dense layer.
    /// int8Bytes = sbyte weights + float32 per-row scales + float32 bias.
    /// fp32Bytes = same weights + bias if stored as float32 (matches DenseLayer storage).
    /// </summary>
    public static (long Quantized, long Original) GetBytes(QuantizedDenseLayer layer)
    {
        if (layer is null) throw new ArgumentNullException(nameof(layer));

        long numWeights = layer.WeightCount;
        long numOutRows = layer.OutputSize;
        long numBiases = layer.OutputSize;

        long quant = numWeights * sizeof(sbyte) + numOutRows * sizeof(float) + numBiases * sizeof(float);
        long orig = numWeights * sizeof(float) + numBiases * sizeof(float);
        return (quant, orig);
    }

    /// <summary>
    /// Returns (int8Bytes, fp32Bytes) for a quantized attention layer. Counts Q/K/V/O
    /// projections plus the output bias.
    /// </summary>
    public static (long Quantized, long Original) GetBytes(QuantizedAttentionLayer layer)
    {
        if (layer is null) throw new ArgumentNullException(nameof(layer));

        long totalQuant = 0;
        long totalOrig = 0;

        // Each of the four projections (Q/K/V/O) is per-row INT8 quantized.
        foreach (var (numWeights, numOutRows) in layer.GetProjectionDimensions())
        {
            totalQuant += numWeights * sizeof(sbyte) + numOutRows * sizeof(float);
            totalOrig += numWeights * sizeof(float);
        }

        // Output bias stays FP32 in both representations; it contributes equally.
        long biasBytes = layer.OutputBiasLength * sizeof(float);
        totalQuant += biasBytes;
        totalOrig += biasBytes;

        return (totalQuant, totalOrig);
    }
}
