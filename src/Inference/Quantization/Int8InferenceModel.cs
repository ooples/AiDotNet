using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Public, inference-only wrapper around a trained <see cref="NeuralNetworkBase{T}"/> that
/// rewrites compatible layers into their INT8-storage quantized counterparts and produces a
/// runnable <see cref="Predict(Tensor{float})"/> entry point.
/// </summary>
/// <remarks>
/// <para>
/// This is the "true INT8 weight-only inference" surface. The existing
/// <c>Int8Quantizer&lt;T, TInput, TOutput&gt;</c> performs fake quantization
/// (round-trips weights through FP32 storage via <c>IParameterizable.WithParameters</c>) and
/// therefore produces no inference speedup or memory reduction at runtime. This wrapper
/// instead delegates to the internal <c>InferenceOptimizer</c> path which swaps
/// <c>MultiHeadAttentionLayer&lt;float&gt;</c>, <c>GroupedQueryAttentionLayer&lt;float&gt;</c>,
/// and <c>DenseLayer&lt;float&gt;</c> in-place for their INT8-storage equivalents
/// (<see cref="QuantizedAttentionLayer"/> and <see cref="QuantizedDenseLayer"/>) that hold
/// <c>sbyte[]</c> weights with per-row scales and dequantize-on-the-fly at MatMul time.
/// </para>
/// <para>
/// The model is cloned by default so the original trained network is left untouched. Pass
/// <c>cloneModel: false</c> to mutate in place when the caller no longer needs the FP32
/// model and wants to save the deep-copy time and memory.
/// </para>
/// <para><b>For Beginners:</b> After you have trained a model, this wrapper lets you produce
/// a smaller and (on memory-bandwidth-bound matmuls) faster inference-only copy that stores
/// weights as 8-bit signed integers (one byte each) instead of 32-bit floats (four bytes each).
/// You typically lose less than 0.1% accuracy and gain ~4x reduction in weight memory.
/// </para>
/// <example>
/// <code>
/// // Train a model as normal
/// var transformer = new Transformer&lt;float&gt;(architecture, lossFn);
/// transformer.Train(features, targets);
///
/// // Wrap for INT8 inference
/// var int8 = Int8InferenceModel.FromTrained(transformer);
///
/// // Predict (uses INT8-stored weights with per-row symmetric scales)
/// var prediction = int8.Predict(input);
///
/// // Inspect quantization stats
/// Console.WriteLine($"INT8 weight bytes: {int8.QuantizedWeightBytes}");
/// Console.WriteLine($"FP32 weight bytes: {int8.OriginalWeightBytes}");
/// Console.WriteLine($"Compression ratio: {int8.CompressionRatio:F2}x");
/// </code>
/// </example>
/// </remarks>
// Internal: per the facade-pattern coding guideline, plumbing types
// stay off the public surface. The supported public entry point for
// INT8 weight-only inference is the AiModelBuilder / NeuralNetworkBase
// facade methods; this class is the in-memory representation those
// facade methods return. Issue #1342 review (PR #1348) flagged the
// previous `public` declarations as a leak of the underlying
// mutable NeuralNetworkBase<float>.
internal sealed class Int8InferenceModel
{
    private readonly NeuralNetworkBase<float> _model;
    private readonly long _quantizedWeightBytes;
    private readonly long _originalWeightBytes;
    private readonly int _quantizedLayerCount;

    private Int8InferenceModel(
        NeuralNetworkBase<float> model,
        long quantizedWeightBytes,
        long originalWeightBytes,
        int quantizedLayerCount)
    {
        _model = model;
        _quantizedWeightBytes = quantizedWeightBytes;
        _originalWeightBytes = originalWeightBytes;
        _quantizedLayerCount = quantizedLayerCount;
    }

    /// <summary>
    /// Gets the underlying neural network (with quantized layers swapped in).
    /// Use <see cref="Predict"/> for the recommended inference entry point.
    /// </summary>
    /// <remarks>
    /// Internal: leaking the mutable NeuralNetworkBase<float> on the
    /// public surface would let callers bypass the inference-only
    /// contract by training the swapped-in quantized layers. Kept
    /// internal so the facade can decide what (if anything) to expose
    /// to consumers.
    /// </remarks>
    internal NeuralNetworkBase<float> InnerModel => _model;

    /// <summary>
    /// Total bytes used by INT8 weight storage (sbyte weights + float32 scales),
    /// summed across all quantized layers.
    /// </summary>
    public long QuantizedWeightBytes => _quantizedWeightBytes;

    /// <summary>
    /// Total bytes the same weights would consume in FP32 storage. Equal to
    /// <c>sum(numWeights * sizeof(float))</c> across all quantized layers.
    /// </summary>
    public long OriginalWeightBytes => _originalWeightBytes;

    /// <summary>
    /// Number of layers whose weights were rewritten to INT8 storage. Other layers
    /// (LayerNorm, embeddings, softmax, dropout-eval, etc.) remain in FP32.
    /// </summary>
    public int QuantizedLayerCount => _quantizedLayerCount;

    /// <summary>
    /// Compression ratio of INT8 weight storage vs original FP32 weight storage. With
    /// per-row symmetric INT8 (1 byte per weight + 4 bytes per output row scale), the
    /// asymptotic ratio approaches 4.0x as <c>inDim</c> grows.
    /// </summary>
    public double CompressionRatio
        => _originalWeightBytes == 0 ? 1.0 : (double)_originalWeightBytes / _quantizedWeightBytes;

    /// <summary>
    /// Run inference. Routes through the underlying network whose attention and dense layers
    /// have been swapped for their INT8-storage equivalents. Dropout, BatchNorm running-stats,
    /// and other train/eval-sensitive layers are placed in eval mode by the
    /// <see cref="NeuralNetworkBase{T}.Predict"/> implementation.
    /// </summary>
    /// <param name="input">Input tensor. Shape contract matches the source model.</param>
    public Tensor<float> Predict(Tensor<float> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        return _model.Predict(input);
    }

    /// <summary>
    /// Builds an INT8 inference-only wrapper from a trained float network.
    /// </summary>
    /// <param name="trained">
    /// The trained source network. Its <c>MultiHeadAttentionLayer&lt;float&gt;</c>,
    /// <c>GroupedQueryAttentionLayer&lt;float&gt;</c>, and <c>DenseLayer&lt;float&gt;</c>
    /// instances will be rewritten to INT8-storage equivalents.
    /// </param>
    /// <param name="cloneModel">
    /// When <c>true</c> (default) the source model is deep-copied via <c>Clone</c> before
    /// rewriting, so the caller's training model is left untouched. Set to <c>false</c>
    /// when the caller no longer needs the original FP32 model and wants to avoid the
    /// deep-copy cost.
    /// </param>
    /// <returns>An INT8 inference-only wrapper.</returns>
    /// <exception cref="ArgumentNullException">When <paramref name="trained"/> is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// When no compatible layers were found and the caller would get zero quantization benefit.
    /// Typically indicates the network has lazy attention layers that never received a forward
    /// pass; run a single training step or warm-up <c>Predict</c> first.
    /// </exception>
    internal static Int8InferenceModel FromTrained(
        NeuralNetworkBase<float> trained,
        bool cloneModel = true)
    {
        if (trained is null) throw new ArgumentNullException(nameof(trained));

        // Build a minimal config that ONLY does INT8 weight quantization. The
        // InferenceOptimizationConfig defaults turn on FlashAttention rewriting and KV-cache
        // which would replace the underlying MultiHeadAttentionLayer<float> instances with
        // cached/paged variants before the INT8 quantization pass runs — leaving zero MHA
        // layers for ApplyWeightOnlyQuantization to find. Explicitly disable both so the
        // INT8 rewrite gets first crack at the original MHA / GQA / Dense layers.
        var config = new InferenceOptimizationConfig
        {
            EnableWeightOnlyQuantization = true,
            InferenceQuantization = InferenceQuantizationMode.WeightOnlyInt8,
            EnableFlashAttention = false,
            EnableKVCache = false,
            EnableBatching = false,
            EnableSpeculativeDecoding = false
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimizedModel, applied) = optimizer.OptimizeForInference(trained, cloneModel: cloneModel);

        if (!applied)
        {
            throw new InvalidOperationException(
                "Int8InferenceModel.FromTrained could not find any layers eligible for INT8 quantization. " +
                "This usually means the model has lazy attention/dense layers that have not yet been " +
                "shape-resolved (their parameter buffers are still zero-sized). Run one training step " +
                "or a warm-up Predict() on the source model first, then call FromTrained again.");
        }

        // Compute artifact stats by scanning the rewritten layers.
        long quantBytes = 0;
        long origBytes = 0;
        int quantLayerCount = 0;
        foreach (var layer in optimizedModel.Layers)
        {
            if (layer is QuantizedDenseLayer qDense)
            {
                var (q, o) = QuantizedLayerStats.GetBytes(qDense);
                quantBytes += q;
                origBytes += o;
                quantLayerCount++;
            }
            else if (layer is QuantizedAttentionLayer qAttn)
            {
                var (q, o) = QuantizedLayerStats.GetBytes(qAttn);
                quantBytes += q;
                origBytes += o;
                quantLayerCount++;
            }
        }

        return new Int8InferenceModel(optimizedModel, quantBytes, origBytes, quantLayerCount);
    }
}
