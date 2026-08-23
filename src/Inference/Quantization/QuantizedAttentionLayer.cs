// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Inference-only attention layer with quantized Q/K/V/O projection weights.
/// Supports INT8, FP8 (E4M3), and NF4 quantization modes.
/// </summary>
/// <remarks>
/// <para>
/// This layer takes a trained <see cref="MultiHeadAttentionLayer{T}"/> or
/// <see cref="GroupedQueryAttentionLayer{T}"/>, extracts its projection weights, and
/// quantizes them using the selected format. During inference the weights are dequantized
/// on the fly and multiplied with FP32 activations, accumulating in FP32.
/// </para>
/// <para><b>For Beginners:</b> Quantization compresses the model weights from 32-bit
/// floating point to a smaller format (8-bit or 4-bit), reducing memory usage while keeping
/// nearly identical accuracy. This layer replaces the original attention layer at inference
/// time so you get faster prediction with less memory.
/// </para>
/// </remarks>
// Shape-preserving, at the two ranks ForwardTraced handles: it reads seqLen from Shape[rank-2] and embDim
// from Shape[rank-1], then rebuilds the result as [seqLen, _embeddingDimension] for rank 2 and as the
// original leading axes with [seqLen, _embeddingDimension] appended for rank 3. Same roles as the
// MultiHeadAttentionLayer / GroupedQueryAttentionLayer this replaces - quantization compresses the stored
// weights, it does not change what the attention block emits, so a differing contract here would be a bug
// in the swap-in rather than a property worth declaring.
//
// No hand-written OutputAxesFor: every axis is carried through, so the generated Same(role) per axis is
// the whole relation. The feature axis is Same rather than Fixed(_embeddingDimension) deliberately -
// the output projection's OutDim equals the width the input had to arrive with (the Q/K/V projections
// take InDim = _embeddingDimension), so the layer returns the width it was given rather than setting one.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class QuantizedAttentionLayer : LayerBase<float>, IShapeContract
{
    private readonly int _embeddingDimension;
    private readonly int _headCount;
    private readonly int _headDimension;
    private readonly int _numKVHeads;
    private readonly bool _isGQA;
    private readonly InferenceQuantizationMode _format;

    // One projection per weight matrix
    private readonly WeightOnlyProjection _qProj;
    private readonly WeightOnlyProjection _kProj;
    private readonly WeightOnlyProjection _vProj;
    private readonly WeightOnlyProjection _oProj;

    // Bias (kept in FP32)
    private readonly float[] _outputBias;

    // Positional encoding settings preserved from source
    private readonly PositionalEncodingType _positionalEncoding;
    private readonly RotaryPositionalEncodingLayer<float>? _ropeLayer;
    private readonly ALiBiPositionalBiasLayer<float>? _alibiLayer;

    /// <summary>Construction state: the 'source' the layer was built with.</summary>
    // The quantized projections own the inference weights after construction. These references
    // are provenance only; traversing them as child layers would expose the original full-precision
    // attention weights through an inference-only layer whose parameter contract is intentionally 0.
    [ExternalState]
    private readonly AiDotNet.NeuralNetworks.Layers.MultiHeadAttentionLayer<float> _source = null!;
    // The two constructors take different source layer types, which cannot share one field.
    [ExternalState]
    private readonly AiDotNet.NeuralNetworks.Layers.GroupedQueryAttentionLayer<float> _sourceGrouped = null!;

    /// <summary>
    /// Creates a quantized attention layer from a trained <see cref="MultiHeadAttentionLayer{T}"/>.
    /// </summary>
    /// <param name="source">The source MHA layer to quantize.</param>
    /// <param name="mode">The quantization format to use (default: INT8).</param>
    public QuantizedAttentionLayer(
        MultiHeadAttentionLayer<float> source,
        InferenceQuantizationMode mode = InferenceQuantizationMode.WeightOnlyInt8)
        : base(
            inputShape: source.GetInputShape(),
            outputShape: source.GetOutputShape())
    {
        _source = source;
        _headCount = source.HeadCount;
        _embeddingDimension = source.GetInputShape()[^1];
        _headDimension = _embeddingDimension / _headCount;
        _numKVHeads = _headCount;
        _isGQA = false;
        _positionalEncoding = source.PositionalEncoding;
        _format = mode == InferenceQuantizationMode.None
            ? InferenceQuantizationMode.WeightOnlyInt8
            : mode;

        if (!source.IsShapeResolved)
            source.ResolveFromShape(new[] { 1, _embeddingDimension });

        _qProj = WeightOnlyProjection.Quantize(source.GetQueryWeights(), _embeddingDimension, _embeddingDimension, _format);
        _kProj = WeightOnlyProjection.Quantize(source.GetKeyWeights(), _embeddingDimension, _embeddingDimension, _format);
        _vProj = WeightOnlyProjection.Quantize(source.GetValueWeights(), _embeddingDimension, _embeddingDimension, _format);
        _oProj = WeightOnlyProjection.Quantize(source.GetOutputWeights(), _embeddingDimension, _embeddingDimension, _format);

        _outputBias = ExtractBias(source);

        (_ropeLayer, _alibiLayer) = CreatePositionalEncodingLayers(
            _positionalEncoding, _headDimension, _headCount, source.GetInputShape()[0], source.RoPETheta);
    }

    /// <summary>
    /// Creates a quantized attention layer from a trained <see cref="GroupedQueryAttentionLayer{T}"/>.
    /// </summary>
    /// <param name="source">The source GQA layer to quantize.</param>
    /// <param name="mode">The quantization format to use (default: INT8).</param>
    public QuantizedAttentionLayer(
        // Not [LayerState]: this layer is non-generic, and ADN0055 reports that the factory only
        // builds layers with one type parameter, so marking its arguments cannot produce one.
        GroupedQueryAttentionLayer<float> source,
        InferenceQuantizationMode mode = InferenceQuantizationMode.WeightOnlyInt8)
        : base(
            inputShape: source.GetInputShape(),
            outputShape: source.GetOutputShape())
    {
        _sourceGrouped = source;
        _headCount = source.NumHeads;
        _numKVHeads = source.NumKVHeads;
        _headDimension = source.HeadDimension;
        _embeddingDimension = source.GetInputShape()[^1];
        _isGQA = true;
        _positionalEncoding = source.PositionalEncoding;
        _format = mode == InferenceQuantizationMode.None
            ? InferenceQuantizationMode.WeightOnlyInt8
            : mode;

        int qOutDim = _headCount * _headDimension;
        int kvOutDim = _numKVHeads * _headDimension;

        _qProj = WeightOnlyProjection.Quantize(source.GetQueryWeights(), qOutDim, _embeddingDimension, _format);
        _kProj = WeightOnlyProjection.Quantize(source.GetKeyWeights(), kvOutDim, _embeddingDimension, _format);
        _vProj = WeightOnlyProjection.Quantize(source.GetValueWeights(), kvOutDim, _embeddingDimension, _format);
        _oProj = WeightOnlyProjection.Quantize(source.GetOutputWeights(), _embeddingDimension, _headCount * _headDimension, _format);

        _outputBias = ExtractBiasFromGQA(source);

        (_ropeLayer, _alibiLayer) = CreatePositionalEncodingLayers(
            _positionalEncoding, _headDimension, _headCount, source.GetInputShape()[0], source.RoPETheta);
    }

    public override bool SupportsTraining => false;

    public override Tensor<float>? GetWeights() => null;

    public override Tensor<float>? GetBiases() => null;

    /// <summary>Gets the number of query heads.</summary>
    public int HeadCount => _headCount;

    /// <summary>Gets the number of KV heads.</summary>
    public int KVHeadCount => _numKVHeads;

    /// <summary>Gets whether this is a GQA layer.</summary>
    public bool IsGQA => _isGQA;

    /// <summary>Gets the positional encoding type.</summary>
    public PositionalEncodingType PositionalEncoding => _positionalEncoding;

    /// <summary>Gets the quantization format used.</summary>
    public InferenceQuantizationMode QuantizationFormat => _format;

    /// <summary>
    /// Enumerates (numWeights, numOutRows) for each projection (Q, K, V, O).
    /// Internal accessor used by <see cref="Int8InferenceModel"/> for storage-byte accounting.
    /// </summary>
    internal IEnumerable<(long NumWeights, int NumOutRows)> GetProjectionDimensions()
    {
        yield return ((long)_qProj.OutDim * _qProj.InDim, _qProj.OutDim);
        yield return ((long)_kProj.OutDim * _kProj.InDim, _kProj.OutDim);
        yield return ((long)_vProj.OutDim * _vProj.InDim, _vProj.OutDim);
        yield return ((long)_oProj.OutDim * _oProj.InDim, _oProj.OutDim);
    }

    /// <summary>
    /// Length of the output bias vector. Internal accessor for stats reporting.
    /// </summary>
    internal int OutputBiasLength => _outputBias.Length;

    protected override Tensor<float> ForwardTraced(Tensor<float> input)
    {
        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int embDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? input.Reshape(1, seqLen, embDim)
            : input.Reshape(batchSize, seqLen, embDim);

        // Flatten to 2D for projection: [batch*seq, embDim]
        var input2D = input3D.Reshape(batchSize * seqLen, embDim);
        var inputSpan = input2D.AsSpan();

        // Dequantize and project Q, K, V
        var qFlat = _qProj.MatMul(inputSpan, batchSize * seqLen);
        var kFlat = _kProj.MatMul(inputSpan, batchSize * seqLen);
        var vFlat = _vProj.MatMul(inputSpan, batchSize * seqLen);

        // Reshape Q to [batch, numHeads, seq, headDim]
        var queries = ReshapeToHeads(qFlat, batchSize, seqLen, _headCount, _headDimension);
        // K/V to [batch, numKVHeads, seq, headDim]
        var keys = ReshapeToHeads(kFlat, batchSize, seqLen, _numKVHeads, _headDimension);
        var values = ReshapeToHeads(vFlat, batchSize, seqLen, _numKVHeads, _headDimension);

        // Apply RoPE if configured
        if (_ropeLayer != null)
        {
            (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: 0);
        }

        // Expand KV heads for GQA
        if (_isGQA && _numKVHeads < _headCount)
        {
            int headsPerGroup = _headCount / _numKVHeads;
            keys = ExpandKVHeads(keys, batchSize, seqLen, _numKVHeads, headsPerGroup, _headCount);
            values = ExpandKVHeads(values, batchSize, seqLen, _numKVHeads, headsPerGroup, _headCount);
        }

        // Compute attention (with ALiBi bias if configured)
        var context = _alibiLayer != null
            ? FlashAttention<float>.Forward(queries, keys, values, FlashAttentionConfig.Default,
                attentionBias: _alibiLayer.ComputeBias(seqLen, seqLen)).Output
            : ComputeStandardAttention(queries, keys, values);

        // Reshape: [batch, numHeads, seq, headDim] -> [batch*seq, numHeads*headDim]
        var contextFlat = new Tensor<float>(new[] { batchSize * seqLen, _headCount * _headDimension });
        var contextSource = context.AsSpan();
        var contextDestination = contextFlat.AsWritableSpan();
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int rowIdx = b * seqLen + s;
                for (int h = 0; h < _headCount; h++)
                {
                    int sourceOffset = ((b * _headCount + h) * seqLen + s) * _headDimension;
                    int destinationOffset = (rowIdx * _headCount + h) * _headDimension;
                    contextSource.Slice(sourceOffset, _headDimension)
                        .CopyTo(contextDestination.Slice(destinationOffset, _headDimension));
                }
            }
        }

        // Output projection with quantized weights
        var contextSpan = contextFlat.AsSpan();
        var outputFlat = _oProj.MatMul(contextSpan, batchSize * seqLen);

        // Add bias
        var output = new Tensor<float>(new[] { batchSize, seqLen, _embeddingDimension });
        var projected = outputFlat.AsSpan();
        var outputDestination = output.AsWritableSpan();
        for (int rowIdx = 0; rowIdx < batchSize * seqLen; rowIdx++)
        {
            int rowOffset = rowIdx * _embeddingDimension;
            for (int e = 0; e < _embeddingDimension; e++)
            {
                outputDestination[rowOffset + e] = projected[rowOffset + e] + _outputBias[e];
            }
        }

        // Reshape back to original rank
        if (rank == 2)
            return output.Reshape(seqLen, _embeddingDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _embeddingDimension;
        return output.Reshape(outputShape);
    }

    public override void UpdateParameters(float learningRate)
        => throw new NotSupportedException("QuantizedAttentionLayer is inference-only.");

    // UpdateParameters refused unconditionally; the base implements it properly.
    public override void ResetState()
    {
        // Inference-only; no recurrent state to clear.
    }

    #region Private Helpers

    private static float[] ExtractBias(MultiHeadAttentionLayer<float> source)
    {
        var params1 = source.GetParameters();
        int embDim = source.GetInputShape()[^1];
        int biasStart = params1.Length - embDim;
        var bias = new float[embDim];
        for (int i = 0; i < embDim; i++)
        {
            bias[i] = params1[biasStart + i];
        }
        return bias;
    }

    private static float[] ExtractBiasFromGQA(GroupedQueryAttentionLayer<float> source)
    {
        var params1 = source.GetParameters();
        int embDim = source.GetInputShape()[^1];
        int biasStart = params1.Length - embDim;
        var bias = new float[embDim];
        for (int i = 0; i < embDim; i++)
        {
            bias[i] = params1[biasStart + i];
        }
        return bias;
    }

    private static (RotaryPositionalEncodingLayer<float>?, ALiBiPositionalBiasLayer<float>?) CreatePositionalEncodingLayers(
        PositionalEncodingType encodingType, int headDimension, int numHeads, int maxSequenceLength,
        double ropeTheta = 10000.0)
    {
        return encodingType switch
        {
            PositionalEncodingType.None => (null, null),
            PositionalEncodingType.Rotary => (
                new RotaryPositionalEncodingLayer<float>(maxSequenceLength, headDimension, ropeTheta), null),
            PositionalEncodingType.ALiBi => (
                null, new ALiBiPositionalBiasLayer<float>(numHeads, maxSequenceLength)),
            _ => throw new ArgumentOutOfRangeException(nameof(encodingType),
                encodingType, $"Unsupported positional encoding type: {encodingType}. " +
                "Supported types are None, Rotary (RoPE), and ALiBi.")
        };
    }

    private static Tensor<float> ReshapeToHeads(Tensor<float> flat, int batchSize, int seqLen, int numHeads, int headDim)
    {
        var result = new Tensor<float>(new[] { batchSize, numHeads, seqLen, headDim });
        var source = flat.AsSpan();
        var destination = result.AsWritableSpan();
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int rowIdx = b * seqLen + s;
                for (int h = 0; h < numHeads; h++)
                {
                    int sourceOffset = (rowIdx * numHeads + h) * headDim;
                    int destinationOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    source.Slice(sourceOffset, headDim)
                        .CopyTo(destination.Slice(destinationOffset, headDim));
                }
            }
        }
        return result;
    }

    private static Tensor<float> ExpandKVHeads(
        Tensor<float> kv, int batchSize, int seqLen,
        int numKVHeads, int headsPerGroup, int totalHeads)
    {
        var expanded = new Tensor<float>(new[] { batchSize, totalHeads, seqLen, kv.Shape[3] });
        int headDim = kv.Shape[3];
        var source = kv.AsSpan();
        var destination = expanded.AsWritableSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int kvh = 0; kvh < numKVHeads; kvh++)
            {
                for (int g = 0; g < headsPerGroup; g++)
                {
                    int qh = kvh * headsPerGroup + g;
                    for (int s = 0; s < seqLen; s++)
                    {
                        int sourceOffset = ((b * numKVHeads + kvh) * seqLen + s) * headDim;
                        int destinationOffset = ((b * totalHeads + qh) * seqLen + s) * headDim;
                        source.Slice(sourceOffset, headDim)
                            .CopyTo(destination.Slice(destinationOffset, headDim));
                    }
                }
            }
        }
        return expanded;
    }

    private Tensor<float> ComputeStandardAttention(Tensor<float> queries, Tensor<float> keys, Tensor<float> values)
    {
        int batchSize = queries.Shape[0];
        int numHeads = queries.Shape[1];
        int seqLenQ = queries.Shape[2];
        int seqLenKV = keys.Shape[2];
        int headDim = queries.Shape[3];

        float scale = 1f / MathF.Sqrt(headDim);
        var output = new Tensor<float>(new[] { batchSize, numHeads, seqLenQ, headDim });
        var querySpan = queries.AsSpan();
        var keySpan = keys.AsSpan();
        var valueSpan = values.AsSpan();
        var outputSpan = output.AsWritableSpan();
        var scores = new float[seqLenKV];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < seqLenQ; i++)
                {
                    float maxScore = float.NegativeInfinity;
                    int queryOffset = ((b * numHeads + h) * seqLenQ + i) * headDim;

                    for (int j = 0; j < seqLenKV; j++)
                    {
                        float dot = 0f;
                        int keyOffset = ((b * numHeads + h) * seqLenKV + j) * headDim;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += querySpan[queryOffset + d] * keySpan[keyOffset + d];
                        }
                        scores[j] = dot * scale;
                        if (scores[j] > maxScore) maxScore = scores[j];
                    }

                    float sumExp = 0f;
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        scores[j] = MathF.Exp(scores[j] - maxScore);
                        sumExp += scores[j];
                    }

                    int outputOffset = ((b * numHeads + h) * seqLenQ + i) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0f;
                        for (int j = 0; j < seqLenKV; j++)
                        {
                            int valueOffset = ((b * numHeads + h) * seqLenKV + j) * headDim;
                            sum += (scores[j] / sumExp) * valueSpan[valueOffset + d];
                        }
                        outputSpan[outputOffset + d] = sum;
                    }
                }
            }
        }
        return output;
    }

    #endregion
}
