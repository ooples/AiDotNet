using System;
using AiDotNet.Configuration;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// A single weight-only-quantized linear projection (one weight matrix) that can be stored in INT8, FP8
/// (E4M3), or NF4 (4-bit NormalFloat) and multiplied against FP32 activations, accumulating in FP32.
/// </summary>
/// <remarks>
/// <para>
/// This is the shared quantized-matmul engine behind both <see cref="QuantizedDenseLayer"/> and
/// <see cref="QuantizedAttentionLayer"/>. Keeping the quantize + dequant-matmul logic in one place means
/// every weight-only-quantized layer honors the user-selected <see cref="InferenceQuantizationMode"/>
/// identically instead of each layer re-implementing (and diverging on) the three formats.
/// </para>
/// <para><b>For Beginners:</b> Quantization stores the model's weights in a smaller number format (8-bit
/// or 4-bit) so the model uses less memory. This type holds one such compressed weight matrix and knows how
/// to multiply it by your data, "unpacking" each weight back to a normal number on the fly.
/// </para>
/// </remarks>
internal readonly struct WeightOnlyProjection
{
    /// <summary>The quantization format this projection is stored in.</summary>
    public InferenceQuantizationMode Format { get; private init; }

    /// <summary>Number of output rows (the projection's output dimension).</summary>
    public int OutDim { get; private init; }

    /// <summary>Number of input columns (the projection's input dimension).</summary>
    public int InDim { get; private init; }

    // INT8 storage (per-row scaling): row-major [OutDim, InDim].
    private readonly sbyte[]? _int8Weights;
    private readonly float[]? _int8Scales;

    // FP8 storage (per-row scaling): row-major [OutDim, InDim] E4M3 bytes.
    private readonly byte[]? _fp8Weights;
    private readonly float[]? _fp8Scales;

    // NF4 storage (per-group scaling): packed 4-bit indices, one scale per group.
    private readonly byte[]? _nf4PackedWeights;
    private readonly float[]? _nf4GroupScales;
    private readonly int _nf4GroupSize;

    private WeightOnlyProjection(
        InferenceQuantizationMode format, int outDim, int inDim,
        sbyte[]? int8Weights, float[]? int8Scales,
        byte[]? fp8Weights, float[]? fp8Scales,
        byte[]? nf4PackedWeights, float[]? nf4GroupScales, int nf4GroupSize)
    {
        Format = format;
        OutDim = outDim;
        InDim = inDim;
        _int8Weights = int8Weights;
        _int8Scales = int8Scales;
        _fp8Weights = fp8Weights;
        _fp8Scales = fp8Scales;
        _nf4PackedWeights = nf4PackedWeights;
        _nf4GroupScales = nf4GroupScales;
        _nf4GroupSize = nf4GroupSize;
    }

    /// <summary>
    /// Quantizes a trained weight matrix stored as <paramref name="weights"/> in <c>[inDim, outDim]</c> layout
    /// (the row-major convention used by <see cref="AiDotNet.NeuralNetworks.Layers.DenseLayer{T}"/> and the
    /// attention projections) into the requested <paramref name="format"/>. <see cref="InferenceQuantizationMode.None"/>
    /// falls back to INT8 (the safest default), matching the layers' historical behavior.
    /// </summary>
    public static WeightOnlyProjection Quantize(
        Tensor<float> weights, int outDim, int inDim, InferenceQuantizationMode format)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));

        // Transpose from [inDim, outDim] to row-major [outDim, inDim] so each output row is contiguous.
        var transposed = new float[(long)outDim * inDim];
        for (int o = 0; o < outDim; o++)
        {
            for (int i = 0; i < inDim; i++)
            {
                transposed[o * inDim + i] = weights[i, o];
            }
        }

        return format switch
        {
            InferenceQuantizationMode.WeightOnlyFP8 => FromFP8(transposed, outDim, inDim),
            InferenceQuantizationMode.WeightOnlyNF4 => FromNF4(transposed, outDim, inDim),
            _ => FromInt8(transposed, outDim, inDim) // INT8 is the default (also for None).
        };
    }

    private static WeightOnlyProjection FromInt8(float[] transposed, int outDim, int inDim)
    {
        var q = Int8WeightOnlyQuantization.QuantizePerRow(transposed, rows: outDim, cols: inDim);
        return new WeightOnlyProjection(
            InferenceQuantizationMode.WeightOnlyInt8, outDim, inDim,
            int8Weights: q.Weights, int8Scales: q.Scales,
            fp8Weights: null, fp8Scales: null,
            nf4PackedWeights: null, nf4GroupScales: null, nf4GroupSize: 0);
    }

    private static WeightOnlyProjection FromFP8(float[] transposed, int outDim, int inDim)
    {
        var q = FP8WeightOnlyQuantization.QuantizePerRow(transposed, rows: outDim, cols: inDim);
        return new WeightOnlyProjection(
            InferenceQuantizationMode.WeightOnlyFP8, outDim, inDim,
            int8Weights: null, int8Scales: null,
            fp8Weights: q.Weights, fp8Scales: q.Scales,
            nf4PackedWeights: null, nf4GroupScales: null, nf4GroupSize: 0);
    }

    private static WeightOnlyProjection FromNF4(float[] transposed, int outDim, int inDim)
    {
        var q = NF4WeightOnlyQuantization.QuantizePerGroup(transposed, rows: outDim, cols: inDim);
        return new WeightOnlyProjection(
            InferenceQuantizationMode.WeightOnlyNF4, outDim, inDim,
            int8Weights: null, int8Scales: null,
            fp8Weights: null, fp8Scales: null,
            nf4PackedWeights: q.PackedWeights, nf4GroupScales: q.GroupScales, nf4GroupSize: q.GroupSize);
    }

    /// <summary>
    /// Computes <c>input[rows, InDim] @ dequant(W)[OutDim, InDim]^T (+ biases) -&gt; output[rows, OutDim]</c>,
    /// dequantizing the weights on the fly and accumulating in FP32. When <paramref name="biases"/> is supplied
    /// it is added per output column (used by dense layers; attention adds its output bias separately).
    /// </summary>
    public Tensor<float> MatMul(ReadOnlySpan<float> input, int rows, float[]? biases = null)
    {
        return Format switch
        {
            InferenceQuantizationMode.WeightOnlyFP8 => MatMulFP8(input, rows, biases),
            InferenceQuantizationMode.WeightOnlyNF4 => MatMulNF4(input, rows, biases),
            _ => MatMulInt8(input, rows, biases)
        };
    }

    private Tensor<float> MatMulInt8(ReadOnlySpan<float> input, int rows, float[]? biases)
    {
        var weights = _int8Weights
            ?? throw new InvalidOperationException("INT8 weights not initialized for quantized projection.");
        var scales = _int8Scales
            ?? throw new InvalidOperationException("INT8 scales not initialized for quantized projection.");
        var output = new Tensor<float>(new[] { rows, OutDim });

        // Routed through AiDotNet.Tensors' tiled SGEMM + AVX2 dequant primitives.
        Int8WeightOnlyMatMul.MultiplyAddBias(
            input: input,
            weightsInt8: weights,
            rowScales: scales,
            biases: biases,
            output: output.AsWritableSpan(),
            rows: rows,
            inputSize: InDim,
            outputSize: OutDim);

        return output;
    }

    private Tensor<float> MatMulFP8(ReadOnlySpan<float> input, int rows, float[]? biases)
    {
        var weights = _fp8Weights
            ?? throw new InvalidOperationException("FP8 weights not initialized for quantized projection.");
        var scales = _fp8Scales
            ?? throw new InvalidOperationException("FP8 scales not initialized for quantized projection.");
        var output = new Tensor<float>(new[] { rows, OutDim });

        for (int r = 0; r < rows; r++)
        {
            int inputBase = r * InDim;
            int outputBase = r * OutDim;
            for (int o = 0; o < OutDim; o++)
            {
                float sum = biases is not null ? biases[o] : 0f;
                float scale = scales[o];
                int wBase = o * InDim;
                for (int i = 0; i < InDim; i++)
                {
                    float w = FP8WeightOnlyQuantization.Dequantize(weights[wBase + i], scale);
                    sum += input[inputBase + i] * w;
                }
                output.SetFlat(outputBase + o, sum);
            }
        }
        return output;
    }

    private Tensor<float> MatMulNF4(ReadOnlySpan<float> input, int rows, float[]? biases)
    {
        var packed = _nf4PackedWeights
            ?? throw new InvalidOperationException("NF4 packed weights not initialized for quantized projection.");
        var groupScales = _nf4GroupScales
            ?? throw new InvalidOperationException("NF4 group scales not initialized for quantized projection.");
        int groupSize = _nf4GroupSize;
        var output = new Tensor<float>(new[] { rows, OutDim });

        for (int r = 0; r < rows; r++)
        {
            int inputBase = r * InDim;
            int outputBase = r * OutDim;
            for (int o = 0; o < OutDim; o++)
            {
                float sum = biases is not null ? biases[o] : 0f;
                int wBaseFlat = o * InDim;
                for (int i = 0; i < InDim; i++)
                {
                    int elementIdx = wBaseFlat + i;
                    int group = elementIdx / groupSize;
                    int nf4Index = NF4WeightOnlyQuantization.ExtractIndex(packed, elementIdx);
                    float w = NF4WeightOnlyQuantization.Dequantize(nf4Index, groupScales[group]);
                    sum += input[inputBase + i] * w;
                }
                output.SetFlat(outputBase + o, sum);
            }
        }
        return output;
    }
}
