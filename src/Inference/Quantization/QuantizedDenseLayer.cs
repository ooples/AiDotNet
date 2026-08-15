// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Configuration;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Inference-only dense layer with weight-only quantization. Supports INT8 (per-row), FP8 (E4M3, per-row),
/// and NF4 (4-bit NormalFloat, per-group) via the shared <see cref="WeightOnlyProjection"/> engine, so a
/// FFN/dense layer honors the same <see cref="InferenceQuantizationMode"/> the user selects for attention.
/// </summary>
// FEATURE-LAST, exactly as DenseLayer declares it, and for the reason ForwardTraced states in its own
// comment: "mirror DenseLayer<T>.Forward: apply the transformation along the LAST dimension and flatten
// every leading dim (batch, sequence, ...) into a single row dimension". Quantization changes how the
// weights are STORED, never what shape the projection produces, so this layer must declare the same
// contract as the DenseLayer it replaces - a quantized swap-in that reported a different shape would
// defeat the point of being a drop-in.
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Per-position projection: the leading axes are carried through untouched.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class QuantizedDenseLayer : LayerBase<float>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// The trailing axis becomes <c>_outputSize</c> and every leading axis is carried through. Read off
    /// the two shape-restoring exits of <c>ForwardTraced</c>: rank 1 returns
    /// <c>activated.Reshape(_outputSize)</c>, and rank &gt; 2 rebuilds <c>outShape</c> by copying
    /// <c>input.Shape[i]</c> for every leading axis and setting only the last to <c>_outputSize</c>.
    /// </para>
    /// <para>
    /// <c>Fixed(_outputSize)</c> reads the field the constructor derived from the source layer's
    /// <c>GetOutputLayerShape()</c>, so a quantized copy of a differently sized dense layer reports its own
    /// width rather than a number observed once. The sequence axis is never pinned - the weight block is
    /// sized by <c>_inputSize</c> and <c>_outputSize</c> alone, so any number of positions is valid.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_outputSize <= 0) return null;

        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputSize));

        return inputRank switch
        {
            1 => new[] { features },
            2 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                features,
            },
            3 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                features,
            },
            _ => null,
        };
    }

    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly WeightOnlyProjection _proj;
    private readonly float[] _biases;

    public QuantizedDenseLayer(DenseLayer<float> source, InferenceQuantizationMode mode = InferenceQuantizationMode.WeightOnlyInt8)
        : base(
            inputShape: source.GetInputShape(),
            outputShape: source.GetOutputShape(),
            scalarActivation: source.ScalarActivation ?? new AiDotNet.ActivationFunctions.IdentityActivation<float>())
    {
        _inputSize = source.GetInputShape()[0];
        _outputSize = source.GetOutputLayerShape().RequireConcrete("Sizing a quantized dense layer's weight block")[0];

        if (source.VectorActivation != null)
            throw new InvalidOperationException("QuantizedDenseLayer scalar-activation ctor called for a vector-activation layer.");

        var weights = source.GetWeights();
        var biases = source.GetBiases();
        if (weights == null || biases == null)
            throw new ArgumentException("Dense layer must expose weights and biases.", nameof(source));

        // DenseLayer stores weights [inputSize, outputSize]; the shared engine transposes to row-major
        // [outputSize, inputSize] and quantizes in the selected format.
        _proj = WeightOnlyProjection.Quantize(weights, outDim: _outputSize, inDim: _inputSize, format: mode);
        _biases = biases.ToArray();
    }

    public QuantizedDenseLayer(DenseLayer<float> source, IVectorActivationFunction<float> vectorActivation, InferenceQuantizationMode mode = InferenceQuantizationMode.WeightOnlyInt8)
        : base(
            inputShape: source.GetInputShape(),
            outputShape: source.GetOutputShape(),
            vectorActivation: vectorActivation)
    {
        _inputSize = source.GetInputShape()[0];
        _outputSize = source.GetOutputLayerShape().RequireConcrete("Sizing a quantized dense layer's weight block")[0];

        var weights = source.GetWeights();
        var biases = source.GetBiases();
        if (weights == null || biases == null)
            throw new ArgumentException("Dense layer must expose weights and biases.", nameof(source));

        _proj = WeightOnlyProjection.Quantize(weights, outDim: _outputSize, inDim: _inputSize, format: mode);
        _biases = biases.ToArray();
    }

    public override bool SupportsTraining => false;

    public override Tensor<float>? GetWeights() => null;

    public override Tensor<float>? GetBiases() => null;

    /// <summary>The weight-only quantization format this layer's weights are stored in.</summary>
    internal InferenceQuantizationMode QuantizationFormat => _proj.Format;

    /// <summary>
    /// Total quantized weight count (rows * cols). Internal accessor used by
    /// <see cref="Int8InferenceModel"/> to compute artifact byte counts.
    /// </summary>
    internal long WeightCount => (long)_outputSize * _inputSize;

    /// <summary>
    /// Output row count. Internal accessor for stats reporting.
    /// </summary>
    internal int OutputSize => _outputSize;

    protected override Tensor<float> ForwardTraced(Tensor<float> input)
    {
        // Industry-standard dense layer rank handling — mirror DenseLayer<T>.Forward:
        // Apply the transformation along the LAST dimension and flatten every leading dim
        // (batch, sequence, ...) into a single row dimension. Without this rank-3
        // [batch, seq, embDim] inputs collapsed to [batch, seq*embDim] and broke the input
        // size invariant (regressed pre-existing tests once we wired Transformer<float> end-
        // to-end through Int8InferenceModel).
        bool inputWas1D = input.Rank == 1;
        int actualInputSize = input.Shape[^1];
        if (actualInputSize != _inputSize)
            throw new ArgumentException(
                $"QuantizedDenseLayer input size mismatch. Expected {_inputSize}, got {actualInputSize}.");

        int rowDim = 1;
        for (int i = 0; i < input.Rank - 1; i++)
            rowDim *= input.Shape[i];

        Tensor<float> flat = inputWas1D
            ? input.Reshape(1, _inputSize)
            : (input.Rank == 2 ? input : input.Reshape(rowDim, _inputSize));

        int batchSize = flat.Shape[0];

        // Weight-only dequant matmul in the layer's stored format (INT8 routes through AiDotNet.Tensors'
        // tiled SGEMM + AVX2 dequant primitives; FP8/NF4 dequantize on the fly). The bias is folded in.
        var output = _proj.MatMul(flat.AsSpan(), batchSize, _biases);

        var activated = ApplyActivation(output);
        if (inputWas1D)
        {
            return activated.Reshape(_outputSize);
        }

        // Restore the original leading-dim layout so downstream layers see the same shape
        // they would have seen from DenseLayer<T> (e.g. [batch, seq, outputSize]).
        if (input.Rank > 2)
        {
            var outShape = new int[input.Rank];
            for (int i = 0; i < input.Rank - 1; i++)
                outShape[i] = input.Shape[i];
            outShape[input.Rank - 1] = _outputSize;
            return activated.Reshape(outShape);
        }

        return activated;
    }

    public override void UpdateParameters(float learningRate)
        => throw new NotSupportedException("QuantizedDenseLayer is inference-only.");

    // UpdateParameters refused unconditionally; the base implements it properly.
    public override void ResetState()
    {
        // Inference-only; no recurrent state to clear.
    }
}
