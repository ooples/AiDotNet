using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// wav2vec 2.0's convolutional positional embedding (Baevski et al. 2020, §2; fairseq/transformers
/// <c>Wav2Vec2PositionalConvEmbedding</c>): a grouped 1-D convolution over time with weight normalization, padded to
/// keep the length (the extra frame of an even kernel removed), its output added to the features after GELU by the
/// caller.
/// </summary>
/// <remarks>
/// The weight is reparametrized as w = g·v/‖v‖ with one norm per kernel tap (PyTorch <c>weight_norm(dim=2)</c>: the
/// norm runs over the output and input channels), so g <c>[K]</c>, v <c>[C, C/G, K]</c> and the bias train, as in the
/// reference.
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class Wav2Vec2PositionalConvLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _kernel;
    private readonly int _groups;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _direction;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _magnitude;

    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    public Wav2Vec2PositionalConvLayer(int channels, int kernel, int groups)
        : base([channels], [channels])
    {
        if (channels <= 0 || kernel <= 0 || groups <= 0 || channels % groups != 0)
            throw new ArgumentException($"The channels ({channels}) must divide into {groups} groups; the kernel ({kernel}) must be positive.");
        _channels = channels;
        _kernel = kernel;
        _groups = groups;
        _direction = new Tensor<T>([channels, channels / groups, kernel]);
        _magnitude = new Tensor<T>([kernel]);
        _bias = new Tensor<T>([channels]);
        RegisterTrainableParameter(_direction, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_magnitude, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 3
            ? [new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), new(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
               new(TensorAxis.Features, AxisRelation.Fixed(_channels))]
            : null;

    public override bool SupportsTraining => true;

    /// <summary>v <c>[C, C/G, K]</c>, g <c>[K]</c> and the bias <c>[C]</c>.</summary>
    internal (Tensor<T> Direction, Tensor<T> Magnitude, Tensor<T> Bias) WeightNormParameters => (_direction, _magnitude, _bias);

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[2] != _channels)
            throw new ArgumentException($"Expected [batch, time, {_channels}], got [{string.Join(",", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], length = input.Shape[1], perGroup = _channels / _groups, pad = _kernel / 2;

        // w = g · v / ‖v‖, the norm per kernel tap over [C, C/G].
        var squares = Engine.ReduceSum(Engine.TensorMultiply(_direction, _direction), new[] { 0, 1 }, keepDims: true);   // [1, 1, K]
        var scale = Engine.TensorDivide(Engine.Reshape(_magnitude, new[] { 1, 1, _kernel }), Engine.TensorSqrt(squares));
        var weight = Engine.TensorMultiply(_direction, Engine.TensorTile(scale, new[] { _channels, perGroup, 1 }));   // [C, C/G, K]

        // Zero padding of K/2 on both sides; output length L + 2·pad − K + 1, then the last frame dropped for even K.
        var zeros = new Tensor<T>(new[] { batch, pad, _channels });
        var padded = Engine.TensorConcatenate(new[] { zeros, input, zeros }, 1);
        int frames = length + 2 * pad - _kernel + 1;
        int kept = _kernel % 2 == 0 ? frames - 1 : frames;
        var indices = new int[kept * _kernel];
        for (int t = 0; t < kept; t++)
            for (int k = 0; k < _kernel; k++) indices[t * _kernel + k] = t + k;
        var index = new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices));
        var outputs = new Tensor<T>[_groups];
        for (int g = 0; g < _groups; g++)
        {
            var slice = Engine.TensorSlice(padded, new[] { 0, 0, g * perGroup }, new[] { batch, padded.Shape[1], perGroup });
            var windows = Engine.Reshape(Engine.TensorGather(slice, index, axis: 1), new[] { batch * kept, _kernel * perGroup });
            // Columns k·(C/G) + i ↔ weight [o, i, k] of this group's outputs.
            var w = Engine.TensorSlice(weight, new[] { g * perGroup, 0, 0 }, new[] { perGroup, perGroup, _kernel });
            var matrix = Engine.Reshape(Engine.TensorPermute(w, new[] { 2, 1, 0 }), new[] { _kernel * perGroup, perGroup });
            outputs[g] = Engine.Reshape(Engine.TensorMatMul(windows, matrix), new[] { batch, kept, perGroup });
        }
        var joined = _groups == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 2);
        return Engine.TensorAdd(joined, Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, 1, _channels }), new[] { batch, kept, 1 }));
    }

    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        var parameters = GetParameters();
        if (gradients.Length != parameters.Length) return;
        for (int i = 0; i < parameters.Length; i++)
            parameters[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(learningRate, gradients[i]));
        SetParameters(parameters);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Kernel"] = _kernel.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Groups"] = _groups.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
