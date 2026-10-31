using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// A grouped 1-D convolution over <c>[batch, time, channels]</c> with "same" zero padding of K/2 (the last frame dropped
/// for an even kernel), as data2vec-audio's positional convolution layers apply it: a plain weight, no weight norm.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class GroupedConv1DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _kernel;
    private readonly int _groups;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weight;

    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    /// <summary>Creates the layer with zero weights; load or initialize them before use.</summary>
    /// <param name="channels">Input and output channels.</param>
    /// <param name="kernel">Kernel width.</param>
    /// <param name="groups">Channel groups; must divide <paramref name="channels"/>.</param>
    public GroupedConv1DLayer(int channels, int kernel, int groups)
        : base([channels], [channels])
    {
        if (channels <= 0 || kernel <= 0 || groups <= 0 || channels % groups != 0)
            throw new ArgumentException($"The channels ({channels}) must divide into {groups} groups; the kernel ({kernel}) must be positive.");
        _channels = channels;
        _kernel = kernel;
        _groups = groups;
        _weight = new Tensor<T>([channels, channels / groups, kernel]);
        _bias = new Tensor<T>([channels]);
        RegisterTrainableParameter(_weight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 3
            ? [new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), new(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
               new(TensorAxis.Features, AxisRelation.Fixed(_channels))]
            : null;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>The weight <c>[C, C/G, K]</c> and bias <c>[C]</c>, PyTorch's <c>Conv1d</c> layout.</summary>
    internal (Tensor<T> Weight, Tensor<T> Bias) ConvParameters => (_weight, _bias);

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[2] != _channels)
            throw new ArgumentException($"Expected [batch, time, {_channels}], got [{string.Join(",", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], length = input.Shape[1], perGroup = _channels / _groups, pad = _kernel / 2;
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
            var w = Engine.TensorSlice(_weight, new[] { g * perGroup, 0, 0 }, new[] { perGroup, perGroup, _kernel });
            var matrix = Engine.Reshape(Engine.TensorPermute(w, new[] { 2, 1, 0 }), new[] { _kernel * perGroup, perGroup });
            outputs[g] = Engine.Reshape(Engine.TensorMatMul(windows, matrix), new[] { batch, kept, perGroup });
        }
        var joined = _groups == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 2);
        return Engine.TensorAdd(joined, Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, 1, _channels }), new[] { batch, kept, 1 }));
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        var parameters = GetParameters();
        if (gradients.Length != parameters.Length) return;
        for (int i = 0; i < parameters.Length; i++)
            parameters[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(learningRate, gradients[i]));
        SetParameters(parameters);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        // Stateless between calls: the forward keeps its intermediates as locals.
        return;
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
