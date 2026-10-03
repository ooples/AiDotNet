using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A 1-D convolution whose kernel is weight-normalized: <c>w = g · v / ‖v‖</c>, with the norm taken per output channel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Weight normalization (Salimans &amp; Kingma 2016) reparameterizes each output channel's filter as a direction
/// <c>v</c> and a length <c>g</c>, which decouples the two in optimization. WaveNet-style networks in flow-based
/// vocoders and acoustic models use it on every convolution (WaveGlow, and Glow-TTS's coupling layers, whose reference
/// implementation wraps each <c>Conv1d</c> in <c>torch.nn.utils.weight_norm</c>). As there, <c>g</c> starts at
/// <c>‖v‖</c>, so the initial kernel equals <c>v</c>.
/// </para>
/// <para>Input <c>[batch, inChannels, time]</c>; output <c>[batch, outChannels, time']</c> with the given padding and
/// dilation.</para>
/// <para><b>For Beginners:</b> An ordinary convolution, except each filter's size and direction are learned separately,
/// which makes training of deep convolution stacks better behaved.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4, 6", TestConstructorArgs = "4, 6, 3, 1, 1")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class WeightNormConv1DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _dilation;
    private readonly int _padding;
    private readonly double _initStandardDeviation;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _direction;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _length;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the convolution.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernelSize">Kernel width.</param>
    /// <param name="dilation">Dilation.</param>
    /// <param name="padding">Zero padding on each side.</param>
    /// <param name="initStandardDeviation">When positive, the direction is drawn from N(0, σ²) and the bias starts at
    /// zero (the convolutional sequence-to-sequence initialization of Gehring et al. 2017, which Deep Voice 3 uses);
    /// otherwise PyTorch's default uniform initialization.</param>
    public WeightNormConv1DLayer(
        [LayerState] int inChannels,
        [LayerState] int outChannels,
        [LayerState] int kernelSize,
        [LayerState] int dilation,
        [LayerState] int padding,
        [LayerState] double initStandardDeviation = 0.0)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _dilation = dilation;
        _padding = padding;
        _initStandardDeviation = initStandardDeviation;

        // PyTorch's Conv1d default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for kernel and bias.
        int fanIn = inChannels * kernelSize;
        double bound = 1.0 / Math.Sqrt(fanIn);
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        _direction = new Tensor<T>(new[] { outChannels, inChannels, 1, kernelSize });
        _bias = new Tensor<T>(new[] { outChannels });
        if (initStandardDeviation > 0)
        {
            for (int i = 0; i < _direction.Length; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                _direction[i] = NumOps.FromDouble(initStandardDeviation * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
            }
        }
        else
        {
            for (int i = 0; i < _direction.Length; i++) _direction[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
            for (int i = 0; i < outChannels; i++) _bias[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        }
        _length = new Tensor<T>(new[] { outChannels });
        int per = inChannels * kernelSize;
        for (int o = 0; o < outChannels; o++)
        {
            double sum = 0;
            for (int i = 0; i < per; i++)
            {
                double v = NumOps.ToDouble(_direction[o * per + i]);
                sum += v * v;
            }
            _length[o] = NumOps.FromDouble(Math.Sqrt(sum));
        }
        RegisterTrainableParameter(_direction, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_length, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <summary>Output channels.</summary>
    public int OutChannels => _outChannels;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 3
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)),
                new OutputAxisContract(TensorAxis.Time, AxisRelation.Window(TensorAxis.Time, _kernelSize, 1, _padding, _dilation)),
            }
            : null;

    /// <summary>Sets the bias and every filter's length to zero, so the convolution outputs zero (a coupling layer's
    /// identity start).</summary>
    internal void ZeroOutput()
    {
        for (int i = 0; i < _length.Length; i++) _length[i] = NumOps.Zero;
        for (int i = 0; i < _bias.Length; i++) _bias[i] = NumOps.Zero;
        Engine.InvalidatePersistentTensor(_length);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != _inChannels)
            throw new ArgumentException($"Expected [batch, {_inChannels}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        // w = g * v / ||v|| per output channel.
        var squared = Engine.TensorMultiply(_direction, _direction);
        var norm = Engine.TensorSqrt(Engine.ReduceSum(squared, new[] { 1, 2, 3 }, keepDims: true));        // [out, 1, 1, 1]
        var scale = Engine.TensorDivide(Engine.Reshape(_length, new[] { _outChannels, 1, 1, 1 }), norm);
        var kernel = Engine.TensorMultiply(_direction, Engine.TensorTile(scale, new[] { 1, _inChannels, 1, _kernelSize }));

        var input4D = Engine.Reshape(input, new[] { input.Shape[0], _inChannels, 1, input.Shape[2] });
        var conv = Engine.Conv2D(input4D, kernel, new[] { 1, 1 }, new[] { 0, _padding }, new[] { 1, _dilation });
        int batch = conv.Shape[0], time = conv.Shape[3];
        var bias = Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _outChannels, 1, 1 }), new[] { batch, 1, 1, time });
        var output = Engine.TensorAdd(conv, bias);
        return Engine.Reshape(output, new[] { batch, _outChannels, time });
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        int total = _direction.Length + _length.Length + _bias.Length;
        if (gradients.Length != total) return;
        int k = 0;
        foreach (var tensor in new[] { _direction, _length, _bias })
        {
            for (int i = 0; i < tensor.Length; i++, k++)
                tensor[i] = NumOps.Subtract(tensor[i], NumOps.Multiply(learningRate, gradients[k]));
            Engine.InvalidatePersistentTensor(tensor);
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _inChannels.ToString(inv);
        metadata["OutChannels"] = _outChannels.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["Dilation"] = _dilation.ToString(inv);
        metadata["Padding"] = _padding.ToString(inv);
        metadata["InitStandardDeviation"] = _initStandardDeviation.ToString("R", inv);
        return metadata;
    }
}
