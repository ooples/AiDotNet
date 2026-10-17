using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A 2-D convolution with explicit input and output channels, kernel, stride and zero padding per axis and an optional
/// bias — PyTorch's <c>nn.Conv2d</c>, with its weight layout <c>[out, in, kernelHeight, kernelWidth]</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The kernel and bias are initialized as PyTorch initializes <c>nn.Conv2d</c>: U(−1/√fan_in, 1/√fan_in) with
/// fan_in = in · kernelHeight · kernelWidth. Input is <c>[batch, in, height, width]</c> or <c>[in, height, width]</c>;
/// output is <c>[batch, out, height', width']</c> (or without the batch axis).</para>
/// <para>With <see cref="ConvolutionNormalization.Weight"/> the kernel is weight-normalized (Salimans and Kingma 2016,
/// PyTorch <c>weight_norm(Conv2d)</c>): W = g · V / ‖V‖ per output channel, g starting at ‖V‖.</para>
/// <para><b>For Beginners:</b> A small filter slides over an image-like grid (here, often a spectrogram) and responds
/// to local patterns; stride skips positions to shrink the grid.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 2, 6, 5",
    TestConstructorArgs = "2, 3, 3, 3, 2, 1, 1, 1, true, AiDotNet.NeuralNetworks.Layers.ConvolutionNormalization.Weight")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class Conv2DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _in;
    private readonly int _out;
    private readonly int _kernelHeight;
    private readonly int _kernelWidth;
    private readonly int _strideHeight;
    private readonly int _strideWidth;
    private readonly int _padHeight;
    private readonly int _padWidth;
    private readonly int _dilationHeight;
    private readonly int _dilationWidth;
    private readonly bool _useBias;
    private readonly ConvolutionNormalization _normalization;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _kernel;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _gain;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernelHeight">Kernel height.</param>
    /// <param name="kernelWidth">Kernel width.</param>
    /// <param name="strideHeight">Stride along the height.</param>
    /// <param name="strideWidth">Stride along the width.</param>
    /// <param name="padHeight">Zero padding at each end of the height.</param>
    /// <param name="padWidth">Zero padding at each end of the width.</param>
    /// <param name="useBias">Whether the layer adds a bias.</param>
    /// <param name="normalization">No normalization (the default) or weight normalization; spectral normalization is not
    /// offered for 2-D kernels.</param>
    /// <param name="dilationHeight">Dilation along the height (1).</param>
    /// <param name="dilationWidth">Dilation along the width (1).</param>
    public Conv2DLayer([LayerState] int inChannels, [LayerState] int outChannels, [LayerState] int kernelHeight, [LayerState] int kernelWidth,
        [LayerState] int strideHeight, [LayerState] int strideWidth, [LayerState] int padHeight, [LayerState] int padWidth,
        [LayerState] bool useBias = true, [LayerState] ConvolutionNormalization normalization = ConvolutionNormalization.None,
        [LayerState] int dilationHeight = 1, [LayerState] int dilationWidth = 1)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        if (normalization == ConvolutionNormalization.Spectral)
            throw new NotSupportedException("Conv2DLayer offers weight normalization or none.");
        _normalization = normalization;
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernelHeight <= 0 || kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelHeight));
        if (strideHeight <= 0 || strideWidth <= 0) throw new ArgumentOutOfRangeException(nameof(strideHeight));
        if (padHeight < 0 || padWidth < 0) throw new ArgumentOutOfRangeException(nameof(padHeight));
        if (dilationHeight <= 0 || dilationWidth <= 0) throw new ArgumentOutOfRangeException(nameof(dilationHeight));
        _dilationHeight = dilationHeight;
        _dilationWidth = dilationWidth;
        _in = inChannels;
        _out = outChannels;
        _kernelHeight = kernelHeight;
        _kernelWidth = kernelWidth;
        _strideHeight = strideHeight;
        _strideWidth = strideWidth;
        _padHeight = padHeight;
        _padWidth = padWidth;
        _useBias = useBias;

        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        double bound = 1.0 / Math.Sqrt(inChannels * kernelHeight * kernelWidth);
        _kernel = new Tensor<T>(new[] { outChannels, inChannels, kernelHeight, kernelWidth });
        for (int i = 0; i < _kernel.Length; i++) _kernel[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        _bias = new Tensor<T>(new[] { outChannels });
        if (useBias)
            for (int i = 0; i < outChannels; i++) _bias[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        int per = inChannels * kernelHeight * kernelWidth;
        _gain = new Tensor<T>(new[] { outChannels });
        for (int o = 0; o < outChannels; o++)
        {
            double sum = 0;
            for (int i = 0; i < per; i++)
            {
                double v = NumOps.ToDouble(_kernel[o * per + i]);
                sum += v * v;
            }
            _gain[o] = NumOps.FromDouble(Math.Sqrt(sum));
        }
        RegisterTrainableParameter(_kernel, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gain, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <summary>The kernel (the direction V under weight normalization) <c>[out, in, kernelHeight, kernelWidth]</c>.</summary>
    internal Tensor<T> Kernel => _kernel;

    // The effective kernel: V itself, or g · V / ‖V‖ per output channel.
    private Tensor<T> EffectiveKernel()
    {
        if (_normalization == ConvolutionNormalization.None) return _kernel;
        int per = _kernel.Length / _out;
        var v = Engine.Reshape(_kernel, new[] { _out, per });
        var norms = Engine.TensorPow(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(v, v), new[] { 1 }, keepDims: true),
            NumOps.FromDouble(1e-24)), NumOps.FromDouble(-0.5));
        var scale = Engine.TensorTile(Engine.TensorMultiply(Engine.Reshape(_gain, new[] { _out, 1 }), norms), new[] { 1, per });
        return Engine.Reshape(Engine.TensorMultiply(v, scale), _kernel._shape);
    }

    /// <summary>The bias <c>[out]</c> (zero and unused without a bias).</summary>
    internal Tensor<T> Bias => _bias;

    /// <summary>Whether the layer adds its bias.</summary>
    internal bool UsesBias => _useBias;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 3;
        if (!(input.Rank == 4 || unbatched) || input.Shape[input.Rank - 3] != _in)
            throw new ArgumentException($"Expected [batch, {_in}, height, width] or [{_in}, height, width], got [{string.Join(", ", input.Shape)}].", nameof(input));
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] }) : input;
        var y = Engine.Conv2D(x, EffectiveKernel(), new[] { _strideHeight, _strideWidth }, new[] { _padHeight, _padWidth }, new[] { _dilationHeight, _dilationWidth });
        if (_useBias)
            y = Engine.TensorAdd(y, Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _out, 1, 1 }), new[] { y.Shape[0], 1, y.Shape[2], y.Shape[3] }));
        return unbatched ? Engine.Reshape(y, new[] { _out, y.Shape[2], y.Shape[3] }) : y;
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
        metadata["InChannels"] = _in.ToString(inv);
        metadata["OutChannels"] = _out.ToString(inv);
        metadata["KernelHeight"] = _kernelHeight.ToString(inv);
        metadata["KernelWidth"] = _kernelWidth.ToString(inv);
        metadata["StrideHeight"] = _strideHeight.ToString(inv);
        metadata["StrideWidth"] = _strideWidth.ToString(inv);
        metadata["PadHeight"] = _padHeight.ToString(inv);
        metadata["PadWidth"] = _padWidth.ToString(inv);
        metadata["UseBias"] = _useBias.ToString(inv);
        metadata["Normalization"] = _normalization.ToString();
        metadata["DilationHeight"] = _dilationHeight.ToString(inv);
        metadata["DilationWidth"] = _dilationWidth.ToString(inv);
        return metadata;
    }
}
