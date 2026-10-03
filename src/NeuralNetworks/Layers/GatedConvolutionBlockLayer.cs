using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Deep Voice 3's convolution block: dropout, a 1-D convolution to twice the channels, a gated linear unit, and an
/// optional residual connection scaled by √0.5.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Ping et al. 2018, §3.3, Fig. 2: "a 1-D convolution filter, a gated-linear unit as a learnable nonlinearity, a residual
/// connection to the input, and a scaling factor of √0.5"; the convolution output of 2c channels splits into the
/// input half and the gate half, <c>a ⊙ σ(b)</c>. Causal convolutions pad k − 1 steps on the left, non-causal ones
/// (k − 1)/2 on both sides; dropout is applied to the input. Weights follow the convolutional sequence-to-sequence
/// initialization the paper cites (Gehring et al. 2017; reference implementation r9y9/deepvoice3_pytorch): weight
/// normalization with directions drawn from N(0, std_mul (1 − p) / (k · c_in)).
/// </para>
/// <para>Input and output are <c>[time, channels]</c> or <c>[batch, time, channels]</c>.</para>
/// <para><b>For Beginners:</b> A convolution whose output decides, channel by channel, how much of itself to let through,
/// which keeps deep stacks of convolutions trainable.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, ChangesShape = true, TestInputShape = "1, 6, 8",
    TestConstructorArgs = "8, 8, 5, 1, false, 0.0, true, 1.0")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GatedConvolutionBlockLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _dilation;
    private readonly bool _causal;
    private readonly double _dropoutRate;
    private readonly bool _residual;
    private readonly double _stdMultiplier;

    private readonly WeightNormConv1DLayer<T> _conv;
    private readonly DropoutLayer<T>? _dropout;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the block.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels (the convolution produces twice as many before the GLU).</param>
    /// <param name="kernelSize">Odd convolution width.</param>
    /// <param name="dilation">Dilation.</param>
    /// <param name="causal">Pad only on the left so outputs never see the future (the decoder).</param>
    /// <param name="dropoutRate">Dropout on the block input.</param>
    /// <param name="residual">Add the input and scale by √0.5 (requires equal channels).</param>
    /// <param name="stdMultiplier">The initialization's std_mul (1 for the first layer after a projection, 4 after a GLU).</param>
    public GatedConvolutionBlockLayer(
        [LayerState] int inChannels,
        [LayerState] int outChannels,
        [LayerState] int kernelSize,
        [LayerState] int dilation,
        [LayerState] bool causal,
        [LayerState] double dropoutRate,
        [LayerState] bool residual,
        [LayerState] double stdMultiplier = 4.0)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernelSize <= 0 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "The width must be odd.");
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        if (residual && inChannels != outChannels) throw new ArgumentException("A residual block keeps its channel count.", nameof(residual));
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _dilation = dilation;
        _causal = causal;
        _dropoutRate = dropoutRate;
        _residual = residual;
        _stdMultiplier = stdMultiplier;

        double std = Math.Sqrt(stdMultiplier * (1.0 - dropoutRate) / (kernelSize * inChannels));
        int padding = causal ? 0 : (kernelSize - 1) / 2 * dilation;
        _conv = new WeightNormConv1DLayer<T>(inChannels, 2 * outChannels, kernelSize, dilation, padding, std);
        RegisterSubLayer(_conv);
        if (dropoutRate > 0)
        {
            _dropout = new DropoutLayer<T>(dropoutRate);
            RegisterSubLayer(_dropout);
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outChannels));
        var time = new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time));
        return inputRank switch
        {
            2 => new[] { time, features },
            3 => new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), time, features },
            _ => null,
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _inChannels)
            throw new ArgumentException($"Expected [time, {_inChannels}] or [batch, time, {_inChannels}], got [{string.Join(", ", input.Shape)}].", nameof(input));
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], _inChannels }) : input;
        int batch = x.Shape[0], time = x.Shape[1];

        var h = _dropout is null ? x : _dropout.Forward(x);
        var channelsFirst = Engine.TensorPermute(h, new[] { 0, 2, 1 }).Contiguous();
        if (_causal && _kernelSize > 1)
        {
            int pad = (_kernelSize - 1) * _dilation;
            channelsFirst = Engine.TensorConcatenate(new[] { new Tensor<T>(new[] { batch, _inChannels, pad }), channelsFirst }, 2);
        }
        var convolved = _conv.Forward(channelsFirst);                                   // [B, 2c, T]
        var a = Engine.TensorSlice(convolved, new[] { 0, 0, 0 }, new[] { batch, _outChannels, time });
        var b = Engine.TensorSlice(convolved, new[] { 0, _outChannels, 0 }, new[] { batch, _outChannels, time });
        var gated = Engine.TensorPermute(Engine.TensorMultiply(a, Engine.Sigmoid(b)), new[] { 0, 2, 1 }).Contiguous();
        var output = _residual
            ? Engine.TensorMultiplyScalar(Engine.TensorAdd(gated, x), NumOps.FromDouble(Math.Sqrt(0.5)))
            : gated;
        return unbatched ? Engine.Reshape(output, new[] { time, _outChannels }) : output;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
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
        metadata["Causal"] = _causal.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        metadata["Residual"] = _residual.ToString(inv);
        metadata["StdMultiplier"] = _stdMultiplier.ToString("R", inv);
        return metadata;
    }
}
