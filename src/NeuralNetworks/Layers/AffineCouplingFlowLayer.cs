using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// An affine coupling layer with a WaveNet-like transform network (WaveGlow; Glow-TTS §3.3): the first half of the
/// channels passes through and parameterizes a shift and log-scale applied to the second half.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reference implementation <c>attentions.CouplingBlock</c> and <c>modules.WN</c>: <c>x₀, x₁ = split(x)</c>;
/// <c>h = start(x₀)</c> (weight-normalized 1×1 convolution); <paramref name="layers"/> gated layers
/// <c>a = tanh(u) ⊙ σ(v)</c> with <c>[u; v] = Dropout(conv_i(h))</c> (weight-normalized, kernel
/// <paramref name="kernelSize"/>, dilation <c>rate^i</c>), each feeding a residual and a skip output through a weight-
/// normalized 1×1 convolution (the last only a skip); <c>[m; s] = end(Σ skips)</c>, an ordinary 1×1 convolution started
/// at zero so the layer starts as the identity; <c>z₁ = m + e^s ⊙ x₁</c>, log-determinant <c>Σ s</c>.
/// </para>
/// <para><b>For Beginners:</b> Half of the data decides how to stretch and shift the other half, which keeps the step
/// exactly reversible however complicated the deciding network is.</para>
/// </remarks>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 4, 6", TestConstructorArgs = "4, 8, 3, 1, 2, 0.0")]
[ElementWiseShape(Note = "Transforms half the channels conditioned on the other half; the shape is carried through.")]
[AutoParameters]
public partial class AffineCouplingFlowLayer<T> : LayerBase<T>, IInvertibleFlowStep<T>
{
    private readonly int _channels;
    private readonly int _hidden;
    private readonly int _kernelSize;
    private readonly int _dilationRate;
    private readonly int _layers;
    private readonly double _dropoutRate;

    private readonly WeightNormConv1DLayer<T> _start;
    private readonly List<WeightNormConv1DLayer<T>> _inLayers = new();
    private readonly List<WeightNormConv1DLayer<T>> _resSkipLayers = new();
    private readonly DropoutLayer<T>? _dropout;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _endWeight;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _endBias;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the coupling layer.</summary>
    /// <param name="channels">Channels of the sequence (even).</param>
    /// <param name="hidden">Width of the transform network (192 in Glow-TTS).</param>
    /// <param name="kernelSize">Odd kernel of the gated convolutions (5).</param>
    /// <param name="dilationRate">Dilation base; layer i uses rate^i (1).</param>
    /// <param name="layers">Gated layers (4).</param>
    /// <param name="dropoutRate">Dropout on the gated convolutions' output (0.05).</param>
    public AffineCouplingFlowLayer(
        [LayerState] int channels,
        [LayerState] int hidden,
        [LayerState] int kernelSize,
        [LayerState] int dilationRate,
        [LayerState] int layers,
        [LayerState] double dropoutRate)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0 || channels % 2 != 0) throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be even.");
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (kernelSize <= 0 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd.");
        if (dilationRate <= 0) throw new ArgumentOutOfRangeException(nameof(dilationRate));
        if (layers <= 0) throw new ArgumentOutOfRangeException(nameof(layers));
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        _channels = channels;
        _hidden = hidden;
        _kernelSize = kernelSize;
        _dilationRate = dilationRate;
        _layers = layers;
        _dropoutRate = dropoutRate;

        _start = new WeightNormConv1DLayer<T>(channels / 2, hidden, 1, 1, 0);
        RegisterSubLayer(_start);
        for (int i = 0; i < layers; i++)
        {
            int dilation = (int)Math.Pow(dilationRate, i);
            int padding = (kernelSize * dilation - dilation) / 2;
            var inLayer = new WeightNormConv1DLayer<T>(hidden, 2 * hidden, kernelSize, dilation, padding);
            var resSkip = new WeightNormConv1DLayer<T>(hidden, i < layers - 1 ? 2 * hidden : hidden, 1, 1, 0);
            _inLayers.Add(inLayer);
            _resSkipLayers.Add(resSkip);
            RegisterSubLayer(inLayer);
            RegisterSubLayer(resSkip);
        }
        if (dropoutRate > 0)
        {
            _dropout = new DropoutLayer<T>(dropoutRate);
            RegisterSubLayer(_dropout);
        }
        _endWeight = new Tensor<T>(new[] { channels, hidden });   // zero: the layer starts as the identity
        _endBias = new Tensor<T>(new[] { channels });
        RegisterTrainableParameter(_endWeight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_endBias, PersistentTensorRole.Biases);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Transform(input, false).Output;

    /// <inheritdoc />
    public (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> input, bool reverse)
    {
        if (input.Rank != 3 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2], half = _channels / 2;
        var x0 = Engine.TensorSlice(input, new[] { 0, 0, 0 }, new[] { batch, half, time });
        var x1 = Engine.TensorSlice(input, new[] { 0, half, 0 }, new[] { batch, half, time });

        var h = _start.Forward(x0);
        Tensor<T>? skip = null;
        for (int i = 0; i < _layers; i++)
        {
            var u = _inLayers[i].Forward(h);
            if (_dropout is not null) u = _dropout.Forward(u);
            var acts = Engine.TensorMultiply(
                Engine.Tanh(Engine.TensorSlice(u, new[] { 0, 0, 0 }, new[] { batch, _hidden, time })),
                Engine.Sigmoid(Engine.TensorSlice(u, new[] { 0, _hidden, 0 }, new[] { batch, _hidden, time })));
            var rs = _resSkipLayers[i].Forward(acts);
            if (i < _layers - 1)
            {
                h = Engine.TensorAdd(h, Engine.TensorSlice(rs, new[] { 0, 0, 0 }, new[] { batch, _hidden, time }));
                var s = Engine.TensorSlice(rs, new[] { 0, _hidden, 0 }, new[] { batch, _hidden, time });
                skip = skip is null ? s : Engine.TensorAdd(skip, s);
            }
            else
            {
                skip = skip is null ? rs : Engine.TensorAdd(skip, rs);
            }
        }

        // end: 1x1 convolution [hidden -> channels] as a matrix product over [batch * time, hidden].
        var rows = Engine.Reshape(Engine.TensorPermute(skip!, new[] { 0, 2, 1 }).Contiguous(), new[] { batch * time, _hidden });
        var projected = Engine.TensorAdd(Engine.TensorMatMul(rows, Engine.TensorTranspose(_endWeight)),
            Engine.TensorTile(Engine.Reshape(_endBias, new[] { 1, _channels }), new[] { batch * time, 1 }));
        var output = Engine.TensorPermute(Engine.Reshape(projected, new[] { batch, time, _channels }), new[] { 0, 2, 1 }).Contiguous();
        var m = Engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { batch, half, time });
        var logs = Engine.TensorSlice(output, new[] { 0, half, 0 }, new[] { batch, half, time });

        Tensor<T> z1;
        Tensor<T>? logDet = null;
        if (reverse)
        {
            z1 = Engine.TensorMultiply(Engine.TensorSubtract(x1, m), Engine.TensorExp(Engine.TensorNegate(logs)));
        }
        else
        {
            z1 = Engine.TensorAdd(m, Engine.TensorMultiply(Engine.TensorExp(logs), x1));
            logDet = Engine.ReduceSum(logs, new[] { 0, 1, 2 }, keepDims: false);
        }
        return (Engine.TensorConcatenate(new[] { x0, z1 }, 1), logDet);
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
        metadata["Channels"] = _channels.ToString(inv);
        metadata["Hidden"] = _hidden.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["DilationRate"] = _dilationRate.ToString(inv);
        metadata["Layers"] = _layers.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        return metadata;
    }
}
