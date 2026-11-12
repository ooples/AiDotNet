using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Batch normalization fixed at its running statistics: <c>y = (x − μ) / √(σ² + ε) · γ + β</c> per channel, with μ, σ²,
/// γ and β all held constant — torchvision's <c>FrozenBatchNorm2d</c>, for a pretrained network used as a fixed feature
/// extractor.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The four statistics are persistent buffers (μ = 0, σ² = 1, γ = 1, β = 0 until a checkpoint is loaded), so the
/// layer behaves the same in training and evaluation and nothing in it is trained. Channels are axis 1 of an input of
/// rank 2 or more (<c>[batch, channels, ...]</c>).</para>
/// <para><b>For Beginners:</b> A pretrained network's normalization, frozen: it rescales each channel exactly as it
/// did when the network was trained.</para>
/// </remarks>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = false, TestInputShape = "1, 3, 4, 5", TestConstructorArgs = "3, 1e-5")]
[ElementWiseShape(Note = "A per-channel affine map; the shape is carried through.")]
[AutoParameters]
public partial class FrozenBatchNormLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly double _epsilon;

    [Buffer]
    private Tensor<T> _runningMean;
    [Buffer]
    private Tensor<T> _runningVariance;
    [Buffer]
    private Tensor<T> _scale;
    [Buffer]
    private Tensor<T> _shift;

    /// <inheritdoc />
    public override bool SupportsTraining => false;

    /// <summary>Creates the layer.</summary>
    /// <param name="channels">Channels (axis 1).</param>
    /// <param name="epsilon">ε added to the variance (1e-5, PyTorch's default).</param>
    public FrozenBatchNormLayer([LayerState] int channels, [LayerState] double epsilon = 1e-5)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        _channels = channels;
        _epsilon = epsilon;
        _runningMean = new Tensor<T>(new[] { channels });
        _runningVariance = new Tensor<T>(new[] { channels });
        _scale = new Tensor<T>(new[] { channels });
        _shift = new Tensor<T>(new[] { channels });
        for (int c = 0; c < channels; c++)
        {
            _runningVariance[c] = NumOps.One;
            _scale[c] = NumOps.One;
        }
    }

    /// <summary>The running mean μ <c>[channels]</c>.</summary>
    internal Tensor<T> RunningMean => _runningMean;

    /// <summary>The running variance σ² <c>[channels]</c>.</summary>
    internal Tensor<T> RunningVariance => _runningVariance;

    /// <summary>The scale γ <c>[channels]</c>.</summary>
    internal Tensor<T> Scale => _scale;

    /// <summary>The shift β <c>[channels]</c>.</summary>
    internal Tensor<T> Shift => _shift;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank < 2 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, ...], got [{string.Join(", ", input.Shape)}].", nameof(input));
        // y = x · a + b with a = γ / √(σ² + ε) and b = β − μ · a, broadcast along every axis but the channels.
        var a = new Tensor<T>(new[] { _channels });
        var b = new Tensor<T>(new[] { _channels });
        for (int c = 0; c < _channels; c++)
        {
            double scale = NumOps.ToDouble(_scale[c]) / Math.Sqrt(NumOps.ToDouble(_runningVariance[c]) + _epsilon);
            a[c] = NumOps.FromDouble(scale);
            b[c] = NumOps.FromDouble(NumOps.ToDouble(_shift[c]) - NumOps.ToDouble(_runningMean[c]) * scale);
        }
        var shape = new int[input.Rank];
        var tiles = new int[input.Rank];
        for (int d = 0; d < input.Rank; d++)
        {
            shape[d] = d == 1 ? _channels : 1;
            tiles[d] = d == 1 ? 1 : input.Shape[d];
        }
        return Engine.TensorAdd(Engine.TensorMultiply(input, Engine.TensorTile(Engine.Reshape(a, shape), tiles)),
            Engine.TensorTile(Engine.Reshape(b, shape), tiles));
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
        metadata["Channels"] = _channels.ToString(inv);
        metadata["Epsilon"] = _epsilon.ToString("R", inv);
        return metadata;
    }
}
