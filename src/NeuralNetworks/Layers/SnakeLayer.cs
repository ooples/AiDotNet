using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The Snake activation (Liu, Hartwig and Ueda 2020): <c>f(x) = x + (1/α) sin²(αx)</c> with a trainable frequency α per
/// channel — a periodic nonlinearity BigVGAN uses to give its generator a periodic inductive bias.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>α starts at 1 and a small constant (1e-9) keeps 1/α finite, as BigVGAN's <c>Snake</c> does. Channels are axis 1
/// of an input <c>[batch, channels, time]</c> (or axis 0 of <c>[channels, time]</c>).</para>
/// <para><b>For Beginners:</b> An activation that wiggles: it is the identity plus a learned sine-squared ripple, which
/// helps a network produce and extend periodic signals such as the harmonics of speech.</para>
/// </remarks>
[LayerCategory(LayerCategory.Activation)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 3, 5", TestConstructorArgs = "3")]
[ElementWiseShape(Note = "A per-channel activation; the shape is carried through.")]
[AutoParameters]
public partial class SnakeLayer<T> : LayerBase<T>
{
    private readonly int _channels;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _alpha;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="channels">The channels, each with its own α.</param>
    public SnakeLayer([LayerState] int channels)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        _channels = channels;
        _alpha = new Tensor<T>(new[] { channels });
        for (int c = 0; c < channels; c++) _alpha[c] = NumOps.One;
        RegisterTrainableParameter(_alpha, PersistentTensorRole.Weights);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 2;
        if (!(input.Rank == 3 || unbatched) || input.Shape[unbatched ? 0 : 1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, time] or [{_channels}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
        int batch = x.Shape[0], time = x.Shape[2];
        var alpha = Engine.TensorTile(Engine.Reshape(_alpha, new[] { 1, _channels, 1 }), new[] { batch, 1, time });
        var sine = Engine.TensorSin(Engine.TensorMultiply(x, alpha));
        var inverse = Engine.TensorPow(Engine.TensorAddScalar(alpha, NumOps.FromDouble(1e-9)), NumOps.FromDouble(-1));
        var y = Engine.TensorAdd(x, Engine.TensorMultiply(inverse, Engine.TensorMultiply(sine, sine)));
        return unbatched ? Engine.Reshape(y, input._shape) : y;
    }

    /// <summary>The channels.</summary>
    internal int Channels => _channels;

    /// <summary>Loads α (a PyTorch Snake's <c>alpha</c>, one per channel).</summary>
    internal void LoadAlpha(double[] alpha)
    {
        if (alpha.Length != _channels) throw new ArgumentException($"Expected {_channels} values, got {alpha.Length}.", nameof(alpha));
        for (int c = 0; c < _channels; c++) _alpha[c] = NumOps.FromDouble(alpha[c]);
        Engine.InvalidatePersistentTensor(_alpha);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
