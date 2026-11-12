using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>UnivNet's averaged multi-period/multi-resolution LSGAN and weighted auxiliary STFT objective.</summary>
internal sealed class UnivNetTrainingLoss<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly UnivNetOptions _options;
    private readonly MultiResolutionSpectrogramDiscriminators<T> _resolutions;
    private readonly HiFiGanDiscriminators<T> _periods;
    private readonly MultiResolutionStftLoss<T> _stft;

    internal UnivNetTrainingLoss(IEngine engine, UnivNetOptions options)
    {
        _engine = engine; _options = options;
        _resolutions = new MultiResolutionSpectrogramDiscriminators<T>(engine, options.StftFftSizes,
            options.StftHopSizes, options.StftWindowSizes, options.ResolutionDiscriminatorChannels, options.LeakyReluSlope);
        _periods = new HiFiGanDiscriminators<T>(engine, options.DiscriminatorPeriods, 0, useScaleDiscriminator: false,
            widthDivisor: options.DiscriminatorWidthDivisor, periodChannels: options.PeriodDiscriminatorChannels, slope: options.LeakyReluSlope);
        _stft = new MultiResolutionStftLoss<T>(engine, options.StftFftSizes, options.StftHopSizes, options.StftWindowSizes);
        Layers = _resolutions.Layers.Concat(_periods.Layers).ToList();
    }

    internal IReadOnlyList<LayerBase<T>> Layers { get; }

    private List<Tensor<T>> Scores(Tensor<T> waveform)
        => _resolutions.Forward(waveform).Concat(_periods.Forward(waveform)).Select(item => item.Score).ToList();

    private Tensor<T> MeanSquare(Tensor<T> value)
        => _engine.ReduceMean(_engine.TensorMultiply(value, value), Enumerable.Range(0, value.Rank).ToArray(), false);

    internal Tensor<T> Generator(Tensor<T> real, Tensor<T> generated, bool adversarial)
    {
        int n = Math.Min(real.Length, generated.Length);
        var target = _engine.TensorNarrow(_engine.Reshape(real, new[] { real.Length }), 0, 0, n);
        var fake = _engine.TensorNarrow(_engine.Reshape(generated, new[] { generated.Length }), 0, 0, n);
        var (convergence, magnitude) = _stft.Forward(new[] { fake }, new[] { target });
        var loss = _engine.TensorMultiplyScalar(_engine.TensorAdd(convergence, magnitude), Ops.FromDouble(_options.StftLossWeight));
        if (!adversarial) return loss;
        var scores = Scores(fake);
        foreach (var score in scores)
            loss = _engine.TensorAdd(loss, _engine.TensorMultiplyScalar(
                MeanSquare(_engine.TensorAddScalar(score, Ops.FromDouble(-1))), Ops.FromDouble(1.0 / scores.Count)));
        return loss;
    }

    internal Tensor<T> Discriminator(Tensor<T> real, Tensor<T> generated)
    {
        int n = Math.Min(real.Length, generated.Length);
        var positive = Scores(_engine.TensorNarrow(_engine.Reshape(real, new[] { real.Length }), 0, 0, n));
        var negative = Scores(_engine.TensorNarrow(_engine.Reshape(generated, new[] { generated.Length }), 0, 0, n));
        Tensor<T>? loss = null;
        for (int i = 0; i < positive.Count; i++)
        {
            var term = _engine.TensorAdd(MeanSquare(_engine.TensorAddScalar(positive[i], Ops.FromDouble(-1))), MeanSquare(negative[i]));
            loss = loss is null ? term : _engine.TensorAdd(loss, term);
        }
        return _engine.TensorMultiplyScalar(loss!, Ops.FromDouble(1.0 / positive.Count));
    }
}
