using AiDotNet.Audio;
using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Codecs;

/// <summary>Section 3.3: multi-scale mel L1, MPD/multi-band STFT LSGAN and feature matching.</summary>
internal sealed class CodecWaveformGan<T>
{
    private readonly IEngine _engine;
    private readonly double _melWeight, _featureWeight;
    private readonly HiFiGanDiscriminators<T> _periods;
    private readonly List<DacBandDiscriminator<T>> _bands = new();
    private readonly List<DifferentiableMel<T>> _mels = new();
    internal List<LayerBase<T>> Layers { get; } = new();
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    internal CodecWaveformGan(IEngine engine, int sampleRate, int[] melWindows, int[] melBins,
        int[] discriminatorWindows, double[] discriminatorBands, int discriminatorChannels,
        int discriminatorWidthDivisor, double melWeight = 15, double featureWeight = 2)
    {
        _engine = engine; _melWeight = melWeight; _featureWeight = featureWeight;
        _periods = new HiFiGanDiscriminators<T>(engine, new[] { 2, 3, 5, 7, 11 }, 0,
            useScaleDiscriminator: false, widthDivisor: discriminatorWidthDivisor,
            spectralFirstScale: false, slope: 0.1, padFullPeriod: true);
        Layers.AddRange(_periods.Layers);
        foreach (int w in discriminatorWindows)
            _bands.Add(new DacBandDiscriminator<T>(engine, w, discriminatorBands, discriminatorChannels, Layers));
        if (melWindows.Length != melBins.Length)
            throw new ArgumentException("Each BiCodec training mel window needs a mel bin count.");
        for (int i = 0; i < melWindows.Length; i++)
        {
            int w = melWindows[i];
            _mels.Add(new DifferentiableMel<T>(engine, sampleRate, w, w / 4, w,
                melBins[i], 0, sampleRate / 2.0));
        }
    }

    private Tensor<T> Mean(Tensor<T> value) => _engine.Reshape(_engine.ReduceMean(value,
        Enumerable.Range(0, value.Rank).ToArray(), false), new[] { 1 });
    private Tensor<T> Square(Tensor<T> value) => _engine.TensorMultiply(value, value);
    private static Tensor<T> Detached(Tensor<T> x) => new(x.Shape.ToArray(), new Vector<T>(x.ToArray()));

    private List<(Tensor<T> Score, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
    {
        var flat = _engine.Reshape(audio, new[] { audio.Length });
        var centered = _engine.TensorSubtract(flat, _engine.TensorBroadcastTo(Mean(flat), flat.Shape.ToArray()));
        var peak = _engine.ReduceMax(_engine.TensorAbs(centered), new[] { 0 }, true);
        var normalized = _engine.TensorMultiplyScalar(_engine.TensorDivide(centered,
            _engine.TensorBroadcastTo(_engine.TensorAddScalar(peak, Ops.FromDouble(1e-9)), centered.Shape.ToArray())), Ops.FromDouble(0.8));
        var result = _periods.Forward(normalized);
        foreach (var band in _bands)
        {
            var value = band.Forward(normalized);
            result.Add((value.Logits, value.Features));
        }
        return result;
    }

    internal Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var r = Discriminate(real); var g = Discriminate(Detached(generated));
        var total = new Tensor<T>(new[] { 1 });
        for (int i = 0; i < r.Count; i++)
            total = _engine.TensorAdd(total, _engine.TensorAdd(Mean(Square(g[i].Score)),
                Mean(Square(_engine.TensorAddScalar(_engine.TensorNegate(r[i].Score), Ops.One)))));
        return total;
    }

    internal Tensor<T> GeneratorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var g = Discriminate(generated);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> r;
        using (new NoGradScope<T>()) r = Discriminate(real);
        var total = new Tensor<T>(new[] { 1 });
        for (int i = 0; i < g.Count; i++)
        {
            total = _engine.TensorAdd(total, Mean(Square(_engine.TensorAddScalar(_engine.TensorNegate(g[i].Score), Ops.One))));
            // DAC excludes the final logit from feature matching.
            for (int j = 0; j < g[i].Features.Count - 1; j++)
                total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(Mean(_engine.TensorAbs(
                    _engine.TensorSubtract(g[i].Features[j], Detached(r[i].Features[j])))), Ops.FromDouble(_featureWeight)));
        }
        foreach (var mel in _mels)
        {
            Tensor<T> target;
            using (new NoGradScope<T>()) target = mel.Forward(real);
            total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(Mean(_engine.TensorAbs(
                _engine.TensorSubtract(mel.Forward(generated), target))), Ops.FromDouble(_melWeight)));
        }
        return total;
    }
}
