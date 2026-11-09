using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The normalized linear-mel and bounded log-mel channels of the released codec critic.</summary>
internal sealed class UniAudioCriticSpectrogram<T>
{
    private readonly IEngine _engine;
    private readonly int _fft, _hop, _bins;
    private readonly Tensor<T> _cos, _sin, _mel;

    internal UniAudioCriticSpectrogram(IEngine engine, int sampleRate, int hop)
    {
        if (sampleRate <= 0 || hop <= 0) throw new ArgumentOutOfRangeException(nameof(hop));
        _engine = engine; _hop = hop; _fft = 4 * hop; _bins = _fft / 2 + 1;
        _cos = new Tensor<T>(new[] { _fft, _bins });
        _sin = new Tensor<T>(new[] { _fft, _bins });
        _mel = new Tensor<T>(new[] { _bins, _bins });
        for (int n = 0; n < _fft; n++)
        {
            double window = (.5 - .5 * Math.Cos(2 * Math.PI * n / _fft)) / Math.Sqrt(_fft);
            for (int k = 0; k < _bins; k++)
            {
                double angle = 2 * Math.PI * n * k / _fft;
                _cos[n, k] = N(window * Math.Cos(angle));
                _sin[n, k] = N(-window * Math.Sin(angle));
            }
        }
        double maximumMel = 2595 * Math.Log10(1 + (sampleRate / 2.0) / 700.0);
        var points = Enumerable.Range(0, _bins + 2)
            .Select(i => 700 * (Math.Pow(10, maximumMel * i / (_bins + 1) / 2595) - 1)).ToArray();
        for (int k = 0; k < _bins; k++)
            for (int m = 0; m < _bins; m++)
            {
                double frequency = (sampleRate / 2.0) * k / (double)(_bins - 1);
                double down = (frequency - points[m]) / (points[m + 1] - points[m]);
                double up = (points[m + 2] - frequency) / (points[m + 2] - points[m + 1]);
                // The author's create_fb_matrix deliberately clamps every coefficient, including its floor.
                _mel[k, m] = N(Math.Min(1, Math.Max(1e-6, Math.Min(down, up))));
            }
    }

    private static T N(double value) => UniAudioOps<T>.NumOps.FromDouble(value);

    internal Tensor<T> Forward(Tensor<T> audio)
    {
        int length = audio.Length;
        if (length <= _fft / 2) throw new ArgumentException("The critic's centered reflection padding requires a longer recording.", nameof(audio));
        int frames = 1 + length / _hop;
        var indices = new Tensor<int>(new[] { frames * _fft });
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < _fft; n++) indices[f * _fft + n] = DifferentiableMel<T>.Reflect(f * _hop + n - _fft / 2, length);
        var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), indices, 0),
            new[] { frames, _fft });
        var real = _engine.TensorMatMul(framed, _cos);
        var imaginary = _engine.TensorMatMul(framed, _sin);
        var power = _engine.TensorAdd(_engine.TensorMultiply(real, real), _engine.TensorMultiply(imaginary, imaginary));
        var magnitude = _engine.TensorPow(_engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(power, N(-1e-7))), N(1e-7)), N(.5));
        var mel = _engine.TensorMatMul(magnitude, _mel);
        // (20 log10(mel) - 20 - (-100)) / 100, clipped to [0,1].
        var logarithm = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(_engine.TensorLog(mel), N(.2 / Math.Log(10))), N(.8));
        logarithm = _engine.TensorSubtract(_engine.ReLU(logarithm), _engine.ReLU(_engine.TensorAddScalar(logarithm, N(-1))));
        Tensor<T> Channel(Tensor<T> values) => _engine.Reshape(_engine.TensorTranspose(values), new[] { 1, 1, _bins, frames });
        return _engine.TensorConcatenate(new[] { Channel(mel), Channel(logarithm) }, 1);
    }
}

/// <summary>The original seven weight-normalized convolutions, including reflection padding and pre-activation.</summary>
internal sealed class UniAudioFrequencyDiscriminator<T>
{
    private readonly IEngine _engine;
    private readonly Conv2DLayer<T>[] _convolutions;

    internal UniAudioFrequencyDiscriminator(IEngine engine, List<LayerBase<T>> owner, int hiddenChannels)
    {
        if (hiddenChannels < 32 || hiddenChannels % 32 != 0) throw new ArgumentOutOfRangeException(nameof(hiddenChannels));
        _engine = engine;
        int[] widths = [2, hiddenChannels / 32, hiddenChannels / 16, hiddenChannels / 8,
            hiddenChannels / 4, hiddenChannels / 2, hiddenChannels, 1];
        _convolutions = new Conv2DLayer<T>[7];
        for (int i = 0; i < _convolutions.Length; i++)
        {
            int stride = i is 1 or 3 or 5 ? 2 : 1;
            _convolutions[i] = new Conv2DLayer<T>(widths[i], widths[i + 1], 3, 3, stride, stride,
                0, 0, true, ConvolutionNormalization.Weight);
            owner.Add(_convolutions[i]);
        }
    }

    internal (Tensor<T> Score, Tensor<T>[] Features) Forward(Tensor<T> input)
    {
        var features = new List<Tensor<T>>();
        var value = input;
        for (int i = 0; i < _convolutions.Length; i++)
        {
            if (i > 0)
            {
                value = VocoderOps.LeakyRelu(_engine, value, .2);
                // The author's in-place activation changes the preceding stored feature map too.
                features.Add(value);
            }
            value = _convolutions[i].Forward(Reflect(value));
        }
        return (value, features.ToArray());
    }

    private Tensor<T> Reflect(Tensor<T> value)
    {
        for (int axis = 2; axis < 4; axis++)
        {
            int length = value.Shape[axis];
            if (length < 2) throw new ArgumentException("The critic's reflection padding requires at least two positions per axis.");
            var indices = new Tensor<int>(new[] { length + 2 });
            indices[0] = 1;
            for (int i = 0; i < length; i++) indices[i + 1] = i;
            indices[length + 1] = length - 2;
            // CPU TensorIndexSelect operates on matrices; move the selected axis to the front
            // and preserve the other dimensions through a reversible reshape/permutation.
            int[] shape = value._shape;
            int[] permutation = new[] { axis }.Concat(Enumerable.Range(0, 4).Where(i => i != axis)).ToArray();
            int[] movedShape = permutation.Select(i => shape[i]).ToArray();
            var rows = _engine.Reshape(_engine.TensorPermute(value, permutation), new[] { length, value.Length / length });
            rows = _engine.TensorIndexSelect(rows, indices, 0);
            movedShape[0] += 2;
            value = _engine.TensorPermute(_engine.Reshape(rows, movedShape),
                Enumerable.Range(0, 4).Select(i => Array.IndexOf(permutation, i)).ToArray());
        }
        return value;
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix) => LoadTorchWeights(TorchTensorSource.From(read), prefix);

    internal void LoadTorchWeights(TorchTensorSource source, string prefix)
    {
        for (int i = 0; i < _convolutions.Length; i++)
        {
            var layer = _convolutions[i];
            string name = prefix + $"discriminator.{i}.{(i == 0 ? 1 : 2)}.";
            int output = layer.Kernel.Shape[0];
            // The critic's convolutions are small; their parameter vector is direction, gain, then bias.
            var values = source.Read(name + "weight_v", layer.Kernel.Shape.ToArray())
                .Concat(source.Read(name + "weight_g", new[] { output, 1, 1, 1 }))
                .Concat(source.Read(name + "bias", new[] { output }));
            layer.SetParameters(new Vector<T>(values.Select(UniAudioOps<T>.NumOps.FromDouble)));
        }
    }
}

internal sealed record UniAudioCriticOutput<T>(Tensor<T>[] Scores, Tensor<T>[][] Features);

/// <summary>UniAudio's mel-based least-squares GAN and full/sub-band spectral reconstruction objective.</summary>
internal sealed class UniAudioCodecGan<T>
{
    private readonly IEngine _engine;
    private readonly UniAudioCodecTrainingOptions _options;
    private readonly UniAudioCriticSpectrogram<T>[] _spectrograms;
    private readonly UniAudioFrequencyDiscriminator<T>[] _critics;
    private readonly Vocoders.MultiResolutionStftLoss<T> _full, _sub;
    private readonly PseudoQmf<T> _pqmf;
    internal List<LayerBase<T>> Layers { get; } = new();

    internal UniAudioCodecGan(IEngine engine, int sampleRate, UniAudioCodecTrainingOptions options)
    {
        _engine = engine; _options = options;
        if (options.DiscriminatorHops.Length == 0 || options.DiscriminatorHops.Length != options.DiscriminatorChannels.Length)
            throw new ArgumentException("Each mel critic requires a hop and hidden width.", nameof(options));
        _spectrograms = options.DiscriminatorHops.Select(hop => new UniAudioCriticSpectrogram<T>(engine, sampleRate, hop)).ToArray();
        _critics = options.DiscriminatorChannels.Select(width => new UniAudioFrequencyDiscriminator<T>(engine, Layers, width)).ToArray();
        _full = new Vocoders.MultiResolutionStftLoss<T>(engine, options.FullFftSizes, options.FullHopSizes, options.FullWindowSizes);
        _sub = new Vocoders.MultiResolutionStftLoss<T>(engine, options.SubFftSizes, options.SubHopSizes, options.SubWindowSizes);
        double cutoff = options.Subbands switch { 2 => .25, 4 => .13, 6 => .10032791, 8 => .07949452,
            _ => throw new ArgumentException("The released PQMF supports two, four, six or eight bands.", nameof(options)) };
        _pqmf = new PseudoQmf<T>(engine, options.Subbands, 62, cutoff, 9);
    }

    private static T N(double value) => UniAudioOps<T>.NumOps.FromDouble(value);
    private Tensor<T> Mean(Tensor<T> value) => _engine.ReduceMean(value, Enumerable.Range(0, value.Rank).ToArray(), false);
    private Tensor<T> Mse(Tensor<T> value, double target)
    {
        var difference = _engine.TensorAddScalar(value, N(-target));
        return Mean(_engine.TensorMultiply(difference, difference));
    }

    internal UniAudioCriticOutput<T> Forward(Tensor<T> audio)
    {
        var scores = new Tensor<T>[_critics.Length];
        var features = new Tensor<T>[_critics.Length][];
        for (int i = 0; i < _critics.Length; i++)
            (scores[i], features[i]) = _critics[i].Forward(_spectrograms[i].Forward(audio));
        return new UniAudioCriticOutput<T>(scores, features);
    }

    internal Tensor<T> AdversarialGeneratorObjective(UniAudioCriticOutput<T> real, UniAudioCriticOutput<T> fake)
    {
        Tensor<T>? adversarial = null, matching = null;
        int count = 0;
        for (int i = 0; i < fake.Scores.Length; i++)
        {
            var term = Mse(fake.Scores[i], 1);
            adversarial = adversarial is null ? term : _engine.TensorAdd(adversarial, term);
            for (int j = 0; j < fake.Features[i].Length; j++)
            {
                var target = new Tensor<T>(real.Features[i][j].Shape.ToArray(), new Vector<T>(real.Features[i][j].ToArray()));
                term = Mean(_engine.TensorAbs(_engine.TensorSubtract(fake.Features[i][j], target)));
                matching = matching is null ? term : _engine.TensorAdd(matching, term);
                count++;
            }
        }
        var result = _engine.TensorMultiplyScalar(adversarial!, N(1.0 / fake.Scores.Length));
        return matching is null ? result : _engine.TensorAdd(result,
            _engine.TensorMultiplyScalar(matching, N(_options.FeatureMatchingWeight / count)));
    }

    internal Tensor<T> DiscriminatorObjective(UniAudioCriticOutput<T> real, UniAudioCriticOutput<T> fake)
    {
        Tensor<T>? loss = null;
        for (int i = 0; i < fake.Scores.Length; i++)
        {
            var term = _engine.TensorAdd(Mse(real.Scores[i], 1), Mse(fake.Scores[i], 0));
            loss = loss is null ? term : _engine.TensorAdd(loss, term);
        }
        return _engine.TensorMultiplyScalar(loss!, N(1.0 / fake.Scores.Length));
    }

    internal Tensor<T> ReconstructionObjective(Tensor<T> real, Tensor<T> generated)
    {
        var (fullSc, fullLog) = _full.Forward(new[] { generated }, new[] { real });
        Tensor<T>[] Bands(Tensor<T> audio)
        {
            var bands = _pqmf.Analysis(_engine.Reshape(audio, new[] { 1, 1, audio.Length }));
            int length = bands.Shape[2];
            return Enumerable.Range(0, _options.Subbands).Select(band => _engine.Reshape(
                _engine.TensorSlice(bands, new[] { 0, band, 0 }, new[] { 1, 1, length }), new[] { length })).ToArray();
        }
        var (subSc, subLog) = _sub.ForwardJointBatch(Bands(generated), Bands(real));
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(_engine.TensorAdd(fullSc, fullLog), N(_options.FullStftWeight)),
            _engine.TensorMultiplyScalar(_engine.TensorAdd(subSc, subLog), N(_options.SubStftWeight)));
    }

    internal Tensor<T> GeneratorObjective(Tensor<T> real, Tensor<T> generated)
    {
        UniAudioCriticOutput<T> reference;
        using (new NoGradScope<T>()) reference = Forward(real);
        return _engine.TensorAdd(ReconstructionObjective(real, generated), AdversarialGeneratorObjective(reference, Forward(generated)));
    }

    internal Tensor<T> DiscriminatorObjective(Tensor<T> real, Tensor<T> generated) => DiscriminatorObjective(Forward(real), Forward(generated));

    internal void LoadTorchWeights(TorchTensorSource source, string prefix = "mfd.")
    {
        for (int i = 0; i < _critics.Length; i++) _critics[i].LoadTorchWeights(source, prefix + $"discriminators.{i}.");
    }
}
