using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// HiFi-GAN's generator (Kong et al. 2020 §2.1, Fig. 1; reference jik876/hifi-gan <c>Generator</c>): a weight-normalized
/// 7-tap input convolution, then per upsampling stage a leaky ReLU (0.1), a weight-normalized transposed convolution and
/// the average of |k_r| multi-receptive-field residual blocks, then a leaky ReLU (0.01), a 7-tap output convolution and
/// tanh. Input <c>[1, channels, frames]</c>, output <c>[1, 1, frames · Πu]</c>.
/// </summary>
internal sealed class HiFiGanGenerator<T>
{
    private const double Slope = 0.1;
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly NormedConv1DLayer<T> _pre;
    private readonly List<NormedConv1DLayer<T>> _ups = new();
    private readonly List<List<(List<NormedConv1DLayer<T>> First, List<NormedConv1DLayer<T>> Second)>> _blocks = new();
    private readonly NormedConv1DLayer<T> _post;
    private readonly bool _resblockType1;
    private readonly bool _tanhOutput;
    private readonly int _outputReflectPad;

    /// <param name="engine">The engine.</param>
    /// <param name="inChannels">Input channels (mel bands, or VITS's latent).</param>
    /// <param name="initialChannels">The first width; every upsampling halves it.</param>
    /// <param name="upsampleRates">Upsampling rates.</param>
    /// <param name="upsampleKernels">Transposed-convolution kernels.</param>
    /// <param name="resblockKernels">Residual block kernels.</param>
    /// <param name="resblockDilations">Residual block dilations (a type-2 block uses the first two, as
    /// <c>ResBlock2</c> does).</param>
    /// <param name="resblockType1">Type-1 (two convolutions per dilation) or type-2 blocks.</param>
    /// <param name="conditionChannels">Global condition width (VITS speakers), or 0.</param>
    /// <param name="plainEnds">VITS's plain input and bias-free output convolutions.</param>
    /// <param name="outputChannels">Output channels (1 for a waveform; iSTFTNet's magnitude and phase bins).</param>
    /// <param name="tanhOutput">Whether tanh bounds the output (false for iSTFTNet's spectral head).</param>
    /// <param name="outputReflectPad">Frames reflect-padded on the left before the output convolution (iSTFTNet: 1).</param>
    /// <param name="initialization">The draws of the reference's <c>init_weights</c> — N(0, 0.01) for the transposed
    /// convolutions, the residual blocks and (unless <paramref name="plainEnds"/>) the output convolution; null keeps
    /// PyTorch's default initialization everywhere.</param>
    public HiFiGanGenerator(IEngine engine, int inChannels, int initialChannels, int[] upsampleRates, int[] upsampleKernels,
        int[] resblockKernels, int[][] resblockDilations, bool resblockType1, int conditionChannels = 0, bool plainEnds = false,
        int outputChannels = 1, bool tanhOutput = true, int outputReflectPad = 0, Random? initialization = null)
    {
        _engine = engine;
        _resblockType1 = resblockType1;
        _tanhOutput = tanhOutput;
        _outputReflectPad = outputReflectPad;
        // VITS keeps HiFi-GAN's body but uses a plain input convolution and a plain, bias-free output one.
        _pre = Add(plainEnds
            ? new NormedConv1DLayer<T>(inChannels, initialChannels, 7, 1, 1, 1, 3, false, ConvolutionNormalization.None)
            : Conv(inChannels, initialChannels, 7, 1, 1, 3));
        if (conditionChannels > 0) Condition = Add(new NormedConv1DLayer<T>(conditionChannels, initialChannels, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
        int channels = initialChannels;
        for (int i = 0; i < upsampleRates.Length; i++)
        {
            int next = initialChannels >> (i + 1);
            _ups.Add(Add(new NormedConv1DLayer<T>(channels, next, upsampleKernels[i], upsampleRates[i], 1, 1,
                (upsampleKernels[i] - upsampleRates[i]) / 2, true, ConvolutionNormalization.Weight)));
            channels = next;
            var stage = new List<(List<NormedConv1DLayer<T>>, List<NormedConv1DLayer<T>>)>();
            for (int j = 0; j < resblockKernels.Length; j++)
            {
                int k = resblockKernels[j];
                var dilations = resblockType1 ? resblockDilations[j] : resblockDilations[j].Take(2).ToArray();
                var first = dilations.Select(d => Add(Conv(channels, channels, k, 1, d, (k * d - d) / 2))).ToList();
                var second = resblockType1
                    ? dilations.Select(_ => Add(Conv(channels, channels, k, 1, 1, (k - 1) / 2))).ToList()
                    : new List<NormedConv1DLayer<T>>();
                stage.Add((first, second));
            }
            _blocks.Add(stage);
        }
        _post = Add(plainEnds
            ? new NormedConv1DLayer<T>(channels, outputChannels, 7, 1, 1, 1, 3, false, ConvolutionNormalization.None, useBias: false)
            : Conv(channels, outputChannels, 7, 1, 1, 3));
        if (initialization is not null)
        {
            Func<double> normal = () =>
            {
                double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
                return 0.01 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            };
            // init_weights touches only the weight; the bias keeps its default draw.
            foreach (var up in _ups) up.Reinitialize(normal, keepBias: true);
            foreach (var stage in _blocks)
                foreach (var (first, second) in stage)
                    foreach (var conv in first.Concat(second)) conv.Reinitialize(normal, keepBias: true);
            if (!plainEnds) _post.Reinitialize(normal, keepBias: true);
        }
    }

    /// <summary>The 1×1 projection of a global condition added after the input convolution (VITS speaker
    /// conditioning), or null.</summary>
    public NormedConv1DLayer<T>? Condition { get; }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private NormedConv1DLayer<T> Conv(int input, int output, int kernel, int stride, int dilation, int padding)
        => new(input, output, kernel, stride, dilation, 1, padding, false, ConvolutionNormalization.Weight);

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    private Tensor<T> LeakyRelu(Tensor<T> x, double slope)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // max(x, slope · x) = slope · x + (1 − slope) · relu(x)
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(x, ops.FromDouble(slope)),
            _engine.TensorMultiplyScalar(_engine.ReLU(x), ops.FromDouble(1 - slope)));
    }

    public Tensor<T> Forward(Tensor<T> features, Tensor<T>? condition = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var x = _pre.Forward(features);
        if (condition is not null && Condition is not null)
            x = _engine.TensorAdd(x, _engine.TensorTile(Condition.Forward(condition), new[] { 1, 1, x.Shape[2] }));
        for (int i = 0; i < _ups.Count; i++)
        {
            x = _ups[i].Forward(LeakyRelu(x, Slope));
            Tensor<T>? sum = null;
            foreach (var (first, second) in _blocks[i])
            {
                var h = x;
                for (int n = 0; n < first.Count; n++)
                {
                    var t = first[n].Forward(LeakyRelu(h, Slope));
                    if (_resblockType1) t = second[n].Forward(LeakyRelu(t, Slope));
                    h = _engine.TensorAdd(t, h);
                }
                sum = sum is null ? h : _engine.TensorAdd(sum, h);
            }
            x = _engine.TensorMultiplyScalar(sum!, ops.FromDouble(1.0 / _blocks[i].Count));
        }
        // F.leaky_relu with its default slope 0.01 before the output convolution (reference Generator.forward).
        x = LeakyRelu(x, 0.01);
        if (_outputReflectPad > 0) x = VocoderOps.ReflectPad(_engine, x, _outputReflectPad, 0);
        var y = _post.Forward(x);
        return _tanhOutput ? _engine.Tanh(y) : y;
    }
}

/// <summary>
/// HiFi-GAN's discriminators (§2.2–2.3): the multi-period discriminator (periods 2, 3, 5, 7, 11; the waveform reshaped
/// to [T/p, p] and convolved along time with (5, 1) kernels, stride 3) and the multi-scale discriminator (raw, ×2 and ×4
/// average-pooled audio; grouped strided convolutions; spectral normalization on the raw scale). Each sub-discriminator
/// returns its score and its feature maps.
/// </summary>
internal sealed class HiFiGanDiscriminators<T>
{
    private readonly double _slope;
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly List<(int Period, List<NormedConv1DLayer<T>> Convs, NormedConv1DLayer<T> Post)> _periods = new();
    private readonly List<(List<NormedConv1DLayer<T>> Convs, NormedConv1DLayer<T> Post)> _scales = new();

    /// <param name="engine">The engine.</param>
    /// <param name="periods">The period discriminators' periods.</param>
    /// <param name="scales">The scale discriminators (HiFi-GAN: 3).</param>
    /// <param name="useScaleDiscriminator">Whether to build the scale discriminators.</param>
    /// <param name="widthDivisor">A divisor on every width (1: the paper's).</param>
    /// <param name="spectralFirstScale">Spectral normalization on the raw-audio scale discriminator (HiFi-GAN).</param>
    /// <param name="periodChannels">The period discriminators' widths: four stride-3 convolutions and a stride-1 one
    /// (HiFi-GAN 32, 128, 512, 1024, 1024; UnivNet 64, 128, 256, 512, 1024).</param>
    /// <param name="slope">The leaky ReLU slope (HiFi-GAN 0.1; UnivNet 0.2).</param>
    public HiFiGanDiscriminators(IEngine engine, int[] periods, int scales, bool useScaleDiscriminator = true, int widthDivisor = 1,
        bool spectralFirstScale = true, int[]? periodChannels = null, double slope = 0.1)
    {
        _engine = engine;
        _slope = slope;
        int W(int c) => Math.Max(1, c / Math.Max(1, widthDivisor));
        var widths = periodChannels ?? new[] { 32, 128, 512, 1024, 1024 };
        if (widths.Length != 5) throw new ArgumentException("A period discriminator has five convolutions.", nameof(periodChannels));
        foreach (int p in periods)
        {
            int[] channels = { 1, W(widths[0]), W(widths[1]), W(widths[2]), W(widths[3]) };
            var convs = new List<NormedConv1DLayer<T>>();
            for (int i = 0; i < 4; i++) convs.Add(Add(new NormedConv1DLayer<T>(channels[i], channels[i + 1], 5, 3, 1, 1, 2, false, ConvolutionNormalization.Weight)));
            convs.Add(Add(new NormedConv1DLayer<T>(W(widths[3]), W(widths[4]), 5, 1, 1, 1, 2, false, ConvolutionNormalization.Weight)));
            _periods.Add((p, convs, Add(new NormedConv1DLayer<T>(W(widths[4]), 1, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Weight))));
        }
        if (!useScaleDiscriminator) return;
        for (int s = 0; s < scales; s++)
        {
            var norm = s == 0 && spectralFirstScale ? ConvolutionNormalization.Spectral : ConvolutionNormalization.Weight;
            var convs = new List<NormedConv1DLayer<T>>
            {
                Add(new NormedConv1DLayer<T>(1, W(128), 15, 1, 1, 1, 7, false, norm)),
                Add(new NormedConv1DLayer<T>(W(128), W(128), 41, 2, 1, Groups(W(128), 4), 20, false, norm)),
                Add(new NormedConv1DLayer<T>(W(128), W(256), 41, 2, 1, Groups(W(128), 16), 20, false, norm)),
                Add(new NormedConv1DLayer<T>(W(256), W(512), 41, 4, 1, Groups(W(256), 16), 20, false, norm)),
                Add(new NormedConv1DLayer<T>(W(512), W(1024), 41, 4, 1, Groups(W(512), 16), 20, false, norm)),
                Add(new NormedConv1DLayer<T>(W(1024), W(1024), 41, 1, 1, Groups(W(1024), 16), 20, false, norm)),
                Add(new NormedConv1DLayer<T>(W(1024), W(1024), 5, 1, 1, 1, 2, false, norm)),
            };
            _scales.Add((convs, Add(new NormedConv1DLayer<T>(W(1024), 1, 3, 1, 1, 1, 1, false, norm))));
        }
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    // The paper's group count, reduced to the largest divisor of a narrowed channel count.
    private static int Groups(int channels, int groups)
    {
        while (channels % groups != 0) groups--;
        return groups;
    }

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    private Tensor<T> LeakyRelu(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(x, ops.FromDouble(_slope)),
            _engine.TensorMultiplyScalar(_engine.ReLU(x), ops.FromDouble(1 - _slope)));
    }

    /// <summary>Every sub-discriminator's score and feature maps for <paramref name="audio"/> <c>[samples]</c>.</summary>
    public List<(Tensor<T> Score, List<Tensor<T>> Features)> Forward(Tensor<T> audio)
    {
        var results = new List<(Tensor<T>, List<Tensor<T>>)>();
        int length = audio.Length;
        foreach (var (period, convs, post) in _periods)
        {
            // Reflect-pad to a multiple of the period, fold to [period, 1, T/p]: each column is one sequence.
            var x = audio;
            if (length % period != 0)
            {
                int pad = period - length % period;
                var index = new Tensor<int>(new[] { pad });
                for (int i = 0; i < pad; i++) index[i] = DifferentiableMel<T>.Reflect(length + i, length);
                var tail = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), index, 0), new[] { pad });
                x = _engine.TensorConcatenate(new[] { audio, tail }, 0);
            }
            int rows = x.Length / period;
            var h = _engine.Reshape(_engine.TensorTranspose(_engine.Reshape(x, new[] { rows, period })), new[] { period, 1, rows });
            var features = new List<Tensor<T>>();
            foreach (var conv in convs)
            {
                h = LeakyRelu(conv.Forward(h));
                features.Add(h);
            }
            h = post.Forward(h);
            features.Add(h);
            results.Add((h, features));
        }
        var scaled = _engine.Reshape(audio, new[] { 1, 1, length });
        for (int s = 0; s < _scales.Count; s++)
        {
            if (s > 0) scaled = AveragePool(scaled);
            var (convs, post) = _scales[s];
            var h = scaled;
            var features = new List<Tensor<T>>();
            foreach (var conv in convs)
            {
                h = LeakyRelu(conv.Forward(h));
                features.Add(h);
            }
            h = post.Forward(h);
            features.Add(h);
            results.Add((h, features));
        }
        return results;
    }

    // AvgPool1d(4, 2, padding = 2) with count_include_pad = True, as a fixed convolution.
    private Tensor<T> AveragePool(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var kernel = new Tensor<T>(new[] { 1, 1, 1, 4 });
        for (int i = 0; i < 4; i++) kernel[0, 0, 0, i] = ops.FromDouble(0.25);
        int length = x.Shape[2];
        var y = _engine.Conv2D(_engine.Reshape(x, new[] { 1, 1, 1, length }), kernel, new[] { 1, 2 }, new[] { 0, 2 }, new[] { 1, 1 });
        return _engine.Reshape(y, new[] { 1, 1, y.Shape[3] });
    }
}

/// <summary>
/// The log-mel spectrogram of a waveform on the autodiff tape (reference <c>meldataset.mel_spectrogram</c>): reflect
/// padding of (n_fft − hop) / 2, a Hann-windowed DFT of each hop-spaced frame as matrix products, magnitude
/// <c>√(re² + im² + 1e-9)</c>, librosa's Slaney mel filterbank, and <c>log(max(x, 1e-5))</c>.
/// </summary>
internal sealed class DifferentiableMel<T>
{
    private readonly IEngine _engine;
    private readonly int _fft;
    private readonly int _hop;
    private readonly Tensor<T> _cos;
    private readonly Tensor<T> _sin;
    private readonly Tensor<T> _mel;

    public DifferentiableMel(IEngine engine, int sampleRate, int fftSize, int hopSize, int windowSize, int melChannels, double fMin, double fMax)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        _engine = engine;
        _fft = fftSize;
        _hop = hopSize;
        int bins = fftSize / 2 + 1;
        // Periodic Hann window of the window length, centred in the FFT frame (torch.stft pads the window).
        var window = new double[fftSize];
        int offset = (fftSize - windowSize) / 2;
        for (int n = 0; n < windowSize; n++) window[offset + n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / windowSize);
        _cos = new Tensor<T>(new[] { fftSize, bins });
        _sin = new Tensor<T>(new[] { fftSize, bins });
        for (int n = 0; n < fftSize; n++)
            for (int k = 0; k < bins; k++)
            {
                double a = 2 * Math.PI * n * k / fftSize;
                _cos[n, k] = ops.FromDouble(window[n] * Math.Cos(a));
                _sin[n, k] = ops.FromDouble(-window[n] * Math.Sin(a));
            }
        var basis = new TacotronSpectrogram(sampleRate, fftSize, hopSize, windowSize, melChannels, fMin, fMax).MelBasis;
        _mel = new Tensor<T>(new[] { bins, melChannels });
        for (int m = 0; m < melChannels; m++)
            for (int k = 0; k < bins; k++) _mel[k, m] = ops.FromDouble(basis[m, k]);
    }

    /// <summary>The source index of position <paramref name="i"/> under reflect padding (repeated reflection when the pad
    /// exceeds the signal, where PyTorch would refuse).</summary>
    internal static int Reflect(int i, int length)
    {
        if (length == 1) return 0;
        int period = 2 * (length - 1);
        int m = ((i % period) + period) % period;
        return m < length ? m : period - m;
    }

    /// <summary>Log-mel <c>[frames, mel]</c> of <paramref name="audio"/> <c>[samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> audio) => LogMel(Magnitude(audio));

    /// <summary>Log-mel <c>[frames, mel]</c> of a magnitude spectrogram <c>[frames, bins]</c> (spec_to_mel).</summary>
    public Tensor<T> LogMel(Tensor<T> magnitude)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var mel = _engine.TensorMatMul(magnitude, _mel);
        // log(max(x, 1e-5)) = log(1e-5 + relu(x − 1e-5))
        var clipped = _engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(mel, ops.FromDouble(-1e-5))), ops.FromDouble(1e-5));
        return _engine.TensorLog(clipped);
    }

    /// <summary>The magnitude spectrogram <c>[frames, fft/2 + 1]</c> of <paramref name="audio"/> <c>[samples]</c>:
    /// reflect padding of (n_fft − hop)/2, a Hann-windowed DFT, <c>√(re² + im² + ε)</c> (reference
    /// <c>spectrogram_torch</c>, ε = 1e-6; HiFi-GAN's mel uses 1e-9, a difference below the log floor).</summary>
    public Tensor<T> Magnitude(Tensor<T> audio, double epsilon = 1e-9)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int length = audio.Length, pad = (_fft - _hop) / 2;
        var column = _engine.Reshape(audio, new[] { length, 1 });
        var head = new Tensor<int>(new[] { pad });
        var tail = new Tensor<int>(new[] { pad });
        for (int i = 0; i < pad; i++)
        {
            head[i] = Reflect(i - pad, length);
            tail[i] = Reflect(length + i, length);
        }
        var padded = _engine.TensorConcatenate(new[] { _engine.TensorIndexSelect(column, head, 0), column, _engine.TensorIndexSelect(column, tail, 0) }, 0);
        int total = length + 2 * pad, frames = 1 + (total - _fft) / _hop;
        var rows = new Tensor<T>[frames];
        for (int f = 0; f < frames; f++)
            rows[f] = _engine.Reshape(_engine.TensorSlice(padded, new[] { f * _hop, 0 }, new[] { _fft, 1 }), new[] { 1, _fft });
        var framed = frames == 1 ? rows[0] : _engine.TensorConcatenate(rows, 0);                       // [frames, fft]
        var re = _engine.TensorMatMul(framed, _cos);
        var im = _engine.TensorMatMul(framed, _sin);
        return _engine.TensorPow(_engine.TensorAddScalar(_engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im)),
            ops.FromDouble(epsilon)), ops.FromDouble(0.5));
    }
}
