using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of CosyVoice's HiFT vocoder (the reference's <c>HiFTGenerator</c>).</summary>
internal sealed record CosyVoiceHiFTConfiguration(
    int MelChannels = 80, int BaseChannels = 512, int Harmonics = 8, int SampleRate = 22050, double SineAmplitude = 0.1,
    double NoiseStd = 0.003, double VoicedThreshold = 10, int[]? UpsampleRates = null, int[]? UpsampleKernels = null,
    int FftSize = 16, int HopSize = 4, int[]? ResblockKernels = null, int[][]? ResblockDilations = null,
    int[]? SourceResblockKernels = null, int[][]? SourceResblockDilations = null, double LeakySlope = 0.1,
    double AudioLimit = 0.99, int F0Channels = 512)
{
    public int[] Upsample => UpsampleRates ?? new[] { 8, 8 };
    public int[] UpsampleKernel => UpsampleKernels ?? new[] { 16, 16 };
    public int[] Kernels => ResblockKernels ?? new[] { 3, 7, 11 };
    public int[][] Dilations => ResblockDilations ?? new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };
    public int[] SourceKernels => SourceResblockKernels ?? new[] { 7, 11 };
    public int[][] SourceDilations => SourceResblockDilations ?? new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>Waveform samples per mel frame: the product of the upsampling rates times the iSTFT hop.</summary>
    public int SamplesPerFrame => Upsample.Aggregate(1, (a, b) => a * b) * HopSize;
}

/// <summary>
/// HiFT, the vocoder CosyVoice's code ships (Li et al. 2023, "HiFTNet", as the reference's <c>HiFTGenerator</c>): a
/// HiFi-GAN body whose last layer predicts an iSTFT magnitude and phase, driven by a neural source filter — an F0
/// predictor (five weight-normalized convolutions with ELU, a linear layer and |·|), a harmonic sine source (SineGen) from
/// that F0 merged by a linear layer and tanh, whose STFT is downsampled into each upsampling stage.
/// </summary>
/// <remarks>
/// The residual blocks use Snake activations (x + sin²(αx)/α, α per channel). The source's sine phases and noise are
/// random; <see cref="Synthesize(Tensor{T}, Random)"/> draws them, and <see cref="Decode"/> takes the source as given.
/// </remarks>
internal sealed class CosyVoiceHiFT<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly CosyVoiceHiFTConfiguration _c;
    private readonly double[] _window;
    private readonly Tensor<T> _stftBasis;           // [2·bins, 1, 1, fft]: windowed cos and −sin rows
    private readonly InverseStft<T> _istft;

    public CosyVoiceHiFT(IEngine engine, List<LayerBase<T>> layers, CosyVoiceHiFTConfiguration configuration)
    {
        _engine = engine;
        _c = configuration;
        var c = configuration;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> => WenetEncoderBlock<T>.Own(layers, layer);
        NormedConv1DLayer<T> Normed(int input, int output, int kernel, int stride, int dilation, int padding, bool transposed = false)
            => Own(new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, 1, padding, transposed, ConvolutionNormalization.Weight));
        NormedConv1DLayer<T> Plain(int input, int output, int kernel, int stride, int padding)
            => Own(new NormedConv1DLayer<T>(input, output, kernel, stride, 1, 1, padding, false, ConvolutionNormalization.None));

        // F0 predictor (ConvRNNF0Predictor).
        F0Convs.Add(Normed(c.MelChannels, c.F0Channels, 3, 1, 1, 1));
        for (int i = 0; i < 4; i++) F0Convs.Add(Normed(c.F0Channels, c.F0Channels, 3, 1, 1, 1));
        F0Classifier = Own(new DenseLayer<T>(1));
        SourceMerge = Own(new DenseLayer<T>(1));

        int bins = c.FftSize / 2 + 1;
        Pre = Normed(c.MelChannels, c.BaseChannels, 7, 1, 1, 3);
        var rates = c.Upsample;
        for (int i = 0; i < rates.Length; i++)
            Ups.Add(Normed(c.BaseChannels >> i, c.BaseChannels >> (i + 1), c.UpsampleKernel[i], rates[i], 1, (c.UpsampleKernel[i] - rates[i]) / 2, transposed: true));
        // downsample_rates = [1] + upsample_rates[::-1][:-1], cumulated and reversed.
        var down = new List<int> { 1 };
        down.AddRange(Enumerable.Reverse(rates).Take(rates.Length - 1));
        var cumulative = new List<int>();
        int product = 1;
        foreach (int r in down) cumulative.Add(product *= r);
        cumulative.Reverse();
        for (int i = 0; i < rates.Length; i++)
        {
            int u = cumulative[i], channels = c.BaseChannels >> (i + 1);
            SourceDowns.Add(u == 1 ? Plain(c.FftSize + 2, channels, 1, 1, 0) : Plain(c.FftSize + 2, channels, u * 2, u, u / 2));
            SourceBlocks.Add(new HiFTResBlock(this, layers, channels, c.SourceKernels[i], c.SourceDilations[i]));
        }
        for (int i = 0; i < rates.Length; i++)
        {
            int channels = c.BaseChannels >> (i + 1);
            for (int j = 0; j < c.Kernels.Length; j++) Blocks.Add(new HiFTResBlock(this, layers, channels, c.Kernels[j], c.Dilations[j]));
        }
        Post = Normed(c.BaseChannels >> rates.Length, c.FftSize + 2, 7, 1, 1, 3);

        // scipy.signal.get_window("hann", n, fftbins=True): the periodic Hann window.
        _window = Enumerable.Range(0, c.FftSize).Select(n => 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / c.FftSize)).ToArray();
        _stftBasis = new Tensor<T>(new[] { 2 * bins, 1, 1, c.FftSize });
        for (int k = 0; k < bins; k++)
            for (int n = 0; n < c.FftSize; n++)
            {
                double a = 2 * Math.PI * k * n / c.FftSize;
                _stftBasis[k, 0, 0, n] = NumOps.FromDouble(_window[n] * Math.Cos(a));
                // The DC and Nyquist bins' imaginary parts are exactly zero.
                bool realOnly = k == 0 || 2 * k == c.FftSize;
                _stftBasis[bins + k, 0, 0, n] = NumOps.FromDouble(realOnly ? 0.0 : -_window[n] * Math.Sin(a));
            }
        _istft = new InverseStft<T>(engine, c.FftSize, c.HopSize, c.FftSize);
    }

    public CosyVoiceHiFTConfiguration Configuration => _c;
    public List<NormedConv1DLayer<T>> F0Convs { get; } = new();
    public DenseLayer<T> F0Classifier { get; }
    public DenseLayer<T> SourceMerge { get; }
    public NormedConv1DLayer<T> Pre { get; }
    public List<NormedConv1DLayer<T>> Ups { get; } = new();
    public List<NormedConv1DLayer<T>> SourceDowns { get; } = new();
    public List<HiFTResBlock> SourceBlocks { get; } = new();
    public List<HiFTResBlock> Blocks { get; } = new();
    public NormedConv1DLayer<T> Post { get; }

    /// <summary>The predicted F0 per mel frame, <c>[frames]</c>, of a mel <c>[1, mel, frames]</c>.</summary>
    public Tensor<T> PredictF0(Tensor<T> mel)
    {
        var h = mel;
        foreach (var conv in F0Convs) h = _engine.ELU(conv.Forward(h), 1.0);
        int frames = h.Shape[2];
        var rows = _engine.TensorTranspose(_engine.Reshape(h, new[] { _c.F0Channels, frames }));               // [frames, ch]
        return _engine.Reshape(_engine.TensorAbs(F0Classifier.Forward(rows)), new[] { frames });
    }

    /// <summary>
    /// SineGen's harmonic sines <c>[samples, harmonics + 1]</c> (with their noise) from the upsampled F0, given the
    /// harmonics' initial phases (the first is ignored: the fundamental starts at 0) and standard-normal noise <c>[samples, harmonics + 1]</c>.
    /// </summary>
    public double[,] Sines(double[] f0, double[] phases, double[,] noise)
    {
        int length = f0.Length, h = _c.Harmonics + 1;
        var sines = new double[length, h];
        for (int i = 0; i < h; i++)
        {
            double cumulative = 0;
            for (int n = 0; n < length; n++)
            {
                cumulative += f0[n] * (i + 1) / _c.SampleRate;
                double theta = 2 * Math.PI * PythonModulo(cumulative, 1.0);
                bool voiced = f0[n] > _c.VoicedThreshold;
                // The fundamental starts at phase 0 (phase_vec[:, 0, :] = 0); the overtones at their drawn phases.
                double sine = _c.SineAmplitude * Math.Sin(theta + (i == 0 ? 0.0 : phases[i]));
                double noiseAmplitude = voiced ? _c.NoiseStd : _c.SineAmplitude / 3;
                sines[n, i] = (voiced ? sine : 0.0) + noiseAmplitude * noise[n, i];
            }
        }
        return sines;
    }

    // torch's % on floats: the result takes the divisor's sign.
    private static double PythonModulo(double a, double b) => a - Math.Floor(a / b) * b;

    /// <summary>The merged source <c>[samples]</c>: tanh of a linear map of the harmonic sines.</summary>
    public Tensor<T> Source(double[,] sines)
    {
        int length = sines.GetLength(0), h = sines.GetLength(1);
        var x = new Tensor<T>(new[] { length, h });
        for (int n = 0; n < length; n++)
            for (int i = 0; i < h; i++) x[n, i] = NumOps.FromDouble(sines[n, i]);
        return _engine.Reshape(_engine.Tanh(SourceMerge.Forward(x)), new[] { length });
    }

    /// <summary>torch.stft(center=True, reflect) of <paramref name="signal"/> <c>[samples]</c>: real then imaginary bins,
    /// <c>[1, 2·bins, frames]</c>.</summary>
    private Tensor<T> Stft(Tensor<T> signal)
    {
        int pad = _c.FftSize / 2;
        var padded = VocoderOps.ReflectPad(_engine, _engine.Reshape(signal, new[] { 1, 1, signal.Length }), pad, pad);
        var x4 = _engine.Reshape(padded, new[] { 1, 1, 1, padded.Shape[2] });
        var spectrum = _engine.Conv2D(x4, _stftBasis, new[] { 1, _c.HopSize }, new[] { 0, 0 }, new[] { 1, 1 });   // [1, 2·bins, 1, frames]
        return _engine.Reshape(spectrum, new[] { 1, spectrum.Shape[1], spectrum.Shape[3] });
    }

    private Tensor<T> LeakyRelu(Tensor<T> x, double slope) =>
        _engine.TensorAdd(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(slope)),
            _engine.TensorMultiplyScalar(_engine.ReLU(x), NumOps.FromDouble(1 - slope)));

    /// <summary>The waveform <c>[samples]</c> for a mel <c>[1, mel, frames]</c> and the merged source <c>[samples]</c>.</summary>
    public Tensor<T> Decode(Tensor<T> mel, Tensor<T> source)
    {
        var c = _c;
        var sourceStft = Stft(source);
        var x = Pre.Forward(mel);
        for (int i = 0; i < Ups.Count; i++)
        {
            x = Ups[i].Forward(LeakyRelu(x, c.LeakySlope));
            if (i == Ups.Count - 1) x = VocoderOps.ReflectPad(_engine, x, 1, 0);
            var si = SourceBlocks[i].Forward(SourceDowns[i].Forward(sourceStft));
            x = _engine.TensorAdd(x, si);
            Tensor<T>? sum = null;
            for (int j = 0; j < c.Kernels.Length; j++)
            {
                var y = Blocks[i * c.Kernels.Length + j].Forward(x);
                sum = sum is null ? y : _engine.TensorAdd(sum, y);
            }
            x = _engine.TensorMultiplyScalar(sum!, NumOps.FromDouble(1.0 / c.Kernels.Length));
        }
        x = Post.Forward(LeakyRelu(x, 0.01));
        int bins = c.FftSize / 2 + 1, frames = x.Shape[2];
        var magnitude = _engine.TensorExp(_engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, bins, frames }));
        magnitude = _engine.TensorClamp(magnitude, NumOps.FromDouble(double.NegativeInfinity), NumOps.FromDouble(100.0));   // torch.clip(max=1e2)
        var phase = _engine.TensorSin(_engine.TensorSlice(x, new[] { 0, bins, 0 }, new[] { 1, bins, frames }));
        var audio = _istft.Forward(magnitude, phase);
        return _engine.TensorClamp(audio, NumOps.FromDouble(-c.AudioLimit), NumOps.FromDouble(c.AudioLimit));
    }

    /// <summary>F0 upsampled by nearest neighbour to the sample rate.</summary>
    public double[] UpsampleF0(Tensor<T> f0)
    {
        int scale = _c.SamplesPerFrame;
        var result = new double[f0.Length * scale];
        for (int i = 0; i < f0.Length; i++)
        {
            double v = NumOps.ToDouble(f0[i]);
            for (int s = 0; s < scale; s++) result[i * scale + s] = v;
        }
        return result;
    }

    /// <summary>Synthesizes the waveform for a mel <c>[1, mel, frames]</c>, drawing the source's random phases and noise.</summary>
    public Tensor<T> Synthesize(Tensor<T> mel, Random random)
    {
        using var _ = new NoGradScope<T>();
        var f0 = UpsampleF0(PredictF0(mel));
        int h = _c.Harmonics + 1;
        var phases = new double[h];
        for (int i = 1; i < h; i++) phases[i] = (random.NextDouble() * 2 - 1) * Math.PI;
        var noise = new double[f0.Length, h];
        for (int n = 0; n < f0.Length; n++)
            for (int i = 0; i < h; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                noise[n, i] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            }
        return Decode(mel, Source(Sines(f0, phases, noise)));
    }

    /// <summary>Loads the reference's <c>hift.pt</c> state dict (weight-norm <c>original0</c>/<c>original1</c>).</summary>
    public void LoadTorch(IReadOnlyDictionary<string, double[]> state)
    {
        void Normed(NormedConv1DLayer<T> conv, string p)
        {
            if (state.ContainsKey(p + ".parametrizations.weight.original1"))
                conv.LoadTorchWeights(state[p + ".parametrizations.weight.original1"], state[p + ".parametrizations.weight.original0"], state[p + ".bias"]);
            else if (state.ContainsKey(p + ".weight_v"))
                conv.LoadTorchWeights(state[p + ".weight_v"], state[p + ".weight_g"], state[p + ".bias"]);
            else
                conv.LoadTorchWeights(state[p + ".weight"], null, state[p + ".bias"]);
        }
        for (int i = 0; i < F0Convs.Count; i++) Normed(F0Convs[i], $"f0_predictor.condnet.{2 * i}");
        TorchParameters.Linear(_engine, F0Classifier, _c.F0Channels, 1, state["f0_predictor.classifier.weight"], state["f0_predictor.classifier.bias"]);
        TorchParameters.Linear(_engine, SourceMerge, _c.Harmonics + 1, 1, state["m_source.l_linear.weight"], state["m_source.l_linear.bias"]);
        Normed(Pre, "conv_pre");
        for (int i = 0; i < Ups.Count; i++) Normed(Ups[i], $"ups.{i}");
        for (int i = 0; i < SourceDowns.Count; i++)
        {
            Normed(SourceDowns[i], $"source_downs.{i}");
            SourceBlocks[i].Load(state, $"source_resblocks.{i}", Normed);
        }
        for (int i = 0; i < Blocks.Count; i++) Blocks[i].Load(state, $"resblocks.{i}", Normed);
        Normed(Post, "conv_post");
    }

    /// <summary>HiFT's residual block: per dilation, Snake → conv(d) → Snake → conv(1), residual.</summary>
    internal sealed class HiFTResBlock
    {
        private readonly CosyVoiceHiFT<T> _owner;
        private readonly int _channels;

        public HiFTResBlock(CosyVoiceHiFT<T> owner, List<LayerBase<T>> layers, int channels, int kernel, int[] dilations)
        {
            _owner = owner;
            _channels = channels;
            foreach (int d in dilations)
            {
                First.Add(WenetEncoderBlock<T>.Own(layers, new NormedConv1DLayer<T>(channels, channels, kernel, 1, d, 1, (kernel * d - d) / 2, false, ConvolutionNormalization.Weight)));
                Second.Add(WenetEncoderBlock<T>.Own(layers, new NormedConv1DLayer<T>(channels, channels, kernel, 1, 1, 1, (kernel - 1) / 2, false, ConvolutionNormalization.Weight)));
                var a1 = WenetEncoderBlock<T>.Own(layers, new TiedEmbeddingLayer<T>(1, channels));
                var a2 = WenetEncoderBlock<T>.Own(layers, new TiedEmbeddingLayer<T>(1, channels));
                a1.Reinitialize(() => 1.0);
                a2.Reinitialize(() => 1.0);
                FirstAlpha.Add(a1);
                SecondAlpha.Add(a2);
            }
        }

        public List<NormedConv1DLayer<T>> First { get; } = new();
        public List<NormedConv1DLayer<T>> Second { get; } = new();
        public List<TiedEmbeddingLayer<T>> FirstAlpha { get; } = new();
        public List<TiedEmbeddingLayer<T>> SecondAlpha { get; } = new();

        // Snake: x + 1/(α + 1e-9) · sin²(αx), α per channel.
        private Tensor<T> Snake(Tensor<T> x, TiedEmbeddingLayer<T> alpha)
        {
            var engine = _owner._engine;
            int frames = x.Shape[2];
            var a = engine.TensorTile(engine.Reshape(alpha.Table, new[] { 1, _channels, 1 }), new[] { 1, 1, frames });
            var s = engine.TensorSin(engine.TensorMultiply(x, a));
            var scale = engine.TensorDivide(engine.TensorMultiply(s, s), engine.TensorAddScalar(a, NumOps.FromDouble(1e-9)));
            return engine.TensorAdd(x, scale);
        }

        public Tensor<T> Forward(Tensor<T> x)
        {
            for (int i = 0; i < First.Count; i++)
            {
                var t = First[i].Forward(Snake(x, FirstAlpha[i]));
                t = Second[i].Forward(Snake(t, SecondAlpha[i]));
                x = _owner._engine.TensorAdd(t, x);
            }
            return x;
        }

        public void Load(IReadOnlyDictionary<string, double[]> state, string p, Action<NormedConv1DLayer<T>, string> normed)
        {
            for (int i = 0; i < First.Count; i++)
            {
                normed(First[i], $"{p}.convs1.{i}");
                normed(Second[i], $"{p}.convs2.{i}");
                FirstAlpha[i].LoadTable(state[$"{p}.activations1.{i}.alpha"]);
                SecondAlpha[i].LoadTable(state[$"{p}.activations2.{i}.alpha"]);
            }
        }
    }
}
