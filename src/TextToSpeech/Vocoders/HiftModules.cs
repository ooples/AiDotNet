using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>The sizes of a HiFT generator (CosyVoice's <c>HiFTGenerator</c> arguments).</summary>
internal sealed record HiftConfiguration(
    int InChannels, int BaseChannels, int Harmonics, int SampleRate, double NsfAlpha, double NsfSigma, double VoicedThreshold,
    int[] UpsampleRates, int[] UpsampleKernels, int FftSize, int HopSize, int[] ResblockKernels, int[][] ResblockDilations,
    int[] SourceKernels, int[][] SourceDilations, double LeakySlope, double AudioLimit, int F0Channels)
{
    /// <summary>Waveform samples per mel frame: the upsampling rates' product times the iSTFT hop.</summary>
    public int UpsampleScale => UpsampleRates.Aggregate(1, (a, b) => a * b) * HopSize;
}

/// <summary>
/// HiFTNet's generator as CosyVoice implements it (FunAudioLLM/CosyVoice <c>cosyvoice/hifigan/generator.py</c>; Li et
/// al. 2023): a neural source filter (an F0 predictor, a harmonic sine source merged to one excitation) driving a HiFi-GAN
/// upsampler whose output is an STFT, inverted to the waveform.
/// </summary>
/// <remarks>
/// <para><b>F0</b> (<c>ConvRNNF0Predictor</c>): five weight-normalized 3-wide convolutions with ELU, a linear map to one
/// value per frame, its absolute value.</para>
/// <para><b>Source</b> (<c>SourceModuleHnNSF</c> with <c>SineGen</c>): F0 repeated to the sample rate; harmonics k = 1 … H+1
/// at phase 2π·frac(Σ k·f0/sr) (harmonics above the first get a random starting phase), amplitude α where F0 exceeds the
/// voiced threshold, plus Gaussian noise (σ where voiced, α/3 where not); a linear merge of the harmonics and tanh. No
/// gradient reaches F0 through the source.</para>
/// <para><b>Decode</b>: the source's STFT (n_fft, hop, Hann window, centred) as real and imaginary channels feeds each
/// upsampling stage through a downsampling convolution and a Snake residual block; each stage is leaky ReLU,
/// a weight-normalized transposed convolution, the source branch added, and the mean of the multi-receptive-field residual
/// blocks (the last stage first reflection-pads one frame on the left). After leaky ReLU (slope 0.01) a
/// weight-normalized convolution gives n_fft + 2 channels: the magnitude's log (exp, clipped at 100) and the phase (sin),
/// which the inverse STFT turns into audio, clamped to ±0.99.</para>
/// </remarks>
internal sealed class HiftGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Tensor<T> _stftBasis;   // [n_fft + 2, 1, 1, n_fft]: the windowed cosines, then the negated sines
    private readonly InverseStft<T> _istft;

    internal sealed record ResBlock(List<SnakeLayer<T>> Snakes1, List<NormedConv1DLayer<T>> Convs1, List<SnakeLayer<T>> Snakes2,
        List<NormedConv1DLayer<T>> Convs2);

    public HiftGenerator(IEngine engine, List<LayerBase<T>> layers, HiftConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        // init_weights: N(0, 0.01) for the convolutions it reaches; the others keep PyTorch's default (Kaiming uniform).
        Func<double> Normal(double std) => () =>
            std * Math.Sqrt(-2.0 * Math.Log(1.0 - initialization.NextDouble())) * Math.Cos(2.0 * Math.PI * initialization.NextDouble());
        Func<double> Kaiming(int fanIn) => () => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int stride, int dilation, int padding, bool weightNorm,
            Func<double> init, bool transposed = false)
        {
            var conv = Add(new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, 1, padding, transposed,
                weightNorm ? ConvolutionNormalization.Weight : ConvolutionNormalization.None));
            conv.Reinitialize(init, bias: 0.0);
            return conv;
        }

        for (int i = 0; i < 5; i++)
        {
            int input = i == 0 ? c.InChannels : c.F0Channels;
            F0Convs.Add(Conv(input, c.F0Channels, 3, 1, 1, 1, true, Kaiming(input * 3)));
        }
        F0Classifier = Conv(c.F0Channels, 1, 1, 1, 1, 0, false, Kaiming(c.F0Channels));
        SourceMerge = Conv(c.Harmonics + 1, 1, 1, 1, 1, 0, false, Kaiming(c.Harmonics + 1));

        ConvPre = Conv(c.InChannels, c.BaseChannels, 7, 1, 1, 3, true, Kaiming(c.InChannels * 7));
        for (int i = 0; i < c.UpsampleRates.Length; i++)
        {
            int u = c.UpsampleRates[i], k = c.UpsampleKernels[i];
            Ups.Add(Conv(c.BaseChannels >> i, c.BaseChannels >> (i + 1), k, u, 1, (k - u) / 2, true, Normal(0.01), transposed: true));
        }
        // Source downsampling: cumulative rates of [1] + reversed rates without its last, reversed again.
        var rates = new List<int> { 1 };
        rates.AddRange(Enumerable.Reverse(c.UpsampleRates).Take(c.UpsampleRates.Length - 1));   // net471: an array's .Reverse() is the in-place void one
        var cumulative = new int[rates.Count];
        for (int i = 0, p = 1; i < rates.Count; i++) cumulative[i] = p *= rates[i];
        Array.Reverse(cumulative);
        for (int i = 0; i < cumulative.Length; i++)
        {
            int u = cumulative[i], channels = c.BaseChannels >> (i + 1);
            SourceDownKernels.Add(u == 1 ? 1 : u * 2);
            SourceDowns.Add(u == 1
                ? Conv(c.FftSize + 2, channels, 1, 1, 1, 0, false, Kaiming(c.FftSize + 2))
                : Conv(c.FftSize + 2, channels, u * 2, u, 1, u / 2, false, Kaiming((c.FftSize + 2) * u * 2)));
            SourceBlocks.Add(NewResBlock(channels, c.SourceKernels[i], c.SourceDilations[i]));
        }
        for (int i = 0; i < c.UpsampleRates.Length; i++)
            for (int j = 0; j < c.ResblockKernels.Length; j++)
                ResBlocks.Add(NewResBlock(c.BaseChannels >> (i + 1), c.ResblockKernels[j], c.ResblockDilations[j]));
        ConvPost = Conv(c.BaseChannels >> c.UpsampleRates.Length, c.FftSize + 2, 7, 1, 1, 3, true, Normal(0.01));

        ResBlock NewResBlock(int channels, int kernel, int[] dilations)
        {
            var block = new ResBlock(new(), new(), new(), new());
            foreach (int d in dilations)
            {
                block.Snakes1.Add(Add(new SnakeLayer<T>(channels)));
                block.Convs1.Add(Conv(channels, channels, kernel, 1, d, (kernel * d - d) / 2, true, Normal(0.01)));
                block.Snakes2.Add(Add(new SnakeLayer<T>(channels)));
                block.Convs2.Add(Conv(channels, channels, kernel, 1, 1, (kernel - 1) / 2, true, Normal(0.01)));
            }
            return block;
        }

        // scipy.signal.get_window("hann", n_fft, fftbins=True), the periodic window, is the STFT basis's window below.
        _istft = new InverseStft<T>(engine, c.FftSize, c.HopSize, c.FftSize);
        int bins = c.FftSize / 2 + 1;
        _stftBasis = new Tensor<T>(new[] { 2 * bins, 1, 1, c.FftSize });
        for (int k = 0; k < bins; k++)
            for (int n = 0; n < c.FftSize; n++)
            {
                double w = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / c.FftSize), a = 2.0 * Math.PI * k * n / c.FftSize;
                _stftBasis[k, 0, 0, n] = NumOps.FromDouble(w * Math.Cos(a));
                // The DC and Nyquist bins' imaginary parts are exactly zero.
                _stftBasis[bins + k, 0, 0, n] = NumOps.FromDouble(k == 0 || 2 * k == c.FftSize ? 0.0 : -w * Math.Sin(a));
            }
    }

    public HiftConfiguration Configuration { get; }
    public List<NormedConv1DLayer<T>> F0Convs { get; } = new();
    public NormedConv1DLayer<T> F0Classifier { get; }
    public NormedConv1DLayer<T> SourceMerge { get; }
    public NormedConv1DLayer<T> ConvPre { get; }
    public List<NormedConv1DLayer<T>> Ups { get; } = new();
    public List<NormedConv1DLayer<T>> SourceDowns { get; } = new();

    /// <summary>Each source-downsampling convolution's width (1, or twice its rate).</summary>
    public List<int> SourceDownKernels { get; } = new();
    public List<ResBlock> SourceBlocks { get; } = new();
    public List<ResBlock> ResBlocks { get; } = new();
    public NormedConv1DLayer<T> ConvPost { get; }

    // ELU(x) = x for x > 0, exp(x) − 1 otherwise: relu(x) − relu(1 − exp(x)).
    private Tensor<T> Elu(Tensor<T> x) =>
        _engine.TensorSubtract(_engine.ReLU(x), _engine.ReLU(_engine.TensorAddScalar(_engine.TensorNegate(_engine.TensorExp(x)), NumOps.One)));

    private Tensor<T> LeakyRelu(Tensor<T> x, double slope) =>
        _engine.TensorSubtract(_engine.ReLU(x), _engine.TensorMultiplyScalar(_engine.ReLU(_engine.TensorNegate(x)), NumOps.FromDouble(slope)));

    /// <summary>The F0 per mel frame <c>[frames]</c> of a mel <c>[1, InChannels, frames]</c>.</summary>
    public Tensor<T> PredictF0(Tensor<T> mel)
    {
        var x = mel;
        foreach (var conv in F0Convs) x = Elu(conv.Forward(x));
        var f0 = _engine.TensorAbs(F0Classifier.Forward(x));                                                // [1, 1, frames]
        return _engine.Reshape(f0, new[] { f0.Shape[2] });
    }

    /// <summary>
    /// The excitation <c>[1, 1, samples]</c> of F0 per frame (<c>SourceModuleHnNSF</c> with <c>SineGen</c>):
    /// <paramref name="random"/> draws the harmonics' starting phases and the noise.
    /// </summary>
    public Tensor<T> Source(Tensor<T> f0, Random random)
    {
        var c = Configuration;
        int samples = f0.Length * c.UpsampleScale, harmonics = c.Harmonics + 1;
        var phases = new double[harmonics];
        for (int h = 1; h < harmonics; h++) phases[h] = -Math.PI + 2 * Math.PI * random.NextDouble();
        var noise = new double[harmonics, samples];
        for (int h = 0; h < harmonics; h++)
            for (int s = 0; s < samples; s++)
                noise[h, s] = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        return Merge(Harmonics(f0, phases, noise));
    }

    /// <summary>
    /// <c>SineGen</c>'s harmonics <c>[1, H+1, samples]</c> of F0 per frame (nearest-upsampled to the samples), given each
    /// harmonic's starting phase (the fundamental's is always 0) and standard-normal noise <c>[H+1, samples]</c>: the
    /// voiced samples' sines plus noise of σ <c>NsfSigma</c>, the unvoiced samples' noise of σ <c>NsfAlpha / 3</c>. F0 is
    /// data here, as SineGen runs without gradients.
    /// </summary>
    public Tensor<T> Harmonics(Tensor<T> f0, double[] phases, double[,] noise)
    {
        var c = Configuration;
        int frames = f0.Length, scale = c.UpsampleScale, samples = frames * scale, harmonics = c.Harmonics + 1;
        var sines = new Tensor<T>(new[] { 1, harmonics, samples });
        for (int h = 0; h < harmonics; h++)
        {
            double cumulative = 0, phase = h == 0 ? 0.0 : phases[h];
            for (int s = 0; s < samples; s++)
            {
                double f = NumOps.ToDouble(f0[s / scale]);
                cumulative += f * (h + 1) / c.SampleRate;
                double theta = 2 * Math.PI * (cumulative - Math.Floor(cumulative));
                bool voiced = f > c.VoicedThreshold;
                double noiseAmplitude = voiced ? c.NsfSigma : c.NsfAlpha / 3;
                sines[0, h, s] = NumOps.FromDouble((voiced ? c.NsfAlpha * Math.Sin(theta + phase) : 0.0) + noiseAmplitude * noise[h, s]);
            }
        }
        return sines;
    }

    /// <summary>The merged excitation <c>[1, 1, samples]</c> of harmonics <c>[1, H+1, samples]</c>: tanh of the linear merge.</summary>
    public Tensor<T> Merge(Tensor<T> harmonics) => _engine.Tanh(SourceMerge.Forward(harmonics));

    private Tensor<T> RunResBlock(ResBlock block, Tensor<T> x)
    {
        for (int i = 0; i < block.Convs1.Count; i++)
        {
            var xt = block.Convs2[i].Forward(block.Snakes2[i].Forward(block.Convs1[i].Forward(block.Snakes1[i].Forward(x))));
            x = _engine.TensorAdd(xt, x);
        }
        return x;
    }

    /// <summary>
    /// The source's STFT (<c>torch.stft</c>, centred with reflection padding, the periodic Hann window) as
    /// <c>[1, n_fft + 2, frames]</c>: the real parts, then the imaginary parts. A fixed convolution, so the gradient reaches
    /// the source merge, which trains through it in the reference.
    /// </summary>
    internal Tensor<T> SourceStft(Tensor<T> source)
    {
        var c = Configuration;
        int pad = c.FftSize / 2;
        var padded = VocoderOps.ReflectPad(_engine, _engine.Reshape(source, new[] { 1, 1, source.Length }), pad, pad);
        var spectrum = _engine.Conv2D(_engine.Reshape(padded, new[] { 1, 1, 1, padded.Shape[2] }), _stftBasis,
            new[] { 1, c.HopSize }, new[] { 0, 0 }, new[] { 1, 1 });                                       // [1, n_fft + 2, 1, frames]
        return _engine.Reshape(spectrum, new[] { 1, spectrum.Shape[1], spectrum.Shape[3] });
    }

    /// <summary>The waveform <c>[samples]</c> of a mel <c>[1, InChannels, frames]</c> and its excitation
    /// <c>[1, 1, frames · UpsampleScale]</c> (<c>HiFTGenerator.decode</c>).</summary>
    public Tensor<T> Decode(Tensor<T> mel, Tensor<T> source)
    {
        var c = Configuration;
        // In the graph: the reference trains m_source.l_linear through the source's STFT and source_downs.
        var stft = SourceStft(source);
        var x = ConvPre.Forward(mel);
        for (int i = 0; i < Ups.Count; i++)
        {
            x = Ups[i].Forward(LeakyRelu(x, c.LeakySlope));
            if (i == Ups.Count - 1)
            {
                // ReflectionPad1d((1, 0)): the second frame, then the frames.
                int channels = x.Shape[1], frames = x.Shape[2];
                var first = _engine.TensorSlice(x, new[] { 0, 0, 1 }, new[] { 1, channels, 1 });
                x = _engine.TensorConcatenate(new[] { first, x }, 2);
            }
            var si = RunResBlock(SourceBlocks[i], SourceDowns[i].Forward(stft));
            x = _engine.TensorAdd(x, si);
            int kernels = c.ResblockKernels.Length;
            var sum = RunResBlock(ResBlocks[i * kernels], x);
            for (int j = 1; j < kernels; j++) sum = _engine.TensorAdd(sum, RunResBlock(ResBlocks[i * kernels + j], x));
            x = _engine.TensorMultiplyScalar(sum, NumOps.FromDouble(1.0 / kernels));
        }
        x = ConvPost.Forward(LeakyRelu(x, 0.01));
        int halfBins = c.FftSize / 2 + 1, outFrames = x.Shape[2];
        var logMagnitude = _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, halfBins, outFrames });
        var phaseInput = _engine.TensorSlice(x, new[] { 0, halfBins, 0 }, new[] { 1, halfBins, outFrames });
        // min(exp(m), 100) = 100 − relu(100 − exp(m)).
        var hundred = NumOps.FromDouble(100);
        var magnitude = _engine.TensorAddScalar(_engine.TensorNegate(_engine.ReLU(_engine.TensorAddScalar(_engine.TensorNegate(_engine.TensorExp(logMagnitude)), hundred))), hundred);
        var audio = _istft.Forward(magnitude, _engine.TensorSin(phaseInput));
        var limit = NumOps.FromDouble(c.AudioLimit);
        return _engine.TensorClamp(audio, NumOps.Negate(limit), limit);
    }

    /// <summary>Mel to waveform (<c>HiFTGenerator.inference</c>): F0, the excitation, then <see cref="Decode"/>.</summary>
    public Tensor<T> Forward(Tensor<T> mel, Random random)
    {
        Tensor<T> source;
        using (new NoGradScope<T>()) source = Source(PredictF0(mel), random);
        return Decode(mel, source);
    }

    /// <summary>Loads a CosyVoice <c>hift.pt</c> state dict (weight-norm parametrizations as <c>original0</c> gain and
    /// <c>original1</c> direction).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        // weight_norm(dim=0): the gain original0 has one value per weight row (dim 0), the direction original1 the weight.
        void WeightNorm(NormedConv1DLayer<T> conv, string name, int[] shape)
        {
            var gainShape = new[] { shape[0] }.Concat(Enumerable.Repeat(1, shape.Length - 1)).ToArray();
            conv.LoadTorchWeights(read(name + ".parametrizations.weight.original1", shape),
                read(name + ".parametrizations.weight.original0", gainShape), read(name + ".bias", new[] { conv.OutputChannels }));
        }
        void Plain(NormedConv1DLayer<T> conv, string name, int[] shape) =>
            conv.LoadTorchWeights(read(name + ".weight", shape), null, read(name + ".bias", new[] { conv.OutputChannels }));

        for (int i = 0; i < F0Convs.Count; i++)
        {
            int input = i == 0 ? c.InChannels : c.F0Channels;
            WeightNorm(F0Convs[i], $"f0_predictor.condnet.{2 * i}", new[] { c.F0Channels, input, 3 });
        }
        F0Classifier.LoadTorchWeights(read("f0_predictor.classifier.weight", new[] { 1, c.F0Channels }), null, read("f0_predictor.classifier.bias", new[] { 1 }));
        SourceMerge.LoadTorchWeights(read("m_source.l_linear.weight", new[] { 1, c.Harmonics + 1 }), null, read("m_source.l_linear.bias", new[] { 1 }));
        WeightNorm(ConvPre, "conv_pre", new[] { c.BaseChannels, c.InChannels, 7 });
        for (int i = 0; i < Ups.Count; i++)
        {
            int input = c.BaseChannels >> i, output = c.BaseChannels >> (i + 1);
            WeightNorm(Ups[i], $"ups.{i}", new[] { input, output, c.UpsampleKernels[i] });
        }
        for (int i = 0; i < SourceDowns.Count; i++)
        {
            var down = SourceDowns[i];
            int channels = c.BaseChannels >> (i + 1), kernel = SourceDownKernels[i];
            Plain(down, $"source_downs.{i}", new[] { channels, c.FftSize + 2, kernel });
            LoadResBlock(SourceBlocks[i], $"source_resblocks.{i}", channels, c.SourceKernels[i]);
        }
        for (int i = 0; i < ResBlocks.Count; i++)
        {
            int stage = i / c.ResblockKernels.Length, channels = c.BaseChannels >> (stage + 1);
            LoadResBlock(ResBlocks[i], $"resblocks.{i}", channels, c.ResblockKernels[i % c.ResblockKernels.Length]);
        }
        int last = c.BaseChannels >> c.UpsampleRates.Length;
        WeightNorm(ConvPost, "conv_post", new[] { c.FftSize + 2, last, 7 });

        void LoadResBlock(ResBlock block, string prefix, int channels, int kernel)
        {
            for (int j = 0; j < block.Convs1.Count; j++)
            {
                WeightNorm(block.Convs1[j], $"{prefix}.convs1.{j}", new[] { channels, channels, kernel });
                WeightNorm(block.Convs2[j], $"{prefix}.convs2.{j}", new[] { channels, channels, kernel });
                block.Snakes1[j].LoadAlpha(read($"{prefix}.activations1.{j}.alpha", new[] { channels }));
                block.Snakes2[j].LoadAlpha(read($"{prefix}.activations2.{j}.alpha", new[] { channels }));
            }
        }
    }
}
