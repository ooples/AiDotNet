using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>A per-channel learnable vector <c>[1, C, 1]</c> held as a one-row table, so it is an ordinary trainable,
/// serialized parameter (StyleTTS 2's Snake α).</summary>
internal static class StyleTts2Ops
{
    /// <summary>Snake: <c>x + sin²(α x) / α</c> with α per channel <c>[1, C]</c>.</summary>
    public static Tensor<T> Snake<T>(IEngine engine, Tensor<T> x, TiedEmbeddingLayer<T> alpha)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var a = engine.TensorTile(engine.Reshape(alpha.Forward(new Tensor<T>(new[] { 1 })), new[] { 1, c, 1 }), new[] { 1, 1, t });
        var s = engine.TensorSin(engine.TensorMultiply(a, x));
        return engine.TensorAdd(x, engine.TensorDivide(engine.TensorMultiply(s, s), a));
    }

    public static TiedEmbeddingLayer<T> Alpha<T>(List<LayerBase<T>> layers, int channels)
    {
        var alpha = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(1, channels));
        alpha.Reinitialize(() => 1.0);
        return alpha;
    }

    /// <summary><c>F.interpolate(mode="linear", align_corners=False)</c> along time with the given scale factor (the
    /// source coordinate <c>(i + ½) / scale − ½</c>, clamped at the ends).</summary>
    public static double[] Linear(double[] x, double scale, int outputLength)
    {
        var y = new double[outputLength];
        int n = x.Length;
        for (int i = 0; i < outputLength; i++)
        {
            double source = Math.Max(0.0, (i + 0.5) / scale - 0.5);
            int left = Math.Min((int)Math.Floor(source), n - 1), right = Math.Min(left + 1, n - 1);
            double w = source - left;
            y[i] = (1 - w) * x[left] + w * x[right];
        }
        return y;
    }
}

/// <summary>
/// StyleTTS 2's <c>AdaINResBlock1</c> (yl4579/StyleTTS2 <c>Modules/istftnet.py</c>, <c>hifigan.py</c>): HiFi-GAN's type-1
/// residual block with AdaIN before each convolution and Snake activations: for each dilation,
/// <c>x += conv₂(snake₂(AdaIN₂(conv₁(snake₁(AdaIN₁(x, s))), s)))</c>.
/// </summary>
internal sealed class StyleTts2AdaInResBlock<T>
{
    private readonly IEngine _engine;

    public StyleTts2AdaInResBlock(IEngine engine, List<LayerBase<T>> layers, int channels, int kernel, int[] dilations, int styleDim)
    {
        _engine = engine;
        foreach (int d in dilations)
        {
            First.Add(StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels, channels, kernel, dilation: d)));
            Second.Add(StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels, channels, kernel)));
            FirstNorm.Add(new StyleTtsAdaIn<T>(engine, layers, styleDim, channels));
            SecondNorm.Add(new StyleTtsAdaIn<T>(engine, layers, styleDim, channels));
            FirstAlpha.Add(StyleTts2Ops.Alpha(layers, channels));
            SecondAlpha.Add(StyleTts2Ops.Alpha(layers, channels));
        }
        Channels = channels;
    }

    public int Channels { get; }
    public List<NormedConv1DLayer<T>> First { get; } = new();
    public List<NormedConv1DLayer<T>> Second { get; } = new();
    public List<StyleTtsAdaIn<T>> FirstNorm { get; } = new();
    public List<StyleTtsAdaIn<T>> SecondNorm { get; } = new();
    public List<TiedEmbeddingLayer<T>> FirstAlpha { get; } = new();
    public List<TiedEmbeddingLayer<T>> SecondAlpha { get; } = new();

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> style)
    {
        for (int i = 0; i < First.Count; i++)
        {
            var h = First[i].Forward(StyleTts2Ops.Snake(_engine, FirstNorm[i].Forward(x, style), FirstAlpha[i]));
            h = Second[i].Forward(StyleTts2Ops.Snake(_engine, SecondNorm[i].Forward(h, style), SecondAlpha[i]));
            x = _engine.TensorAdd(h, x);
        }
        return x;
    }

    public void Load(TorchReader read, string name)
    {
        for (int i = 0; i < First.Count; i++)
        {
            StyleTtsOps.LoadWeightNorm(First[i], read, $"{name}.convs1.{i}");
            StyleTtsOps.LoadWeightNorm(Second[i], read, $"{name}.convs2.{i}");
            FirstNorm[i].Load(read, $"{name}.adain1.{i}");
            SecondNorm[i].Load(read, $"{name}.adain2.{i}");
            FirstAlpha[i].LoadTable(read($"{name}.alpha1.{i}", new[] { 1, Channels, 1 }));
            SecondAlpha[i].LoadTable(read($"{name}.alpha2.{i}", new[] { 1, Channels, 1 }));
        }
    }
}

/// <summary>
/// The harmonic-plus-noise source of StyleTTS 2's generators (the reference's <c>SineGen</c> and
/// <c>SourceModuleHnNSF</c>, after the neural source-filter model of Wang et al.): the F0 contour at the waveform rate,
/// its fundamental and eight overtones as sines whose phase is integrated at the frame rate and interpolated back,
/// unvoiced frames (F0 ≤ 10 Hz) replaced by noise, merged by a linear layer and tanh.
/// </summary>
internal sealed class StyleTts2Source<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private const int Harmonics = 9;
    private const double SineAmplitude = 0.1, NoiseStd = 0.003, VoicedThreshold = 10.0;

    public StyleTts2Source(IEngine engine, List<LayerBase<T>> layers, int sampleRate, int upsampleScale)
    {
        _engine = engine;
        SampleRate = sampleRate;
        UpsampleScale = upsampleScale;
        Merge = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(1));
    }

    public int SampleRate { get; }
    public int UpsampleScale { get; }
    public DenseLayer<T> Merge { get; }

    /// <summary>When false, the random initial phases and the noise are zero (for comparison with a reference whose draws
    /// are not reproducible).</summary>
    internal bool Stochastic { get; set; } = true;

    private static double Gaussian(Random random) =>
        Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());

    /// <summary>The merged harmonic source <c>[1, 1, frames · scale]</c> of an F0 contour <c>[1, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> f0, Random random)
    {
        int frames = f0.Length, samples = frames * UpsampleScale;
        var waves = new double[Harmonics][];
        var uv = new double[samples];
        for (int h = 0; h < Harmonics; h++)
        {
            // f0 upsampled (nearest), times the harmonic number, as a fraction of a cycle per sample.
            var rad = new double[samples];
            for (int i = 0; i < samples; i++)
            {
                double f = NumOps.ToDouble(f0[i / UpsampleScale]) * (h + 1) / SampleRate;
                rad[i] = f - Math.Floor(f);
            }
            if (h > 0 && Stochastic) rad[0] += random.NextDouble();
            // The phase integrated at the frame rate (linear interpolation down), scaled back and interpolated up.
            var coarse = StyleTts2Ops.Linear(rad, 1.0 / UpsampleScale, samples / UpsampleScale);
            var phase = new double[coarse.Length];
            double sum = 0;
            for (int i = 0; i < coarse.Length; i++)
            {
                sum += coarse[i];
                phase[i] = sum * 2 * Math.PI * UpsampleScale;
            }
            var fine = StyleTts2Ops.Linear(phase, UpsampleScale, samples);
            waves[h] = new double[samples];
            for (int i = 0; i < samples; i++) waves[h][i] = Math.Sin(fine[i]) * SineAmplitude;
        }
        for (int i = 0; i < samples; i++) uv[i] = NumOps.ToDouble(f0[i / UpsampleScale]) > VoicedThreshold ? 1.0 : 0.0;
        var rows = new Tensor<T>(new[] { samples, Harmonics });
        for (int i = 0; i < samples; i++)
        {
            double noiseAmp = uv[i] * NoiseStd + (1 - uv[i]) * SineAmplitude / 3;
            for (int h = 0; h < Harmonics; h++)
            {
                double noise = Stochastic ? noiseAmp * Gaussian(random) : 0.0;
                rows[i, h] = NumOps.FromDouble(waves[h][i] * uv[i] + noise);
            }
        }
        var merged = _engine.Tanh(Merge.Forward(rows));                                                    // [S, 1]
        return _engine.Reshape(merged, new[] { 1, 1, samples });
    }

    public void Load(TorchReader read, string name) =>
        StyleTtsOps.LoadLinear(_engine, Merge, read, name + ".l_linear", Harmonics, 1);
}

/// <summary>Which waveform generator StyleTTS 2's decoder ends in (the released configurations use iSTFTNet for LJSpeech
/// and LibriTTS; HiFi-GAN is the code's other option).</summary>
public enum StyleTts2DecoderType
{
    /// <summary>iSTFTNet (Kaneko et al. 2022): upsampling to a 20-point STFT's frame rate, then the inverse STFT.</summary>
    IstftNet,

    /// <summary>HiFi-GAN (Kong et al. 2020) to the waveform, with Snake activations.</summary>
    HiFiGan,
}

/// <summary>
/// StyleTTS 2's waveform generator (the reference's <c>Generator</c> in <c>Modules/istftnet.py</c> or
/// <c>Modules/hifigan.py</c>): HiFi-GAN's upsampling stack with AdaIN-Snake residual blocks, the harmonic source read at
/// every stage through a strided convolution and its own residual block, and either an inverse STFT of predicted log
/// magnitude and phase (iSTFTNet) or a tanh waveform head (HiFi-GAN).
/// </summary>
internal sealed class StyleTts2Generator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly StyleTts2DecoderType _type;
    private readonly int[] _rates;
    private readonly int _fft;
    private readonly CenteredComplexStft<T>? _stft;
    private readonly InverseStft<T>? _istft;

    public StyleTts2Generator(IEngine engine, List<LayerBase<T>> layers, StyleTts2DecoderType type, int styleDim, int[] kernels,
        int[] rates, int initialChannels, int[][] dilations, int[] upsampleKernels, int istftFft, int istftHop, int sampleRate)
    {
        _engine = engine;
        _type = type;
        _rates = rates;
        _fft = istftFft;
        int scale = rates.Aggregate(1, (a, b) => a * b) * (type == StyleTts2DecoderType.IstftNet ? istftHop : 1);
        Source = new StyleTts2Source<T>(engine, layers, sampleRate, scale);
        int sourceChannels = type == StyleTts2DecoderType.IstftNet ? istftFft + 2 : 1;
        int channels = initialChannels;
        if (type == StyleTts2DecoderType.HiFiGan) Alphas.Add(StyleTts2Ops.Alpha(layers, initialChannels));
        for (int i = 0; i < rates.Length; i++)
        {
            int u = rates[i], k = upsampleKernels[i], next = initialChannels >> (i + 1);
            Ups.Add(StyleTtsOps.Own(layers, type == StyleTts2DecoderType.IstftNet
                ? new NormedConv1DLayer<T>(channels, next, k, u, 1, 1, (k - u) / 2, true, ConvolutionNormalization.Weight)
                : new NormedConv1DLayer<T>(channels, next, k, u, 1, 1, u / 2 + u % 2, true, ConvolutionNormalization.Weight, outputPadding: u % 2)));
            if (i + 1 < rates.Length)
            {
                int strideF0 = rates.Skip(i + 1).Aggregate(1, (a, b) => a * b);
                NoiseConvs.Add(StyleTtsOps.Own(layers, StyleTtsOps.PlainConv<T>(sourceChannels, next, strideF0 * 2, strideF0, (strideF0 + 1) / 2)));
                NoiseRes.Add(new StyleTts2AdaInResBlock<T>(engine, layers, next, 7, new[] { 1, 3, 5 }, styleDim));
            }
            else
            {
                NoiseConvs.Add(StyleTtsOps.Own(layers, StyleTtsOps.PlainConv<T>(sourceChannels, next, 1)));
                NoiseRes.Add(new StyleTts2AdaInResBlock<T>(engine, layers, next, 11, new[] { 1, 3, 5 }, styleDim));
            }
            if (type == StyleTts2DecoderType.HiFiGan) Alphas.Add(StyleTts2Ops.Alpha(layers, next));
            channels = next;
        }
        for (int i = 0; i < rates.Length; i++)
            for (int j = 0; j < kernels.Length; j++)
                ResBlocks.Add(new StyleTts2AdaInResBlock<T>(engine, layers, initialChannels >> (i + 1), kernels[j], dilations[j], styleDim));
        Post = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels, type == StyleTts2DecoderType.IstftNet ? istftFft + 2 : 1, 7));
        if (type == StyleTts2DecoderType.IstftNet)
        {
            _stft = new CenteredComplexStft<T>(engine, istftFft, istftHop, istftFft);
            _istft = new InverseStft<T>(engine, istftFft, istftHop, istftFft);
        }
        KernelCount = kernels.Length;
    }

    public StyleTts2Source<T> Source { get; }
    public List<NormedConv1DLayer<T>> Ups { get; } = new();
    public List<NormedConv1DLayer<T>> NoiseConvs { get; } = new();
    public List<StyleTts2AdaInResBlock<T>> NoiseRes { get; } = new();
    public List<StyleTts2AdaInResBlock<T>> ResBlocks { get; } = new();
    public List<TiedEmbeddingLayer<T>> Alphas { get; } = new();
    public NormedConv1DLayer<T> Post { get; }
    public int KernelCount { get; }

    private Tensor<T> LeakyRelu(Tensor<T> x, double slope) => StyleTtsOps.LeakyRelu(_engine, x, slope);

    /// <summary>The iSTFTNet source features: the magnitude and phase <c>[1, n_fft + 2, frames]</c> of the harmonic
    /// source's STFT (<c>torch.abs</c> and <c>torch.angle</c> of <c>torch.stft</c>).</summary>
    /// <remarks>An FFT returns exact zeros where the transform is exactly real or zero — the DC and Nyquist bins, every
    /// bin of the first frame (reflect padding makes it symmetric), the bins a smooth source leaves empty — and the angle
    /// there is 0 or π. The matrix DFT leaves rounding residue instead, whose sign would pick the angle, so components
    /// below 1e-12 of the frame's largest magnitude are taken as zero.</remarks>
    internal Tensor<T> SourceSpectrum(Tensor<T> source)
    {
        var (re, im) = _stft!.Forward(_engine.Reshape(source, new[] { source.Length }));
        int bins = re.Shape[1], frames = re.Shape[2];
        var features = new Tensor<T>(new[] { 1, 2 * bins, frames });
        for (int f = 0; f < frames; f++)
        {
            double peak = 0;
            for (int k = 0; k < bins; k++)
                peak = Math.Max(peak, Math.Max(Math.Abs(NumOps.ToDouble(re[0, k, f])), Math.Abs(NumOps.ToDouble(im[0, k, f]))));
            double floor = 1e-12 * peak;
            for (int k = 0; k < bins; k++)
            {
                double a = NumOps.ToDouble(re[0, k, f]), b = NumOps.ToDouble(im[0, k, f]);
                if (Math.Abs(a) <= floor) a = 0.0;
                if (Math.Abs(b) <= floor || k == 0 || 2 * k == _fft) b = 0.0;
                features[0, k, f] = NumOps.FromDouble(Math.Sqrt(a * a + b * b));
                features[0, bins + k, f] = NumOps.FromDouble(Math.Atan2(b, a));
            }
        }
        return features;
    }

    /// <summary>The waveform <c>[samples]</c> of features <c>[1, C, T]</c>, a style <c>[1, style]</c> and the F0 contour
    /// <c>[1, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> style, Tensor<T> f0, Random random)
    {
        // iSTFTNet computes its whole source, merge layer included, without gradients; the HiFi-GAN variant trains the merge.
        Tensor<T> har;
        if (_type == StyleTts2DecoderType.IstftNet)
        {
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                har = SourceSpectrum(Source.Forward(f0, random));
        }
        else
        {
            har = Source.Forward(f0, random);
        }
        for (int i = 0; i < Ups.Count; i++)
        {
            x = _type == StyleTts2DecoderType.IstftNet ? LeakyRelu(x, 0.1) : StyleTts2Ops.Snake(_engine, x, Alphas[i]);
            var xSource = NoiseRes[i].Forward(NoiseConvs[i].Forward(har), style);
            x = Ups[i].Forward(x);
            if (_type == StyleTts2DecoderType.IstftNet && i == Ups.Count - 1)
                x = VocoderOps.ReflectPad(_engine, x, 1, 0);
            x = _engine.TensorAdd(x, xSource);
            Tensor<T>? sum = null;
            for (int j = 0; j < KernelCount; j++)
            {
                var y = ResBlocks[i * KernelCount + j].Forward(x, style);
                sum = sum is null ? y : _engine.TensorAdd(sum, y);
            }
            x = _engine.TensorMultiplyScalar(sum!, NumOps.FromDouble(1.0 / KernelCount));
        }
        if (_type == StyleTts2DecoderType.HiFiGan)
        {
            x = Post.Forward(StyleTts2Ops.Snake(_engine, x, Alphas[Ups.Count]));
            var wave = _engine.Tanh(x);
            return _engine.Reshape(wave, new[] { wave.Length });
        }
        x = Post.Forward(LeakyRelu(x, 0.01));
        int bins = _fft / 2 + 1, frames = x.Shape[2];
        var magnitude = _engine.TensorExp(_engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, bins, frames }));
        var phase = _engine.TensorSin(_engine.TensorSlice(x, new[] { 0, bins, 0 }, new[] { 1, bins, frames }));
        return _istft!.Forward(magnitude, phase);
    }

    public void Load(TorchReader read, string name)
    {
        Source.Load(read, name + ".m_source");
        for (int i = 0; i < Ups.Count; i++)
        {
            StyleTtsOps.LoadWeightNorm(Ups[i], read, $"{name}.ups.{i}");
            StyleTtsOps.LoadPlain(NoiseConvs[i], read, $"{name}.noise_convs.{i}");
            NoiseRes[i].Load(read, $"{name}.noise_res.{i}");
        }
        for (int i = 0; i < ResBlocks.Count; i++) ResBlocks[i].Load(read, $"{name}.resblocks.{i}");
        for (int i = 0; i < Alphas.Count; i++)
            Alphas[i].LoadTable(read($"{name}.alphas.{i}", new[] { 1, Alphas[i].Dimension, 1 }));
        StyleTtsOps.LoadWeightNorm(Post, read, name + ".conv_post");
    }
}

/// <summary>
/// StyleTTS 2's decoder (the reference's <c>Decoder</c>): the pitch and energy contours each downsampled by a strided
/// 3-tap convolution; an AdaIN residual block over the aligned phoneme features with them; four AdaIN residual blocks
/// reading the contours and a 64-wide residual projection of the phoneme features again, the fourth upsampling by 2;
/// and the waveform generator, which also reads the F0 contour for its harmonic source.
/// </summary>
internal sealed class StyleTts2Decoder<T>
{
    private readonly IEngine _engine;

    public StyleTts2Decoder(IEngine engine, List<LayerBase<T>> layers, StyleTts2DecoderType type, int dimIn, int styleDim,
        int bottleneck, int residualDim, int[] kernels, int[] rates, int initialChannels, int[][] dilations, int[] upsampleKernels,
        int istftFft, int istftHop, int sampleRate)
    {
        _engine = engine;
        Encode = new StyleTtsAdaInResBlock<T>(engine, layers, dimIn + 2, bottleneck, styleDim, false);
        for (int i = 0; i < 3; i++) Decode.Add(new StyleTtsAdaInResBlock<T>(engine, layers, bottleneck + residualDim + 2, bottleneck, styleDim, false));
        Decode.Add(new StyleTtsAdaInResBlock<T>(engine, layers, bottleneck + residualDim + 2, initialChannels, styleDim, true));
        F0Conv = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(1, 1, 3, stride: 2, padding: 1));
        NConv = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(1, 1, 3, stride: 2, padding: 1));
        AsrResidual = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, residualDim, 1));
        Generator = new StyleTts2Generator<T>(engine, layers, type, styleDim, kernels, rates, initialChannels, dilations, upsampleKernels,
            istftFft, istftHop, sampleRate);
    }

    public StyleTtsAdaInResBlock<T> Encode { get; }
    public List<StyleTtsAdaInResBlock<T>> Decode { get; } = new();
    public NormedConv1DLayer<T> F0Conv { get; }
    public NormedConv1DLayer<T> NConv { get; }
    public NormedConv1DLayer<T> AsrResidual { get; }
    public StyleTts2Generator<T> Generator { get; }

    /// <summary>
    /// In training the reference smooths the contours with a box filter of a random width (pitch 0, 3 or 7; energy 0, 3,
    /// 7 or 15) before the decoder reads them.
    /// </summary>
    private Tensor<T> Smooth(Tensor<T> contour, int width)
    {
        if (width == 0) return contour;
        int t = contour.Length;
        var kernel = new Tensor<T>(new[] { 1, 1, 1, width });
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < width; i++) kernel[0, 0, 0, i] = ops.FromDouble(1.0 / width);
        var y = _engine.Conv2D(_engine.Reshape(contour, new[] { 1, 1, 1, t }), kernel, new[] { 1, 1 }, new[] { 0, width / 2 }, new[] { 1, 1 });
        return _engine.Reshape(y, new[] { 1, y.Length });
    }

    /// <summary>The waveform of aligned phoneme features <c>[1, dimIn, T]</c>, pitch and energy <c>[1, 2T]</c> and the
    /// acoustic style <c>[1, style]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> asr, Tensor<T> f0Curve, Tensor<T> energy, Tensor<T> style, bool training, Random random)
    {
        if (training)
        {
            int[] f0Widths = { 0, 3, 7 }, nWidths = { 0, 3, 7, 15 };
            f0Curve = Smooth(f0Curve, f0Widths[random.Next(3)]);
            energy = Smooth(energy, nWidths[random.Next(4)]);
        }
        var f0 = F0Conv.Forward(_engine.Reshape(f0Curve, new[] { 1, 1, f0Curve.Length }));
        var n = NConv.Forward(_engine.Reshape(energy, new[] { 1, 1, energy.Length }));
        var x = Encode.Forward(_engine.TensorConcatenate(new[] { asr, f0, n }, 1), style, training, random);
        var residual = AsrResidual.Forward(asr);
        bool concatenate = true;
        foreach (var block in Decode)
        {
            if (concatenate) x = _engine.TensorConcatenate(new[] { x, residual, f0, n }, 1);
            x = block.Forward(x, style, training, random);
            if (block.Upsample) concatenate = false;
        }
        return Generator.Forward(x, style, f0Curve, random);
    }

    public void Load(TorchReader read, string name)
    {
        Encode.Load(read, name + ".encode");
        for (int i = 0; i < Decode.Count; i++) Decode[i].Load(read, $"{name}.decode.{i}");
        StyleTtsOps.LoadWeightNorm(F0Conv, read, name + ".F0_conv");
        StyleTtsOps.LoadWeightNorm(NConv, read, name + ".N_conv");
        StyleTtsOps.LoadWeightNorm(AsrResidual, read, name + ".asr_res.0");
        Generator.Load(read, name + ".generator");
    }
}
