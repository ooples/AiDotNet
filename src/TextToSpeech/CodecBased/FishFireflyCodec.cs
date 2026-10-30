using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of Fish Speech's Firefly-GAN codec (the release's <c>firefly_gan_vq.yaml</c>).</summary>
internal sealed record FishFireflyConfiguration(int SampleRate, int FftSize, int HopSize, int MelChannels, int[] EncoderDepths, int[] EncoderDims,
    int QuantizerDim, int Groups, int[] Levels, int[] DownsampleFactors, int[] UpsampleRates, int[] UpsampleKernels, int[] ResBlockKernels,
    int[][] ResBlockDilations, int HeadChannels, int PreKernel, int PostKernel);

/// <summary>
/// Fish Speech's Firefly-GAN codec (Liao et al. 2024, §3.2; the release's <c>FireflyArchitecture</c>): a log-mel
/// spectrogram, a ConvNeXt encoder, the grouped finite scalar quantizer (§3.2.2: downsampled twice by strided
/// convolutions with ConvNeXt blocks, eight groups of FSQ with levels 8·5·5·5, upsampled back) and a HiFi-GAN generator
/// whose multi-receptive-field fusion is a ParallelBlock (the mean of three residual blocks with SiLU). Its convolutions
/// pad on the left (the release's <c>FishConvNet</c>), transposed ones trim on the right.
/// </summary>
internal sealed class FishFireflyCodec<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Random _random;
    private readonly double[,] _melBasis;      // [bins, mels]

    internal sealed record FishConv(NormedConv1DLayer<T> Conv, int Kernel, int Dilation, int Stride);

    internal sealed record ConvNeXt(FishConv Depthwise, LayerNormalizationLayer<T> Norm, DenseLayer<T> Expand, DenseLayer<T> Project,
        BiasFreeLinearLayer<T> Gamma);

    public FishFireflyCodec(IEngine engine, List<LayerBase<T>> layers, FishFireflyConfiguration configuration, Random random)
    {
        _engine = engine;
        _random = random;
        Configuration = configuration;
        var c = configuration;
        _melBasis = MelBasis(c.SampleRate, c.FftSize, c.MelChannels);
        int dim = c.EncoderDims[^1];
        if (dim != c.QuantizerDim) throw new ArgumentException("The quantizer reads the encoder's last width.", nameof(configuration));
        Stem = Conv(layers, c.MelChannels, c.EncoderDims[0], 7, 1, 1, 1, ConvolutionNormalization.None);
        StemNorm = Norm(layers, c.EncoderDims[0]);
        for (int i = 1; i < c.EncoderDims.Length; i++)
        {
            var norm = Norm(layers, c.EncoderDims[i - 1]);
            var conv = new NormedConv1DLayer<T>(c.EncoderDims[i - 1], c.EncoderDims[i], 1, 1, 1, 1, 0, false, ConvolutionNormalization.None);
            layers.Add(conv);
            Initialize(conv, 0.02);
            Downsample.Add((norm, conv));
        }
        for (int i = 0; i < c.EncoderDims.Length; i++)
            Stages.Add(Enumerable.Range(0, c.EncoderDepths[i]).Select(_ => NewConvNeXt(layers, c.EncoderDims[i])).ToList());
        EncoderNorm = Norm(layers, dim);
        foreach (int factor in c.DownsampleFactors)
            QuantizerDown.Add((Conv(layers, dim, dim, factor, factor, 1, 1, ConvolutionNormalization.None), NewConvNeXt(layers, dim)));
        Quantizer = new ChatTtsGroupedFsq<T>(engine, layers, dim, c.Levels, c.Groups, 1, random);
        foreach (int factor in Enumerable.Reverse(c.DownsampleFactors))
            QuantizerUp.Add((Transposed(layers, dim, dim, factor, factor, ConvolutionNormalization.None), NewConvNeXt(layers, dim)));
        Pre = Conv(layers, dim, c.HeadChannels, c.PreKernel, 1, 1, 1, ConvolutionNormalization.Weight, kaiming: true);
        int channels = c.HeadChannels;
        for (int i = 0; i < c.UpsampleRates.Length; i++)
        {
            int next = c.HeadChannels >> (i + 1);
            Ups.Add(Transposed(layers, channels, next, c.UpsampleKernels[i], c.UpsampleRates[i], ConvolutionNormalization.Weight, kaiming: true));
            channels = next;
            var blocks = new List<List<(FishConv C1, FishConv C2)>>();
            for (int j = 0; j < c.ResBlockKernels.Length; j++)
                blocks.Add(c.ResBlockDilations[j].Select(d => (Conv(layers, next, next, c.ResBlockKernels[j], 1, d, 1, ConvolutionNormalization.Weight, kaiming: true),
                    Conv(layers, next, next, c.ResBlockKernels[j], 1, d, 1, ConvolutionNormalization.Weight, kaiming: true))).ToList());
            Parallel.Add(blocks);
        }
        Post = Conv(layers, channels, 1, c.PostKernel, 1, 1, 1, ConvolutionNormalization.Weight, kaiming: true);
    }

    public FishFireflyConfiguration Configuration { get; }
    public FishConv Stem { get; }
    public LayerNormalizationLayer<T> StemNorm { get; }
    public List<(LayerNormalizationLayer<T> Norm, NormedConv1DLayer<T> Conv)> Downsample { get; } = new();
    public List<List<ConvNeXt>> Stages { get; } = new();
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public List<(FishConv Conv, ConvNeXt Block)> QuantizerDown { get; } = new();
    public ChatTtsGroupedFsq<T> Quantizer { get; }
    public List<(FishConv Conv, ConvNeXt Block)> QuantizerUp { get; } = new();
    public FishConv Pre { get; }
    public List<FishConv> Ups { get; } = new();
    public List<List<List<(FishConv C1, FishConv C2)>>> Parallel { get; } = new();
    public FishConv Post { get; }

    /// <summary>Codes per second.</summary>
    public double FrameRate => Configuration.SampleRate / (double)(Configuration.HopSize * Configuration.DownsampleFactors.Aggregate(1, (a, b) => a * b));

    // ---------------------------------------------------------------- construction

    private void Initialize(NormedConv1DLayer<T> conv, double std)
    {
        // trunc_normal_(std) within ±2 (absolute bounds, 100σ away) and zero biases.
        conv.LoadTorchWeights(IndexTtsOps.Normal(_random, conv.TorchWeightShape.Aggregate(1, (a, b) => a * b), std), null, new double[conv.OutputChannels]);
    }

    private FishConv Conv(List<LayerBase<T>> layers, int inputs, int outputs, int kernel, int stride, int dilation, int groups,
        ConvolutionNormalization normalization, bool kaiming = false)
    {
        var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, stride, dilation, groups, 0, false, normalization);
        layers.Add(conv);
        if (kaiming)
        {
            // nn.Conv1d's default: U(±1/√fan_in) weights and biases (the release's init_weights matches no FishConvNet).
            int fanIn = inputs / groups * kernel;
            conv.LoadTorchWeights(IndexTtsOps.Uniform(_random, conv.TorchWeightShape.Aggregate(1, (a, b) => a * b), fanIn), null,
                IndexTtsOps.Uniform(_random, outputs, fanIn));
        }
        else
        {
            Initialize(conv, 0.02);
        }
        return new FishConv(conv, kernel, dilation, stride);
    }

    private FishConv Transposed(List<LayerBase<T>> layers, int inputs, int outputs, int kernel, int stride, ConvolutionNormalization normalization,
        bool kaiming = false)
    {
        var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, stride, 1, 1, 0, true, normalization);
        layers.Add(conv);
        if (kaiming)
        {
            int fanIn = outputs * kernel;
            conv.LoadTorchWeights(IndexTtsOps.Uniform(_random, conv.TorchWeightShape.Aggregate(1, (a, b) => a * b), fanIn), null,
                IndexTtsOps.Uniform(_random, outputs, fanIn));
        }
        else
        {
            Initialize(conv, 0.02);
        }
        return new FishConv(conv, kernel, 1, stride);
    }

    private static LayerNormalizationLayer<T> Norm(List<LayerBase<T>> layers, int width)
    {
        var norm = new LayerNormalizationLayer<T>(width, 1e-6);
        layers.Add(norm);
        return norm;
    }

    private ConvNeXt NewConvNeXt(List<LayerBase<T>> layers, int dim)
    {
        var depthwise = Conv(layers, dim, dim, 7, 1, 1, dim, ConvolutionNormalization.None);
        var norm = Norm(layers, dim);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(layer);
            TorchParameters.Linear(_engine, layer, inputs, outputs, IndexTtsOps.Normal(_random, inputs * outputs, 0.02), new double[outputs]);
            return layer;
        }
        return new ConvNeXt(depthwise, norm, Linear(dim, 4 * dim), Linear(4 * dim, dim), IndexTtsOps.Values(layers, Enumerable.Repeat(1e-6, dim).ToArray()));
    }

    /// <summary>torchaudio's <c>melscale_fbanks</c> with Slaney's scale and area normalization, <c>[bins, mels]</c> (the
    /// release computes it in float32; this in double).</summary>
    internal static double[,] MelBasis(int sampleRate, int fftSize, int mels)
    {
        int bins = fftSize / 2 + 1;
        double top = sampleRate / 2;
        double HzToMel(double hz) => hz >= 1000 ? 15.0 + Math.Log(hz / 1000.0) / (Math.Log(6.4) / 27.0) : hz / (200.0 / 3);
        double MelToHz(double mel) => mel >= 15.0 ? 1000.0 * Math.Exp(Math.Log(6.4) / 27.0 * (mel - 15.0)) : mel * 200.0 / 3;
        var freqs = Enumerable.Range(0, bins).Select(k => top * k / (bins - 1)).ToArray();
        double low = HzToMel(0), high = HzToMel(top);
        var points = Enumerable.Range(0, mels + 2).Select(i => MelToHz(low + (high - low) * i / (mels + 1))).ToArray();
        var basis = new double[bins, mels];
        for (int m = 0; m < mels; m++)
        {
            double enorm = 2.0 / (points[m + 2] - points[m]);
            for (int k = 0; k < bins; k++)
            {
                double down = (freqs[k] - points[m]) / (points[m + 1] - points[m]);
                double up = (points[m + 2] - freqs[k]) / (points[m + 2] - points[m + 1]);
                basis[k, m] = Math.Max(0, Math.Min(down, up)) * enorm;
            }
        }
        return basis;
    }

    // ---------------------------------------------------------------- the pieces

    private Tensor<T> ZeroPad(Tensor<T> x, int left, int right)
    {
        int channels = x.Shape[1], length = x.Shape[2];
        var parts = new List<Tensor<T>>();
        if (left > 0) parts.Add(new Tensor<T>(new[] { 1, channels, left }));
        parts.Add(x);
        if (right > 0) parts.Add(new Tensor<T>(new[] { 1, channels, right }));
        return parts.Count == 1 ? x : _engine.TensorConcatenate(parts.ToArray(), 2);
    }

    /// <summary><c>FishConvNet</c>: kernel − stride zeros on the left and enough on the right for whole frames.</summary>
    private Tensor<T> Apply(FishConv conv, Tensor<T> x)
    {
        int kernel = (conv.Kernel - 1) * conv.Dilation + 1, length = x.Shape[2], pad = kernel - conv.Stride;
        double frames = (length - kernel + pad) / (double)conv.Stride + 1;
        int ideal = ((int)Math.Ceiling(frames) - 1) * conv.Stride + (kernel - pad);
        return conv.Conv.Forward(ZeroPad(x, pad, ideal - length));
    }

    /// <summary><c>FishTransConvNet</c>: the transposed convolution, kernel − stride samples trimmed on the right.</summary>
    private Tensor<T> ApplyTransposed(FishConv conv, Tensor<T> x)
    {
        var y = conv.Conv.Forward(x);
        int trim = conv.Kernel - conv.Stride;
        return trim > 0 ? _engine.TensorNarrow(y, 2, 0, y.Shape[2] - trim) : y;
    }

    // A layer norm over channels of [1, C, L] (channels_first).
    private Tensor<T> ChannelNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int channels = x.Shape[1], length = x.Shape[2];
        var rows = norm.Forward(_engine.TensorTranspose(_engine.Reshape(x, new[] { channels, length })));
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, channels, length });
    }

    private Tensor<T> Apply(ConvNeXt block, Tensor<T> x)
    {
        int channels = x.Shape[1];
        var h = Apply(block.Depthwise, x);
        int length = h.Shape[2];
        var rows = block.Norm.Forward(_engine.TensorTranspose(_engine.Reshape(h, new[] { channels, length })));       // [L, C]
        rows = block.Project.Forward(ActivationHelper.ExactGelu(block.Expand.Forward(rows), _engine));
        rows = _engine.TensorMultiply(rows, _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(block.Gamma), new[] { 1, channels }), new[] { length, channels }));
        return _engine.TensorAdd(x, _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, channels, length }));
    }

    private Tensor<T> Silu(Tensor<T> x) => _engine.TensorMultiply(x, _engine.Sigmoid(x));

    // ---------------------------------------------------------------- encode and decode

    /// <summary>The log mel <c>[1, mels, frames]</c> of a waveform (<c>LogMelSpectrogram</c>: reflect padding of
    /// (n_fft − hop)/2 and (n_fft − hop + 1)/2, a Hann-windowed, uncentred STFT, √(re² + im² + 1e-6), Slaney filters,
    /// <c>log(max(x, 1e-5))</c>).</summary>
    public Tensor<T> Mel(Tensor<T> audio)
    {
        var c = Configuration;
        int length = audio.Length, fft = c.FftSize, hop = c.HopSize, bins = fft / 2 + 1;
        int left = (fft - hop) / 2, right = (fft - hop + 1) / 2, total = length + left + right, frames = (total - fft) / hop + 1;
        var index = new Tensor<int>(new[] { frames * fft });
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < fft; n++) index[f * fft + n] = Vocoders.DifferentiableMel<T>.Reflect(f * hop + n - left, length);
        var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), index, 0), new[] { frames, fft });
        var cos = new Tensor<T>(new[] { fft, bins });
        var sin = new Tensor<T>(new[] { fft, bins });
        for (int n = 0; n < fft; n++)
        {
            double w = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / fft);
            for (int k = 0; k < bins; k++)
            {
                double a = 2 * Math.PI * n * k / fft;
                cos[n, k] = NumOps.FromDouble(w * Math.Cos(a));
                sin[n, k] = NumOps.FromDouble(-w * Math.Sin(a));
            }
        }
        var re = _engine.TensorMatMul(framed, cos);
        var im = _engine.TensorMatMul(framed, sin);
        var magnitude = _engine.TensorSqrt(_engine.TensorAddScalar(_engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im)), NumOps.FromDouble(1e-6)));
        var basis = new Tensor<T>(new[] { bins, c.MelChannels });
        for (int k = 0; k < bins; k++)
            for (int m = 0; m < c.MelChannels; m++) basis[k, m] = NumOps.FromDouble(_melBasis[k, m]);
        var mel = _engine.TensorMatMul(magnitude, basis);
        var floored = _engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(mel, NumOps.FromDouble(-1e-5))), NumOps.FromDouble(1e-5));
        return _engine.Reshape(_engine.TensorTranspose(_engine.TensorLog(floored)), new[] { 1, c.MelChannels, frames });
    }

    /// <summary>The codes <c>[groups, frames]</c> of a waveform (<c>encode</c>).</summary>
    public int[,] Encode(Tensor<T> audio) => EncodeMel(Mel(audio));

    /// <summary>The codes <c>[groups, frames]</c> of a log mel <c>[1, mels, frames]</c>.</summary>
    public int[,] EncodeMel(Tensor<T> mel)
    {
        var x = ChannelNorm(StemNorm, Apply(Stem, mel));
        foreach (var block in Stages[0]) x = Apply(block, x);
        for (int i = 0; i < Downsample.Count; i++)
        {
            x = Downsample[i].Conv.Forward(ChannelNorm(Downsample[i].Norm, x));
            foreach (var block in Stages[i + 1]) x = Apply(block, x);
        }
        x = ChannelNorm(EncoderNorm, x);
        foreach (var (conv, block) in QuantizerDown) x = Apply(block, Apply(conv, x));
        int dim = x.Shape[1], frames = x.Shape[2];
        var codes = Quantizer.Encode(_engine.TensorTranspose(_engine.Reshape(x, new[] { dim, frames })));       // [frames, groups]
        int groups = codes.GetLength(1);
        var result = new int[groups, frames];
        for (int g = 0; g < groups; g++)
            for (int t = 0; t < frames; t++) result[g, t] = codes[t, g];
        return result;
    }

    /// <summary>The waveform <c>[frames · 4 · hop]</c> of codes <c>[groups, frames]</c> (<c>decode</c>).</summary>
    public Tensor<T> Decode(int[,] codes)
    {
        int groups = codes.GetLength(0), frames = codes.GetLength(1);
        var transposed = new int[frames, groups];
        for (int g = 0; g < groups; g++)
            for (int t = 0; t < frames; t++) transposed[t, g] = codes[g, t];
        var z = Quantizer.Decode(transposed);                                                                  // [frames, dim]
        var x = _engine.Reshape(_engine.TensorTranspose(z), new[] { 1, z.Shape[1], frames });
        foreach (var (conv, block) in QuantizerUp) x = Apply(block, ApplyTransposed(conv, x));
        x = Apply(Pre, x);
        for (int i = 0; i < Ups.Count; i++)
        {
            x = ApplyTransposed(Ups[i], Silu(x));
            Tensor<T>? sum = null;
            foreach (var block in Parallel[i])
            {
                var h = x;
                foreach (var (c1, c2) in block) h = _engine.TensorAdd(h, Apply(c2, Silu(Apply(c1, Silu(h)))));
                sum = sum is null ? h : _engine.TensorAdd(sum, h);
            }
            x = _engine.TensorMultiplyScalar(sum ?? throw new InvalidOperationException("A ParallelBlock has no residual blocks."),
                NumOps.FromDouble(1.0 / Parallel[i].Count));
        }
        var y = _engine.Tanh(Apply(Post, Silu(x)));
        return _engine.Reshape(y, new[] { y.Length });
    }

    // ---------------------------------------------------------------- loading

    /// <summary>Loads the release's generator checkpoint names (<c>backbone.*</c>, <c>quantizer.*</c>, <c>head.*</c>; weight
    /// norm as parametrizations or <c>weight_g</c>/<c>weight_v</c>).</summary>
    public void Load(Func<string, double[]?> take)
    {
        double[] Need(string name) => take(name) ?? throw new InvalidDataException($"The Firefly checkpoint has no tensor '{name}'.");
        void Conv(NormedConv1DLayer<T> conv, string name)
        {
            var bias = Need(name + ".bias");
            var v = take(name + ".parametrizations.weight.original1") ?? take(name + ".weight_v");
            if (v is not null)
                conv.LoadTorchWeights(v, take(name + ".parametrizations.weight.original0") ?? Need(name + ".weight_g"), bias);
            else
                conv.LoadTorchWeights(Need(name + ".weight"), null, bias);
        }
        void LayerNorm(LayerNormalizationLayer<T> norm, string name) => TorchParameters.LayerNorm(_engine, norm, Need(name + ".weight"), Need(name + ".bias"));
        void Block(ConvNeXt block, string name)
        {
            Conv(block.Depthwise.Conv, name + ".dwconv.conv");
            LayerNorm(block.Norm, name + ".norm");
            int d = block.Depthwise.Conv.OutputChannels;
            TorchParameters.Linear(_engine, block.Expand, d, 4 * d, Need(name + ".pwconv1.weight"), Need(name + ".pwconv1.bias"));
            TorchParameters.Linear(_engine, block.Project, 4 * d, d, Need(name + ".pwconv2.weight"), Need(name + ".pwconv2.bias"));
            TorchParameters.Linear(block.Gamma, 1, d, Need(name + ".gamma"));
        }
        Conv(Stem.Conv, "backbone.downsample_layers.0.0.conv");
        LayerNorm(StemNorm, "backbone.downsample_layers.0.1");
        for (int i = 0; i < Downsample.Count; i++)
        {
            LayerNorm(Downsample[i].Norm, $"backbone.downsample_layers.{i + 1}.0");
            Conv(Downsample[i].Conv, $"backbone.downsample_layers.{i + 1}.1");
        }
        for (int i = 0; i < Stages.Count; i++)
            for (int j = 0; j < Stages[i].Count; j++) Block(Stages[i][j], $"backbone.stages.{i}.{j}");
        LayerNorm(EncoderNorm, "backbone.norm");
        for (int i = 0; i < QuantizerDown.Count; i++)
        {
            Conv(QuantizerDown[i].Conv.Conv, $"quantizer.downsample.{i}.0.conv");
            Block(QuantizerDown[i].Block, $"quantizer.downsample.{i}.1");
        }
        Quantizer.LoadTorchWeights((name, _) => Need(name.Replace("quantizer.rvqs.", "quantizer.residual_fsq.rvqs.")), "");
        for (int i = 0; i < QuantizerUp.Count; i++)
        {
            Conv(QuantizerUp[i].Conv.Conv, $"quantizer.upsample.{i}.0.conv");
            Block(QuantizerUp[i].Block, $"quantizer.upsample.{i}.1");
        }
        Conv(Pre.Conv, "head.conv_pre.conv");
        for (int i = 0; i < Ups.Count; i++)
        {
            Conv(Ups[i].Conv, $"head.ups.{i}.conv");
            for (int j = 0; j < Parallel[i].Count; j++)
                for (int m = 0; m < Parallel[i][j].Count; m++)
                {
                    Conv(Parallel[i][j][m].C1.Conv, $"head.resblocks.{i}.blocks.{j}.convs1.{m}.conv");
                    Conv(Parallel[i][j][m].C2.Conv, $"head.resblocks.{i}.blocks.{j}.convs2.{m}.conv");
                }
        }
        Conv(Post.Conv, "head.conv_post.conv");
    }
}
