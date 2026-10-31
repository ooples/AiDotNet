using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of one of ChatTTS's convolutional stacks (<c>DVAEDecoder</c>): its input, output, ConvNeXt width,
/// depth and bottleneck.</summary>
internal sealed record ChatTtsStackConfiguration(int InputDim, int OutputDim, int Hidden, int Layers, int BottleneckDim,
    int Kernel = 7, int Dilation = 2);

/// <summary>
/// ChatTTS's convolutional stack (2noise/ChatTTS <c>DVAEDecoder</c>): an input convolution pair through a bottleneck
/// (<c>conv_in</c>: 3-wide convolutions with GELU between), ConvNeXt blocks, and a 1-wide output convolution without bias.
/// Each block is Vocos's ConvNeXt with a dilated depthwise convolution: x + γ ⊙ pw₂(GELU(pw₁(LN(dw(x))))).
/// </summary>
/// <remarks>The DVAE's encoder and decoder and the GPT-hidden-state decoder are all this stack. Tensors are
/// <c>[1, channels, frames]</c>; layer norms use ε = 1e-6 and GELU is exact, as in the reference.</remarks>
internal sealed class ChatTtsConvStack<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(NormedConv1DLayer<T> Depthwise, LayerNormalizationLayer<T> Norm, NormedConv1DLayer<T> Expand,
        NormedConv1DLayer<T> Project, ChannelScaleLayer<T> Scale);

    public ChatTtsConvStack(IEngine engine, List<LayerBase<T>> layers, ChatTtsStackConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int dilation, int groups, int padding, bool bias = true)
        {
            var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, dilation, groups, padding, false, ConvolutionNormalization.None, bias);
            layers.Add(conv);
            return conv;
        }
        InputConv = Conv(c.InputDim, c.BottleneckDim, 3, 1, 1, 1);
        BottleneckConv = Conv(c.BottleneckDim, c.Hidden, 3, 1, 1, 1);
        for (int i = 0; i < c.Layers; i++)
        {
            var depthwise = Conv(c.Hidden, c.Hidden, c.Kernel, c.Dilation, c.Hidden, c.Dilation * (c.Kernel / 2));
            var norm = new LayerNormalizationLayer<T>(c.Hidden, 1e-6);
            layers.Add(norm);
            var expand = Conv(c.Hidden, 4 * c.Hidden, 1, 1, 1, 0);
            var project = Conv(4 * c.Hidden, c.Hidden, 1, 1, 1, 0);
            // layer_scale_init_value = 1e-6.
            var scale = new ChannelScaleLayer<T>(c.Hidden, 1e-6);
            layers.Add(scale);
            Blocks.Add(new Block(depthwise, norm, expand, project, scale));
        }
        OutputConv = Conv(c.Hidden, c.OutputDim, 1, 1, 1, 0, bias: false);
    }

    public ChatTtsStackConfiguration Configuration { get; }
    public NormedConv1DLayer<T> InputConv { get; }
    public NormedConv1DLayer<T> BottleneckConv { get; }
    public List<Block> Blocks { get; } = new();
    public NormedConv1DLayer<T> OutputConv { get; }

    // Layer norm over the channels of [1, C, T].
    private Tensor<T> ChannelNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var rows = norm.Forward(_engine.TensorTranspose(_engine.Reshape(x, new[] { c, t })));
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, c, t });
    }

    private Tensor<T> Gelu(Tensor<T> x) => ActivationHelper.ExactGelu(x, _engine);

    /// <summary>The stack on <c>[1, InputDim, T]</c>, giving <c>[1, OutputDim, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = BottleneckConv.Forward(Gelu(InputConv.Forward(x)));
        foreach (var block in Blocks)
        {
            var h = block.Project.Forward(Gelu(block.Expand.Forward(ChannelNorm(block.Norm, block.Depthwise.Forward(y)))));
            y = _engine.TensorAdd(y, block.Scale.Forward(h));
        }
        return OutputConv.Forward(y);
    }

    /// <summary>Loads <c>{prefix}conv_in.0/2</c>, <c>{prefix}decoder_block.{i}.*</c> and <c>{prefix}conv_out</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        void Conv(NormedConv1DLayer<T> conv, string name, int output, int input, int kernel, bool bias = true) =>
            conv.LoadTorchWeights(read(name + ".weight", new[] { output, input, kernel }), null,
                bias ? read(name + ".bias", new[] { output }) : new double[output]);
        Conv(InputConv, prefix + "conv_in.0", c.BottleneckDim, c.InputDim, 3);
        Conv(BottleneckConv, prefix + "conv_in.2", c.Hidden, c.BottleneckDim, 3);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}decoder_block.{i}";
            Conv(b.Depthwise, p + ".dwconv", c.Hidden, 1, c.Kernel);
            TorchParameters.LayerNorm(_engine, b.Norm, read(p + ".norm.weight", new[] { c.Hidden }), read(p + ".norm.bias", new[] { c.Hidden }));
            // pwconv1 / pwconv2 are nn.Linear over channels; the 1-wide convolutions take them as [out, in, 1].
            b.Expand.LoadTorchWeights(read(p + ".pwconv1.weight", new[] { 4 * c.Hidden, c.Hidden }), null, read(p + ".pwconv1.bias", new[] { 4 * c.Hidden }));
            b.Project.LoadTorchWeights(read(p + ".pwconv2.weight", new[] { c.Hidden, 4 * c.Hidden }), null, read(p + ".pwconv2.bias", new[] { c.Hidden }));
            var gamma = read(p + ".weight", new[] { c.Hidden });
            for (int k = 0; k < c.Hidden; k++) b.Scale.Scale[k] = NumOps.FromDouble(gamma[k]);
            _engine.InvalidatePersistentTensor(b.Scale.Scale);
        }
        Conv(OutputConv, prefix + "conv_out", c.OutputDim, c.Hidden, 1, bias: false);
    }
}

/// <summary>
/// ChatTTS's grouped residual finite scalar quantizer (<c>GFSQ</c> over vector-quantize-pytorch 1.14's
/// <c>GroupedResidualFSQ</c>): the feature is split into G groups; each group is projected to len(levels) dimensions,
/// bounded, and quantized by R residual FSQ stages whose scales are (L − 1)^−r, then projected back.
/// </summary>
/// <remarks>
/// <para>FSQ (Mentzer et al. 2023): bound(z) = tanh(z + atanh(o / h)) · h − o with h = (L − 1)(1 + 1e-3)/2 and o = 0.5 for
/// even L (0 for odd); round; divide by ⌊L/2⌋. A code's index is Σ (digit + ⌊L/2⌋) · basis with basis the running
/// product of the levels.</para>
/// <para>The residual starts from the bounded projection, as 1.14 does (later versions start from a soft-clamped input
/// and scale by L^−r instead, which decodes ChatTTS's codes to different features).</para>
/// </remarks>
internal sealed class ChatTtsGroupedFsq<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int[] _levels;
    private readonly int[] _basis;

    public ChatTtsGroupedFsq(IEngine engine, List<LayerBase<T>> layers, int dim, int[] levels, int groups, int residuals, Random initialization)
    {
        _engine = engine;
        if (dim % groups != 0) throw new ArgumentException("The feature width must divide by the groups.", nameof(groups));
        _levels = (int[])levels.Clone();
        _basis = new int[levels.Length];
        int product = 1;
        for (int i = 0; i < levels.Length; i++) { _basis[i] = product; product *= levels[i]; }
        CodebookSize = product;
        Dim = dim;
        Groups = groups;
        Residuals = residuals;
        int perGroup = dim / groups;
        for (int g = 0; g < groups; g++)
        {
            var projectIn = new DenseLayer<T>(levels.Length, (IActivationFunction<T>)new IdentityActivation<T>());
            var projectOut = new DenseLayer<T>(perGroup, (IActivationFunction<T>)new IdentityActivation<T>());
            // nn.Linear's initialization: weights and biases from U(±1/√fan_in).
            double[] Uniform(int count, int fanIn)
            {
                var values = new double[count];
                double bound = 1.0 / Math.Sqrt(fanIn);
                for (int i = 0; i < count; i++) values[i] = bound * (2 * initialization.NextDouble() - 1);
                return values;
            }
            TorchParameters.Linear(engine, projectIn, perGroup, levels.Length, Uniform(levels.Length * perGroup, perGroup), Uniform(levels.Length, perGroup));
            TorchParameters.Linear(engine, projectOut, levels.Length, perGroup, Uniform(perGroup * levels.Length, levels.Length), Uniform(perGroup, levels.Length));
            layers.Add(projectIn);
            layers.Add(projectOut);
            ProjectIn.Add(projectIn);
            ProjectOut.Add(projectOut);
        }
    }

    public int Dim { get; }
    public int Groups { get; }
    public int Residuals { get; }

    /// <summary>Codes per stage (the product of the levels: 625 for (5, 5, 5, 5)).</summary>
    public int CodebookSize { get; }

    public List<DenseLayer<T>> ProjectIn { get; } = new();
    public List<DenseLayer<T>> ProjectOut { get; } = new();

    private double Scale(int residual, int d) => Math.Pow(_levels[d] - 1, -residual);

    // FSQ's bound of one value at dimension d.
    private double Bound(double z, int d)
    {
        double half = (_levels[d] - 1) * (1 + 1e-3) / 2, offset = _levels[d] % 2 == 0 ? 0.5 : 0.0;
        double shift = Atanh(offset / half);
        return Math.Tanh(z + shift) * half - offset;
    }

    private static double Atanh(double x) => 0.5 * Math.Log((1 + x) / (1 - x));

    /// <summary>
    /// The codes of features <c>[T, Dim]</c>: <c>[T, G · R]</c>, the group's stages consecutive (ChatTTS's
    /// <c>b t (g r)</c>). Quantization reads the features without a gradient.
    /// </summary>
    public int[,] Encode(Tensor<T> features)
    {
        int frames = features.Shape[0], perGroup = Dim / Groups, d = _levels.Length;
        var codes = new int[frames, Groups * Residuals];
        for (int g = 0; g < Groups; g++)
        {
            var slice = _engine.TensorSlice(features, new[] { 0, g * perGroup }, new[] { frames, perGroup });
            var projected = ProjectIn[g].Forward(slice);                                                    // [T, d]
            for (int t = 0; t < frames; t++)
            {
                var residual = new double[d];
                for (int k = 0; k < d; k++) residual[k] = Bound(NumOps.ToDouble(projected[t, k]), k);
                for (int r = 0; r < Residuals; r++)
                {
                    int index = 0;
                    for (int k = 0; k < d; k++)
                    {
                        double scale = Scale(r, k);
                        int half = _levels[k] / 2;
                        double digit = Math.Round(Bound(residual[k] / scale, k), MidpointRounding.ToEven);
                        residual[k] -= digit / half * scale;
                        index += (int)(digit + half) * _basis[k];
                    }
                    codes[t, g * Residuals + r] = index;
                }
            }
        }
        return codes;
    }

    /// <summary>The features <c>[T, Dim]</c> of codes <c>[T, G · R]</c> (<c>get_output_from_indices</c>).</summary>
    public Tensor<T> Decode(int[,] codes)
    {
        int frames = codes.GetLength(0), d = _levels.Length;
        if (codes.GetLength(1) != Groups * Residuals)
            throw new ArgumentException($"Expected {Groups * Residuals} codes per frame.", nameof(codes));
        var outputs = new List<Tensor<T>>();
        for (int g = 0; g < Groups; g++)
        {
            var summed = new Tensor<T>(new[] { frames, d });
            for (int t = 0; t < frames; t++)
                for (int r = 0; r < Residuals; r++)
                {
                    int index = codes[t, g * Residuals + r];
                    if (index < 0 || index >= CodebookSize)
                        throw new ArgumentOutOfRangeException(nameof(codes), $"Codes lie in [0, {CodebookSize}).");
                    for (int k = 0; k < d; k++)
                    {
                        int half = _levels[k] / 2, digit = index / _basis[k] % _levels[k];
                        summed[t, k] = NumOps.Add(summed[t, k], NumOps.FromDouble((digit - half) / (double)half * Scale(r, k)));
                    }
                }
            outputs.Add(ProjectOut[g].Forward(summed));
        }
        return _engine.TensorConcatenate(outputs.ToArray(), 1);
    }

    /// <summary>Loads <c>{prefix}quantizer.rvqs.{g}.project_in/out</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        int perGroup = Dim / Groups, d = _levels.Length;
        for (int g = 0; g < Groups; g++)
        {
            string p = $"{prefix}quantizer.rvqs.{g}";
            TorchParameters.Linear(_engine, ProjectIn[g], perGroup, d, read(p + ".project_in.weight", new[] { d, perGroup }), read(p + ".project_in.bias", new[] { d }));
            TorchParameters.Linear(_engine, ProjectOut[g], d, perGroup, read(p + ".project_out.weight", new[] { perGroup, d }), read(p + ".project_out.bias", new[] { perGroup }));
        }
    }
}

/// <summary>
/// A ChatTTS mel decoder (2noise/ChatTTS <c>DVAE</c> decode path): features <c>[1, 2·dim, T]</c> are split into two
/// halves interleaved in time (<c>view(B, 2, dim, T).permute(0, 2, 3, 1).flatten(2)</c>, so each frame becomes two),
/// run through a <see cref="ChatTtsConvStack{T}"/>, mapped to 100 mel bands by a bias-free 3-wide convolution and
/// multiplied by the per-band coefficients <c>coef</c>. The DVAE decodes quantized codes this way and the "Decoder"
/// decodes the GPT's hidden states.
/// </summary>
internal sealed class ChatTtsMelDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _mels;

    public ChatTtsMelDecoder(IEngine engine, List<LayerBase<T>> layers, ChatTtsStackConfiguration stack, int dim, int mels)
    {
        _engine = engine;
        _mels = mels;
        Dim = dim;
        Stack = new ChatTtsConvStack<T>(engine, layers, stack);
        OutputConv = new NormedConv1DLayer<T>(dim, mels, 3, 1, 1, 1, 1, false, ConvolutionNormalization.None, false);
        layers.Add(OutputConv);
        // The reference draws coef from U(0, 1) when none is given; a checkpoint stores it.
        Coefficients = new ChannelScaleLayer<T>(mels, 1.0);
        layers.Add(Coefficients);
    }

    public int Dim { get; }
    public ChatTtsConvStack<T> Stack { get; }
    public NormedConv1DLayer<T> OutputConv { get; }

    /// <summary>The per-band <c>coef</c> buffer.</summary>
    public ChannelScaleLayer<T> Coefficients { get; }

    /// <summary>The mel <c>[1, mels, 2T]</c> of features <c>[1, 2·Dim, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> features)
    {
        int t = features.Shape[2];
        if (features.Shape[1] != 2 * Dim) throw new ArgumentException($"Expected {2 * Dim} feature channels.", nameof(features));
        var interleaved = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(features, new[] { 1, 2, Dim, t }), new[] { 0, 2, 3, 1 }),
            new[] { 1, Dim, 2 * t });
        return Coefficients.Forward(OutputConv.Forward(Stack.Forward(interleaved)));
    }

    /// <summary>Loads <c>{prefix}decoder.*</c>, <c>{prefix}out_conv.weight</c> and <c>{prefix}coef</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        Stack.LoadTorchWeights(read, prefix + "decoder.");
        OutputConv.LoadTorchWeights(read(prefix + "out_conv.weight", new[] { _mels, Dim, 3 }), null, new double[_mels]);
        var coef = read(prefix + "coef", new[] { 1, _mels, 1 });
        for (int m = 0; m < _mels; m++) Coefficients.Scale[m] = NumOps.FromDouble(coef[m]);
        _engine.InvalidatePersistentTensor(Coefficients.Scale);
    }
}

/// <summary>
/// ChatTTS's DVAE encoder (2noise/ChatTTS <c>DVAE</c> encode path): the waveform's log-mel (torchaudio
/// <c>MelSpectrogram</c> at 24 kHz, n_fft 1024, hop 256, 100 HTK bands, power 1, centred; <c>log(max(x, 1e-5))</c>)
/// divided by <c>coef</c>, a downsampling convolution pair (3-wide, then 4-wide with stride 2, GELU after each) to half
/// the frame rate, the encoder stack, and the grouped FSQ, giving <c>G · R</c> codes per frame.
/// </summary>
internal sealed class ChatTtsDvaeEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _fft, _hop, _mels;
    private readonly Tensor<T> _window;
    private readonly Tensor<T> _filterbank;   // [bins, mels]

    public ChatTtsDvaeEncoder(IEngine engine, List<LayerBase<T>> layers, ChatTtsStackConfiguration stack, int dim, int sampleRate,
        int fftSize, int hopSize, int mels, ChatTtsGroupedFsq<T> quantizer)
    {
        _engine = engine;
        _fft = fftSize;
        _hop = hopSize;
        _mels = mels;
        Dim = dim;
        Quantizer = quantizer;
        _window = new Tensor<T>(new[] { fftSize });
        // torch.hann_window: the periodic window.
        for (int i = 0; i < fftSize; i++) _window[i] = NumOps.FromDouble(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / fftSize));
        var weights = MelFilterbanks.Htk(sampleRate, fftSize, mels, 0.0, sampleRate / 2.0);
        int bins = fftSize / 2 + 1;
        _filterbank = new Tensor<T>(new[] { bins, mels });
        for (int m = 0; m < mels; m++)
            for (int k = 0; k < bins; k++) _filterbank[k, m] = NumOps.FromDouble(weights[m, k]);
        Downsample1 = new NormedConv1DLayer<T>(mels, dim, 3, 1, 1, 1, 1, false, ConvolutionNormalization.None);
        Downsample2 = new NormedConv1DLayer<T>(dim, dim, 4, 2, 1, 1, 1, false, ConvolutionNormalization.None);
        layers.Add(Downsample1);
        layers.Add(Downsample2);
        Stack = new ChatTtsConvStack<T>(engine, layers, stack);
    }

    public int Dim { get; }
    public NormedConv1DLayer<T> Downsample1 { get; }
    public NormedConv1DLayer<T> Downsample2 { get; }
    public ChatTtsConvStack<T> Stack { get; }
    public ChatTtsGroupedFsq<T> Quantizer { get; }

    /// <summary>The log-mel <c>[1, mels, frames]</c> of a waveform <c>[samples]</c>.</summary>
    public Tensor<T> LogMel(Tensor<T> audio)
    {
        var waveform = _engine.Reshape(audio, new[] { 1, audio.Length });
        _engine.STFT(waveform, _fft, _hop, _window, center: true, out var magnitude, out _);
        int bins = _fft / 2 + 1, frames = magnitude.Length / bins;
        var framesFirst = _engine.TensorTranspose(_engine.Reshape(magnitude, new[] { bins, frames }));       // [T, bins]
        var mel = _engine.TensorMatMul(framesFirst, _filterbank);                                         // [T, mels]
        var log = _engine.TensorLog(_engine.TensorClampMin(mel, NumOps.FromDouble(1e-5)));
        return _engine.Reshape(_engine.TensorTranspose(log), new[] { 1, _mels, frames });
    }

    /// <summary>The codes <c>[T, G · R]</c> of a waveform, from <paramref name="coefficients"/>-scaled mels.</summary>
    public int[,] Encode(Tensor<T> audio, Tensor<T> coefficients)
    {
        var mel = LogMel(audio);
        int frames = mel.Shape[2];
        var scaled = _engine.TensorDivide(mel, _engine.TensorBroadcastTo(_engine.Reshape(coefficients, new[] { 1, _mels, 1 }), new[] { 1, _mels, frames }));
        var x = ActivationHelper.ExactGelu(Downsample2.Forward(ActivationHelper.ExactGelu(Downsample1.Forward(scaled), _engine)), _engine);
        var encoded = Stack.Forward(x);                                                                   // [1, 2·dim·?, T/2]
        var features = _engine.TensorTranspose(_engine.Reshape(encoded, new[] { encoded.Shape[1], encoded.Shape[2] }));
        return Quantizer.Encode(features);
    }

    /// <summary>Loads <c>{prefix}downsample_conv.0/2</c>, <c>{prefix}encoder.*</c> and the quantizer.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        Downsample1.LoadTorchWeights(read(prefix + "downsample_conv.0.weight", new[] { Dim, _mels, 3 }), null, read(prefix + "downsample_conv.0.bias", new[] { Dim }));
        Downsample2.LoadTorchWeights(read(prefix + "downsample_conv.2.weight", new[] { Dim, Dim, 4 }), null, read(prefix + "downsample_conv.2.bias", new[] { Dim }));
        Stack.LoadTorchWeights(read, prefix + "encoder.");
        Quantizer.LoadTorchWeights(read, prefix + "vq_layer.");
    }
}
