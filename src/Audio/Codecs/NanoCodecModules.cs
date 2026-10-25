using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using EnginePadMode = AiDotNet.Tensors.Engines.PadMode;

namespace AiDotNet.Audio.Codecs;

/// <summary>The shape of NVIDIA NeMo's NanoCodec; the defaults are <c>nemo-nano-codec-22khz-0.6kbps-12.5fps</c>, the
/// codec KaniTTS speaks through.</summary>
internal sealed record NanoCodecConfiguration
{
    public int SampleRate { get; init; } = 22050;

    /// <summary>The encoder's first width, doubled by every downsampling (<c>HiFiGANEncoder.base_channels</c>).</summary>
    public int EncoderChannels { get; init; } = 24;

    public int[] DownsampleRates { get; init; } = { 2, 3, 6, 7, 7 };

    /// <summary>The encoder's output width, split over the quantizer's groups (<c>encoded_dim</c>).</summary>
    public int EncodedDim { get; init; } = 16;

    /// <summary>The decoder's first width, halved by every upsampling (<c>CausalHiFiGANDecoder.base_channels</c>).</summary>
    public int DecoderChannels { get; init; } = 864;

    public int[] UpsampleRates { get; init; } = { 7, 7, 6, 3, 2 };

    public int[] ResidualKernels { get; init; } = { 3, 7, 11 };

    public int[] ResidualDilations { get; init; } = { 1, 3, 5 };

    public int InputKernel { get; init; } = 7;

    public int EncoderOutputKernel { get; init; } = 7;

    public int DecoderOutputKernel { get; init; } = 3;

    /// <summary>The encoder's activation: "lrelu", "elu", "snake" or "half_snake".</summary>
    public string EncoderActivation { get; init; } = "lrelu";

    /// <summary>The decoder's activation.</summary>
    public string DecoderActivation { get; init; } = "half_snake";

    /// <summary>The encoder's convolution padding (<c>pad_mode</c>): "replicate", "reflect" or "zeros".</summary>
    public string EncoderPadMode { get; init; } = "replicate";

    /// <summary>The decoder's output activation: "clamp" (to [−1, 1]) or "tanh".</summary>
    public string OutputActivation { get; init; } = "clamp";

    /// <summary>Whether each upsampling convolution has one group per output channel (<c>n_groups_equal_to_out_channels</c>).</summary>
    public bool GroupedUpsampling { get; init; } = true;

    /// <summary>The finite scalar quantizer's groups, one codebook each.</summary>
    public int QuantizerGroups { get; init; } = 4;

    /// <summary>The levels of each dimension of a group's code (their product is the codebook size: 9·8·8·7 = 4032).</summary>
    public int[] QuantizerLevels { get; init; } = { 9, 8, 8, 7 };

    /// <summary>FSQ's scale-down before rounding (<c>eps</c>).</summary>
    public double QuantizerEpsilon { get; init; } = 1e-3;

    /// <summary>Audio samples per code frame (the product of the downsampling rates).</summary>
    internal int SamplesPerFrame => DownsampleRates.Aggregate(1, (a, r) => a * r);

    /// <summary>Codes per codebook.</summary>
    internal int CodebookSize => QuantizerLevels.Aggregate(1, (a, l) => a * l);
}

/// <summary>NeMo's <c>CodecActivation</c>: leaky ReLU (slope 0.01), ELU, Snake, or HalfSnake (Snake on the first half of
/// the channels, leaky ReLU on the rest).</summary>
internal sealed class NanoCodecActivation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly string _kind;
    private readonly int _channels;

    public NanoCodecActivation(IEngine engine, string kind, int channels, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _kind = kind.ToLowerInvariant();
        _channels = channels;
        switch (_kind)
        {
            case "lrelu":
            case "elu":
                break;
            case "snake":
                Snake = new SnakeLayer<T>(channels);
                owner.Add(Snake);
                break;
            case "half_snake":
                Snake = new SnakeLayer<T>(channels / 2);
                owner.Add(Snake);
                break;
            default:
                throw new ArgumentException($"Unknown activation '{kind}'.", nameof(kind));
        }
    }

    /// <summary>Whether Snake covers only the first half of the channels.</summary>
    public bool IsHalfSnake => _kind == "half_snake";

    /// <summary>The Snake part's α, null for leaky ReLU and ELU.</summary>
    public SnakeLayer<T>? Snake { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        switch (_kind)
        {
            case "lrelu":
                return _engine.LeakyReLU(x, NumOps.FromDouble(0.01));
            case "elu":
                return _engine.ELU(x, 1.0);
            case "snake":
                return Snake!.Forward(x);
            default:
                int batch = x.Shape[0], time = x.Shape[2], half = _channels / 2;
                var snake = Snake!.Forward(_engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, half, time }));
                var relu = _engine.LeakyReLU(_engine.TensorSlice(x, new[] { 0, half, 0 }, new[] { batch, _channels - half, time }), NumOps.FromDouble(0.01));
                return _engine.TensorConcatenate(new[] { snake, relu }, 1);
        }
    }
}

/// <summary>NeMo's <c>Conv1dNorm</c> (a weight-normalized convolution padded by <c>(k·d − d) / 2</c>, or a given padding,
/// on both sides with the given mode) and <c>CausalConv1dNorm</c> (padded on the left by the effective kernel minus the
/// stride, and on the right as far as a whole last frame needs, with zeros).</summary>
internal sealed class NanoCodecConv<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly bool _causal;
    private readonly int _padding, _stride, _effectiveKernel;
    private readonly EnginePadMode _mode;

    public NanoCodecConv(IEngine engine, int input, int output, int kernel, int stride, int dilation, bool causal, string padMode,
        List<LayerBase<T>> owner, int? padding = null)
    {
        _engine = engine;
        _causal = causal;
        _stride = stride;
        _effectiveKernel = (kernel - 1) * dilation + 1;
        _padding = padding is > 0 ? padding.Value : (kernel * dilation - dilation) / 2;
        _mode = padMode switch
        {
            "replicate" => EnginePadMode.Replicate,
            "reflect" => EnginePadMode.Reflect,
            "zeros" => EnginePadMode.Constant,
            _ => throw new ArgumentException($"Unknown padding '{padMode}'.", nameof(padMode)),
        };
        Conv = new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, 1, 0, false, ConvolutionNormalization.Weight);
        owner.Add(Conv);
    }

    public NormedConv1DLayer<T> Conv { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        if (!_causal)
            return Conv.Forward(_padding > 0 ? _engine.PadNd(x, new[] { _padding, _padding }, _mode, NumOps.Zero) : x);
        int length = x.Shape[2], total = _effectiveKernel - _stride;
        // EncodecConv1d._get_extra_padding_for_conv1d: enough on the right for a whole last frame.
        double frames = (length - _effectiveKernel + total) / (double)_stride + 1;
        int ideal = ((int)Math.Ceiling(frames) - 1) * _stride + _effectiveKernel - total;
        int extra = ideal - length;
        return Conv.Forward(total > 0 || extra > 0 ? _engine.PadNd(x, new[] { total, extra }, EnginePadMode.Constant, NumOps.Zero) : x);
    }
}

/// <summary>NeMo's <c>CausalConvTranspose1dNorm</c>: a weight-normalized transposed convolution (one group per output
/// channel when grouped) with its <c>kernel − stride</c> extra samples trimmed from the right.</summary>
internal sealed class NanoCodecUpsample<T>
{
    private readonly IEngine _engine;
    private readonly int _stride;

    public NanoCodecUpsample(IEngine engine, int input, int output, int stride, bool grouped, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _stride = stride;
        Conv = new NormedConv1DLayer<T>(input, output, 2 * stride, stride, 1, grouped ? output : 1, 0, true, ConvolutionNormalization.Weight);
        owner.Add(Conv);
    }

    public NormedConv1DLayer<T> Conv { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Conv.Forward(x);
        return _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { y.Shape[0], y.Shape[1], x.Shape[2] * _stride });
    }
}

/// <summary>NeMo's <c>ResidualBlock</c>: activation, a dilated convolution, activation, a convolution, added to the input.</summary>
internal sealed class NanoCodecResidualBlock<T>
{
    private readonly IEngine _engine;

    public NanoCodecResidualBlock(IEngine engine, int channels, int kernel, int dilation, string activation, bool causal, string padMode,
        List<LayerBase<T>> owner)
    {
        _engine = engine;
        InputActivation = new NanoCodecActivation<T>(engine, activation, channels, owner);
        SkipActivation = new NanoCodecActivation<T>(engine, activation, channels, owner);
        InputConv = new NanoCodecConv<T>(engine, channels, channels, kernel, 1, dilation, causal, padMode, owner);
        SkipConv = new NanoCodecConv<T>(engine, channels, channels, kernel, 1, 1, causal, padMode, owner);
    }

    public NanoCodecActivation<T> InputActivation { get; }
    public NanoCodecActivation<T> SkipActivation { get; }
    public NanoCodecConv<T> InputConv { get; }
    public NanoCodecConv<T> SkipConv { get; }

    public Tensor<T> Forward(Tensor<T> x) =>
        _engine.TensorAdd(x, SkipConv.Forward(SkipActivation.Forward(InputConv.Forward(InputActivation.Forward(x)))));
}

/// <summary>NeMo's <c>HiFiGANResLayer</c>: one chain of residual blocks (one per dilation) per kernel size, averaged.</summary>
internal sealed class NanoCodecResidualLayer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public NanoCodecResidualLayer(IEngine engine, int channels, int[] kernels, int[] dilations, string activation, bool causal, string padMode,
        List<LayerBase<T>> owner)
    {
        _engine = engine;
        Chains = kernels.Select(k => dilations.Select(d => new NanoCodecResidualBlock<T>(engine, channels, k, d, activation, causal, padMode, owner)).ToArray()).ToArray();
    }

    public NanoCodecResidualBlock<T>[][] Chains { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        Tensor<T>? sum = null;
        foreach (var chain in Chains)
        {
            var y = x;
            foreach (var block in chain) y = block.Forward(y);
            sum = sum is null ? y : _engine.TensorAdd(sum, y);
        }
        return _engine.TensorMultiplyScalar(sum!, NumOps.FromDouble(1.0 / Chains.Length));
    }
}

/// <summary>NeMo's <c>HiFiGANEncoder</c>: a convolution to the base width, then per rate a residual layer, activation and a
/// strided convolution (kernel 2·rate) doubling the width, then activation and a convolution to the encoded width.</summary>
internal sealed class NanoCodecEncoder<T>
{
    public NanoCodecEncoder(IEngine engine, NanoCodecConfiguration c, List<LayerBase<T>> owner)
    {
        PreConv = new NanoCodecConv<T>(engine, 1, c.EncoderChannels, c.InputKernel, 1, 1, false, c.EncoderPadMode, owner);
        int channels = c.EncoderChannels;
        foreach (int rate in c.DownsampleRates)
        {
            var residual = new NanoCodecResidualLayer<T>(engine, channels, c.ResidualKernels, c.ResidualDilations, c.EncoderActivation, false, c.EncoderPadMode, owner);
            var activation = new NanoCodecActivation<T>(engine, c.EncoderActivation, channels, owner);
            var down = new NanoCodecConv<T>(engine, channels, 2 * channels, 2 * rate, rate, 1, false, c.EncoderPadMode, owner, padding: (rate + 1) / 2);
            Blocks.Add((residual, activation, down));
            channels *= 2;
        }
        PostActivation = new NanoCodecActivation<T>(engine, c.EncoderActivation, channels, owner);
        PostConv = new NanoCodecConv<T>(engine, channels, c.EncodedDim, c.EncoderOutputKernel, 1, 1, false, c.EncoderPadMode, owner);
    }

    public NanoCodecConv<T> PreConv { get; }
    public List<(NanoCodecResidualLayer<T> Residual, NanoCodecActivation<T> Activation, NanoCodecConv<T> Down)> Blocks { get; } = new();
    public NanoCodecActivation<T> PostActivation { get; }
    public NanoCodecConv<T> PostConv { get; }

    /// <summary>The encoding <c>[B, encoded, frames]</c> of audio <c>[B, 1, samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> audio)
    {
        var x = PreConv.Forward(audio);
        foreach (var (residual, activation, down) in Blocks) x = down.Forward(activation.Forward(residual.Forward(x)));
        return PostConv.Forward(PostActivation.Forward(x));
    }
}

/// <summary>NeMo's <c>CausalHiFiGANDecoder</c>: a causal convolution to the base width, then per rate activation, a causal
/// transposed convolution halving the width and a causal residual layer, then activation, a causal convolution to one
/// channel and the output activation.</summary>
internal sealed class NanoCodecDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly string _output;

    public NanoCodecDecoder(IEngine engine, NanoCodecConfiguration c, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _output = c.OutputActivation;
        if (_output is not ("clamp" or "tanh")) throw new ArgumentException($"Unknown output activation '{_output}'.", nameof(c));
        PreConv = new NanoCodecConv<T>(engine, c.EncodedDim, c.DecoderChannels, c.InputKernel, 1, 1, true, "zeros", owner);
        int channels = c.DecoderChannels;
        foreach (int rate in c.UpsampleRates)
        {
            var activation = new NanoCodecActivation<T>(engine, c.DecoderActivation, channels, owner);
            var up = new NanoCodecUpsample<T>(engine, channels, channels / 2, rate, c.GroupedUpsampling, owner);
            channels /= 2;
            var residual = new NanoCodecResidualLayer<T>(engine, channels, c.ResidualKernels, c.ResidualDilations, c.DecoderActivation, true, "zeros", owner);
            Blocks.Add((activation, up, residual));
        }
        PostActivation = new NanoCodecActivation<T>(engine, c.DecoderActivation, channels, owner);
        PostConv = new NanoCodecConv<T>(engine, channels, 1, c.DecoderOutputKernel, 1, 1, true, "zeros", owner);
    }

    public NanoCodecConv<T> PreConv { get; }
    public List<(NanoCodecActivation<T> Activation, NanoCodecUpsample<T> Up, NanoCodecResidualLayer<T> Residual)> Blocks { get; } = new();
    public NanoCodecActivation<T> PostActivation { get; }
    public NanoCodecConv<T> PostConv { get; }

    /// <summary>The audio <c>[B, samples]</c> of a dequantized encoding <c>[B, encoded, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> encoding)
    {
        var x = PreConv.Forward(encoding);
        foreach (var (activation, up, residual) in Blocks) x = residual.Forward(up.Forward(activation.Forward(x)));
        var y = PostConv.Forward(PostActivation.Forward(x));
        y = _output == "clamp" ? _engine.TensorClamp(y, NumOps.FromDouble(-1), NumOps.FromDouble(1)) : _engine.Tanh(y);
        return _engine.Reshape(y, new[] { y.Shape[0], y.Shape[2] });
    }
}

/// <summary>
/// NeMo's <c>GroupFiniteScalarQuantizer</c> (Mentzer et al. 2023, per group as in HiFi-Codec): the encoding splits into
/// groups, each dimension is compressed to <c>s·tanh(x + tan(o / s)) − o</c> (s = (L − 1)/2 · (1 − ε), o = ½ for an even
/// L), rounded and scaled by <c>⌊L/2⌋</c>, and a group's code is its digits in the mixed radix of the levels.
/// </summary>
internal sealed class NanoCodecQuantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly NanoCodecConfiguration _c;

    public NanoCodecQuantizer(NanoCodecConfiguration configuration)
    {
        _c = configuration;
        if (_c.EncodedDim != _c.QuantizerGroups * _c.QuantizerLevels.Length)
            throw new ArgumentException("The encoded width must be the groups times the levels per group.", nameof(configuration));
    }

    /// <summary>The codes <c>[groups, frames]</c> of an encoding <c>[1, encoded, frames]</c>.</summary>
    public int[,] Encode(Tensor<T> encoding)
    {
        int dims = _c.QuantizerLevels.Length, frames = encoding.Shape[2];
        var codes = new int[_c.QuantizerGroups, frames];
        var values = encoding.ToArray();
        for (int g = 0; g < _c.QuantizerGroups; g++)
            for (int t = 0; t < frames; t++)
            {
                int index = 0, radix = 1;
                for (int d = 0; d < dims; d++)
                {
                    int levels = _c.QuantizerLevels[d];
                    double scale = (levels - 1) / 2.0 * (1 - _c.QuantizerEpsilon);
                    double offset = levels % 2 == 0 ? 0.5 : 0;
                    double shift = Math.Tan(offset / scale);
                    double x = NumOps.ToDouble(values[(g * dims + d) * frames + t]);
                    double compressed = scale * Math.Tanh(x + shift) - offset;
                    // torch.round rounds half to even; codes / ⌊L/2⌋ then back: the digit is the rounded value plus ⌊L/2⌋.
                    int digit = (int)Math.Round(compressed, MidpointRounding.ToEven) + levels / 2;
                    index += digit * radix;
                    radix *= levels;
                }
                codes[g, t] = index;
            }
        return codes;
    }

    /// <summary>The dequantized encoding <c>[1, encoded, frames]</c> of codes <c>[groups, frames]</c>
    /// (<c>FiniteScalarQuantizer.decode</c>).</summary>
    public Tensor<T> Decode(int[,] codes)
    {
        int dims = _c.QuantizerLevels.Length, frames = codes.GetLength(1);
        if (codes.GetLength(0) != _c.QuantizerGroups) throw new ArgumentException($"Expected {_c.QuantizerGroups} codebooks.", nameof(codes));
        var output = new Tensor<T>(new[] { 1, _c.EncodedDim, frames });
        for (int g = 0; g < _c.QuantizerGroups; g++)
            for (int t = 0; t < frames; t++)
            {
                int index = codes[g, t], radix = 1;
                // NeMo reads each digit as (index // radix) % levels, so a code past the codebook wraps its last digit.
                if (index < 0) throw new ArgumentOutOfRangeException(nameof(codes), $"Code {index} is negative.");
                for (int d = 0; d < dims; d++)
                {
                    int levels = _c.QuantizerLevels[d], half = levels / 2;
                    int digit = index / radix % levels;
                    output[0, g * dims + d, t] = NumOps.FromDouble((digit - half) / (double)half);
                    radix *= levels;
                }
            }
        return output;
    }
}

/// <summary>
/// NVIDIA NeMo's NanoCodec (<c>AudioCodecModel</c> with a HiFi-GAN encoder, finite scalar quantization in groups and a
/// causal HiFi-GAN decoder): audio to codes (zero-padded to whole frames) and codes to audio. Parameter names follow the
/// NeMo checkpoint (<c>audio_encoder.*</c>, <c>audio_decoder.*</c>; weight norm as <c>conv.weight_g</c> and
/// <c>conv.weight_v</c>, or as <c>conv.parametrizations.weight.original0</c> and <c>original1</c>).
/// </summary>
internal sealed class NanoCodec<T>
{
    private readonly IEngine _engine;

    public NanoCodec(IEngine engine, NanoCodecConfiguration configuration, List<LayerBase<T>> encoderLayers, List<LayerBase<T>> decoderLayers)
    {
        _engine = engine;
        Configuration = configuration;
        if (configuration.SamplesPerFrame != configuration.UpsampleRates.Aggregate(1, (a, r) => a * r))
            throw new ArgumentException("The decoder must upsample as far as the encoder downsamples.", nameof(configuration));
        Encoder = new NanoCodecEncoder<T>(engine, configuration, encoderLayers);
        Decoder = new NanoCodecDecoder<T>(engine, configuration, decoderLayers);
        Quantizer = new NanoCodecQuantizer<T>(configuration);
    }

    public NanoCodecConfiguration Configuration { get; }
    public NanoCodecEncoder<T> Encoder { get; }
    public NanoCodecDecoder<T> Decoder { get; }
    public NanoCodecQuantizer<T> Quantizer { get; }

    /// <summary>The codes <c>[groups, frames]</c> of audio at the codec's rate (<c>AudioCodecModel.encode</c>: zero-padded
    /// to whole frames, encoded and quantized).</summary>
    public int[,] Encode(Tensor<T> audio)
    {
        int length = audio.Length, spf = Configuration.SamplesPerFrame;
        if (length == 0) throw new ArgumentException("There is no audio.", nameof(audio));
        int padded = spf * ((length + spf - 1) / spf);
        var x = _engine.Reshape(audio, new[] { 1, 1, length });
        if (padded > length) x = _engine.PadNd(x, new[] { 0, padded - length }, EnginePadMode.Constant, MathHelper.GetNumericOperations<T>().Zero);
        return Quantizer.Encode(Encoder.Forward(x));
    }

    /// <summary>The audio <c>[samples]</c> of codes <c>[groups, frames]</c> (<c>AudioCodecModel.decode</c>).</summary>
    public Tensor<T> Decode(int[,] codes)
    {
        var audio = Decoder.Forward(Quantizer.Decode(codes));
        return _engine.Reshape(audio, new[] { audio.Length });
    }

    /// <summary>Loads a NeMo NanoCodec state dict (the encoder's and decoder's parameters; the quantizer has none).</summary>
    public void LoadNemoWeights(Func<string, double[]?> read)
    {
        double[] Need(string name) => read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        // torch's weight-norm parametrization (original0 the gain, original1 the direction), or the older weight_norm's
        // weight_g and weight_v, which the released checkpoint uses.
        void Conv(NormedConv1DLayer<T> conv, string prefix)
        {
            var direction = read($"{prefix}.conv.parametrizations.weight.original1") ?? Need($"{prefix}.conv.weight_v");
            var gain = read($"{prefix}.conv.parametrizations.weight.original0") ?? Need($"{prefix}.conv.weight_g");
            conv.LoadTorchWeights(direction, gain, Need($"{prefix}.conv.bias"));
        }
        void Activation(NanoCodecActivation<T> activation, string prefix)
        {
            if (activation.Snake is null) return;
            activation.Snake.LoadAlpha(Need(activation.IsHalfSnake ? $"{prefix}.activation.snake_act.alpha" : $"{prefix}.activation.alpha"));
        }
        void Residual(NanoCodecResidualLayer<T> layer, string prefix)
        {
            for (int k = 0; k < layer.Chains.Length; k++)
                for (int d = 0; d < layer.Chains[k].Length; d++)
                {
                    var block = layer.Chains[k][d];
                    string p = $"{prefix}.res_blocks.{k}.res_blocks.{d}";
                    Activation(block.InputActivation, $"{p}.input_activation");
                    Conv(block.InputConv.Conv, $"{p}.input_conv");
                    Activation(block.SkipActivation, $"{p}.skip_activation");
                    Conv(block.SkipConv.Conv, $"{p}.skip_conv");
                }
        }
        Conv(Encoder.PreConv.Conv, "audio_encoder.pre_conv");
        for (int i = 0; i < Encoder.Blocks.Count; i++)
        {
            Residual(Encoder.Blocks[i].Residual, $"audio_encoder.res_layers.{i}");
            Activation(Encoder.Blocks[i].Activation, $"audio_encoder.activations.{i}");
            Conv(Encoder.Blocks[i].Down.Conv, $"audio_encoder.down_sample_conv_layers.{i}");
        }
        Activation(Encoder.PostActivation, "audio_encoder.post_activation");
        Conv(Encoder.PostConv.Conv, "audio_encoder.post_conv");
        Conv(Decoder.PreConv.Conv, "audio_decoder.pre_conv");
        for (int i = 0; i < Decoder.Blocks.Count; i++)
        {
            Activation(Decoder.Blocks[i].Activation, $"audio_decoder.activations.{i}");
            Conv(Decoder.Blocks[i].Up.Conv, $"audio_decoder.up_sample_conv_layers.{i}");
            Residual(Decoder.Blocks[i].Residual, $"audio_decoder.res_layers.{i}");
        }
        Activation(Decoder.PostActivation, "audio_decoder.post_activation");
        Conv(Decoder.PostConv.Conv, "audio_decoder.post_conv");
    }
}
