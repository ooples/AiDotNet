using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using EnginePadMode = AiDotNet.Tensors.Engines.PadMode;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>The shape of MegaTTS 3's WaveVAE; the defaults are the release's (<c>WavVAE_V3</c>).</summary>
internal sealed record MegaTts3WaveVaeConfiguration
{
    public int[] EncoderRatios { get; init; } = { 6, 5, 4, 4, 2 };
    public int EncoderFilters { get; init; } = 32;
    public int EncoderDimension { get; init; } = 512;
    public int LatentChannels { get; init; } = 32;
    public int UpsamplerChannels { get; init; } = 320;
    public int LatentUpsampling { get; init; } = 4;
    public int GeneratorFilters { get; init; } = 128;
    public int[] GeneratorRatios { get; init; } = { 5, 4, 4, 3 };
}

/// <summary>The release's <c>Upsample</c>: <c>x + sin x</c>, then the sum of a nearest-neighbour upsampling followed by
/// leaky ReLU, reflection padding and a weight-normalized convolution, and a leaky ReLU followed by a weight-normalized
/// transposed convolution; each halves the channels.</summary>
internal sealed class MegaTts3Upsample<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _rate;

    public MegaTts3Upsample(IEngine engine, List<LayerBase<T>> layers, int channels, int rate)
    {
        _engine = engine;
        _rate = rate;
        int rk = rate >= 5 ? rate : 5;
        Conv = new NormedConv1DLayer<T>(channels, channels / 2, 7, 1, 1, 1, 0, false, ConvolutionNormalization.Weight);
        Transposed = new NormedConv1DLayer<T>(channels, channels / 2, rk * 2, rate, 1, 1, rk - rate / 2, true, ConvolutionNormalization.Weight,
            outputPadding: rate % 2);
        layers.Add(Conv);
        layers.Add(Transposed);
    }

    public NormedConv1DLayer<T> Conv { get; }
    public NormedConv1DLayer<T> Transposed { get; }

    /// <summary>Upsamples <c>[B, C, T]</c> to <c>[B, C/2, T·rate]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _engine.TensorAdd(_engine.TensorSin(x), x);
        int b = x.Shape[0], c = x.Shape[1], t = x.Shape[2];
        var repeated = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(x, new[] { b, c, t, 1 }), new[] { b, c, t, _rate }), new[] { b, c, t * _rate });
        var slope = NumOps.FromDouble(0.2);
        var near = Conv.Forward(_engine.PadNd(_engine.LeakyReLU(repeated, slope), new[] { 3, 3 }, EnginePadMode.Reflect, NumOps.Zero));
        var trans = Transposed.Forward(_engine.LeakyReLU(x, slope));
        return _engine.TensorAdd(near, trans);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        void Load(NormedConv1DLayer<T> conv, string name) =>
            conv.LoadTorchWeights(read($"{name}.weight_v"), read($"{name}.weight_g"), read($"{name}.bias"));
        Load(Conv, $"{prefix}.upsample.3");
        Load(Transposed, $"{prefix}.trans_upsample.1");
    }
}

/// <summary>The release's <c>ResBlockMRFV2Inter</c>: three HiFi-GAN v2 residual stacks (kernels 3, 7, 11; each three
/// dilated (1, 3, 5) then plain convolutions with leaky ReLU 0.2), averaged.</summary>
internal sealed class MegaTts3ResidualStacks<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTts3ResidualStacks(IEngine engine, List<LayerBase<T>> layers, int channels)
    {
        _engine = engine;
        foreach (int kernel in new[] { 3, 7, 11 })
        {
            var stack = new List<(NormedConv1DLayer<T>, NormedConv1DLayer<T>)>();
            foreach (int dilation in new[] { 1, 3, 5 })
            {
                var c1 = new NormedConv1DLayer<T>(channels, channels, kernel, 1, dilation, 1, (kernel * dilation - dilation) / 2, false, ConvolutionNormalization.Weight);
                var c2 = new NormedConv1DLayer<T>(channels, channels, kernel, 1, 1, 1, (kernel - 1) / 2, false, ConvolutionNormalization.Weight);
                layers.Add(c1);
                layers.Add(c2);
                stack.Add((c1, c2));
            }
            Stacks.Add(stack);
        }
    }

    public List<List<(NormedConv1DLayer<T> Dilated, NormedConv1DLayer<T> Plain)>> Stacks { get; } = new();

    public Tensor<T> Forward(Tensor<T> x)
    {
        var slope = NumOps.FromDouble(0.2);
        Tensor<T>? sum = null;
        foreach (var stack in Stacks)
        {
            var y = x;
            foreach (var (dilated, plain) in stack)
                y = _engine.TensorAdd(plain.Forward(_engine.LeakyReLU(dilated.Forward(_engine.LeakyReLU(y, slope)), slope)), y);
            sum = sum is null ? y : _engine.TensorAdd(sum, y);
        }
        return _engine.TensorMultiplyScalar(sum!, NumOps.FromDouble(1.0 / 3));
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        for (int s = 0; s < Stacks.Count; s++)
            for (int j = 0; j < Stacks[s].Count; j++)
            {
                string block = $"{prefix}.block{s + 1}";
                Stacks[s][j].Dilated.LoadTorchWeights(read($"{block}.convs1.{j}.weight_v"), read($"{block}.convs1.{j}.weight_g"), read($"{block}.convs1.{j}.bias"));
                Stacks[s][j].Plain.LoadTorchWeights(read($"{block}.convs2.{j}.weight_v"), read($"{block}.convs2.{j}.weight_g"), read($"{block}.convs2.{j}.bias"));
            }
    }
}

/// <summary>
/// MegaTTS 3's WaveVAE (<c>WavVAE_V3</c>): a SEANet encoder (EnCodec's, with a two-layer LSTM) to 512 channels at 25 Hz and
/// a linear map to a diagonal Gaussian (mean and log-variance, clamped to [−30, 20]) over a 32-wide latent; the decoder maps
/// a latent to 320 channels, upsamples by 4 and runs the HiFi-GAN-style generator (ngf 128, upsampling 5·4·4·3, tanh) to
/// 24 kHz audio.
/// </summary>
internal sealed class MegaTts3WaveVae<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTts3WaveVae(IEngine engine, List<LayerBase<T>> encoderLayers, List<LayerBase<T>> decoderLayers, MegaTts3WaveVaeConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Encoder = new SeanetEncoder<T>(engine, new SeanetConfig
        {
            Channels = 1, Dimension = c.EncoderDimension, Filters = c.EncoderFilters, ResidualLayers = 1, Ratios = (int[])c.EncoderRatios.Clone(),
            KernelSize = 7, LastKernelSize = 7, DilationBase = 2, ResidualKernelSizes = new[] { 3, 1 }, Causal = false, Reflect = true,
            TrueSkip = false, Compress = 2, LstmLayers = 2, Normalization = ConvolutionNormalization.Weight,
        }, encoderLayers);
        ToPosterior = MegaTts3Ops.Linear(encoderLayers, 2 * c.LatentChannels);
        ToDecoder = MegaTts3Ops.Linear(decoderLayers, c.UpsamplerChannels);
        LatentUpsampler = new MegaTts3Upsample<T>(engine, decoderLayers, c.UpsamplerChannels, c.LatentUpsampling);
        int mult = 1 << c.GeneratorRatios.Length;
        GeneratorInput = new NormedConv1DLayer<T>(c.UpsamplerChannels / 2, mult * c.GeneratorFilters, 7, 1, 1, 1, 0, false, ConvolutionNormalization.Weight);
        decoderLayers.Add(GeneratorInput);
        foreach (int rate in c.GeneratorRatios)
        {
            var up = new MegaTts3Upsample<T>(engine, decoderLayers, mult * c.GeneratorFilters, rate);
            var stacks = new MegaTts3ResidualStacks<T>(engine, decoderLayers, mult * c.GeneratorFilters / 2);
            Stages.Add((up, stacks));
            mult /= 2;
        }
        GeneratorOutput = new NormedConv1DLayer<T>(c.GeneratorFilters, 1, 7, 1, 1, 1, 0, false, ConvolutionNormalization.Weight);
        decoderLayers.Add(GeneratorOutput);
    }

    public MegaTts3WaveVaeConfiguration Configuration { get; }
    public SeanetEncoder<T> Encoder { get; }
    public DenseLayer<T> ToPosterior { get; }
    public DenseLayer<T> ToDecoder { get; }
    public MegaTts3Upsample<T> LatentUpsampler { get; }
    public NormedConv1DLayer<T> GeneratorInput { get; }
    public List<(MegaTts3Upsample<T> Up, MegaTts3ResidualStacks<T> Stacks)> Stages { get; } = new();
    public NormedConv1DLayer<T> GeneratorOutput { get; }

    /// <summary>Samples per latent frame (960: 40 ms at 24 kHz).</summary>
    public int Hop => Configuration.EncoderRatios.Aggregate(1, (a, r) => a * r);

    /// <summary><c>encode</c>: the posterior mean and log-variance <c>[1, frames, latent]</c> of audio <c>[samples]</c>.</summary>
    public (Tensor<T> Mean, Tensor<T> LogVariance) Encode(Tensor<T> audio)
    {
        int n = Configuration.LatentChannels;
        var h = Encoder.Forward(_engine.Reshape(audio, new[] { 1, 1, audio.Length }));                       // [1, D, frames]
        var z = MegaTts3Ops.Rows(_engine, ToPosterior, _engine.TensorPermute(h, new[] { 0, 2, 1 }));           // [1, frames, 2n]
        int frames = z.Shape[1];
        var mean = _engine.TensorSlice(z, new[] { 0, 0, 0 }, new[] { 1, frames, n });
        var logVariance = _engine.TensorClamp(_engine.TensorSlice(z, new[] { 0, 0, n }, new[] { 1, frames, n }), NumOps.FromDouble(-30), NumOps.FromDouble(20));
        return (mean, logVariance);
    }

    /// <summary><c>encode_latent</c>: a latent <c>[1, frames, latent]</c> drawn from the posterior with Gaussian
    /// <paramref name="noise"/> of the same shape (the mean when null).</summary>
    public Tensor<T> EncodeLatent(Tensor<T> audio, Func<int[], Tensor<T>> draw)
    {
        var (mean, logVariance) = Encode(audio);
        var std = _engine.TensorExp(_engine.TensorMultiplyScalar(logVariance, NumOps.FromDouble(0.5)));
        return _engine.TensorAdd(mean, _engine.TensorMultiply(std, draw(mean.Shape.ToArray())));
    }

    /// <summary><c>encode_latent</c> with a given draw <paramref name="noise"/> (the mean when null).</summary>
    public Tensor<T> EncodeLatent(Tensor<T> audio, Tensor<T>? noise)
    {
        var (mean, logVariance) = Encode(audio);
        if (noise is null) return mean;
        var std = _engine.TensorExp(_engine.TensorMultiplyScalar(logVariance, NumOps.FromDouble(0.5)));
        return _engine.TensorAdd(mean, _engine.TensorMultiply(std, noise));
    }

    /// <summary><c>decode</c>: the audio <c>[samples]</c> of a latent <c>[1, frames, latent]</c>.</summary>
    public Tensor<T> Decode(Tensor<T> latent)
    {
        var x = _engine.TensorPermute(MegaTts3Ops.Rows(_engine, ToDecoder, latent), new[] { 0, 2, 1 });       // [1, 320, frames]
        x = LatentUpsampler.Forward(x);
        x = GeneratorInput.Forward(_engine.PadNd(x, new[] { 3, 3 }, EnginePadMode.Reflect, NumOps.Zero));
        foreach (var (up, stacks) in Stages) x = stacks.Forward(up.Forward(x));
        x = GeneratorOutput.Forward(_engine.PadNd(_engine.LeakyReLU(x, NumOps.FromDouble(0.2)), new[] { 3, 3 }, EnginePadMode.Reflect, NumOps.Zero));
        x = _engine.Tanh(x);
        return _engine.Reshape(x, new[] { x.Length });
    }

    /// <summary>Loads a release checkpoint (<c>model_gen</c>'s state; the encoder is optional, as the release ships only
    /// the decoder).</summary>
    public void LoadTorch(IEngine engine, Func<string, double[]?> read, bool includeEncoder)
    {
        double[] Need(string name) => read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        if (includeEncoder)
        {
            foreach (var (name, conv) in Encoder.NamedConvolutions("encoder.encoder.model"))
                conv.Conv.LoadTorchWeights(Need($"{name}.conv.conv.weight_v"), Need($"{name}.conv.conv.weight_g"), Need($"{name}.conv.conv.bias"));
            if (Encoder.NamedLstm("encoder.encoder.model") is { } lstm)
            {
                var cells = lstm.Lstm.ForwardCells;
                for (int l = 0; l < cells.Count; l++)
                    cells[l].LoadTorchWeights(Need($"{lstm.Name}.lstm.weight_ih_l{l}"), Need($"{lstm.Name}.lstm.weight_hh_l{l}"),
                        Need($"{lstm.Name}.lstm.bias_ih_l{l}"), Need($"{lstm.Name}.lstm.bias_hh_l{l}"));
            }
            int d = Configuration.EncoderDimension, n = 2 * Configuration.LatentChannels;
            TorchParameters.Linear(engine, ToPosterior, d, n, Need("proj_to_z.weight"), Need("proj_to_z.bias"));
        }
        TorchParameters.Linear(engine, ToDecoder, Configuration.LatentChannels, Configuration.UpsamplerChannels, Need("proj_to_decoder.weight"), Need("proj_to_decoder.bias"));
        LatentUpsampler.LoadTorch(Need, "latent_upsampler");
        GeneratorInput.LoadTorchWeights(Need("decoder.model_up.1.weight_v"), Need("decoder.model_up.1.weight_g"), Need("decoder.model_up.1.bias"));
        int index = 2;
        foreach (var (up, stacks) in Stages)
        {
            up.LoadTorch(Need, $"decoder.model_up.{index++}");
            stacks.LoadTorch(Need, $"decoder.model_up.{index++}");
        }
        string last = $"decoder.model_up.{index + 2}";
        GeneratorOutput.LoadTorchWeights(Need($"{last}.weight_v"), Need($"{last}.weight_g"), Need($"{last}.bias"));
    }
}
