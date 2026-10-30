using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>The sizes of a unit-based HiFi-GAN (fairseq's <c>CodeGenerator</c> config); the defaults are the released
/// mHuBERT layer-11, 1000-unit LJSpeech vocoder (<c>mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj</c>), 16 kHz with a
/// 320-sample code hop.</summary>
internal sealed record CodeHiFiGanConfiguration
{
    public int Units { get; init; } = 1000;
    public int EmbeddingDim { get; init; } = 128;
    public int InitialChannels { get; init; } = 512;
    public int[] UpsampleRates { get; init; } = { 5, 4, 4, 2, 2 };
    public int[] UpsampleKernels { get; init; } = { 11, 8, 8, 4, 4 };
    public int[] ResblockKernels { get; init; } = { 3, 7, 11 };
    public int[][] ResblockDilations { get; init; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>The duration predictor's hidden width and kernel (<c>dur_predictor_params</c>: 128, 3); 0 for none.</summary>
    public int DurationHidden { get; init; } = 128;

    /// <inheritdoc cref="DurationHidden"/>
    public int DurationKernel { get; init; } = 3;

    /// <summary>Samples per unit frame: the upsampling rates' product (320).</summary>
    public int HopSize => UpsampleRates.Aggregate(1, (a, b) => a * b);
}

/// <summary>
/// A unit-based HiFi-GAN with a duration predictor (Polyak et al. 2021; fairseq <c>CodeHiFiGANVocoder</c>): discrete
/// speech units, embedded, optionally repeated by their predicted durations, through HiFi-GAN's generator to a waveform.
/// LLaMA-Omni and SpeechGPT voice deduplicated mHuBERT units with it.
/// </summary>
/// <remarks>
/// <para>The duration predictor is FastSpeech 2's <c>VariancePredictor</c> on the unit embeddings: a convolution with
/// ReLU, LayerNorm, a second convolution with ReLU, LayerNorm, and a linear map to a log duration. A unit lasts
/// <c>max(1, round(exp(d) − 1))</c> frames, with torch's round-half-to-even. The generator is HiFi-GAN V1 with type-1
/// residual blocks on the embeddings (<see cref="HiFiGanGenerator{T}"/>).</para>
/// <para>Inference only: the speech models that use it load a released vocoder and keep it frozen.</para>
/// </remarks>
internal sealed class CodeHiFiGan<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public CodeHiFiGan(IEngine engine, List<LayerBase<T>> layers, CodeHiFiGanConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - initialization.NextDouble())) * Math.Cos(2.0 * Math.PI * initialization.NextDouble());
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        Embedding = Add(new TiedEmbeddingLayer<T>(c.Units, c.EmbeddingDim));
        Embedding.Reinitialize(Normal);
        if (c.DurationHidden > 0)
        {
            int h = c.DurationHidden, k = c.DurationKernel;
            DurationConv1 = Add(new NormedConv1DLayer<T>(c.EmbeddingDim, h, k, 1, 1, 1, (k - 1) / 2, false, ConvolutionNormalization.None));
            DurationConv1.Reinitialize(() => Uniform(c.EmbeddingDim * k), bias: 0.0);
            DurationNorm1 = Add(new LayerNormalizationLayer<T>(h, 1e-5));
            // VariancePredictor's second convolution pads by 1 whatever its kernel.
            DurationConv2 = Add(new NormedConv1DLayer<T>(h, h, k, 1, 1, 1, 1, false, ConvolutionNormalization.None));
            DurationConv2.Reinitialize(() => Uniform(h * k), bias: 0.0);
            DurationNorm2 = Add(new LayerNormalizationLayer<T>(h, 1e-5));
            DurationProjection = Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));
            TorchParameters.Linear(engine, DurationProjection, h, 1, Enumerable.Range(0, h).Select(_ => Uniform(h)).ToArray(), new[] { Uniform(h) });
        }
        Generator = new HiFiGanGenerator<T>(engine, c.EmbeddingDim, c.InitialChannels, c.UpsampleRates, c.UpsampleKernels, c.ResblockKernels,
            c.ResblockDilations, resblockType1: true, initialization: initialization);
        layers.AddRange(Generator.Layers);
    }

    public CodeHiFiGanConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public NormedConv1DLayer<T>? DurationConv1 { get; }
    public LayerNormalizationLayer<T>? DurationNorm1 { get; }
    public NormedConv1DLayer<T>? DurationConv2 { get; }
    public LayerNormalizationLayer<T>? DurationNorm2 { get; }
    public DenseLayer<T>? DurationProjection { get; }
    public HiFiGanGenerator<T> Generator { get; }

    private Tensor<T> Embed(IReadOnlyList<int> units)
    {
        var ids = new Tensor<T>(new[] { units.Count });
        for (int i = 0; i < units.Count; i++)
        {
            if (units[i] < 0 || units[i] >= Configuration.Units) throw new ArgumentOutOfRangeException(nameof(units), $"Units lie in [0, {Configuration.Units}).");
            ids[i] = NumOps.FromDouble(units[i]);
        }
        return Embedding.Forward(ids);                                                                            // [L, E]
    }

    /// <summary>The predicted log durations <c>[L]</c> of units (<c>VariancePredictor</c> at inference).</summary>
    public Tensor<T> LogDurations(IReadOnlyList<int> units)
    {
        if (DurationConv1 is null || DurationNorm1 is null || DurationConv2 is null || DurationNorm2 is null || DurationProjection is null)
            throw new InvalidOperationException("This vocoder has no duration predictor.");
        int length = units.Count;
        var x = _engine.Reshape(_engine.TensorTranspose(Embed(units)), new[] { 1, Configuration.EmbeddingDim, length });
        Tensor<T> Rows(Tensor<T> channels) => _engine.TensorTranspose(_engine.Reshape(channels, new[] { channels.Shape[1], length }));
        Tensor<T> Channels(Tensor<T> rows) => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], length });
        var h = DurationNorm1.Forward(Rows(_engine.ReLU(DurationConv1.Forward(x))));
        h = DurationNorm2.Forward(Rows(_engine.ReLU(DurationConv2.Forward(Channels(h)))));
        return _engine.Reshape(DurationProjection.Forward(h), new[] { length });
    }

    /// <summary>Each unit's frames: <c>max(1, round(exp(d) − 1))</c> of its log duration, rounding half to even.</summary>
    public int[] Durations(IReadOnlyList<int> units)
    {
        var log = LogDurations(units);
        var frames = new int[units.Count];
        for (int i = 0; i < frames.Length; i++) frames[i] = Math.Max(1, (int)Math.Round(Math.Exp(NumOps.ToDouble(log[i])) - 1, MidpointRounding.ToEven));
        return frames;
    }

    /// <summary>
    /// The waveform <c>[frames · hop]</c> of discrete units (<c>CodeHiFiGANVocoder.forward</c>): the embeddings, each
    /// repeated by its predicted duration when <paramref name="durationPrediction"/>, through the generator.
    /// </summary>
    public Tensor<T> Forward(IReadOnlyList<int> units, bool durationPrediction = true)
    {
        if (units is null || units.Count == 0) throw new ArgumentException("There are no units.", nameof(units));
        var rows = Embed(units);
        if (durationPrediction)
        {
            var durations = Durations(units);
            var index = new Tensor<int>(new[] { durations.Sum() });
            int at = 0;
            for (int i = 0; i < durations.Length; i++)
                for (int r = 0; r < durations[i]; r++) index[at++] = i;
            rows = _engine.TensorIndexSelect(rows, index, 0);
        }
        int frames = rows.Shape[0];
        var x = _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, Configuration.EmbeddingDim, frames });
        var wave = Generator.Forward(x);
        return _engine.Reshape(wave, new[] { wave.Length });
    }

    /// <summary>Loads a fairseq CodeHiFiGAN generator state (the checkpoint's <c>generator</c> entry: <c>dict</c>,
    /// <c>dur_predictor.*</c> and HiFi-GAN's weight-normalized convolutions); <paramref name="read"/> returns null for a
    /// missing tensor.</summary>
    public void LoadTorchWeights(Func<string, double[]?> read, string prefix = "")
    {
        var c = Configuration;
        double[] Need(string name) => read(prefix + name) ?? throw new InvalidDataException($"The vocoder checkpoint has no tensor '{prefix}{name}'.");
        Embedding.LoadTable(Need("dict.weight"));
        if (DurationConv1 is not null && DurationNorm1 is not null && DurationConv2 is not null && DurationNorm2 is not null && DurationProjection is not null)
        {
            DurationConv1.LoadTorchWeights(Need("dur_predictor.conv1.0.weight"), null, Need("dur_predictor.conv1.0.bias"));
            TorchParameters.LayerNorm(_engine, DurationNorm1, Need("dur_predictor.ln1.weight"), Need("dur_predictor.ln1.bias"));
            DurationConv2.LoadTorchWeights(Need("dur_predictor.conv2.0.weight"), null, Need("dur_predictor.conv2.0.bias"));
            TorchParameters.LayerNorm(_engine, DurationNorm2, Need("dur_predictor.ln2.weight"), Need("dur_predictor.ln2.bias"));
            TorchParameters.Linear(_engine, DurationProjection, c.DurationHidden, 1, Need("dur_predictor.proj.weight"), Need("dur_predictor.proj.bias"));
        }
        Generator.LoadTorch(name => read(prefix + name), "");
    }
}
