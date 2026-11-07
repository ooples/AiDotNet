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

    /// <summary>The duration predictor's dropout in training (<c>var_pred_dropout</c>: 0.5).</summary>
    public double DurationDropout { get; init; } = 0.5;

    /// <summary>Speaker embeddings (speech-resynthesis' <c>multispkr</c>: a table of 200); 0 for a single-speaker
    /// vocoder.</summary>
    public int Speakers { get; init; }

    /// <summary>Utterance style embeddings (expresso's <c>multistyle</c>: SpiritLM Base's vocoder has 34, its
    /// <c>styles.txt</c>); 0 for none.</summary>
    public int Styles { get; init; }

    /// <summary>Pitch-token embeddings (<c>num_f0_tokens</c>: SpiritLM Expressive's 64 F0 VQ-VAE codes), concatenated per
    /// frame; 0 for none.</summary>
    public int PitchTokens { get; init; }

    /// <summary>Style-token embeddings (<c>num_style_tokens</c>: SpiritLM Expressive's 100), concatenated per frame; 0 for
    /// none.</summary>
    public int StyleTokens { get; init; }

    /// <summary>Unit frames per pitch token (<c>round(f0_hop_size / code_hop_size)</c>: 1280 / 640 = 2).</summary>
    public int PitchRatio { get; init; } = 2;

    /// <summary>Unit frames per style token (<c>round(style_hop_size / code_hop_size)</c>: 16000 / 640 = 25).</summary>
    public int StyleRatio { get; init; } = 25;

    /// <summary>The generator's input width: the unit embedding and each conditioning embedding concatenated.</summary>
    public int InputDim => EmbeddingDim * (1 + (PitchTokens > 0 ? 1 : 0) + (StyleTokens > 0 ? 1 : 0) + (Speakers > 0 ? 1 : 0) + (Styles > 0 ? 1 : 0));

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
/// <para>Multi-speaker (<see cref="CodeHiFiGanConfiguration.Speakers"/>; speech-resynthesis'
/// <c>DurationCodeGenerator</c>): a speaker's embedding, repeated over the frames, is concatenated to the unit embeddings
/// before the generator.</para>
/// <para>Expresso's and SpiritLM's <c>CodeGenerator</c> adds per-frame pitch and style tokens
/// (<see cref="CodeHiFiGanConfiguration.PitchTokens"/>, <see cref="CodeHiFiGanConfiguration.StyleTokens"/>), each embedded and
/// repeated with its unit, and a per-utterance style (<see cref="CodeHiFiGanConfiguration.Styles"/>), repeated over the frames
/// like the speaker. The generator reads the unit, pitch-token, style-token, speaker and style embeddings concatenated in that
/// order.</para>
/// <para>Training (<see cref="TrainingForward"/>) reads frame-level codes; the duration predictor learns the log
/// (duration + 1) of each run of a code (<c>process_duration</c>: the first and last runs, cut by the segment, left out)
/// by mean squared error.</para>
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
        TiedEmbeddingLayer<T>? Table(int rows)
        {
            if (rows <= 0) return null;
            var table = Add(new TiedEmbeddingLayer<T>(rows, c.EmbeddingDim));
            table.Reinitialize(Normal);
            return table;
        }
        PitchEmbedding = Table(c.PitchTokens);
        StyleTokenEmbedding = Table(c.StyleTokens);
        SpeakerEmbedding = Table(c.Speakers);
        StyleEmbedding = Table(c.Styles);
        Generator = new HiFiGanGenerator<T>(engine, c.InputDim, c.InitialChannels, c.UpsampleRates, c.UpsampleKernels, c.ResblockKernels,
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

    /// <summary>The speaker table; null for a single-speaker vocoder.</summary>
    public TiedEmbeddingLayer<T>? SpeakerEmbedding { get; }

    /// <summary>The per-frame pitch-token table; null when the vocoder reads none.</summary>
    public TiedEmbeddingLayer<T>? PitchEmbedding { get; }

    /// <summary>The per-frame style-token table; null when the vocoder reads none.</summary>
    public TiedEmbeddingLayer<T>? StyleTokenEmbedding { get; }

    /// <summary>The per-utterance style table; null when the vocoder reads none.</summary>
    public TiedEmbeddingLayer<T>? StyleEmbedding { get; }

    public HiFiGanGenerator<T> Generator { get; }

    private Tensor<T> Embed(IReadOnlyList<int> units) => Lookup(Embedding, units, Configuration.Units, nameof(units));

    private static Tensor<T> Lookup(TiedEmbeddingLayer<T> table, IReadOnlyList<int> ids, int count, string name)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++)
        {
            if (ids[i] < 0 || ids[i] >= count) throw new ArgumentOutOfRangeException(name, $"Ids lie in [0, {count}).");
            tensor[i] = NumOps.FromDouble(ids[i]);
        }
        return table.Forward(tensor);                                                                             // [L, E]
    }

    /// <summary>
    /// Tokens at a lower rate brought to <paramref name="frames"/> unit frames (<c>CodeHiFiGANVocoderModel.upsample</c>):
    /// each repeated <paramref name="ratio"/> times, the last repeated or the tail cut to fit; the counts must agree to
    /// within one token.
    /// </summary>
    public static int[] Upsample(IReadOnlyList<int> tokens, int frames, int ratio)
    {
        if (tokens.Count == 0 || Math.Abs((double)tokens.Count * ratio - frames) / ratio > 1)
            throw new ArgumentException($"{tokens.Count} tokens at {ratio} frames each do not cover {frames} frames.", nameof(tokens));
        var result = new int[frames];
        for (int f = 0; f < frames; f++) result[f] = tokens[Math.Min(f / ratio, tokens.Count - 1)];
        return result;
    }

    /// <summary>The predicted log durations <c>[L]</c> of units (<c>VariancePredictor</c>; its dropout when
    /// <paramref name="random"/> is given).</summary>
    public Tensor<T> LogDurations(IReadOnlyList<int> units, Random? random = null)
    {
        if (DurationConv1 is null || DurationNorm1 is null || DurationConv2 is null || DurationNorm2 is null || DurationProjection is null)
            throw new InvalidOperationException("This vocoder has no duration predictor.");
        int length = units.Count;
        var x = _engine.Reshape(_engine.TensorTranspose(Embed(units)), new[] { 1, Configuration.EmbeddingDim, length });
        Tensor<T> Rows(Tensor<T> channels) => _engine.TensorTranspose(_engine.Reshape(channels, new[] { channels.Shape[1], length }));
        Tensor<T> Channels(Tensor<T> rows) => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], length });
        Tensor<T> Drop(Tensor<T> t) => random is not null && Configuration.DurationDropout > 0
            ? AiDotNet.TextToSpeech.CodecBased.T5Seq2Seq<T>.Dropout(_engine, t, Configuration.DurationDropout, random) : t;
        var h = Drop(DurationNorm1.Forward(Rows(_engine.ReLU(DurationConv1.Forward(x)))));
        h = Drop(DurationNorm2.Forward(Rows(_engine.ReLU(DurationConv2.Forward(Channels(h))))));
        return _engine.Reshape(DurationProjection.Forward(h), new[] { length });
    }

    // The frame rows [L, E] with the per-frame pitch- and style-token rows [L, E] and the speaker's and style's embeddings
    // concatenated to each, as [1, channels, L].
    private Tensor<T> GeneratorInput(Tensor<T> rows, Tensor<T>? pitchRows, Tensor<T>? styleTokenRows, int? speaker, int? style)
    {
        int frames = rows.Shape[0], e = Configuration.EmbeddingDim;
        var parts = new List<Tensor<T>> { rows };
        if (pitchRows is not null) parts.Add(pitchRows);
        if (styleTokenRows is not null) parts.Add(styleTokenRows);
        Tensor<T> Utterance(TiedEmbeddingLayer<T> table, int? id, int count, string name)
        {
            if (id is null || id < 0 || id >= count) throw new ArgumentOutOfRangeException(name, $"This vocoder reads {name} 0 … {count - 1}.");
            var embedding = table.Forward(new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(id.Value) })));
            return _engine.TensorBroadcastTo(_engine.Reshape(embedding, new[] { 1, e }), new[] { frames, e });
        }
        if (SpeakerEmbedding is not null) parts.Add(Utterance(SpeakerEmbedding, speaker, Configuration.Speakers, nameof(speaker)));
        if (StyleEmbedding is not null) parts.Add(Utterance(StyleEmbedding, style, Configuration.Styles, nameof(style)));
        if (parts.Count > 1) rows = _engine.TensorConcatenate(parts.ToArray(), 1);
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], frames });
    }

    // The embedded per-frame pitch and style tokens [L, E] (null for a table the vocoder lacks).
    private (Tensor<T>? Pitch, Tensor<T>? StyleTokens) ConditionRows(IReadOnlyList<int>? pitch, IReadOnlyList<int>? styleTokens, int frames)
    {
        Tensor<T>? Rows(TiedEmbeddingLayer<T>? table, IReadOnlyList<int>? ids, int count, string name)
        {
            if (table is null) return null;
            if (ids is null) throw new ArgumentNullException(name, $"This vocoder reads {name}.");
            if (ids.Count != frames) throw new ArgumentException($"{ids.Count} {name} for {frames} frames.", name);
            return Lookup(table, ids, count, name);
        }
        return (Rows(PitchEmbedding, pitch, Configuration.PitchTokens, nameof(pitch)),
            Rows(StyleTokenEmbedding, styleTokens, Configuration.StyleTokens, nameof(styleTokens)));
    }

    /// <summary>
    /// The training forward (speech-resynthesis' <c>DurationCodeGenerator.forward</c> in training): the waveform of
    /// frame-level codes, as they are, and the duration predictor's loss, the mean squared error of its log durations
    /// against log(run length + 1) over each run of a code (the first and last runs left out when there are more than
    /// two, since a segment cuts them).
    /// </summary>
    /// <remarks>Pitch and style tokens, when the vocoder reads them, are given per code frame (see <see cref="Upsample"/>).</remarks>
    public (Tensor<T> Waveform, Tensor<T>? DurationLoss) TrainingForward(IReadOnlyList<int> codes, int? speaker, Random random,
        int? style = null, IReadOnlyList<int>? pitch = null, IReadOnlyList<int>? styleTokens = null)
    {
        if (codes is null || codes.Count == 0) throw new ArgumentException("There are no codes.", nameof(codes));
        Tensor<T>? durationLoss = null;
        if (DurationConv1 is not null)
        {
            var runs = new List<(int Code, int Start, int Count)>();
            for (int i = 0; i < codes.Count; i++)
            {
                if (i > 0 && codes[i] == codes[i - 1]) runs[runs.Count - 1] = (runs[^1].Code, runs[^1].Start, runs[^1].Count + 1);
                else runs.Add((codes[i], i, 1));
            }
            var kept = runs.Count > 2 ? runs.Skip(1).Take(runs.Count - 2).ToList() : runs;
            var logDurations = LogDurations(kept.Select(r => r.Code).ToArray(), random);
            var targets = new Tensor<T>(new[] { kept.Count }, new Vector<T>(kept.Select(r => NumOps.FromDouble(Math.Log(r.Count + 1))).ToArray()));
            var difference = _engine.TensorSubtract(logDurations, targets);
            durationLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), null, false),
                NumOps.FromDouble(1.0 / kept.Count));
        }
        var (pitchRows, styleRows) = ConditionRows(pitch, styleTokens, codes.Count);
        var wave = Generator.Forward(GeneratorInput(Embed(codes), pitchRows, styleRows, speaker, style));
        return (_engine.Reshape(wave, new[] { wave.Length }), durationLoss);
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
    /// <remarks>Pitch and style tokens, when the vocoder reads them: with duration prediction one per unit, repeated with
    /// it; without, at their own rates, brought to the units' frames by <see cref="Upsample"/> (as the release does).</remarks>
    public Tensor<T> Forward(IReadOnlyList<int> units, bool durationPrediction = true, int? speaker = null, int? style = null,
        IReadOnlyList<int>? pitch = null, IReadOnlyList<int>? styleTokens = null)
    {
        if (units is null || units.Count == 0) throw new ArgumentException("There are no units.", nameof(units));
        if (!durationPrediction)
        {
            if (PitchEmbedding is not null && pitch is not null) pitch = Upsample(pitch, units.Count, Configuration.PitchRatio);
            if (StyleTokenEmbedding is not null && styleTokens is not null) styleTokens = Upsample(styleTokens, units.Count, Configuration.StyleRatio);
        }
        var rows = Embed(units);
        var (pitchRows, styleRows) = ConditionRows(pitch, styleTokens, units.Count);
        if (durationPrediction)
        {
            var durations = Durations(units);
            var index = new Tensor<int>(new[] { durations.Sum() });
            int at = 0;
            for (int i = 0; i < durations.Length; i++)
                for (int r = 0; r < durations[i]; r++) index[at++] = i;
            rows = _engine.TensorIndexSelect(rows, index, 0);
            if (pitchRows is not null) pitchRows = _engine.TensorIndexSelect(pitchRows, index, 0);
            if (styleRows is not null) styleRows = _engine.TensorIndexSelect(styleRows, index, 0);
        }
        var wave = Generator.Forward(GeneratorInput(rows, pitchRows, styleRows, speaker, style));
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
        if (SpeakerEmbedding is not null) SpeakerEmbedding.LoadTable(Need("spkr.weight"));
        if (StyleEmbedding is not null) StyleEmbedding.LoadTable(Need("style.weight"));
        if (PitchEmbedding is not null) PitchEmbedding.LoadTable(Need("f0_dict.weight"));
        if (StyleTokenEmbedding is not null) StyleTokenEmbedding.LoadTable(Need("style_dict.weight"));
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
