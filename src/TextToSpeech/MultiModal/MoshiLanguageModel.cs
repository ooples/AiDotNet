using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes of Moshi's RQ-Transformer (the release's <c>LMModel</c> arguments; Table 1 of the paper).</summary>
internal sealed record MoshiLanguageModelConfiguration
{
    /// <summary>Mimi codebooks per speaker, Q (8): each speaker's first is semantic.</summary>
    public int Codebooks { get; init; } = 8;

    /// <summary>Audio streams read by the Temporal Transformer (Moshi's 8 and the user's 8).</summary>
    public int AudioStreams { get; init; } = 16;

    /// <summary>Audio streams the Depth Transformer predicts: the paper's all 16 (Eq. 6), the release's Moshi 8.</summary>
    public int DepthStreams { get; init; } = 16;

    /// <summary>Audio codebook entries (2048) and text vocabulary (32000).</summary>
    public int AudioCardinality { get; init; } = 2048;

    /// <inheritdoc cref="AudioCardinality"/>
    public int TextCardinality { get; init; } = 32000;

    /// <summary>The delay of each stream, text first (the release's 0, then 0 and 1×7 for each speaker's audio).</summary>
    public int[] Delays { get; init; } = new[] { 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 };

    /// <summary>The text padding token (PAD, 3) and end-of-padding token (EPAD, 0).</summary>
    public int TextPadding { get; init; } = 3;

    /// <inheritdoc cref="TextPadding"/>
    public int TextEndPadding { get; init; } = 0;

    public MoshiTransformerConfiguration Temporal { get; init; } = new();

    public MoshiTransformerConfiguration Depth { get; init; } = new()
    {
        Dim = 1024, Heads = 16, Layers = 6, FeedForward = 4224, Context = 0, Rotary = false, WeightsPerStep = 16,
    };

    public int Streams => 1 + AudioStreams;
}

/// <summary>
/// Moshi's RQ-Transformer (Défossez et al. 2024 §3.4; the release's <c>LMModel</c>): the Temporal Transformer reads the
/// sum of each stream's embedding of the previous step and produces a context z_s; the text logits are a linear map of z_s
/// (l_{s,1}), and the Depth Transformer, given z_s projected per codebook plus the embedding of the previous codebook's
/// value at step s, produces each audio codebook's logits in turn (Eq. 1–3).
/// </summary>
/// <remarks>Stream values of −1 embed as zeros (no input) and are not scored. The audio streams of each speaker are delayed
/// (acoustic delay τ, Eq. 4), the delayed positions filled with the start token (<c>card</c>, or <c>text_card</c> for
/// text).</remarks>
internal sealed class MoshiLanguageModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MoshiLanguageModel(IEngine engine, List<LayerBase<T>> layers, MoshiLanguageModelConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Delays.Length != c.Streams) throw new ArgumentException($"{c.Streams} streams need {c.Streams} delays.");
        if (c.DepthStreams > c.AudioStreams || c.Depth.WeightsPerStep != c.DepthStreams && c.Depth.WeightsPerStep != 1)
            throw new ArgumentException("The Depth Transformer predicts at most the audio streams, with weights per predicted stream.");
        int d = c.Temporal.Dim, dd = c.Depth.Dim;
        BiasFreeLinearLayer<T> Add(BiasFreeLinearLayer<T> layer) { layers.Add(layer); return layer; }
        TextEmbedding = Add(new BiasFreeLinearLayer<T>(1, (c.TextCardinality + 1) * d));
        AudioEmbeddings = Enumerable.Range(0, c.AudioStreams).Select(_ => Add(new BiasFreeLinearLayer<T>(1, (c.AudioCardinality + 1) * d))).ToArray();
        Temporal = new MoshiTransformer<T>(engine, layers, c.Temporal);
        OutNorm = Add(new BiasFreeLinearLayer<T>(1, d));
        TextLinear = Add(new BiasFreeLinearLayer<T>(d, c.TextCardinality));
        DepthIn = Enumerable.Range(0, c.DepthStreams).Select(_ => Add(new BiasFreeLinearLayer<T>(d, dd))).ToArray();
        DepthTextEmbedding = Add(new BiasFreeLinearLayer<T>(1, (c.TextCardinality + 1) * dd));
        DepthEmbeddings = Enumerable.Range(0, c.DepthStreams - 1).Select(_ => Add(new BiasFreeLinearLayer<T>(1, (c.AudioCardinality + 1) * dd))).ToArray();
        Depth = new MoshiTransformer<T>(engine, layers, c.Depth);
        Heads = Enumerable.Range(0, c.DepthStreams).Select(_ => Add(new BiasFreeLinearLayer<T>(dd, c.AudioCardinality))).ToArray();
        Initialize(random);
    }

    public MoshiLanguageModelConfiguration Configuration { get; }
    public BiasFreeLinearLayer<T> TextEmbedding { get; }
    public BiasFreeLinearLayer<T>[] AudioEmbeddings { get; }
    public MoshiTransformer<T> Temporal { get; }
    public BiasFreeLinearLayer<T> OutNorm { get; }
    public BiasFreeLinearLayer<T> TextLinear { get; }
    public BiasFreeLinearLayer<T>[] DepthIn { get; }
    public BiasFreeLinearLayer<T> DepthTextEmbedding { get; }
    public BiasFreeLinearLayer<T>[] DepthEmbeddings { get; }
    public MoshiTransformer<T> Depth { get; }
    public BiasFreeLinearLayer<T>[] Heads { get; }

    /// <summary>The Depth Transformer's layers with its input projections, embeddings and heads (their own learning
    /// rate in Table 1).</summary>
    public IEnumerable<LayerBase<T>> DepthLayers() =>
        DepthIn.Cast<LayerBase<T>>().Append(DepthTextEmbedding).Concat(DepthEmbeddings).Concat(Depth.AllLayers).Concat(Heads);

    /// <summary>torch's <c>trunc_normal_</c> at ±3 std, by rejection.</summary>
    internal static double TruncatedNormal(Random random, double std)
    {
        while (true)
        {
            double z = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
            if (Math.Abs(z) <= 3) return z * std;
        }
    }

    private void FillTable(BiasFreeLinearLayer<T> table, Func<double> sample)
    {
        var w = table.Weights;
        for (int i = 0; i < w.Length; i++) w[i] = NumOps.FromDouble(sample());
        _engine.InvalidatePersistentTensor(w);
    }

    // The release's _init_weights: embeddings, the text linear, the heads and the Temporal Transformer truncated-normal
    // (std 1/√fan_in, embeddings 1/√dim); the depth input projections and Depth Transformer at torch's defaults.
    private void Initialize(Random random)
    {
        var c = Configuration;
        int d = c.Temporal.Dim, dd = c.Depth.Dim;
        FillTable(TextEmbedding, () => TruncatedNormal(random, 1.0 / Math.Sqrt(d)));
        foreach (var e in AudioEmbeddings) FillTable(e, () => TruncatedNormal(random, 1.0 / Math.Sqrt(d)));
        FillTable(DepthTextEmbedding, () => TruncatedNormal(random, 1.0 / Math.Sqrt(dd)));
        foreach (var e in DepthEmbeddings) FillTable(e, () => TruncatedNormal(random, 1.0 / Math.Sqrt(dd)));
        FillTable(OutNorm, () => 1.0);
        void Linear(BiasFreeLinearLayer<T> layer, int inputs, int outputs, bool truncated) => TorchParameters.Linear(layer, inputs, outputs,
            Enumerable.Range(0, inputs * outputs).Select(_ => truncated ? TruncatedNormal(random, 1.0 / Math.Sqrt(inputs)) : (2 * random.NextDouble() - 1) / Math.Sqrt(inputs)).ToArray());
        Linear(TextLinear, d, c.TextCardinality, true);
        foreach (var h in Heads) Linear(h, dd, c.AudioCardinality, true);
        foreach (var p in DepthIn) Linear(p, d, dd, false);
        Temporal.Initialize(random, truncatedNormal: true);
        Depth.Initialize(random, truncatedNormal: false);
    }

    // Rows of a table [rows, dim] for ids (−1 → zeros).
    private Tensor<T> Embed(BiasFreeLinearLayer<T> table, IReadOnlyList<int> ids, int rows, int dim)
    {
        var index = new Tensor<int>(new[] { ids.Count });
        var keep = new Tensor<T>(new[] { ids.Count, 1 });
        bool any = false;
        for (int i = 0; i < ids.Count; i++)
        {
            if (ids[i] < -1 || ids[i] >= rows) throw new ArgumentOutOfRangeException(nameof(ids), $"Ids lie in [-1, {rows}).");
            index[i] = Math.Max(0, ids[i]);
            keep[i, 0] = ids[i] < 0 ? NumOps.Zero : NumOps.One;
            any |= ids[i] < 0;
        }
        var e = _engine.TensorIndexSelect(_engine.Reshape(MoshiTransformer<T>.Value(table), new[] { rows, dim }), index, 0);
        return any ? _engine.TensorMultiply(e, _engine.TensorBroadcastTo(keep, new[] { ids.Count, dim })) : e;
    }

    /// <summary>The Temporal Transformer's input <c>[L, D]</c>: the sum of each stream's embedding of tokens
    /// <c>[streams, L]</c>.</summary>
    public Tensor<T> InputEmbeddings(int[,] tokens)
    {
        var c = Configuration;
        int l = tokens.GetLength(1), d = c.Temporal.Dim;
        int[] Row(int k) => Enumerable.Range(0, l).Select(t => tokens[k, t]).ToArray();
        var x = Embed(TextEmbedding, Row(0), c.TextCardinality + 1, d);
        for (int q = 0; q < c.AudioStreams; q++) x = _engine.TensorAdd(x, Embed(AudioEmbeddings[q], Row(q + 1), c.AudioCardinality + 1, d));
        return x;
    }

    /// <summary>The context z <c>[L, D]</c> (after the output norm) and text logits <c>[L, text]</c> of tokens.</summary>
    public (Tensor<T> Context, Tensor<T> TextLogits) TemporalForward(int[,] tokens, MoshiTransformer<T>.Cache? cache = null)
    {
        var z = Temporal.RmsNorm(OutNorm, Temporal.Forward(InputEmbeddings(tokens), cache));
        return (z, TextLinear.Forward(z));
    }

    /// <summary>The Depth Transformer's logits <c>[streams, L, card]</c> for contexts <c>[L, D]</c> and the step's values
    /// <c>[1 + depth streams − 1, L]</c> (text, then the audio streams before the last), teacher-forced.</summary>
    public Tensor<T> DepthForward(Tensor<T> context, int[,] stepTokens)
    {
        var c = Configuration;
        int l = context.Shape[0], dd = c.Depth.Dim, k = c.DepthStreams;
        var outputs = new Tensor<T>[l];
        for (int t = 0; t < l; t++)
        {
            var z = _engine.TensorSlice(context, new[] { t, 0 }, new[] { 1, c.Temporal.Dim });
            var inputs = new Tensor<T>[k];
            for (int s = 0; s < k; s++) inputs[s] = DepthInput(s, stepTokens[s, t], z);
            var h = Depth.Forward(_engine.TensorConcatenate(inputs, 0));                                        // [K, dd]
            var logits = new Tensor<T>[k];
            for (int s = 0; s < k; s++) logits[s] = Heads[s].Forward(_engine.TensorSlice(h, new[] { s, 0 }, new[] { 1, dd }));
            outputs[t] = _engine.Reshape(_engine.TensorConcatenate(logits, 0), new[] { k, 1, c.AudioCardinality });
        }
        return l == 1 ? outputs[0] : _engine.TensorConcatenate(outputs, 1);
    }

    /// <summary>The Depth Transformer's input <c>[1, depth dim]</c> for predicted stream <paramref name="s"/>: the context
    /// projected for it, plus the embedding of the token before it (the step's text for the first, else the previous
    /// predicted stream's).</summary>
    public Tensor<T> DepthInput(int s, int previous, Tensor<T> z)
    {
        var c = Configuration;
        int dd = c.Depth.Dim;
        var token = s == 0
            ? Embed(DepthTextEmbedding, new[] { previous }, c.TextCardinality + 1, dd)
            : Embed(DepthEmbeddings[s - 1], new[] { previous }, c.AudioCardinality + 1, dd);
        return _engine.TensorAdd(DepthIn[DepthIn.Length == 1 ? 0 : s].Forward(z), token);
    }

    /// <summary>The release's <c>_delay_sequence</c> of codes <c>[streams, T]</c>: stream k shifted right by its delay, the
    /// first positions filled with its start token.</summary>
    public int[,] Delay(int[,] codes)
    {
        var c = Configuration;
        int streams = codes.GetLength(0), length = codes.GetLength(1);
        var delayed = new int[streams, length];
        for (int k = 0; k < streams; k++)
        {
            int start = k == 0 ? c.TextCardinality : c.AudioCardinality;
            for (int t = 0; t < length; t++) delayed[k, t] = t < c.Delays[k] ? start : codes[k, t - c.Delays[k]];
        }
        return delayed;
    }

    /// <summary>
    /// The training logits (the release's <c>LMModel.forward</c>) of codes <c>[streams, T]</c> (−1 where a value is absent):
    /// the delayed sequence after the start step, the text logits <c>[T, text]</c> and the predicted audio streams' logits
    /// <c>[depth, T, card]</c>, undelayed to align with <paramref name="codes"/>, and which of each are scored (valid
    /// context and a present target).
    /// </summary>
    public (Tensor<T> TextLogits, bool[] TextMask, Tensor<T> AudioLogits, bool[,] AudioMask) Forward(int[,] codes)
    {
        var c = Configuration;
        int streams = codes.GetLength(0), length = codes.GetLength(1);
        if (streams != c.Streams) throw new ArgumentException($"Expected {c.Streams} streams, got {streams}.", nameof(codes));
        var delayed = Delay(codes);
        var input = new int[streams, length];
        var step = new int[c.DepthStreams, length];
        for (int k = 0; k < streams; k++)
        {
            input[k, 0] = k == 0 ? c.TextCardinality : c.AudioCardinality;
            for (int t = 1; t < length; t++) input[k, t] = delayed[k, t - 1];
        }
        for (int s = 0; s < c.DepthStreams; s++)
            for (int t = 0; t < length; t++) step[s, t] = delayed[s, t];
        var (z, textLogits) = TemporalForward(input);
        var audio = DepthForward(z, step);
        // Undelay: stream q's logits at position t + delay predict codes at t.
        var undelayed = new Tensor<T>[c.DepthStreams];
        var audioMask = new bool[c.DepthStreams, length];
        for (int s = 0; s < c.DepthStreams; s++)
        {
            int delay = c.Delays[1 + s];
            var row = _engine.TensorSlice(audio, new[] { s, 0, 0 }, new[] { 1, length, c.AudioCardinality });
            if (delay > 0)
            {
                var shifted = _engine.TensorSlice(row, new[] { 0, delay, 0 }, new[] { 1, length - delay, c.AudioCardinality });
                row = _engine.TensorConcatenate(new[] { shifted, new Tensor<T>(new[] { 1, delay, c.AudioCardinality }) }, 1);
            }
            undelayed[s] = row;
            for (int t = 0; t < length; t++) audioMask[s, t] = t < length - delay && codes[1 + s, t] != -1;
        }
        int textDelay = c.Delays[0];
        if (textDelay > 0)
            textLogits = _engine.TensorConcatenate(new[] { _engine.TensorSlice(textLogits, new[] { textDelay, 0 }, new[] { length - textDelay, c.TextCardinality }),
                new Tensor<T>(new[] { textDelay, c.TextCardinality }) }, 0);
        var textMask = Enumerable.Range(0, length).Select(t => t < length - textDelay && codes[0, t] != -1).ToArray();
        return (textLogits, textMask, _engine.TensorConcatenate(undelayed, 0), audioMask);
    }

    // Weighted cross-entropy of logits [N, V] against targets with weights (0 skips): Σ w·CE / Σ w.
    private Tensor<T> WeightedCrossEntropy(Tensor<T> logits, int[] targets, double[] weights)
    {
        int n = targets.Length, vocabulary = logits.Shape[1];
        var log = _engine.Reshape(_engine.TensorLogSoftmax(logits, axis: 1), new[] { n * vocabulary, 1 });
        var picks = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(i => i * vocabulary + Math.Max(0, targets[i])).ToArray()));
        var w = new Tensor<T>(new[] { n, 1 }, new Vector<T>(weights.Select(NumOps.FromDouble).ToArray()));
        double total = weights.Sum();
        if (total <= 0) return new Tensor<T>(new[] { 1 });
        return _engine.TensorMultiplyScalar(_engine.TensorNegate(_engine.ReduceSum(_engine.TensorMultiply(_engine.TensorIndexSelect(log, picks, 0), w), null, false)),
            NumOps.FromDouble(1.0 / total));
    }

    /// <summary>
    /// The loss of Eq. 7 on codes <c>[streams, T]</c>: the text cross-entropy (PAD and EPAD weighted by
    /// <paramref name="textPaddingWeight"/>, §4.4's 0.5) plus the audio cross-entropy weighted α = <paramref name="semanticWeight"/>
    /// for each speaker's semantic stream and 1 for the acoustic ones, normalized by the weights.
    /// </summary>
    public Tensor<T> Loss(int[,] codes, double semanticWeight = 100.0, double textPaddingWeight = 0.5)
    {
        var c = Configuration;
        int length = codes.GetLength(1);
        var (textLogits, textMask, audioLogits, audioMask) = Forward(codes);
        var textTargets = Enumerable.Range(0, length).Select(t => codes[0, t]).ToArray();
        var textWeights = Enumerable.Range(0, length).Select(t => !textMask[t] ? 0.0
            : textTargets[t] == c.TextPadding || textTargets[t] == c.TextEndPadding ? textPaddingWeight : 1.0).ToArray();
        var loss = WeightedCrossEntropy(textLogits, textTargets, textWeights);
        var audioTargets = new int[c.DepthStreams * length];
        var audioWeights = new double[c.DepthStreams * length];
        for (int s = 0; s < c.DepthStreams; s++)
        {
            bool semantic = s % c.Codebooks == 0;
            for (int t = 0; t < length; t++)
            {
                audioTargets[s * length + t] = codes[1 + s, t];
                audioWeights[s * length + t] = audioMask[s, t] ? (semantic ? semanticWeight : 1.0) : 0.0;
            }
        }
        var audio = WeightedCrossEntropy(_engine.Reshape(audioLogits, new[] { c.DepthStreams * length, c.AudioCardinality }), audioTargets, audioWeights);
        return _engine.TensorAdd(loss, audio);
    }

    /// <summary>Loads the release's <c>LMModel</c> state (<c>text_emb</c>, <c>emb.{q}</c>, <c>transformer.</c>,
    /// <c>out_norm.alpha</c>, <c>text_linear</c>, <c>depformer_in.{s}</c>, <c>depformer_text_emb</c>,
    /// <c>depformer_emb.{s}</c>, <c>depformer.</c>, <c>linears.{s}</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> take)
    {
        var c = Configuration;
        int d = c.Temporal.Dim, dd = c.Depth.Dim;
        double[] Read(string name) => take(name) ?? throw new InvalidDataException($"The Moshi checkpoint has no tensor '{name}'.");
        void Values(BiasFreeLinearLayer<T> holder, string name)
        {
            var v = Read(name);
            var w = holder.Weights;
            if (v.Length != w.Length) throw new InvalidDataException($"'{name}' holds {v.Length} values; expected {w.Length}.");
            for (int i = 0; i < v.Length; i++) w[i] = NumOps.FromDouble(v[i]);
            _engine.InvalidatePersistentTensor(w);
        }
        Values(TextEmbedding, "text_emb.weight");
        for (int q = 0; q < c.AudioStreams; q++) Values(AudioEmbeddings[q], $"emb.{q}.weight");
        Temporal.LoadTorchWeights(take, "transformer.");
        Values(OutNorm, "out_norm.alpha");
        TorchParameters.Linear(TextLinear, d, c.TextCardinality, Read("text_linear.weight"));
        for (int s = 0; s < DepthIn.Length; s++) TorchParameters.Linear(DepthIn[s], d, dd, Read($"depformer_in.{s}.weight"));
        Values(DepthTextEmbedding, "depformer_text_emb.weight");
        for (int s = 0; s < DepthEmbeddings.Length; s++) Values(DepthEmbeddings[s], $"depformer_emb.{s}.weight");
        Depth.LoadTorchWeights(take, "depformer.");
        for (int s = 0; s < Heads.Length; s++) TorchParameters.Linear(Heads[s], dd, c.AudioCardinality, Read($"linears.{s}.weight"));
    }
}
