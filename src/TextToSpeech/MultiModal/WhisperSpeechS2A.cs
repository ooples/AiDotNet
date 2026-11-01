using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes and tunables of WhisperSpeech's semantic-to-acoustic model.</summary>
internal sealed record WhisperSpeechS2AConfiguration
{
    public int Depth { get; init; } = 12;
    public int Context { get; init; } = 2250;
    public int SemanticLength { get; init; } = 750;
    public int SemanticCodes { get; init; } = 513;
    public int SemanticWidth { get; init; } = 64;
    public int SpeakerWidth { get; init; } = 192;
    public int Heads { get; init; } = 12;
    public int HeadWidth { get; init; } = 64;
    public int FfnMult { get; init; } = 4;
    public int Quantizers { get; init; } = 4;
    public int AcousticCodes { get; init; } = 1024;
    public double InitStd { get; init; } = 9;
    public double EmbeddingsStd { get; init; } = 0.2;
    public double OutputMult { get; init; } = 5.6;
    public double QueryMult { get; init; } = 0.3;
    public double EncoderDepthRatio { get; init; } = 0.25;
    public bool Rope { get; init; } = true;
    public bool CausalEncoder { get; init; }
    public bool HiddenToEmbedding { get; init; } = true;

    /// <summary>The conditioned variant (s2a_delar_mup_wds_mlang_cond): language, SNR, C50 and speaker embeddings summed.</summary>
    public bool Conditioned { get; init; }

    public int Languages { get; init; } = 99;
    public int Width => Heads * HeadWidth;
    public int BaseWidth => 3 * HeadWidth;
    public int EncoderDepth => (int)(Depth * 2 * EncoderDepthRatio);
    public int DecoderDepth => Depth * 2 - EncoderDepth;
    public double QkScale => QueryMult * 8 / Math.Sqrt(HeadWidth);
}

/// <summary>The cond variant's conditions; null marks a condition as missing (its own learned code).</summary>
public sealed record WhisperSpeechConditions(int? Language = null, double? Snr = null, double? C50 = null);

/// <summary>
/// WhisperSpeech's semantic-to-acoustic model (<c>SADelARTransformer</c>): the semantic tokens (the VQ codebook's frozen
/// 64-wide rows, projected) through a rotary encoder, plus the L2-normalized speaker embedding projected to the width (or,
/// for the cond variant, the sum of language, binned SNR, binned C50 and speaker embeddings); a rotary decoder over the
/// sum of each EnCodec quantizer's embedding (two shared special rows: 1024 ignore/pad, 1025 start), cross-attending with
/// key positions ×3 (three acoustic frames per semantic one); a GELU splitter into one state per quantizer, each scored
/// against its quantizer's table.
/// </summary>
internal sealed class WhisperSpeechS2A<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public WhisperSpeechS2A(IEngine engine, List<LayerBase<T>> layers, WhisperSpeechS2AConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int width = c.Width;
        DenseLayer<T> Dense(int outputs) { var l = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()); layers.Add(l); return l; }
        BiasFreeLinearLayer<T> Table(int rows, int dim) { var l = new BiasFreeLinearLayer<T>(1, rows * dim); layers.Add(l); return l; }
        SemanticEmbedding = Table(c.SemanticCodes, c.SemanticWidth);
        if (c.SemanticWidth != width)
        {
            EmbedToHidden = Dense(width);
            if (c.CausalEncoder || c.HiddenToEmbedding) HiddenToEmbed = Dense(c.SemanticWidth);
        }
        if (c.Conditioned)
        {
            LanguageEmbedding = Table(c.Languages + 1, width);
            SnrEmbedding = Table(11, width);
            C50Embedding = Table(7, width);
        }
        if (c.SpeakerWidth != width) SpeakerToHidden = Dense(width);
        for (int i = 0; i < c.EncoderDepth; i++)
            EncoderBlocks.Add(new WhisperSpeechBlock<T>(engine, layers, width, c.Heads, c.QkScale, c.FfnMult, cross: false, rope: c.Rope));
        EncoderNorm = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(EncoderNorm);
        WhisperSpeechEmbeddings<T>? first = null;
        for (int q = 0; q < c.Quantizers; q++)
        {
            var e = new WhisperSpeechEmbeddings<T>(engine, layers, c.AcousticCodes, width, 2, width, unembed: true, sharedSpecial: first);
            first ??= e;
            AcousticEmbeddings.Add(e);
        }
        for (int i = 0; i < c.DecoderDepth; i++)
        {
            var block = new WhisperSpeechBlock<T>(engine, layers, width, c.Heads, c.QkScale, c.FfnMult, cross: true, rope: c.Rope);
            block.CrossAttention!.KeySubsampling = 3;
            DecoderBlocks.Add(block);
        }
        DecoderNorm = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(DecoderNorm);
        Splitter = Dense(width * c.Quantizers);
        Initialize(random);
    }

    public WhisperSpeechS2AConfiguration Configuration { get; }
    public BiasFreeLinearLayer<T> SemanticEmbedding { get; }
    public DenseLayer<T>? EmbedToHidden { get; }
    public DenseLayer<T>? HiddenToEmbed { get; }
    public BiasFreeLinearLayer<T>? LanguageEmbedding { get; }
    public BiasFreeLinearLayer<T>? SnrEmbedding { get; }
    public BiasFreeLinearLayer<T>? C50Embedding { get; }
    public DenseLayer<T>? SpeakerToHidden { get; }
    public List<WhisperSpeechBlock<T>> EncoderBlocks { get; } = new();
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public List<WhisperSpeechEmbeddings<T>> AcousticEmbeddings { get; } = new();
    public List<WhisperSpeechBlock<T>> DecoderBlocks { get; } = new();
    public LayerNormalizationLayer<T> DecoderNorm { get; }
    public DenseLayer<T> Splitter { get; }

    // init_transformer: linears truncated-normal at init_std / fan_in, query heads zero, embeddings at embeddings_std.
    private void Initialize(Random random)
    {
        var c = Configuration;
        int w = c.Width;
        void Fill(BiasFreeLinearLayer<T> table, double std)
        {
            var values = table.Weights;
            for (int i = 0; i < values.Length; i++) values[i] = NumOps.FromDouble(WhisperSpeechOps.TruncatedNormal(random, std));
            _engine.InvalidatePersistentTensor(values);
        }
        void Linear(DenseLayer<T> layer, int inputs, int outputs) => TorchParameters.Linear(_engine, layer, inputs, outputs,
            Enumerable.Range(0, inputs * outputs).Select(_ => WhisperSpeechOps.TruncatedNormal(random, c.InitStd / inputs)).ToArray(),
            Enumerable.Range(0, outputs).Select(_ => WhisperSpeechOps.TruncatedNormal(random, c.InitStd / inputs)).ToArray());
        Fill(SemanticEmbedding, c.EmbeddingsStd);
        foreach (var t in new[] { LanguageEmbedding, SnrEmbedding, C50Embedding }) if (t is not null) Fill(t, c.EmbeddingsStd);
        foreach (var e in AcousticEmbeddings)
        {
            Fill(e.Main, c.EmbeddingsStd);
            if (e == AcousticEmbeddings[0] && e.Special is not null) Fill(e.Special, c.EmbeddingsStd);
        }
        if (EmbedToHidden is not null) Linear(EmbedToHidden, c.SemanticWidth, w);
        if (HiddenToEmbed is not null) Linear(HiddenToEmbed, w, c.SemanticWidth);
        if (SpeakerToHidden is not null) Linear(SpeakerToHidden, c.SpeakerWidth, w);
        foreach (var block in EncoderBlocks.Concat(DecoderBlocks))
        {
            foreach (var attention in new[] { block.Attention, block.CrossAttention }.Where(a => a is not null))
            {
                TorchParameters.Linear(_engine, attention!.Query, w, w, new double[w * w], new double[w]);
                TorchParameters.Linear(attention.Key, w, w, Enumerable.Range(0, w * w).Select(_ => WhisperSpeechOps.TruncatedNormal(random, c.InitStd / w)).ToArray());
                Linear(attention.Value, w, w);
                Linear(attention.Output, w, w);
            }
            Linear(block.Mlp1, w, w * c.FfnMult);
            Linear(block.Mlp2, w * c.FfnMult, w);
        }
        Linear(Splitter, w, w * c.Quantizers);
    }

    private Tensor<T> Row(BiasFreeLinearLayer<T> table, int rows, int index) =>
        _engine.TensorSlice(_engine.Reshape(table.Weights, new[] { rows, Configuration.Width }), new[] { index, 0 }, new[] { 1, Configuration.Width });

    /// <summary>The encoder's states <c>[SemanticLength, width]</c> plus the speaker (or conditions) of padded semantic ids
    /// and a speaker embedding.</summary>
    public Tensor<T> Encode(IReadOnlyList<int> semantic, Tensor<T> speaker, WhisperSpeechConditions? conditions = null)
    {
        var c = Configuration;
        int n = semantic.Count;
        var index = new Tensor<int>(new[] { n }, new Vector<int>(semantic.ToArray()));
        var x = _engine.TensorIndexSelect(_engine.Reshape(SemanticEmbedding.Weights, new[] { c.SemanticCodes, c.SemanticWidth }), index, 0);
        if (EmbedToHidden is not null) x = EmbedToHidden.Forward(x);
        var positions = Enumerable.Range(0, n).ToArray();
        foreach (var block in EncoderBlocks) x = block.Forward(x, positions, null, null, c.CausalEncoder);
        x = EncoderNorm.Forward(x);
        // F.normalize(speaker): x / max(‖x‖, 1e-12).
        var s = _engine.Reshape(speaker, new[] { 1, c.SpeakerWidth });
        double norm = Math.Sqrt(Enumerable.Range(0, c.SpeakerWidth).Sum(i => Math.Pow(NumOps.ToDouble(s[0, i]), 2)));
        s = _engine.TensorMultiplyScalar(s, NumOps.FromDouble(1.0 / Math.Max(norm, 1e-12)));
        var condition = SpeakerToHidden is not null ? SpeakerToHidden.Forward(s) : s;
        if (c.Conditioned)
        {
            var cond = conditions ?? new WhisperSpeechConditions();
            int Binned(double? v, double min, double max, int bins) =>
                v is double value ? Math.Min(bins - 1, Math.Max(0, (int)Math.Truncate((value - min) / (max - min) * bins))) : bins;
            condition = _engine.TensorAdd(condition, Row(LanguageEmbedding!, c.Languages + 1, cond.Language ?? c.Languages));
            condition = _engine.TensorAdd(condition, Row(SnrEmbedding!, 11, Binned(cond.Snr, -10, 70, 10)));
            condition = _engine.TensorAdd(condition, Row(C50Embedding!, 7, Binned(cond.C50, 0, 60, 6)));
        }
        return _engine.TensorAdd(x, _engine.TensorBroadcastTo(condition, new[] { n, c.Width }));
    }

    /// <summary>Logits <c>[quantizers, n, codes + 2]</c> of acoustic ids <c>[quantizers][n]</c> at positions.</summary>
    public Tensor<T> Decode(int[][] acoustic, int[] positions, Tensor<T> encoded,
        List<(WhisperSpeechAttention<T>.Cache Self, WhisperSpeechAttention<T>.Cache Cross)>? caches = null)
    {
        var c = Configuration;
        int n = positions.Length, width = c.Width;
        Tensor<T>? x = null;
        for (int q = 0; q < c.Quantizers; q++)
        {
            var e = AcousticEmbeddings[q].Embed(acoustic[q]);
            x = x is null ? e : _engine.TensorAdd(x, e);
        }
        var semanticPositions = Enumerable.Range(0, encoded.Shape[0]).ToArray();
        for (int i = 0; i < DecoderBlocks.Count; i++)
            x = DecoderBlocks[i].Forward(x!, positions, encoded, semanticPositions, causal: true, caches?[i].Self, caches?[i].Cross);
        x = DecoderNorm.Forward(x!);
        var split = ActivationHelper.ExactGelu(Splitter.Forward(x), _engine);                            // [n, q · width]
        var scale = NumOps.FromDouble(c.OutputMult / ((double)width / c.BaseWidth));
        var logits = new Tensor<T>[c.Quantizers];
        for (int q = 0; q < c.Quantizers; q++)
        {
            var part = _engine.TensorSlice(split, new[] { 0, q * width }, new[] { n, width });
            var l = _engine.TensorMultiplyScalar(AcousticEmbeddings[q].Unembed(part), scale);
            logits[q] = _engine.Reshape(l, new[] { 1, n, l.Shape[1] });
        }
        return _engine.TensorConcatenate(logits, 0);
    }

    /// <summary>The semantic ids as the release feeds them: a padding (start) token, the tokens, padding to SemanticLength.</summary>
    public int[] SemanticInput(IReadOnlyList<int> semantic)
    {
        var c = Configuration;
        if (semantic.Count + 1 > c.SemanticLength) throw new ArgumentException($"At most {c.SemanticLength - 1} semantic tokens.", nameof(semantic));
        var input = Enumerable.Repeat(c.SemanticCodes - 1, c.SemanticLength).ToArray();
        for (int i = 0; i < semantic.Count; i++) input[i + 1] = semantic[i];
        return input;
    }

    /// <summary>
    /// The training loss (<c>forward</c>): for each quantizer, the cross-entropy of its next acoustic token (1024 ignored),
    /// averaged over the quantizers. <paramref name="acoustic"/> is already delayed (quantizer i shifted right by i + 1,
    /// the start token 1025 before it).
    /// </summary>
    public Tensor<T> Loss(IReadOnlyList<int> semanticInput, int[][] acoustic, Tensor<T> speaker, WhisperSpeechConditions? conditions = null)
    {
        var c = Configuration;
        int n = acoustic[0].Length;
        var encoded = Encode(semanticInput, speaker, conditions);
        var logits = Decode(acoustic, Enumerable.Range(0, n).ToArray(), encoded);
        int vocabulary = logits.Shape[2];
        Tensor<T>? total = null;
        for (int q = 0; q < c.Quantizers; q++)
        {
            var rows = _engine.Reshape(_engine.TensorSlice(logits, new[] { q, 0, 0 }, new[] { 1, n - 1, vocabulary }), new[] { n - 1, vocabulary });
            var log = _engine.TensorLogSoftmax(rows, axis: 1);
            int scored = Enumerable.Range(1, n - 1).Count(t => acoustic[q][t] != c.AcousticCodes);
            var picks = new Tensor<T>(log._shape);
            for (int t = 1; t < n; t++)
                if (acoustic[q][t] != c.AcousticCodes) picks[t - 1, acoustic[q][t]] = NumOps.FromDouble(-1.0 / Math.Max(1, scored));
            var term = _engine.ReduceSum(_engine.TensorMultiply(log, picks), new[] { 0, 1 }, false);
            total = total is null ? term : _engine.TensorAdd(total, term);
        }
        return _engine.TensorMultiplyScalar(total!, NumOps.FromDouble(1.0 / c.Quantizers));
    }

    /// <summary>The release's delayed layout of codes <c>[quantizers][frames]</c>: quantizer i's row starts with i + 1 start
    /// tokens (1025), its codes after, padded with 1024 to <paramref name="length"/>.</summary>
    public int[][] Delayed(int[][] codes, int length)
    {
        var c = Configuration;
        var result = new int[c.Quantizers][];
        for (int q = 0; q < c.Quantizers; q++)
        {
            result[q] = Enumerable.Repeat(c.AcousticCodes, length).ToArray();
            for (int t = 0; t <= q && t < length; t++) result[q][t] = c.AcousticCodes + 1;
            for (int t = 0; t < codes[q].Length && q + 1 + t < length; t++) result[q][q + 1 + t] = codes[q][t];
        }
        return result;
    }

    /// <summary>
    /// Acoustic codes <c>[quantizers][N − 4]</c> for semantic tokens (<c>generate</c>): the decoder over the delayed layout
    /// from the start tokens, each position filling the quantizers below it (quantizer i first at position i + 1), every
    /// row sampled at temperature T over the top k by the exponential race; then each row shifted back by its delay and
    /// cut to N − 4.
    /// </summary>
    public int[][] Generate(IReadOnlyList<int> semantic, Tensor<T> speaker, int? maxLength, double temperature, int? topK, Random random,
        WhisperSpeechConditions? conditions = null)
    {
        var c = Configuration;
        int n = maxLength ?? semantic.Count * 3;
        var encoded = Encode(SemanticInput(semantic), speaker, conditions);
        var toks = new int[c.Quantizers][];
        for (int q = 0; q < c.Quantizers; q++) toks[q] = Enumerable.Repeat(c.AcousticCodes + 1, c.Context).ToArray();
        var caches = DecoderBlocks.Select(_ => (new WhisperSpeechAttention<T>.Cache(), new WhisperSpeechAttention<T>.Cache())).ToList();
        int[] SampleRows(Tensor<T> logits)
        {
            int vocabulary = logits.Shape[2], last = logits.Shape[1] - 1;
            var result = new int[c.Quantizers];
            for (int q = 0; q < c.Quantizers; q++)
            {
                var row = new double[vocabulary];
                for (int v = 0; v < vocabulary; v++) row[v] = NumOps.ToDouble(logits[q, last, v]) / Math.Max(temperature, 1e-5);
                if (topK is int k && k < vocabulary)
                {
                    double pivot = row.OrderByDescending(v => v).ElementAt(k - 1);
                    for (int v = 0; v < vocabulary; v++) if (row[v] < pivot) row[v] = double.NegativeInfinity;
                }
                double max = row.Max();
                var probabilities = row.Select(v => Math.Exp(v - max)).ToArray();
                double total = probabilities.Sum(), bestScore = double.NegativeInfinity;
                for (int v = 0; v < vocabulary; v++)
                {
                    double score = probabilities[v] / total / -Math.Log(1.0 - random.NextDouble());
                    if (score > bestScore) { bestScore = score; result[q] = v; }
                }
            }
            return result;
        }
        int[][] Column(int start, int count) => toks.Select(row => row.Skip(start).Take(count).ToArray()).ToArray();
        int start1 = 1;
        var initial = SampleRows(Decode(Column(0, start1), Enumerable.Range(0, start1).ToArray(), encoded, caches));
        for (int q = 0; q < Math.Min(start1, c.Quantizers); q++) toks[q][start1] = initial[q];
        for (int i = start1 + 1; i < Math.Min(n, c.Context - 1); i++)
        {
            var sampled = SampleRows(Decode(Column(i - 1, 1), new[] { i - 1 }, encoded, caches));
            for (int q = 0; q < Math.Min(i, c.Quantizers); q++) toks[q][i] = sampled[q];
        }
        var output = new int[c.Quantizers][];
        // Python's [:N − 4] of the N − 1 shifted positions: a negative bound counts from the end.
        int shifted = Math.Max(0, n - 1), keep = n - 4 >= 0 ? Math.Min(n - 4, shifted) : Math.Max(0, shifted + n - 4);
        for (int q = 0; q < c.Quantizers; q++)
        {
            // toks[:, q, 1:N] rolled by −q, then the first N − 4.
            var row = toks[q].Skip(1).Take(n - 1).ToArray();
            output[q] = row.Length == 0 ? Array.Empty<int>() : Enumerable.Range(0, keep).Select(t => row[(t + q) % row.Length]).ToArray();
        }
        return output;
    }

    /// <summary>Loads the release's state dict.</summary>
    public void LoadTorchWeights(Func<string, double[]?> read)
    {
        var c = Configuration;
        int w = c.Width;
        double[] Read(string name) => read(name) ?? throw new InvalidDataException($"The S2A checkpoint has no tensor '{name}'.");
        TorchParameters.Linear(SemanticEmbedding, 1, c.SemanticCodes * c.SemanticWidth, Read("semantic_embedding.weight"));
        if (EmbedToHidden is not null) TorchParameters.Linear(_engine, EmbedToHidden, c.SemanticWidth, w, Read("emb_to_hidden.weight"), Read("emb_to_hidden.bias"));
        if (HiddenToEmbed is not null) TorchParameters.Linear(_engine, HiddenToEmbed, w, c.SemanticWidth, Read("hidden_to_emb.weight"), Read("hidden_to_emb.bias"));
        string speakerName = c.Conditioned ? "cond_embeddings.speaker.spk_to_hidden" : "spk_to_hidden";
        if (SpeakerToHidden is not null) TorchParameters.Linear(_engine, SpeakerToHidden, c.SpeakerWidth, w, Read(speakerName + ".weight"), Read(speakerName + ".bias"));
        if (c.Conditioned)
        {
            TorchParameters.Linear(LanguageEmbedding!, 1, (c.Languages + 1) * w, Read("cond_embeddings.lang.embed.weight"));
            TorchParameters.Linear(SnrEmbedding!, 1, 11 * w, Read("cond_embeddings.snr.embed.weight"));
            TorchParameters.Linear(C50Embedding!, 1, 7 * w, Read("cond_embeddings.c50.embed.weight"));
        }
        for (int i = 0; i < EncoderBlocks.Count; i++) EncoderBlocks[i].LoadTorchWeights(read, $"encoder.{i}.");
        TorchParameters.LayerNorm(_engine, EncoderNorm, Read("ln_post.weight"), Read("ln_post.bias"));
        for (int q = 0; q < c.Quantizers; q++) AcousticEmbeddings[q].LoadTorchWeights(read, $"embds.embeddings.{q}.", loadSpecial: q == 0);
        for (int i = 0; i < DecoderBlocks.Count; i++) DecoderBlocks[i].LoadTorchWeights(read, $"decoder.layers.{i}.");
        TorchParameters.LayerNorm(_engine, DecoderNorm, Read("decoder.ln_post.weight"), Read("decoder.ln_post.bias"));
        TorchParameters.Linear(_engine, Splitter, w, w * c.Quantizers, Read("head.splitter.0.weight"), Read("head.splitter.0.bias"));
    }
}
