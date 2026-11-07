using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes and tunables of WhisperSpeech's text-to-semantic model (the checkpoint's <c>config</c> and
/// <c>tunables</c>).</summary>
internal sealed record WhisperSpeechT2SConfiguration
{
    public int Depth { get; init; } = 12;
    public int Heads { get; init; } = 12;
    public int HeadWidth { get; init; } = 64;
    public int FfnMult { get; init; } = 4;
    public int TextLength { get; init; } = 550;
    public int TextCodes { get; init; } = 256;
    public int SemanticLength { get; init; } = 750;
    public int SemanticCodes { get; init; } = 513;
    public int SemanticWidth { get; init; } = 64;
    public int Languages { get; init; } = 99;
    public double QueryMult { get; init; } = 1;
    public double OutputMult { get; init; } = 0.35;
    public double EncoderDepthRatio { get; init; } = 0.25;
    public bool CausalEncoder { get; init; } = true;
    public bool CpsInput { get; init; } = true;
    public int CpsBins { get; init; } = 32;
    public int PaddingTokenOffset { get; init; } = -1;
    public double InitStd { get; init; } = 1;
    public double EmbeddingsStd { get; init; } = 0.01;

    public int Width => Heads * HeadWidth;
    public int BaseWidth => 3 * HeadWidth;
    public int EncoderDepth => (int)(Depth * 2.0 * EncoderDepthRatio);
    public int DecoderDepth => Depth * 2 - EncoderDepth;
    public double QkScale => QueryMult * 8 / Math.Sqrt(HeadWidth);

    /// <summary>The semantic stream's padding and start token (<c>stoks_codes + padding_token_offset</c>).</summary>
    public int PaddingToken => SemanticCodes + PaddingTokenOffset;
}

/// <summary>
/// WhisperSpeech's text-to-semantic model (<c>TSARTransformer</c>, t2s_up_wds_mlang_enclm.py): UTF-8 bytes through a causal
/// encoder (plus a language embedding), and an autoregressive decoder over Whisper-derived semantic tokens that
/// cross-attends to it, conditioned on a characters-per-second embedding; the semantic table is the VQ tokenizer's frozen
/// codebook (64 wide, projected to the model width), its logits scaled by <c>output_mult / (width / base width)</c>.
/// </summary>
internal sealed class WhisperSpeechT2S<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Tensor<T> _textPositions, _semanticPositions;

    public WhisperSpeechT2S(IEngine engine, List<LayerBase<T>> layers, WhisperSpeechT2SConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int width = c.Width;
        LanguageEmbeddings = new BiasFreeLinearLayer<T>(1, c.Languages * width);
        layers.Add(LanguageEmbeddings);
        if (c.CpsInput)
        {
            CpsEmbeddings = new BiasFreeLinearLayer<T>(1, c.CpsBins * width);
            layers.Add(CpsEmbeddings);
        }
        TextEmbeddings = new WhisperSpeechEmbeddings<T>(engine, layers, c.TextCodes, width, 0, width, unembed: true);
        for (int i = 0; i < c.EncoderDepth; i++)
            EncoderBlocks.Add(new WhisperSpeechBlock<T>(engine, layers, width, c.Heads, c.QkScale, c.FfnMult, cross: false, rope: false));
        EncoderNorm = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(EncoderNorm);
        SemanticEmbeddings = new WhisperSpeechEmbeddings<T>(engine, layers, c.SemanticCodes, width, 1, c.SemanticWidth, unembed: true);
        for (int i = 0; i < c.DecoderDepth; i++)
            DecoderBlocks.Add(new WhisperSpeechBlock<T>(engine, layers, width, c.Heads, c.QkScale, c.FfnMult, cross: true, rope: false));
        DecoderNorm = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(DecoderNorm);
        _textPositions = Table(WhisperSpeechOps.Sinusoids(c.TextLength, width));
        _semanticPositions = Table(WhisperSpeechOps.Sinusoids(c.SemanticLength, width));
        Initialize(random);
    }

    public WhisperSpeechT2SConfiguration Configuration { get; }
    public BiasFreeLinearLayer<T> LanguageEmbeddings { get; }
    public BiasFreeLinearLayer<T>? CpsEmbeddings { get; }
    public WhisperSpeechEmbeddings<T> TextEmbeddings { get; }
    public List<WhisperSpeechBlock<T>> EncoderBlocks { get; } = new();
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public WhisperSpeechEmbeddings<T> SemanticEmbeddings { get; }
    public List<WhisperSpeechBlock<T>> DecoderBlocks { get; } = new();
    public LayerNormalizationLayer<T> DecoderNorm { get; }

    private static Tensor<T> Table(double[,] values)
    {
        var t = new Tensor<T>(new[] { values.GetLength(0), values.GetLength(1) });
        for (int i = 0; i < values.GetLength(0); i++)
            for (int j = 0; j < values.GetLength(1); j++) t[i, j] = NumOps.FromDouble(values[i, j]);
        return t;
    }

    private Tensor<T> Row(BiasFreeLinearLayer<T> table, int rows, int index)
    {
        int width = Configuration.Width;
        return _engine.TensorSlice(_engine.Reshape(table.Weights, new[] { rows, width }), new[] { index, 0 }, new[] { 1, width });
    }

    private Tensor<T> Positions(Tensor<T> table, int start, int count) =>
        _engine.TensorSlice(table, new[] { start, 0 }, new[] { count, Configuration.Width });

    // The release's init_transformer: linears truncated-normal at init_std / fan_in (biases too), embeddings at
    // embeddings_std, projectors at init_std, query heads and layer norms at their constants.
    private void Initialize(Random random)
    {
        var c = Configuration;
        void Fill(BiasFreeLinearLayer<T> table, double std)
        {
            var w = table.Weights;
            for (int i = 0; i < w.Length; i++) w[i] = NumOps.FromDouble(WhisperSpeechOps.TruncatedNormal(random, std));
            _engine.InvalidatePersistentTensor(w);
        }
        Fill(LanguageEmbeddings, c.EmbeddingsStd);
        if (CpsEmbeddings is not null) Fill(CpsEmbeddings, c.EmbeddingsStd);
        Fill(TextEmbeddings.Main, c.EmbeddingsStd);
        Fill(SemanticEmbeddings.Main, c.EmbeddingsStd);
        if (SemanticEmbeddings.Special is not null) Fill(SemanticEmbeddings.Special, c.EmbeddingsStd);
        void Linear(DenseLayer<T> layer, int inputs, int outputs, double std)
        {
            TorchParameters.Linear(_engine, layer, inputs, outputs,
                Enumerable.Range(0, inputs * outputs).Select(_ => WhisperSpeechOps.TruncatedNormal(random, std)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => WhisperSpeechOps.TruncatedNormal(random, std)).ToArray());
        }
        int w = c.Width;
        foreach (var block in EncoderBlocks.Concat(DecoderBlocks))
        {
            foreach (var attention in new[] { block.Attention, block.CrossAttention }.Where(a => a is not null))
            {
                TorchParameters.Linear(_engine, attention!.Query, w, w, new double[w * w], new double[w]);
                TorchParameters.Linear(attention.Key, w, w, Enumerable.Range(0, w * w).Select(_ => WhisperSpeechOps.TruncatedNormal(random, c.InitStd / w)).ToArray());
                Linear(attention.Value, w, w, c.InitStd / w);
                Linear(attention.Output, w, w, c.InitStd / w);
            }
            Linear(block.Mlp1, w, w * c.FfnMult, c.InitStd / w);
            Linear(block.Mlp2, w * c.FfnMult, w, c.InitStd / (w * (double)c.FfnMult));
        }
        if (SemanticEmbeddings.EmbedToHidden is not null)
            Linear(SemanticEmbeddings.EmbedToHidden, c.SemanticWidth, w, c.InitStd);
        if (SemanticEmbeddings.HiddenToEmbed is not null)
            Linear(SemanticEmbeddings.HiddenToEmbed, w, c.SemanticWidth, c.InitStd);
    }

    /// <summary>The characters-per-second bin's embedding (<c>_embed_cps</c>: ⌊cps / 20 · bins⌋, capped).</summary>
    public Tensor<T>? Cps(double cps)
    {
        if (CpsEmbeddings is null) return null;
        int bin = Math.Min(Configuration.CpsBins - 1, Math.Max(0, (int)(cps / 20 * Configuration.CpsBins)));
        return Row(CpsEmbeddings, Configuration.CpsBins, bin);
    }

    /// <summary>The encoder's states <c>[TextLength, width]</c> of text ids (already padded) in a language; causal when the
    /// tunables say so (the release always is, in training and inference).</summary>
    public Tensor<T> Encode(IReadOnlyList<int> textIds, int language)
    {
        var c = Configuration;
        int length = textIds.Count;
        var x = TextEmbeddings.Embed(textIds);
        x = _engine.TensorAdd(x, _engine.TensorBroadcastTo(Row(LanguageEmbeddings, c.Languages, language), new[] { length, c.Width }));
        x = _engine.TensorAdd(x, Positions(_textPositions, 0, length));
        var positions = Enumerable.Range(0, length).ToArray();
        foreach (var block in EncoderBlocks) x = block.Forward(x, positions, null, null, c.CausalEncoder);
        return EncoderNorm.Forward(x);
    }

    private Tensor<T> Scale(Tensor<T> logits)
    {
        var c = Configuration;
        return _engine.TensorMultiplyScalar(logits, NumOps.FromDouble(c.OutputMult / ((double)c.Width / c.BaseWidth)));
    }

    /// <summary>The decoder's logits <c>[n, codes + 1]</c> for semantic ids at positions (teacher-forced or cached).</summary>
    public Tensor<T> Decode(IReadOnlyList<int> semanticIds, int[] positions, Tensor<T> encoded, Tensor<T>? cps,
        List<(WhisperSpeechAttention<T>.Cache Self, WhisperSpeechAttention<T>.Cache Cross)>? caches = null)
    {
        var c = Configuration;
        int n = semanticIds.Count, textLength = encoded.Shape[0];
        var x = SemanticEmbeddings.Embed(semanticIds);
        var pos = _engine.TensorConcatenate(positions.Select(p => Positions(_semanticPositions, p, 1)).ToArray(), 0);
        x = _engine.TensorAdd(x, pos);
        if (cps is not null) x = _engine.TensorAdd(x, _engine.TensorBroadcastTo(cps, new[] { n, c.Width }));
        var textPositions = Enumerable.Range(0, textLength).ToArray();
        for (int i = 0; i < DecoderBlocks.Count; i++)
            x = DecoderBlocks[i].Forward(x, positions, encoded, textPositions, causal: true, caches?[i].Self, caches?[i].Cross);
        return Scale(SemanticEmbeddings.Unembed(DecoderNorm.Forward(x)));
    }

    /// <summary>The text ids as the release feeds them: UTF-8 bytes of the stripped text, one end-of-text (0) before them
    /// and padding (0) to TextLength; also the unshifted target form.</summary>
    public (int[] Input, int[] Target) TextIds(string text)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(text.Trim());
        int length = Configuration.TextLength;
        if (bytes.Length + 1 > length) throw new ArgumentException($"The text is longer than {length - 1} bytes.", nameof(text));
        var input = new int[length];
        var target = new int[length];
        for (int i = 0; i < bytes.Length; i++)
        {
            input[i + 1] = bytes[i];
            target[i] = bytes[i];
        }
        return (input, target);
    }

    /// <summary>
    /// The training loss (<c>forward</c> in train mode): the semantic tokens padded (a padding token before them as the
    /// start, padding after), their cross-entropy over every position, plus 0.1 × the causal encoder's own next-byte
    /// cross-entropy.
    /// </summary>
    public Tensor<T> Loss(string text, IReadOnlyList<int> semantic, int language, double cps)
    {
        var c = Configuration;
        var (inText, outText) = TextIds(text);
        if (semantic.Count + 1 > c.SemanticLength) throw new ArgumentException($"At most {c.SemanticLength - 1} semantic tokens.", nameof(semantic));
        var input = Enumerable.Repeat(c.PaddingToken, c.SemanticLength).ToArray();
        var target = Enumerable.Repeat(c.PaddingToken, c.SemanticLength).ToArray();
        for (int i = 0; i < semantic.Count; i++)
        {
            input[i + 1] = semantic[i];
            target[i] = semantic[i];
        }
        var encoded = Encode(inText, language);
        var logits = Decode(input, Enumerable.Range(0, c.SemanticLength).ToArray(), encoded, Cps(cps));
        var loss = CrossEntropy(logits, target);
        if (c.CausalEncoder)
        {
            var encoderLogits = Scale(TextEmbeddings.Unembed(encoded));
            loss = _engine.TensorAdd(loss, _engine.TensorMultiplyScalar(CrossEntropy(encoderLogits, outText), NumOps.FromDouble(0.1)));
        }
        return loss;
    }

    private Tensor<T> CrossEntropy(Tensor<T> logits, int[] targets)
    {
        int n = targets.Length, vocabulary = logits.Shape[1];
        var log = _engine.TensorLogSoftmax(logits, axis: 1);
        var picks = new Tensor<T>(log._shape);
        for (int i = 0; i < n; i++) picks[i, targets[i]] = NumOps.FromDouble(-1.0 / n);
        return _engine.ReduceSum(_engine.TensorMultiply(log, picks), new[] { 0, 1 }, false);
    }

    /// <summary>
    /// Semantic tokens for text (<c>generate</c>): the encoder once; the decoder from the start token (padding) and the
    /// prompt, one token at a time with its caches, sampled at temperature T over the top k (all when null) by the
    /// exponential race; it stops at the padding token or after N − 1 positions. The release's mask of the special logit
    /// indexes the batch, so the special token stays possible, as there.
    /// </summary>
    public int[] Generate(string text, int language, double cps, IReadOnlyList<int>? prompt, int maxLength, double temperature, int? topK, Random random)
    {
        var c = Configuration;
        var (inText, _) = TextIds(text);
        var encoded = Encode(inText, language);
        var cpsEmbedding = Cps(cps);
        var caches = DecoderBlocks.Select(_ => (new WhisperSpeechAttention<T>.Cache(), new WhisperSpeechAttention<T>.Cache())).ToList();
        int n = maxLength;
        var tokens = new List<int> { c.PaddingToken };
        if (prompt is not null) tokens.AddRange(prompt);
        int start = tokens.Count - 1;
        int Sample(Tensor<T> logits)
        {
            int vocabulary = logits.Shape[1], last = logits.Shape[0] - 1;
            var row = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++) row[v] = NumOps.ToDouble(logits[last, v]) / Math.Max(temperature, 1e-5);
            if (topK is int k && k < vocabulary)
            {
                double pivot = row.OrderByDescending(v => v).ElementAt(k - 1);
                for (int v = 0; v < vocabulary; v++) if (row[v] < pivot) row[v] = double.NegativeInfinity;
            }
            double max = row.Max();
            var probabilities = row.Select(v => Math.Exp(v - max)).ToArray();
            double total = probabilities.Sum();
            int best = 0;
            double bestScore = double.NegativeInfinity;
            for (int v = 0; v < vocabulary; v++)
            {
                double exponential = -Math.Log(1.0 - random.NextDouble());
                double score = probabilities[v] / total / exponential;
                if (score > bestScore) { bestScore = score; best = v; }
            }
            return best;
        }
        // Prefill: the start token and the prompt.
        var logits = Decode(tokens, Enumerable.Range(0, tokens.Count).ToArray(), encoded, cpsEmbedding, caches);
        tokens.Add(Sample(logits));
        for (int i = start + 1; i < n - 1; i++)
        {
            logits = Decode(new[] { tokens[i] }, new[] { i }, encoded, cpsEmbedding, caches);
            int next = Sample(logits);
            tokens.Add(next);
            if (next == c.PaddingToken) return tokens.Skip(1).Take(i).ToArray();
        }
        return tokens.Skip(1).ToArray();
    }

    /// <summary>Loads the release's state dict (<c>lang_embeddings</c>, <c>cps_embeddings</c>, <c>encoder.*</c>,
    /// <c>embeddings.embedding.*</c>, <c>decoder.*</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> read)
    {
        var c = Configuration;
        double[] Read(string name) => read(name) ?? throw new InvalidDataException($"The T2S checkpoint has no tensor '{name}'.");
        TorchParameters.Linear(LanguageEmbeddings, 1, c.Languages * c.Width, Read("lang_embeddings.weight"));
        if (CpsEmbeddings is not null) TorchParameters.Linear(CpsEmbeddings, 1, c.CpsBins * c.Width, Read("cps_embeddings.weight"));
        TextEmbeddings.LoadTorchWeights(read, "encoder.embedding.");
        for (int i = 0; i < EncoderBlocks.Count; i++) EncoderBlocks[i].LoadTorchWeights(read, $"encoder.layers.{i}.");
        TorchParameters.LayerNorm(_engine, EncoderNorm, Read("encoder.ln_post.weight"), Read("encoder.ln_post.bias"));
        SemanticEmbeddings.LoadTorchWeights(read, "embeddings.embedding.");
        for (int i = 0; i < DecoderBlocks.Count; i++) DecoderBlocks[i].LoadTorchWeights(read, $"decoder.layers.{i}.");
        TorchParameters.LayerNorm(_engine, DecoderNorm, Read("decoder.ln_post.weight"), Read("decoder.ln_post.bias"));
    }
}
