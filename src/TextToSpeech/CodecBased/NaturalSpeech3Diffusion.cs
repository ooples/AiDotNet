using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// NaturalSpeech 3's diffusion Transformer (App. A.1): Transformer blocks whose feed-forward is a 1-D convolution and whose
/// LayerNorms are conditional on the diffusion time. The time embedding and the condition MLP the paper leaves out follow
/// MaskGCT: <c>SinusoidalPosEmb(t)</c> through Linear-SiLU-Linear, and the condition through Linear-SiLU-Linear added to
/// the input.
/// </summary>
internal sealed class Ns3Backbone<T>
{
    private readonly IEngine _engine;
    private readonly int _hidden;

    public Ns3Backbone(IEngine engine, List<LayerBase<T>> layers, Random random, int hidden, int heads, int filter, int kernel, int count, double dropout)
    {
        _engine = engine;
        _hidden = hidden;
        StepIn = IndexTtsOps.Linear(engine, layers, random, hidden, 4 * hidden);
        StepOut = IndexTtsOps.Linear(engine, layers, random, 4 * hidden, hidden);
        ConditionIn = IndexTtsOps.Linear(engine, layers, random, hidden, 4 * hidden);
        ConditionOut = IndexTtsOps.Linear(engine, layers, random, 4 * hidden, hidden);
        Encoder = new Ns2TransformerEncoder<T>(engine, layers, random, 0, hidden, heads, filter, kernel, count, dropout, true, NaturalSpeech2Positions.Sinusoidal);
    }

    public DenseLayer<T> StepIn { get; }
    public DenseLayer<T> StepOut { get; }
    public DenseLayer<T> ConditionIn { get; }
    public DenseLayer<T> ConditionOut { get; }
    public Ns2TransformerEncoder<T> Encoder { get; }

    private Tensor<T> Silu(Tensor<T> x) => _engine.TensorMultiply(x, _engine.Sigmoid(x));

    /// <summary>The output rows <c>[N, hidden]</c> of input rows plus the condition rows, at time t.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> condition, double t, bool training, Random? random)
    {
        var step = StepOut.Forward(Silu(StepIn.Forward(Ns2Ops.StepEmbedding<T>(t, _hidden))));
        x = _engine.TensorAdd(x, ConditionOut.Forward(Silu(ConditionIn.Forward(condition))));
        return Encoder.Forward(x, step, training, random);
    }
}

/// <summary>
/// One attribute's masked discrete diffusion (§3.3.2) on a shared backbone, with MaskGCT's semantic-to-acoustic conventions
/// for what the paper leaves out: the input sums the attribute's codebooks (those below the predicted one given, the
/// predicted one masked where unknown, mask embeddings above), a layer embedding joins the condition, the prompt is an
/// unmasked prefix, and each codebook has its own head.
/// </summary>
internal sealed class Ns3MaskedDiffusion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _hidden;

    public Ns3MaskedDiffusion(IEngine engine, List<LayerBase<T>> layers, Random random, Ns3Backbone<T> backbone, int hidden, int codebooks, int vocabulary)
    {
        _engine = engine;
        _hidden = hidden;
        Backbone = backbone;
        Codebooks = codebooks;
        Vocabulary = vocabulary;
        Tokens = Enumerable.Range(0, codebooks).Select(_ => IndexTtsOps.Embedding(layers, random, vocabulary, hidden, 0.02)).ToArray();
        Mask = IndexTtsOps.Embedding(layers, random, 1, hidden, 0.02);
        LayerEmbedding = IndexTtsOps.Embedding(layers, random, codebooks, hidden, 0.02);
        Heads = Enumerable.Range(0, codebooks).Select(_ => IndexTtsOps.Linear(engine, layers, random, hidden, vocabulary)).ToArray();
    }

    public Ns3Backbone<T> Backbone { get; }
    public int Codebooks { get; }
    public int Vocabulary { get; }
    public TiedEmbeddingLayer<T>[] Tokens { get; }
    public TiedEmbeddingLayer<T> Mask { get; }
    public TiedEmbeddingLayer<T> LayerEmbedding { get; }
    public DenseLayer<T>[] Heads { get; }

    private Tensor<T> MaskRows(int rows) => _engine.TensorTile(Mask.Forward(IndexTtsOps.Ids<T>(new[] { 0 })), new[] { rows, 1 });

    private Tensor<T> Column(int layer, int[,] codes, int start, int count) =>
        Tokens[layer].Forward(IndexTtsOps.Ids<T>(Enumerable.Range(start, count).Select(i => codes[i, layer]).ToArray()));

    private Tensor<T> Rows(IReadOnlyList<bool> flags)
    {
        var m = new Tensor<T>(new[] { flags.Count, _hidden });
        for (int i = 0; i < flags.Count; i++)
            if (flags[i]) for (int j = 0; j < _hidden; j++) m[i, j] = NumOps.One;
        return m;
    }

    private Tensor<T> WithLayer(Tensor<T> condition, int layer) =>
        _engine.TensorAdd(condition, _engine.TensorTile(LayerEmbedding.Forward(IndexTtsOps.Ids<T>(new[] { layer })), new[] { condition.Shape[0], 1 }));

    /// <summary>
    /// The training logits <c>[N, V]</c> of codebook <paramref name="layer"/> for codes <c>[N, codebooks]</c> whose first
    /// <paramref name="prompt"/> frames are the prompt, masked at <paramref name="masked"/>, under the condition rows
    /// <c>[N, hidden]</c> at time t.
    /// </summary>
    public Tensor<T> TrainingLogits(Tensor<T> condition, int[,] codes, int layer, int prompt, IReadOnlyList<bool> masked, double t, bool training, Random? random)
    {
        int n = codes.GetLength(0);
        var x = new Tensor<T>(new[] { n, _hidden });
        var maskRows = MaskRows(n);
        var promptRows = Rows(Enumerable.Range(0, n).Select(i => i < prompt).ToArray());
        var notPrompt = _engine.TensorAddScalar(_engine.TensorNegate(promptRows), NumOps.One);
        var m = Rows(masked);
        var keep = _engine.TensorAddScalar(_engine.TensorNegate(m), NumOps.One);
        for (int q = 0; q < Codebooks; q++)
        {
            var e = Column(q, codes, 0, n);
            if (q < layer) x = _engine.TensorAdd(x, e);
            else if (q == layer) x = _engine.TensorAdd(x, _engine.TensorAdd(_engine.TensorMultiply(m, maskRows), _engine.TensorMultiply(keep, e)));
            else x = _engine.TensorAdd(x, _engine.TensorAdd(_engine.TensorMultiply(e, promptRows), _engine.TensorMultiply(maskRows, notPrompt)));
        }
        return Heads[layer].Forward(Backbone.Forward(x, WithLayer(condition, layer), t, training, random));
    }

    /// <summary>
    /// Generation (§3.3.2's inference): after a prompt's codes <c>[Lp, codebooks]</c> with condition rows covering the prompt
    /// and the target (<c>[Lp + N, hidden]</c>), each codebook coarse to fine by confidence-based remasking; classifier-free
    /// guidance contrasts with the target alone. Returns <c>[N, codebooks]</c>.
    /// </summary>
    public int[,] Generate(Tensor<T> condition, int[,] prompt, MaskGctDecoding decoding, Func<int, double[]> uniforms)
    {
        int lp = prompt.GetLength(0), n = condition.Shape[0] - lp;
        if (n <= 0) throw new ArgumentException("The condition must cover the prompt and a target.", nameof(condition));
        var result = new int[n, Codebooks];
        var cumulative = new Tensor<T>(new[] { n, _hidden });
        Tensor<T>? promptSum = null;
        for (int q = 0; q < Codebooks && lp > 0; q++)
            promptSum = promptSum is null ? Column(q, prompt, 0, lp) : _engine.TensorAdd(promptSum, Column(q, prompt, 0, lp));
        for (int layer = 0; layer < Codebooks; layer++)
        {
            var layered = WithLayer(condition, layer);
            var targetCondition = _engine.TensorSlice(layered, new[] { lp, 0 }, new[] { n, _hidden });
            var maskRows = MaskRows(n);
            int current = layer;
            var known = cumulative;
            var seq = MaskGctSampling.Decode(n, Vocabulary, decoding, (tokens, mask, t) =>
            {
                var m = Rows(mask);
                var keep = _engine.TensorAddScalar(_engine.TensorNegate(m), NumOps.One);
                var cur = _engine.TensorAdd(known, _engine.TensorAdd(_engine.TensorMultiply(m, maskRows),
                    _engine.TensorMultiply(keep, Tokens[current].Forward(IndexTtsOps.Ids<T>(tokens)))));
                cur = _engine.TensorAdd(cur, _engine.TensorMultiplyScalar(maskRows, NumOps.FromDouble(Codebooks - 1 - current)));
                var embeds = Backbone.Forward(promptSum is null ? cur : _engine.TensorConcatenate(new[] { promptSum, cur }, 0), layered, t, false, null);
                embeds = _engine.TensorSlice(embeds, new[] { lp, 0 }, new[] { n, _hidden });
                if (decoding.Guidance > 0 && lp > 0)
                {
                    var unconditional = Backbone.Forward(cur, targetCondition, t, false, null);
                    embeds = MaskGctSampling.Guide(_engine, embeds, unconditional, decoding.Guidance, decoding.GuidanceRescale);
                }
                return MaskGctSampling.Matrix(Heads[current].Forward(embeds));
            }, uniforms);
            for (int i = 0; i < n; i++) result[i, layer] = seq[i];
            cumulative = _engine.TensorAdd(cumulative, Tokens[layer].Forward(IndexTtsOps.Ids<T>(seq)));
        }
        return result;
    }
}

/// <summary>NaturalSpeech 3's factorized diffusion sizes.</summary>
internal sealed record Ns3Configuration
{
    public int Vocab { get; init; } = 100;
    public int PhonemeDim { get; init; } = 512;
    public int PhonemeHeads { get; init; } = 8;
    public int PhonemeLayers { get; init; } = 6;
    public int PhonemeFilter { get; init; } = 2048;
    public int PhonemeKernel { get; init; } = 9;
    public double Dropout { get; init; } = 0.1;
    public int Hidden { get; init; } = 1024;
    public int Heads { get; init; } = 8;
    public int Filter { get; init; } = 2048;
    public int Kernel { get; init; } = 3;
    public int FrameLayers { get; init; } = 12;
    public int PhonemeDiffusionLayers { get; init; } = 6;
    public int MaxDuration { get; init; } = 63;
    public int ProsodyCodebooks { get; init; } = 1;
    public int ContentCodebooks { get; init; } = 2;
    public int DetailCodebooks { get; init; } = 3;
    public int CodebookSize { get; init; } = 1024;
    public bool PhonemeProsody { get; init; }
}

/// <summary>
/// NaturalSpeech 3's factorized diffusion model (§3.3, Fig. 3): the phoneme encoder; duration diffusion at the phoneme
/// level and the length regulator; prosody, content and acoustic-detail diffusions sharing one Transformer, each conditioned
/// on the phonemes and the attributes generated before it (content on prosody, detail on prosody and content), each
/// following its own prompt. The optional phoneme-level prosody diffusion conditions the duration and prosody stages.
/// </summary>
internal sealed class Ns3Core<T>
{
    private readonly IEngine _engine;
    private readonly Ns3Configuration _c;

    public Ns3Core(IEngine engine, Ns3Configuration c, Random random, List<LayerBase<T>> layers)
    {
        _engine = engine;
        _c = c;
        int h = c.Hidden, v = c.CodebookSize;
        PhonemeEncoder = new Ns2TransformerEncoder<T>(engine, layers, random, c.Vocab, c.PhonemeDim, c.PhonemeHeads, c.PhonemeFilter, c.PhonemeKernel,
            c.PhonemeLayers, c.Dropout, false, NaturalSpeech2Positions.Sinusoidal, embeddingStd: 0.02);
        PhonemeProjection = IndexTtsOps.Linear(engine, layers, random, c.PhonemeDim, h);
        PhonemeBackbone = new Ns3Backbone<T>(engine, layers, random, h, c.Heads, c.Filter, c.Kernel, c.PhonemeDiffusionLayers, c.Dropout);
        FrameBackbone = new Ns3Backbone<T>(engine, layers, random, h, c.Heads, c.Filter, c.Kernel, c.FrameLayers, c.Dropout);
        Duration = new Ns3MaskedDiffusion<T>(engine, layers, random, PhonemeBackbone, h, 1, c.MaxDuration + 1);
        if (c.PhonemeProsody)
        {
            PhonemeProsody = new Ns3MaskedDiffusion<T>(engine, layers, random, PhonemeBackbone, h, 1, v);
            PhonemeProsodyCondition = IndexTtsOps.Embedding(layers, random, v, h, 0.02);
        }
        Prosody = new Ns3MaskedDiffusion<T>(engine, layers, random, FrameBackbone, h, c.ProsodyCodebooks, v);
        Content = new Ns3MaskedDiffusion<T>(engine, layers, random, FrameBackbone, h, c.ContentCodebooks, v);
        Detail = new Ns3MaskedDiffusion<T>(engine, layers, random, FrameBackbone, h, c.DetailCodebooks, v);
        ProsodyCondition = Enumerable.Range(0, c.ProsodyCodebooks).Select(_ => IndexTtsOps.Embedding(layers, random, v, h, 0.02)).ToArray();
        ContentCondition = Enumerable.Range(0, c.ContentCodebooks).Select(_ => IndexTtsOps.Embedding(layers, random, v, h, 0.02)).ToArray();
    }

    public Ns2TransformerEncoder<T> PhonemeEncoder { get; }
    public DenseLayer<T> PhonemeProjection { get; }
    public Ns3Backbone<T> PhonemeBackbone { get; }
    public Ns3Backbone<T> FrameBackbone { get; }
    public Ns3MaskedDiffusion<T> Duration { get; }
    public Ns3MaskedDiffusion<T>? PhonemeProsody { get; }
    public TiedEmbeddingLayer<T>? PhonemeProsodyCondition { get; }
    public Ns3MaskedDiffusion<T> Prosody { get; }
    public Ns3MaskedDiffusion<T> Content { get; }
    public Ns3MaskedDiffusion<T> Detail { get; }
    public TiedEmbeddingLayer<T>[] ProsodyCondition { get; }
    public TiedEmbeddingLayer<T>[] ContentCondition { get; }

    /// <summary>The phonemes' condition c_ph <c>[n, hidden]</c>: the phoneme encoder (its embedding drawn from N(0, 0.02²),
    /// MaskGCT's), projected to the diffusion width.</summary>
    public Tensor<T> Phonemes(IReadOnlyList<int> phonemes, bool training, Random? random) =>
        PhonemeProjection.Forward(PhonemeEncoder.Forward(phonemes, null, training, random));

    /// <summary>c_ph plus the embedding of each phoneme's prosody code, when the phoneme-level prosody stage is on.</summary>
    public Tensor<T> WithPhonemeProsody(Tensor<T> phonemes, IReadOnlyList<int>? prosody) =>
        prosody is null || PhonemeProsodyCondition is null ? phonemes : _engine.TensorAdd(phonemes, PhonemeProsodyCondition.Forward(IndexTtsOps.Ids<T>(prosody)));

    /// <summary>Rows <c>[n, hidden]</c> repeated by integer durations (the length regulator).</summary>
    public Tensor<T> Expand(Tensor<T> rows, IReadOnlyList<int> durations)
    {
        var index = new List<int>();
        for (int j = 0; j < durations.Count; j++)
            for (int r = 0; r < durations[j]; r++) index.Add(j);
        if (index.Count == 0) throw new ArgumentException("The durations expand to no frames.", nameof(durations));
        return _engine.TensorIndexSelect(rows, new Tensor<int>(new[] { index.Count }, new Vector<int>(index.ToArray())), 0);
    }

    /// <summary>The frame condition plus the embeddings of codes <c>[frames, ·]</c> through the given tables.</summary>
    public Tensor<T> Plus(Tensor<T> condition, int[,] codes, int firstColumn, TiedEmbeddingLayer<T>[] tables)
    {
        for (int q = 0; q < tables.Length; q++)
        {
            var ids = Enumerable.Range(0, codes.GetLength(0)).Select(f => codes[f, firstColumn + q]).ToArray();
            condition = _engine.TensorAdd(condition, tables[q].Forward(IndexTtsOps.Ids<T>(ids)));
        }
        return condition;
    }
}
