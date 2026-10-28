using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>How IndexTTS2's text-to-semantic model receives the target length.</summary>
public enum IndexTts2DurationConditioning
{
    /// <summary>The paper's (§ Duration Control): one token <c>p = W_sem h(T)</c> after the conditioning, the target
    /// length's row of the semantic position table (tied), or zeros for free-length generation. The input is
    /// <c>[c + e, p, BT, text, BA, sem]</c>.</summary>
    Paper,

    /// <summary>The released checkpoint's: two learned tokens (<c>speed_emb</c> rows 1 then 0) after the conditioning and
    /// no length control. Use it to run the released <c>gpt.pth</c>.</summary>
    ReleaseSpeedTokens,
}

/// <summary>Per-layer keys and values <c>[beams·heads, cached, dk]</c> of a GPT-2 decode.</summary>
internal sealed class IndexTtsGpt2Cache<T>
{
    internal readonly List<(Tensor<T> Keys, Tensor<T> Values)> Layers = new();

    /// <summary>Rows cached so far.</summary>
    public int Length => Layers.Count == 0 ? 0 : Layers[0].Keys.Shape[1];
}

/// <summary>
/// The GPT-2 transformer stack IndexTTS decodes with (Hugging Face <c>GPT2Model</c> with its position embeddings replaced
/// by zeros): pre-norm blocks of causal multi-head self-attention (packed <c>c_attn</c>) and a 4× <c>gelu_new</c>
/// feed-forward, residual dropout 0.1, and the final <c>ln_f</c>. LayerNorm ε = 1e-5.
/// </summary>
internal sealed class IndexTtsGpt2<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Attention, DenseLayer<T> AttentionProjection,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> FeedForward, DenseLayer<T> FeedForwardProjection);

    public IndexTtsGpt2(IEngine engine, List<LayerBase<T>> layers, Random random, int dim, int heads, int depth, double dropout)
    {
        if (dim % heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(heads));
        _engine = engine;
        Dim = dim;
        Heads = heads;
        Dropout = dropout;
        // GPT2PreTrainedModel._init_weights: N(0, 0.02) weights, zero biases, and the residual projections scaled by
        // 1/√(2·layers).
        DenseLayer<T> Conv1D(int inputs, int outputs, double std)
        {
            var layer = new DenseLayer<T>(outputs, (AiDotNet.Interfaces.IActivationFunction<T>)new AiDotNet.ActivationFunctions.IdentityActivation<T>());
            layers.Add(layer);
            TorchParameters.Linear(engine, layer, inputs, outputs, IndexTtsOps.Normal(random, inputs * outputs, std), new double[outputs]);
            return layer;
        }
        double residual = 0.02 / Math.Sqrt(2.0 * depth);
        for (int i = 0; i < depth; i++)
            Blocks.Add(new Block(IndexTtsOps.LayerNorm(layers, dim), Conv1D(dim, 3 * dim, 0.02), Conv1D(dim, dim, residual),
                IndexTtsOps.LayerNorm(layers, dim), Conv1D(dim, 4 * dim, 0.02), Conv1D(4 * dim, dim, residual)));
        FinalNorm = IndexTtsOps.LayerNorm(layers, dim);
    }

    public int Dim { get; }
    public int Heads { get; }
    public double Dropout { get; }
    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random? random) =>
        training && Dropout > 0 && random is not null ? T5Seq2Seq<T>.Dropout(_engine, x, Dropout, random) : x;

    /// <summary>
    /// The hidden states <c>[B, L, d]</c> (after <c>ln_f</c>) of input embeddings <c>[B, L, d]</c>, attending causally
    /// over <paramref name="cache"/>'s rows and the new ones, which are appended to it.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, IndexTtsGpt2Cache<T>? cache, bool training = false, Random? random = null)
    {
        int batch = x.Shape[0], length = x.Shape[1], d = Dim, heads = Heads, dk = d / heads;
        int past = cache?.Length ?? 0, total = past + length;
        var mask = new Tensor<T>(new[] { length, total });
        var blocked = NumOps.FromDouble(double.MinValue);
        for (int i = 0; i < length; i++)
            for (int j = past + i + 1; j < total; j++) mask[i, j] = blocked;
        bool masked = length > 1;

        Tensor<T> Split(Tensor<T> t) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(t, new[] { batch, length, heads, dk }),
            new[] { 0, 2, 1, 3 }), new[] { batch * heads, length, dk });
        var h = Drop(x, training, random);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            var qkv = MegaTts3Ops.Rows(_engine, b.Attention, b.Norm1.Forward(h));                  // [B, L, 3d]
            var q = Split(_engine.TensorNarrow(qkv, 2, 0, d));
            var k = Split(_engine.TensorNarrow(qkv, 2, d, d));
            var v = Split(_engine.TensorNarrow(qkv, 2, 2 * d, d));
            if (cache is not null)
            {
                if (i < cache.Layers.Count)
                {
                    k = _engine.TensorConcatenate(new[] { cache.Layers[i].Keys, k }, 1);
                    v = _engine.TensorConcatenate(new[] { cache.Layers[i].Values, v }, 1);
                    cache.Layers[i] = (k, v);
                }
                else
                {
                    cache.Layers.Add((k, v));
                }
            }
            var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
            if (masked) scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(_engine.Reshape(mask, new[] { 1, length, total }), scores.Shape.ToArray()));
            var weights = Drop(_engine.Softmax(scores, axis: 2), training, random);
            var context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(weights, v), new[] { batch, heads, length, dk }),
                new[] { 0, 2, 1, 3 }), new[] { batch, length, d });
            h = _engine.TensorAdd(h, Drop(MegaTts3Ops.Rows(_engine, b.AttentionProjection, context), training, random));
            var ff = IndexTtsOps.NewGelu(_engine, MegaTts3Ops.Rows(_engine, b.FeedForward, b.Norm2.Forward(h)));
            h = _engine.TensorAdd(h, Drop(MegaTts3Ops.Rows(_engine, b.FeedForwardProjection, ff), training, random));
        }
        return FinalNorm.Forward(h);
    }

    /// <summary>Keeps the cache rows of <paramref name="beams"/> (one source beam per new beam).</summary>
    public void Reorder(IndexTtsGpt2Cache<T> cache, IReadOnlyList<int> beams)
    {
        int heads = Heads;
        var rows = new Tensor<int>(new[] { beams.Count * heads });
        for (int b = 0; b < beams.Count; b++)
            for (int hd = 0; hd < heads; hd++) rows[b * heads + hd] = beams[b] * heads + hd;
        Tensor<T> Select(Tensor<T> t)
        {
            int length = t.Shape[1], dk = t.Shape[2];
            var picked = _engine.TensorIndexSelect(_engine.Reshape(t, new[] { t.Shape[0], length * dk }), rows, 0);
            return _engine.Reshape(picked, new[] { rows.Length, length, dk });
        }
        for (int i = 0; i < cache.Layers.Count; i++)
            cache.Layers[i] = (Select(cache.Layers[i].Keys), Select(cache.Layers[i].Values));
    }

    /// <summary>Loads <c>{prefix}h.{i}.*</c> and <c>{prefix}ln_f</c> (Hugging Face's <c>Conv1D</c> stores
    /// <c>[in, out]</c>).</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int d = Dim;
        void Conv1D(DenseLayer<T> layer, string name, int inputs, int outputs)
        {
            var weight = read(name + ".weight", new[] { inputs, outputs });
            var transposed = new double[weight.Length];
            for (int r = 0; r < inputs; r++)
                for (int c = 0; c < outputs; c++) transposed[c * inputs + r] = weight[r * outputs + c];
            TorchParameters.Linear(_engine, layer, inputs, outputs, transposed, read(name + ".bias", new[] { outputs }));
        }
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}h.{i}.";
            IndexTtsOps.LoadNorm(_engine, read, b.Norm1, p + "ln_1", d);
            Conv1D(b.Attention, p + "attn.c_attn", d, 3 * d);
            Conv1D(b.AttentionProjection, p + "attn.c_proj", d, d);
            IndexTtsOps.LoadNorm(_engine, read, b.Norm2, p + "ln_2", d);
            Conv1D(b.FeedForward, p + "mlp.c_fc", d, 4 * d);
            Conv1D(b.FeedForwardProjection, p + "mlp.c_proj", 4 * d, d);
        }
        IndexTtsOps.LoadNorm(_engine, read, FinalNorm, prefix + "ln_f", d);
    }
}

/// <summary>The sizes and special tokens of IndexTTS2's text-to-semantic model (the release's <c>gpt</c> config).</summary>
internal sealed record IndexTtsUnifiedVoiceConfiguration(int ModelDim, int Heads, int Layers, int MaxTextTokens, int MaxMelTokens,
    int TextTokens, int MelCodes, int StartText, int StopText, int StartMel, int StopMel, int ConditionLatents,
    IndexTtsConformerConfiguration Condition, int ConditionPerceiverMultiple, IndexTtsConformerConfiguration Emotion,
    int EmotionPerceiverMultiple, int EmotionDim, IndexTts2DurationConditioning Duration, double Dropout);

/// <summary>The decoding settings of IndexTTS2's semantic tokens (Hugging Face <c>generate</c> arguments).</summary>
internal sealed record IndexTtsDecoding(bool Sample, int Beams, int TopK, double TopP, double Temperature, double RepetitionPenalty,
    int MaxTokens);

/// <summary>
/// IndexTTS2's autoregressive text-to-semantic model (the release's <c>UnifiedVoice</c>, v2): a conformer and perceiver
/// pool the speaker prompt's w2v-BERT features into 32 latents c; a second conformer and a one-latent perceiver pool the
/// style prompt into the emotion vector e (then <c>emovec_layer</c> and <c>emo_layer</c>); the GPT-2 stack reads
/// <c>[c + e, duration, BT text ET, BA semantic EA]</c> with learned text and semantic positions and predicts each
/// semantic token through <c>final_norm</c> and <c>mel_head</c>.
/// </summary>
internal sealed class IndexTtsUnifiedVoice<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public IndexTtsUnifiedVoice(IEngine engine, List<LayerBase<T>> layers, IndexTtsUnifiedVoiceConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int d = c.ModelDim;
        int start = layers.Count;
        Conditioning = new IndexTtsConformerEncoder<T>(engine, layers, c.Condition, random);
        Perceiver = new IndexTtsPerceiver<T>(engine, layers, random, d, c.Condition.OutputSize, c.ConditionLatents, c.Condition.Heads,
            c.ConditionPerceiverMultiple);
        SpeakerConditionerLayers = layers.GetRange(start, layers.Count - start);
        if (c.Duration == IndexTts2DurationConditioning.ReleaseSpeedTokens)
            SpeedEmbedding = IndexTtsOps.Values(layers, new double[2 * d]);             // normal_(std = 0.0): zeros
        start = layers.Count;
        EmotionConditioning = new IndexTtsConformerEncoder<T>(engine, layers, c.Emotion, random);
        EmotionPerceiver = new IndexTtsPerceiver<T>(engine, layers, random, c.EmotionDim, c.Emotion.OutputSize, 1, c.Emotion.Heads,
            c.EmotionPerceiverMultiple);
        EmotionConditionerLayers = layers.GetRange(start, layers.Count - start);
        TextEmbedding = IndexTtsOps.Embedding(layers, random, c.TextTokens + 1, d, 0.02);
        EmotionLayer = IndexTtsOps.Linear(engine, layers, random, d, d);
        EmotionVectorLayer = IndexTtsOps.Linear(engine, layers, random, c.EmotionDim, d);
        MelEmbedding = IndexTtsOps.Embedding(layers, random, c.MelCodes, d, 0.02);
        Gpt = new IndexTtsGpt2<T>(engine, layers, random, d, c.Heads, c.Layers, c.Dropout);
        MelPositions = IndexTtsOps.Embedding(layers, random, c.MaxMelTokens + 2 + 1, d, 0.02);
        TextPositions = IndexTtsOps.Embedding(layers, random, c.MaxTextTokens + 2, d, 0.02);
        FinalNorm = IndexTtsOps.LayerNorm(layers, d);
        MelHead = IndexTtsOps.Linear(engine, layers, random, d, c.MelCodes);
    }

    public IndexTtsUnifiedVoiceConfiguration Configuration { get; }

    /// <summary>The speaker conditioner's layers (conformer and perceiver).</summary>
    public List<LayerBase<T>> SpeakerConditionerLayers { get; }

    /// <summary>The emotion conditioner's layers (conformer and perceiver).</summary>
    public List<LayerBase<T>> EmotionConditionerLayers { get; }
    public IndexTtsConformerEncoder<T> Conditioning { get; }
    public IndexTtsPerceiver<T> Perceiver { get; }
    public BiasFreeLinearLayer<T>? SpeedEmbedding { get; }
    public IndexTtsConformerEncoder<T> EmotionConditioning { get; }
    public IndexTtsPerceiver<T> EmotionPerceiver { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public DenseLayer<T> EmotionLayer { get; }
    public DenseLayer<T> EmotionVectorLayer { get; }
    public TiedEmbeddingLayer<T> MelEmbedding { get; }
    public IndexTtsGpt2<T> Gpt { get; }
    public TiedEmbeddingLayer<T> MelPositions { get; }
    public TiedEmbeddingLayer<T> TextPositions { get; }
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public DenseLayer<T> MelHead { get; }

    /// <summary>The longest semantic sequence a duration token can name (rows of the semantic position table).</summary>
    public int MaxDurationTokens => Configuration.MaxMelTokens + 2;

    /// <summary>The speaker latents c <c>[latents, d]</c> of normalized w2v-BERT features <c>[T, 1024]</c>.</summary>
    public Tensor<T> SpeakerConditioning(Tensor<T> features, bool training = false, Random? random = null) =>
        Perceiver.Forward(Conditioning.Forward(features, training, random));

    /// <summary>The emotion perceiver's pooled style <c>[EmotionDim]</c> of normalized w2v-BERT features.</summary>
    public Tensor<T> EmotionEmbedding(Tensor<T> features, bool training = false, Random? random = null) =>
        _engine.Reshape(EmotionPerceiver.Forward(EmotionConditioning.Forward(features, training, random)), new[] { 1, Configuration.EmotionDim });

    /// <summary>The emotion vector e <c>[1, d]</c> (<c>get_emovec</c>: the pooled style through <c>emovec_layer</c> and
    /// <c>emo_layer</c>).</summary>
    public Tensor<T> EmotionVector(Tensor<T> features, bool training = false, Random? random = null) =>
        EmotionLayer.Forward(EmotionVectorLayer.Forward(EmotionEmbedding(features, training, random)));

    /// <summary>
    /// The conditioning rows: <c>c + e</c> and the duration token(s). In <see cref="IndexTts2DurationConditioning.Paper"/>
    /// the token is the semantic position row of <paramref name="targetTokens"/> (zeros when null); the release's layout
    /// appends <c>speed_emb[1]</c> and <c>speed_emb[0]</c>.
    /// </summary>
    public Tensor<T> Conditions(Tensor<T> speaker, Tensor<T> emotion, int? targetTokens)
    {
        int d = Configuration.ModelDim, latents = speaker.Shape[0];
        var fused = _engine.TensorAdd(speaker, _engine.TensorBroadcastTo(emotion, new[] { latents, d }));
        Tensor<T> duration;
        if (SpeedEmbedding is not null)
        {
            var speed = _engine.Reshape(IndexTtsOps.Value(SpeedEmbedding), new[] { 2, d });
            duration = _engine.TensorConcatenate(new[] { _engine.TensorNarrow(speed, 0, 1, 1), _engine.TensorNarrow(speed, 0, 0, 1) }, 0);
        }
        else if (targetTokens is int length)
        {
            if (length < 0 || length >= MaxDurationTokens)
                throw new ArgumentOutOfRangeException(nameof(targetTokens), $"A target length must be below {MaxDurationTokens} tokens.");
            duration = MelPositions.Forward(IndexTtsOps.Ids<T>(new[] { length }));
        }
        else
        {
            duration = new Tensor<T>(new[] { 1, d });
        }
        return _engine.TensorConcatenate(new[] { fused, duration }, 0);
    }

    /// <summary>The text rows <c>[L + 2, d]</c>: <c>[BT, text…, ET]</c> embedded with text positions 0 ….</summary>
    public Tensor<T> TextRows(IReadOnlyList<int> text)
    {
        var c = Configuration;
        var ids = new List<int>(text.Count + 2) { c.StartText };
        ids.AddRange(text);
        ids.Add(c.StopText);
        if (ids.Count > TextPositions.VocabularySize) throw new ArgumentException($"At most {c.MaxTextTokens} text tokens fit.", nameof(text));
        return _engine.TensorAdd(TextEmbedding.Forward(IndexTtsOps.Ids<T>(ids)),
            TextPositions.Forward(IndexTtsOps.Ids<T>(Enumerable.Range(0, ids.Count).ToArray())));
    }

    /// <summary>Semantic rows <c>[n, d]</c> of tokens at semantic positions <paramref name="start"/> ….</summary>
    public Tensor<T> MelRows(IReadOnlyList<int> tokens, int start = 0)
    {
        if (start + tokens.Count > MelPositions.VocabularySize)
            throw new ArgumentException($"At most {Configuration.MaxMelTokens} semantic tokens fit.", nameof(tokens));
        return _engine.TensorAdd(MelEmbedding.Forward(IndexTtsOps.Ids<T>(tokens)),
            MelPositions.Forward(IndexTtsOps.Ids<T>(Enumerable.Range(start, tokens.Count).ToArray())));
    }

    /// <summary>
    /// The latents <c>[n + 1, d]</c> (after <c>final_norm</c>) at the semantic inputs <c>[BA, codes…]</c> following
    /// the conditions and text: row i predicts code i + 1 (the last row predicts EA).
    /// </summary>
    public Tensor<T> Latents(Tensor<T> conditions, IReadOnlyList<int> text, IReadOnlyList<int> codes, bool training = false, Random? random = null)
    {
        var inputs = new List<int>(codes.Count + 1) { Configuration.StartMel };
        inputs.AddRange(codes);
        var mel = MelRows(inputs);
        var x = _engine.TensorConcatenate(new[] { conditions, TextRows(text), mel }, 0);
        int length = x.Shape[0], d = Configuration.ModelDim;
        var hidden = Gpt.Forward(_engine.Reshape(x, new[] { 1, length, d }), null, training, random);
        var rows = _engine.TensorNarrow(_engine.Reshape(hidden, new[] { length, d }), 0, length - inputs.Count, inputs.Count);
        return FinalNorm.Forward(rows);
    }

    /// <summary>Semantic logits <c>[n, MelCodes]</c> of latents.</summary>
    public Tensor<T> Logits(Tensor<T> latents) => MelHead.Forward(latents);

    /// <summary>
    /// Generates semantic codes (the release's <c>inference_speech</c> through Hugging Face <c>generate</c>): beam search
    /// over <see cref="IndexTtsDecoding.Beams"/> beams, sampling the candidates when <see cref="IndexTtsDecoding.Sample"/>,
    /// with the repetition penalty, temperature, top-k and top-p applied to the log-probabilities. The result stops before
    /// EA.
    /// </summary>
    public int[] Generate(Tensor<T> conditions, IReadOnlyList<int> text, IndexTtsDecoding decoding, Random random)
    {
        var c = Configuration;
        int d = c.ModelDim, beams = decoding.Beams, vocabulary = c.MelCodes;
        var prefix = _engine.TensorConcatenate(new[] { conditions, TextRows(text), MelRows(new[] { c.StartMel }) }, 0);
        int prompt = prefix.Shape[0];
        var cache = new IndexTtsGpt2Cache<T>();
        var hidden = Gpt.Forward(_engine.Reshape(prefix, new[] { 1, prompt, d }), cache);
        var last = _engine.TensorNarrow(_engine.Reshape(hidden, new[] { prompt, d }), 0, prompt - 1, 1);
        // Every beam starts from the same prompt; only beam 0 scores (the others start at -1e9).
        Gpt.Reorder(cache, Enumerable.Repeat(0, beams).ToArray());
        var lastRows = Enumerable.Repeat(last, beams).ToArray();

        // The ids the repetition penalty sees: generate's placeholder prompt (all 1) and BA, then the codes.
        var sequences = new List<List<int>>();
        for (int b = 0; b < beams; b++) sequences.Add(new List<int>());
        var beamScores = new double[beams];
        for (int b = 1; b < beams; b++) beamScores[b] = -1e9;
        var hypotheses = new List<(double Score, int[] Codes)>();
        double worst = 1e9;
        bool done = false;
        int keep = 2 * beams;                                           // max(2, 1 + one end token) per beam
        for (int step = 0; step < decoding.MaxTokens && !done; step++)
        {
            var logits = Logits(FinalNorm.Forward(_engine.TensorConcatenate(lastRows, 0)));
            var candidates = new List<(double Score, int Token, int Beam)>();
            var flat = new double[beams * vocabulary];
            for (int b = 0; b < beams; b++)
            {
                var row = new double[vocabulary];
                for (int t = 0; t < vocabulary; t++) row[t] = NumOps.ToDouble(logits[b, t]);
                var scores = Warp(LogSoftmax(row), sequences[b], decoding);
                for (int t = 0; t < vocabulary; t++) flat[b * vocabulary + t] = scores[t] + beamScores[b];
            }
            int[] picked = decoding.Sample ? SampleWithoutReplacement(flat, keep, random) : TopIndices(flat, keep);
            var ordered = picked.Select(i => (Score: flat[i], Index: i)).OrderByDescending(p => p.Score).ToList();

            int cur = step + 1;                                             // tokens generated after this step
            var nextScores = new double[beams];
            var nextTokens = new int[beams];
            var nextBeams = new int[beams];
            int filled = 0;
            for (int rank = 0; rank < ordered.Count && filled < beams; rank++)
            {
                int token = ordered[rank].Index % vocabulary, beam = ordered[rank].Index / vocabulary;
                if (token == c.StopMel)
                {
                    if (rank >= beams) continue;
                    AddHypothesis(hypotheses, ref worst, beams, ordered[rank].Score, sequences[beam].ToArray());
                }
                else
                {
                    nextScores[filled] = ordered[rank].Score;
                    nextTokens[filled] = token;
                    nextBeams[filled] = beam;
                    filled++;
                }
            }
            if (filled < beams) throw new InvalidOperationException("Too many end tokens among the beam candidates.");
            // BeamHypotheses.is_done with early_stopping False and length_penalty 0.
            done = hypotheses.Count >= beams && worst >= ordered.Max(p => p.Score);

            sequences = nextBeams.Select((source, i) => new List<int>(sequences[source]) { nextTokens[i] }).ToList();
            beamScores = nextScores;
            if (done || step + 1 >= decoding.MaxTokens) break;
            Gpt.Reorder(cache, nextBeams);
            // GPT2InferenceModel embeds the k-th generated code at semantic position k + 1 (attention_mask length minus the
            // prompt), one past where the teacher-forced forward puts it; the release generates this way.
            var rows = new List<Tensor<T>>();
            for (int b = 0; b < beams; b++) rows.Add(MelRows(new[] { nextTokens[b] }, cur + 1));
            var next = Gpt.Forward(_engine.Reshape(_engine.TensorConcatenate(rows.ToArray(), 0), new[] { beams, 1, d }), cache);
            lastRows = Enumerable.Range(0, beams).Select(b => _engine.Reshape(_engine.TensorNarrow(next, 0, b, 1), new[] { 1, d })).ToArray();
        }
        if (!done)
            for (int b = 0; b < beams; b++) AddHypothesis(hypotheses, ref worst, beams, beamScores[b], sequences[b].ToArray());
        // finalize: the best hypothesis; Python's sorted() is stable, so among equal scores the last added wins.
        var best = hypotheses.Select((h, i) => (h, i)).OrderBy(p => p.h.Score).ThenBy(p => p.i).Last().h;
        return best.Codes;
    }

    private static void AddHypothesis(List<(double Score, int[] Codes)> hypotheses, ref double worst, int beams, double score, int[] codes)
    {
        if (hypotheses.Count < beams || score > worst)
        {
            hypotheses.Add((score, codes));
            if (hypotheses.Count > beams)
            {
                var sorted = hypotheses.Select((h, i) => (h.Score, i)).OrderBy(p => p.Score).ThenBy(p => p.i).ToList();
                hypotheses.RemoveAt(sorted[0].i);
                worst = sorted[1].Score;
            }
            else
            {
                worst = Math.Min(score, worst);
            }
        }
    }

    private static double[] LogSoftmax(double[] row)
    {
        double max = row.Max(), sum = 0;
        foreach (var v in row) sum += Math.Exp(v - max);
        double log = max + Math.Log(sum);
        return row.Select(v => v - log).ToArray();
    }

    /// <summary>The logits processors in <c>generate</c>'s order: repetition penalty, then (sampling) temperature, top-k
    /// and top-p, each keeping at least two tokens.</summary>
    private double[] Warp(double[] scores, IReadOnlyList<int> generated, IndexTtsDecoding decoding)
    {
        var c = Configuration;
        if (decoding.RepetitionPenalty != 1.0)
        {
            var seen = new HashSet<int>(generated) { 1, c.StartMel };
            foreach (int t in seen)
            {
                if (t >= scores.Length) continue;
                scores[t] = scores[t] < 0 ? scores[t] * decoding.RepetitionPenalty : scores[t] / decoding.RepetitionPenalty;
            }
        }
        if (!decoding.Sample) return scores;
        if (decoding.Temperature != 1.0)
            for (int t = 0; t < scores.Length; t++) scores[t] /= decoding.Temperature;
        if (decoding.TopK > 0)
        {
            int k = Math.Min(Math.Max(decoding.TopK, 2), scores.Length);
            double threshold = scores.OrderByDescending(v => v).ElementAt(k - 1);
            for (int t = 0; t < scores.Length; t++)
                if (scores[t] < threshold) scores[t] = double.NegativeInfinity;
        }
        if (decoding.TopP < 1.0)
        {
            var order = Enumerable.Range(0, scores.Length).OrderBy(t => scores[t]).ToArray();
            double max = scores.Max(), total = 0;
            var probabilities = order.Select(t => double.IsNegativeInfinity(scores[t]) ? 0 : Math.Exp(scores[t] - max)).ToArray();
            total = probabilities.Sum();
            double cumulative = 0;
            var remove = new bool[order.Length];
            for (int i = 0; i < order.Length; i++)
            {
                cumulative += probabilities[i] / total;
                remove[i] = cumulative <= 1 - decoding.TopP;
            }
            for (int i = Math.Max(0, order.Length - 2); i < order.Length; i++) remove[i] = false;
            for (int i = 0; i < order.Length; i++)
                if (remove[i]) scores[order[i]] = double.NegativeInfinity;
        }
        return scores;
    }

    private static int[] TopIndices(double[] scores, int count) =>
        Enumerable.Range(0, scores.Length).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(count).ToArray();

    /// <summary><c>torch.multinomial(softmax(scores), count)</c> without replacement: successive draws from the
    /// renormalized remainder.</summary>
    private static int[] SampleWithoutReplacement(double[] scores, int count, Random random)
    {
        double max = scores.Where(v => !double.IsNegativeInfinity(v)).DefaultIfEmpty(0).Max();
        var weights = scores.Select(v => double.IsNegativeInfinity(v) ? 0 : Math.Exp(v - max)).ToArray();
        var picked = new List<int>(count);
        for (int n = 0; n < count; n++)
        {
            double total = weights.Sum();
            if (total <= 0) break;
            double u = random.NextDouble() * total, acc = 0;
            int chosen = -1;
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] <= 0) continue;
                acc += weights[i];
                chosen = i;
                if (acc > u) break;
            }
            picked.Add(chosen);
            weights[chosen] = 0;
        }
        return picked.ToArray();
    }

    /// <summary>Loads the release's <c>gpt.pth</c> names (the text head, which inference never reads, is not loaded).</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.ModelDim;
        Conditioning.Load(read, "conditioning_encoder.");
        Perceiver.Load(_engine, read, "perceiver_encoder.");
        if (SpeedEmbedding is not null) TorchParameters.Linear(SpeedEmbedding, 1, 2 * d, read("speed_emb.weight", new[] { 2, d }));
        EmotionConditioning.Load(read, "emo_conditioning_encoder.");
        EmotionPerceiver.Load(_engine, read, "emo_perceiver_encoder.");
        TextEmbedding.LoadTable(read("text_embedding.weight", new[] { c.TextTokens + 1, d }));
        IndexTtsOps.LoadLinear(_engine, read, EmotionLayer, "emo_layer", d, d);
        IndexTtsOps.LoadLinear(_engine, read, EmotionVectorLayer, "emovec_layer", c.EmotionDim, d);
        MelEmbedding.LoadTable(read("mel_embedding.weight", new[] { c.MelCodes, d }));
        Gpt.Load(read, "gpt.");
        MelPositions.LoadTable(read("mel_pos_embedding.emb.weight", new[] { MelPositions.VocabularySize, d }));
        TextPositions.LoadTable(read("text_pos_embedding.emb.weight", new[] { TextPositions.VocabularySize, d }));
        IndexTtsOps.LoadNorm(_engine, read, FinalNorm, "final_norm", d);
        IndexTtsOps.LoadLinear(_engine, read, MelHead, "mel_head", d, c.MelCodes);
    }
}
