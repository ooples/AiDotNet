using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of GPT-SoVITS's text-to-semantic model (the s1 checkpoint's <c>config["model"]</c>).</summary>
/// <param name="Hidden">hidden_dim (= embedding_dim, 512).</param>
/// <param name="Heads">head (16).</param>
/// <param name="Layers">n_layer (24).</param>
/// <param name="Vocab">vocab_size: the semantic tokens and EOS (1025).</param>
/// <param name="PhonemeVocab">phoneme_vocab_size (512 for v1, 732 after).</param>
/// <param name="EmbeddingDropout">dropout: the token embeddings' (0).</param>
/// <param name="LayerDropout">The Transformer's attention, residual and feed-forward dropout and the positional
/// embeddings' (0.1, fixed in the release's code).</param>
/// <param name="BertDim">The BERT features' width (1024).</param>
internal sealed record GptSoVitsT2SConfiguration(int Hidden = 512, int Heads = 16, int Layers = 24, int Vocab = 1025, int PhonemeVocab = 732,
    double EmbeddingDropout = 0, double LayerDropout = 0.1, int BertDim = 1024)
{
    /// <summary>The end-of-semantic token (vocab_size − 1).</summary>
    public int Eos => Vocab - 1;
}

/// <summary>How GPT-SoVITS samples semantic tokens (<c>infer_panel_naive</c>'s <c>sample</c>).</summary>
internal sealed record GptSoVitsSampling(int TopK = 15, double TopP = 1.0, double Temperature = 1.0, double RepetitionPenalty = 1.35,
    int EarlyStopTokens = -1, int MaxSteps = 1500);

/// <summary>
/// GPT-SoVITS's text-to-semantic model (RVC-Boss/GPT-SoVITS <c>AR/models/t2s_model.py</c> <c>Text2SemanticDecoder</c>,
/// after SoundStorm's s1 and VALL-E): phoneme embeddings plus projected BERT features and the semantic-token embeddings,
/// each with sinusoidal positions scaled by a learned α, through a post-norm Transformer (ReLU feed-forward) where the
/// text attends to itself and each semantic token to the text and the tokens before it, and a bias-free head over the
/// 1,024 semantic tokens and EOS.
/// </summary>
internal sealed class GptSoVitsT2S<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly float[] _positions;
    private const int PositionRows = 4000;

    internal sealed record Block(DenseLayer<T> InProjection, DenseLayer<T> OutProjection, DenseLayer<T> Linear1, DenseLayer<T> Linear2,
        LayerNormalizationLayer<T> Norm1, LayerNormalizationLayer<T> Norm2);

    public GptSoVitsT2S(IEngine engine, List<LayerBase<T>> layers, GptSoVitsT2SConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int d = c.Hidden;
        if (d % c.Heads != 0) throw new ArgumentException("The width must divide by the head count.", nameof(configuration));
        BertProjection = IndexTtsOps.Linear(engine, layers, random, c.BertDim, d);
        TextEmbedding = IndexTtsOps.Embedding(layers, random, c.PhonemeVocab, d, 1.0);
        TextAlpha = IndexTtsOps.Values(layers, new[] { 1.0 });
        AudioEmbedding = IndexTtsOps.Embedding(layers, random, c.Vocab, d, 1.0);
        AudioAlpha = IndexTtsOps.Values(layers, new[] { 1.0 });
        for (int i = 0; i < c.Layers; i++)
        {
            // MultiheadAttention: xavier_uniform in_proj and zero biases; the feed-forward nn.Linear defaults.
            var inProjection = IndexTtsOps.Linear(engine, layers, random, d, 3 * d);
            double bound = Math.Sqrt(6.0 / (d + 3 * d));
            TorchParameters.Linear(engine, inProjection, d, 3 * d,
                Enumerable.Range(0, 3 * d * d).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(), new double[3 * d]);
            var outProjection = IndexTtsOps.Linear(engine, layers, random, d, d);
            TorchParameters.Linear(engine, outProjection, d, d, IndexTtsOps.Uniform(random, d * d, d), new double[d]);
            Blocks.Add(new Block(inProjection, outProjection, IndexTtsOps.Linear(engine, layers, random, d, 4 * d),
                IndexTtsOps.Linear(engine, layers, random, 4 * d, d), IndexTtsOps.LayerNorm(layers, d), IndexTtsOps.LayerNorm(layers, d)));
        }
        Head = IndexTtsOps.BiasFree(layers, random, d, c.Vocab);
        _positions = SinePositions(d);
    }

    public GptSoVitsT2SConfiguration Configuration { get; }
    public DenseLayer<T> BertProjection { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public BiasFreeLinearLayer<T> TextAlpha { get; }
    public TiedEmbeddingLayer<T> AudioEmbedding { get; }
    public BiasFreeLinearLayer<T> AudioAlpha { get; }
    public List<Block> Blocks { get; } = new();
    public BiasFreeLinearLayer<T> Head { get; }

    // SinePositionalEmbedding.extend_pe over 4000 rows, in float32 as the release builds it.
    private static float[] SinePositions(int dim)
    {
        var pe = new float[PositionRows * dim];
        var divTerm = new float[dim / 2];
        for (int i = 0; i < divTerm.Length; i++) divTerm[i] = MathF.Exp((2 * i) * (float)(-(Math.Log(10000.0) / dim)));
        for (int p = 0; p < PositionRows; p++)
            for (int i = 0; i < divTerm.Length; i++)
            {
                float angle = p * divTerm[i];
                pe[p * dim + 2 * i] = MathF.Sin(angle);
                if (2 * i + 1 < dim) pe[p * dim + 2 * i + 1] = MathF.Cos(angle);
            }
        return pe;
    }

    private Tensor<T> Positions(int start, int count)
    {
        int d = Configuration.Hidden;
        if (start + count > PositionRows) throw new ArgumentException($"At most {PositionRows} positions.");
        var t = new Tensor<T>(new[] { count, d });
        for (int i = 0; i < count; i++)
            for (int j = 0; j < d; j++) t[i, j] = NumOps.FromDouble(_positions[(start + i) * d + j]);
        return t;
    }

    private Tensor<T> Drop(Tensor<T> x, double p, bool training, Random? random) =>
        training && p > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, p, random ?? throw new ArgumentNullException(nameof(random))) : x;

    // x * x_scale + α * pe[start : start + n]
    private Tensor<T> Positioned(Tensor<T> x, BiasFreeLinearLayer<T> alpha, int start)
    {
        int n = x.Shape[0], d = Configuration.Hidden;
        var a = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(alpha), new[] { 1, 1 }), new[] { n, d });
        return _engine.TensorAdd(x, _engine.TensorMultiply(a, Positions(start, n)));
    }

    /// <summary>The text rows <c>[Tx, d]</c>: phoneme embeddings plus projected BERT features <c>[Tx, 1024]</c>, then
    /// α-scaled positions (and the positional dropout in training).</summary>
    public Tensor<T> TextRows(IReadOnlyList<int> phonemes, Tensor<T> bert, bool training = false, Random? random = null)
    {
        var c = Configuration;
        if (phonemes.Any(p => p < 0 || p >= c.PhonemeVocab)) throw new ArgumentOutOfRangeException(nameof(phonemes), $"Phoneme ids lie in [0, {c.PhonemeVocab}).");
        if (bert.Rank != 2 || bert.Shape[0] != phonemes.Count || bert.Shape[1] != c.BertDim)
            throw new ArgumentException($"The BERT features are [{phonemes.Count}, {c.BertDim}].", nameof(bert));
        var x = Drop(TextEmbedding.Forward(IndexTtsOps.Ids<T>(phonemes)), c.EmbeddingDropout, training, random);
        x = _engine.TensorAdd(x, BertProjection.Forward(bert));
        return Drop(Positioned(x, TextAlpha, 0), c.LayerDropout, training, random);
    }

    /// <summary>Semantic-token rows <c>[n, d]</c> at positions <paramref name="start"/> ….</summary>
    public Tensor<T> AudioRows(IReadOnlyList<int> tokens, int start = 0, bool training = false, Random? random = null)
    {
        var c = Configuration;
        if (tokens.Any(t => t < 0 || t >= c.Vocab)) throw new ArgumentOutOfRangeException(nameof(tokens), $"Semantic ids lie in [0, {c.Vocab}).");
        var y = Drop(AudioEmbedding.Forward(IndexTtsOps.Ids<T>(tokens)), c.EmbeddingDropout, training, random);
        return Drop(Positioned(y, AudioAlpha, start), c.LayerDropout, training, random);
    }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    /// <summary>One post-norm block over <c>[B, L, d]</c> with an additive bias <c>[B, L, Lk]</c> (or none), extending the
    /// layer's key and value cache <c>[B, Lk, d]</c> when one is given.</summary>
    private Tensor<T> RunBlock(Block block, Tensor<T> x, Tensor<T>? bias, ref Tensor<T>? keys, ref Tensor<T>? values, bool training, Random? random)
    {
        var c = Configuration;
        int b = x.Shape[0], l = x.Shape[1], d = c.Hidden, h = c.Heads, dk = d / h;
        var qkv = Rows(block.InProjection, x);
        var q = _engine.TensorNarrow(qkv, 2, 0, d);
        var k = _engine.TensorNarrow(qkv, 2, d, d);
        var v = _engine.TensorNarrow(qkv, 2, 2 * d, d);
        if (keys is not null)
        {
            k = _engine.TensorConcatenate(new[] { keys, k }, 1);
            v = _engine.TensorConcatenate(new[] { values!, v }, 1);
        }
        keys = k;
        values = v;
        int lk = k.Shape[1];
        Tensor<T> Split(Tensor<T> t, int length) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(t, new[] { b, length, h, dk }),
            new[] { 0, 2, 1, 3 }), new[] { b * h, length, dk });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Split(q, l), _engine.TensorPermute(Split(k, lk), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
        if (bias is not null)
        {
            var expanded = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(bias, new[] { b, 1, l, lk }), new[] { b, h, l, lk }),
                new[] { b * h, l, lk });
            scores = _engine.TensorAdd(scores, expanded);
        }
        var weights = Drop(_engine.Softmax(scores, axis: 2), c.LayerDropout, training, random);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(weights, Split(v, lk)), new[] { b, h, l, dk }),
            new[] { 0, 2, 1, 3 }), new[] { b, l, d });
        var attended = Rows(block.Norm1, _engine.TensorAdd(x, Drop(Rows(block.OutProjection, context), c.LayerDropout, training, random)));
        var hidden = Drop(_engine.ReLU(Rows(block.Linear1, attended)), c.LayerDropout, training, random);
        return Rows(block.Norm2, _engine.TensorAdd(attended, Drop(Rows(block.Linear2, hidden), c.LayerDropout, training, random)));
    }

    private Tensor<T> Logits(Tensor<T> rows) => Rows(Head, rows);

    // ---------------------------------------------------------------- inference

    /// <summary>
    /// <c>infer_panel_naive</c> (not streaming): the text <c>[Tx, d]</c> and the prompt's semantic tokens (or none, the
    /// reference-free mode) run once, then one token at a time with a key and value cache; EOS may not be chosen in the
    /// first 11 steps. Returns what the release's caller keeps, <c>y[:, -idx:]</c> with <c>idx</c> the last step: the
    /// generated tokens when EOS ended the run (EOS dropped); one fewer when the step limit or early stop did; all of
    /// <c>y</c> (the prompt included) when the run ended at step 0, and all generated tokens in the reference-free mode.
    /// </summary>
    /// <param name="noise">The Exp(1) noise of each step's <c>multinomial_sample_one_no_sync</c> over <paramref name="vocab"/>
    /// candidates (the step and the candidate count given); null draws it from <paramref name="random"/>.</param>
    public List<int> Generate(Tensor<T> textRows, IReadOnlyList<int>? prompt, GptSoVitsSampling sampling, Random random,
        Func<int, int, double[]>? noise = null)
    {
        var c = Configuration;
        int tx = textRows.Shape[0], d = c.Hidden;
        var y = prompt?.ToList() ?? new List<int>();
        int prefix = y.Count;
        var xy = prompt is { Count: > 0 } ? _engine.TensorConcatenate(new[] { textRows, AudioRows(prompt) }, 0) : textRows;
        int src = tx + prefix;
        // The text attends to the text only; each prompt token to the text and the tokens up to itself.
        var bias = new Tensor<T>(new[] { 1, src, src });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < src; i++)
            for (int j = 0; j < src; j++)
                if ((i < tx && j >= tx) || (i >= tx && j >= tx && j > i)) bias[0, i, j] = blocked;
        var keys = new Tensor<T>?[c.Layers];
        var values = new Tensor<T>?[c.Layers];
        var x = _engine.Reshape(xy, new[] { 1, src, d });
        Tensor<T>? stepBias = bias;
        int idx = 0;
        for (int step = 0; step < sampling.MaxSteps; step++)
        {
            idx = step;
            for (int i = 0; i < c.Layers; i++) x = RunBlock(Blocks[i], x, stepBias, ref keys[i], ref values[i], false, null);
            stepBias = null;
            var last = _engine.Reshape(_engine.TensorNarrow(x, 1, x.Shape[1] - 1, 1), new[] { 1, d });
            var logits = Logits(last).ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
            if (step < 11) Array.Resize(ref logits, logits.Length - 1);
            var probabilities = Probabilities(logits, y, sampling, out int penalizedArgmax);
            var draw = noise?.Invoke(step, probabilities.Length) ?? Enumerable.Range(0, probabilities.Length).Select(_ => -Math.Log(1 - random.NextDouble())).ToArray();
            int sample = 0;
            double best = double.NegativeInfinity;
            for (int j = 0; j < probabilities.Length; j++)
            {
                double ratio = probabilities[j] / draw[j];
                if (ratio > best) { best = ratio; sample = j; }
            }
            y.Add(sample);
            bool stop = sampling.EarlyStopTokens != -1 && y.Count - prefix > sampling.EarlyStopTokens;
            if (penalizedArgmax == c.Eos || sample == c.Eos)
            {
                stop = true;
                y.RemoveAt(y.Count - 1);
            }
            if (step == sampling.MaxSteps - 1) stop = true;
            if (stop)
            {
                if (y.Count == 0) y.Add(0);
                break;
            }
            x = _engine.Reshape(AudioRows(new[] { y[y.Count - 1] }, prefix + step), new[] { 1, 1, d });
        }
        if (prompt is not { Count: > 0 } || idx == 0) return y;
        return y.Skip(y.Count - idx).ToList();
    }

    /// <summary>
    /// <c>logits_to_probs</c>: the repetition penalty on every earlier token (divided if positive, multiplied if
    /// negative), nucleus filtering (keeping the most likely), temperature, top-k, softmax. Also returns the argmax of the
    /// penalized logits (the release checks it for EOS).
    /// </summary>
    internal static double[] Probabilities(double[] logits, IReadOnlyList<int> previous, GptSoVitsSampling sampling, out int penalizedArgmax)
    {
        var l = (double[])logits.Clone();
        if (previous.Count > 0 && sampling.RepetitionPenalty != 1.0)
        {
            var original = (double[])logits.Clone();
            foreach (int token in previous.Distinct())
            {
                if (token >= l.Length) throw new ArgumentOutOfRangeException(nameof(previous), "A previous token lies outside the logits.");
                double s = original[token];
                l[token] = s < 0 ? s * sampling.RepetitionPenalty : s / sampling.RepetitionPenalty;
            }
        }
        penalizedArgmax = 0;
        for (int i = 1; i < l.Length; i++) if (l[i] > l[penalizedArgmax]) penalizedArgmax = i;
        if (sampling.TopP < 1.0)
        {
            var order = Enumerable.Range(0, l.Length).OrderByDescending(i => l[i]).ToArray();
            double max = l[order[0]], z = order.Sum(i => Math.Exp(l[i] - max)), cumulative = 0;
            var remove = new bool[l.Length];
            for (int r = 0; r < order.Length; r++)
            {
                cumulative += Math.Exp(l[order[r]] - max) / z;
                remove[order[r]] = r > 0 && cumulative > sampling.TopP;
            }
            for (int i = 0; i < l.Length; i++) if (remove[i]) l[i] = double.NegativeInfinity;
        }
        double temperature = Math.Max(sampling.Temperature, 1e-5);
        for (int i = 0; i < l.Length; i++) l[i] /= temperature;
        if (sampling.TopK > 0)
        {
            double pivot = l.OrderByDescending(v => v).ElementAt(Math.Min(sampling.TopK, l.Length) - 1);
            for (int i = 0; i < l.Length; i++) if (l[i] < pivot) l[i] = double.NegativeInfinity;
        }
        double m = l.Max(), sum = 0;
        var p = new double[l.Length];
        for (int i = 0; i < l.Length; i++) { p[i] = Math.Exp(l[i] - m); sum += p[i]; }
        for (int i = 0; i < l.Length; i++) p[i] /= sum;
        return p;
    }

    // ---------------------------------------------------------------- training

    /// <summary>One training example: phonemes, their BERT features <c>[Tx, 1024]</c> and the target semantic tokens.</summary>
    internal sealed record Example(int[] Phonemes, Tensor<T> Bert, int[] Semantic);

    /// <summary>
    /// <c>make_input_data</c> and the forward pass for a batch: the text right-padded with zeros, the semantic tokens with
    /// EOS (the collate's pad), the targets the tokens then EOS (EOS over the padding), and the logits <c>[B, Ty + 1, V]</c>
    /// from the last text position on. The padding mask treats the text as left-padded, as the release's
    /// <c>make_pad_mask_left</c> does although its collate pads on the right; <paramref name="maskTextPaddingOnTheRight"/>
    /// masks the actual padding instead.
    /// </summary>
    internal (Tensor<T> Logits, int[,] Targets) TeacherForced(IReadOnlyList<Example> batch, IReadOnlyList<int[]> semantic, bool training,
        Random? random, bool maskTextPaddingOnTheRight = false)
    {
        var c = Configuration;
        int b = batch.Count, d = c.Hidden;
        int tx = batch.Max(e => e.Phonemes.Length), ty = semantic.Max(s => s.Length);
        var xRows = new List<Tensor<T>>();
        var yRows = new List<Tensor<T>>();
        var targets = new int[b, ty + 1];
        for (int i = 0; i < b; i++)
        {
            var e = batch[i];
            var phonemes = e.Phonemes.Concat(Enumerable.Repeat(0, tx - e.Phonemes.Length)).ToArray();
            var bert = new Tensor<T>(new[] { tx, c.BertDim });
            for (int r = 0; r < e.Phonemes.Length; r++)
                for (int j = 0; j < c.BertDim; j++) bert[r, j] = e.Bert[r, j];
            xRows.Add(TextRows(phonemes, bert, training, random));
            var s = semantic[i];
            var input = new int[ty];
            for (int t = 0; t <= ty; t++) targets[i, t] = t < s.Length ? s[t] : c.Eos;
            for (int t = 0; t < ty; t++) input[t] = targets[i, t];
            yRows.Add(AudioRows(input, 0, training, random));
        }
        int src = tx + ty;
        var x = _engine.TensorConcatenate(xRows.Select(r => _engine.Reshape(r, new[] { 1, tx, d })).ToArray(), 0);
        var y = _engine.TensorConcatenate(yRows.Select(r => _engine.Reshape(r, new[] { 1, ty, d })).ToArray(), 0);
        var xy = _engine.TensorConcatenate(new[] { x, y }, 1);
        var bias = new Tensor<T>(new[] { b, src, src });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < b; i++)
        {
            int textLength = batch[i].Phonemes.Length, semanticLength = semantic[i].Length;
            for (int q = 0; q < src; q++)
                for (int k = 0; k < src; k++)
                {
                    bool padded = k < tx
                        ? (maskTextPaddingOnTheRight ? k >= textLength : k < tx - textLength)
                        : k - tx >= semanticLength;
                    bool causal = (q < tx && k >= tx) || (q >= tx && k >= tx && k > q);
                    if (padded || causal) bias[i, q, k] = blocked;
                }
        }
        var h = xy;
        for (int i = 0; i < c.Layers; i++)
        {
            Tensor<T>? keys = null, values = null;
            h = RunBlock(Blocks[i], h, bias, ref keys, ref values, training, random);
        }
        return (Logits(_engine.TensorNarrow(h, 1, tx - 1, ty + 1)), targets);
    }

    /// <summary>The summed cross-entropy of the logits <c>[B, n, V]</c> against the targets (<c>forward_old</c>'s loss).</summary>
    internal Tensor<T> SummedCrossEntropy(Tensor<T> logits, int[,] targets) => _engine.TensorNegate(SummedLogProbabilities(logits, targets, null));

    // Per example (or summed over the batch when perExample is null) Σ log p(target).
    private Tensor<T> SummedLogProbabilities(Tensor<T> logits, int[,] targets, int? example)
    {
        int b = logits.Shape[0], n = logits.Shape[1], v = logits.Shape[2];
        var logProbabilities = _engine.Reshape(_engine.TensorLogSoftmax(logits, axis: 2), new[] { b * n * v, 1 });
        var picks = new List<int>();
        for (int i = 0; i < b; i++)
        {
            if (example is { } only && i != only) continue;
            for (int t = 0; t < n; t++) picks.Add((i * n + t) * v + targets[i, t]);
        }
        var index = new Tensor<int>(new[] { picks.Count }, new Vector<int>(picks.ToArray()));
        return _engine.ReduceSum(_engine.TensorIndexSelect(logProbabilities, index, 0), null);
    }

    /// <summary><c>make_reject_y</c>: per example, a random span repeated (or, with equal chance, dropped) from the padded
    /// token row; the rows padded with zeros to the longest.</summary>
    internal static List<int[]> RejectedSequences(IReadOnlyList<int[]> padded, Random random)
    {
        var result = new List<int[]>();
        foreach (var row in padded)
        {
            int a = random.Next(row.Length), z = random.Next(row.Length);
            if (a > z) (a, z) = (z, a);
            var pre = row.Take(a);
            var span = row.Skip(a).Take(z - a).ToArray();
            var rest = row.Skip(z);
            result.Add(random.Next(2) == 0 ? pre.Concat(span).Concat(span).Concat(rest).ToArray() : pre.Concat(rest).ToArray());
        }
        return result;
    }

    /// <summary>
    /// The s1 loss: <c>forward_old</c>'s summed cross-entropy, plus with <paramref name="dpo"/> (<c>if_dpo</c>) the
    /// reference-free DPO term −log σ(0.2 (log p(chosen) − log p(rejected))) averaged over the batch, the rejected tokens
    /// a span repeated or dropped.
    /// </summary>
    internal Tensor<T> Loss(IReadOnlyList<Example> batch, bool dpo, bool training, Random random, bool maskTextPaddingOnTheRight = false,
        IReadOnlyList<int[]>? rejectedSequences = null)
    {
        var c = Configuration;
        var semantic = batch.Select(e => e.Semantic).ToList();
        var (logits, targets) = TeacherForced(batch, semantic, training, random, maskTextPaddingOnTheRight);
        var loss = SummedCrossEntropy(logits, targets);
        if (!dpo) return loss;
        int ty = semantic.Max(s => s.Length);
        var padded = semantic.Select(s => s.Concat(Enumerable.Repeat(c.Eos, ty - s.Length)).ToArray()).ToList();
        var rejected = rejectedSequences?.ToList() ?? RejectedSequences(padded, random);
        int longest = rejected.Max(r => r.Length);
        var rejectedPadded = rejected.Select(r => r.Concat(Enumerable.Repeat(0, longest - r.Length)).ToArray()).ToList();
        // make_input_data multiplies the padding out and adds EOS there, so a rejected row's own length decides it.
        var rejectedRows = rejectedPadded.Select((r, i) => r.Take(rejected[i].Length).ToArray()).ToList();
        var (rejectedLogits, rejectedTargets) = TeacherForced(batch, rejectedRows, training, random, maskTextPaddingOnTheRight);
        Tensor<T>? dpoSum = null;
        for (int i = 0; i < batch.Count; i++)
        {
            var margin = _engine.TensorSubtract(SummedLogProbabilities(logits, targets, i), SummedLogProbabilities(rejectedLogits, rejectedTargets, i));
            // −log σ(βx) = softplus(−βx)
            var term = _engine.Softplus(_engine.TensorMultiplyScalar(margin, NumOps.FromDouble(-0.2)));
            dpoSum = dpoSum is null ? term : _engine.TensorAdd(dpoSum, term);
        }
        return _engine.TensorAdd(loss, _engine.TensorMultiplyScalar(dpoSum!, NumOps.FromDouble(1.0 / batch.Count)));
    }

    // ---------------------------------------------------------------- weights

    /// <summary>Loads the s1 checkpoint's state (names after <c>model.</c>).</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.Hidden;
        IndexTtsOps.LoadLinear(_engine, read, BertProjection, "bert_proj", c.BertDim, d);
        TextEmbedding.LoadLeadingRows(read("ar_text_embedding.word_embeddings.weight", new[] { c.PhonemeVocab, d }));
        AudioEmbedding.LoadLeadingRows(read("ar_audio_embedding.word_embeddings.weight", new[] { c.Vocab, d }));
        TorchParameters.Linear(TextAlpha, 1, 1, read("ar_text_position.alpha", new[] { 1 }));
        TorchParameters.Linear(AudioAlpha, 1, 1, read("ar_audio_position.alpha", new[] { 1 }));
        for (int i = 0; i < c.Layers; i++)
        {
            var block = Blocks[i];
            string n = $"h.layers.{i}";
            TorchParameters.Linear(_engine, block.InProjection, d, 3 * d, read($"{n}.self_attn.in_proj_weight", new[] { 3 * d, d }),
                read($"{n}.self_attn.in_proj_bias", new[] { 3 * d }));
            IndexTtsOps.LoadLinear(_engine, read, block.OutProjection, $"{n}.self_attn.out_proj", d, d);
            IndexTtsOps.LoadLinear(_engine, read, block.Linear1, $"{n}.linear1", d, 4 * d);
            IndexTtsOps.LoadLinear(_engine, read, block.Linear2, $"{n}.linear2", 4 * d, d);
            IndexTtsOps.LoadNorm(_engine, read, block.Norm1, $"{n}.norm1", d);
            IndexTtsOps.LoadNorm(_engine, read, block.Norm2, $"{n}.norm2", d);
        }
        IndexTtsOps.LoadLinear(read, Head, "ar_predict_layer", d, c.Vocab);
    }
}
