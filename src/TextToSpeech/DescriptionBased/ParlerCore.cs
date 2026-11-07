using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>The sizes of Parler-TTS's decoder (huggingface/parler-tts <c>ParlerTTSDecoderConfig</c>; the released
/// parler-tts-mini-v1 in parentheses).</summary>
/// <param name="AudioVocabulary">Logits per codebook (1088: 1024 codes, the end token 1024, the start token 1025 and
/// spare rows); each codebook's embedding has one row more.</param>
/// <param name="Codebooks">Codebooks (9: DAC 44 kHz at 8 kbps).</param>
/// <param name="ModelDim">Width (1024).</param>
/// <param name="Heads">Attention heads (16).</param>
/// <param name="KeyValueHeads">Key/value heads of the self-attention (16; fewer is grouped-query attention).</param>
/// <param name="CrossKeyValueHeads">Key/value heads of the cross-attention (16).</param>
/// <param name="Layers">Decoder layers (24).</param>
/// <param name="FeedForwardDim">Feed-forward width (4096).</param>
/// <param name="PromptVocabulary">Rows of the transcript embedding <c>embed_prompts</c> (32128, Flan-T5's).</param>
/// <param name="EncoderDim">Width of the description encoder's states (1024, Flan-T5-large); a different width adds the
/// projection <c>enc_to_dec_proj</c>.</param>
/// <param name="Dropout">Dropout on the embeddings and each residual branch (0.1).</param>
/// <param name="AttentionDropout">Dropout on the attention weights (0).</param>
/// <param name="ActivationDropout">Dropout after the feed-forward activation (0).</param>
/// <param name="LayerDrop">Probability of skipping a decoder layer in training (0).</param>
/// <param name="RotaryEmbeddings">Rotary positions in place of the added sinusoidal ones (off).</param>
/// <param name="RopeTheta">Rotary base (10,000).</param>
/// <param name="PromptCrossAttention">Whether the transcript is cross-attended next to the description instead of
/// prepended to the codes (off).</param>
/// <param name="BeginToken">The start token of every codebook (1025).</param>
/// <param name="EndToken">The end and padding token (1024).</param>
/// <param name="CodebookWeights">Per-codebook loss weights (none: the mean over codebooks).</param>
internal sealed record ParlerConfiguration(
    int AudioVocabulary, int Codebooks, int ModelDim, int Heads, int KeyValueHeads, int CrossKeyValueHeads, int Layers,
    int FeedForwardDim, int PromptVocabulary, int EncoderDim, double Dropout, double AttentionDropout,
    double ActivationDropout, double LayerDrop, bool RotaryEmbeddings, double RopeTheta, bool PromptCrossAttention,
    int BeginToken, int EndToken, double[]? CodebookWeights);

/// <summary>
/// The rotation of a rotary position embedding at each position, <c>[positions, headDim]</c>: the reference's
/// <c>ParlerTTSRotaryEmbedding</c> (Llama's), whose halves repeat the frequencies <c>θ^(−2i/headDim)</c>.
/// </summary>
internal sealed class ParlerRotation<T>
{
    public ParlerRotation(int positions, int headDim, double theta)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Cos = new Tensor<T>(new[] { positions, headDim });
        Sin = new Tensor<T>(new[] { positions, headDim });
        int half = headDim / 2;
        for (int p = 0; p < positions; p++)
            for (int i = 0; i < half; i++)
            {
                double angle = p / Math.Pow(theta, 2.0 * i / headDim);
                Cos[p, i] = Cos[p, i + half] = ops.FromDouble(Math.Cos(angle));
                Sin[p, i] = Sin[p, i + half] = ops.FromDouble(Math.Sin(angle));
            }
    }

    public Tensor<T> Cos { get; }
    public Tensor<T> Sin { get; }

    /// <summary><c>x · cos + rotate_half(x) · sin</c> on heads <c>[H, L, headDim]</c>.</summary>
    public Tensor<T> Apply(IEngine engine, Tensor<T> x)
    {
        int heads = x.Shape[0], length = x.Shape[1], dim = x.Shape[2], half = dim / 2;
        var ops = MathHelper.GetNumericOperations<T>();
        var first = engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { heads, length, half });
        var second = engine.TensorSlice(x, new[] { 0, 0, half }, new[] { heads, length, half });
        var rotated = engine.TensorConcatenate(new[] { engine.TensorMultiplyScalar(second, ops.FromDouble(-1.0)), first }, 2);
        Tensor<T> Tiled(Tensor<T> table) => engine.TensorTile(engine.Reshape(table, new[] { 1, length, dim }), new[] { heads, 1, 1 });
        return engine.TensorAdd(engine.TensorMultiply(x, Tiled(Cos)), engine.TensorMultiply(rotated, Tiled(Sin)));
    }
}

/// <summary>
/// The reference's <c>ParlerTTSAttention</c>: bias-free query, key, value and output projections, queries scaled by
/// <c>headDim^-1/2</c>, keys and values shared by groups of heads when there are fewer of them (<c>repeat_kv</c>), an
/// additive mask, and dropout on the attention weights.
/// </summary>
internal sealed class ParlerAttention<T>
{
    private readonly IEngine _engine;
    private readonly int _dim, _heads, _keyValueHeads;
    private readonly double _dropout;

    public ParlerAttention(IEngine engine, List<LayerBase<T>> layers, int dim, int heads, int keyValueHeads, double dropout)
    {
        if (dim % heads != 0) throw new ArgumentException($"Width {dim} must divide into {heads} heads.");
        if (heads % keyValueHeads != 0) throw new ArgumentException($"{heads} heads must divide into {keyValueHeads} key/value heads.");
        _engine = engine;
        _dim = dim;
        _heads = heads;
        _keyValueHeads = keyValueHeads;
        _dropout = dropout;
        int headDim = dim / heads;
        Query = ParlerCore<T>.Own(layers, new BiasFreeLinearLayer<T>(dim, dim));
        Key = ParlerCore<T>.Own(layers, new BiasFreeLinearLayer<T>(dim, keyValueHeads * headDim));
        Value = ParlerCore<T>.Own(layers, new BiasFreeLinearLayer<T>(dim, keyValueHeads * headDim));
        Output = ParlerCore<T>.Own(layers, new BiasFreeLinearLayer<T>(dim, dim));
    }

    public BiasFreeLinearLayer<T> Query { get; }
    public BiasFreeLinearLayer<T> Key { get; }
    public BiasFreeLinearLayer<T> Value { get; }
    public BiasFreeLinearLayer<T> Output { get; }
    public int KeyValueWidth => _keyValueHeads * (_dim / _heads);

    /// <summary>Attention of <paramref name="x"/> <c>[Lq, d]</c> over <paramref name="source"/> <c>[Lk, d]</c>; the
    /// rotation, when given, turns the queries and (in self-attention) the keys.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> source, Tensor<T>? mask, ParlerRotation<T>? rotation, bool rotateKeys,
        bool training, Random random)
    {
        int queries = x.Shape[0], keys = source.Shape[0], headDim = _dim / _heads, groups = _heads / _keyValueHeads;
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> Split(Tensor<T> projected, int length, int heads) =>
            _engine.TensorPermute(_engine.Reshape(projected, new[] { length, heads, headDim }), new[] { 1, 0, 2 });
        Tensor<T> Repeat(Tensor<T> kv)
        {
            if (groups == 1) return kv;
            var expanded = _engine.TensorTile(_engine.Reshape(kv, new[] { _keyValueHeads, 1, keys, headDim }), new[] { 1, groups, 1, 1 });
            return _engine.Reshape(expanded, new[] { _heads, keys, headDim });
        }
        var q = Split(_engine.TensorMultiplyScalar(Query.Forward(x), ops.FromDouble(1.0 / Math.Sqrt(headDim))), queries, _heads);
        var k = Split(Key.Forward(source), keys, _keyValueHeads);
        var v = Split(Value.Forward(source), keys, _keyValueHeads);
        if (rotation is not null)
        {
            q = rotation.Apply(_engine, q);
            if (rotateKeys) k = rotation.Apply(_engine, k);
        }
        k = Repeat(k);
        v = Repeat(v);
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                    // [H, Lq, Lk]
        if (mask is not null)
            scores = _engine.TensorAdd(scores, _engine.TensorTile(_engine.Reshape(mask, new[] { 1, queries, keys }), new[] { _heads, 1, 1 }));
        var weights = _engine.Softmax(scores, axis: 2);
        if (training && _dropout > 0) weights = T5Seq2Seq<T>.Dropout(_engine, weights, _dropout, random);
        var context = _engine.BatchMatMul(weights, v);                                                       // [H, Lq, hd]
        var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { queries, _dim });
        return Output.Forward(merged);
    }
}

/// <summary>
/// The reference's <c>ParlerTTSDecoderLayer</c>, pre-norm: <c>x + SelfAttention(LN(x))</c>,
/// <c>x + CrossAttention(LN(x), description)</c>, <c>x + fc2(GELU(fc1(LN(x))))</c>, every projection bias-free, the
/// exact (erf) GELU, dropout on each residual branch.
/// </summary>
internal sealed class ParlerDecoderLayer<T>
{
    private readonly IEngine _engine;
    private readonly ParlerConfiguration _c;

    public ParlerDecoderLayer(IEngine engine, List<LayerBase<T>> layers, ParlerConfiguration c)
    {
        _engine = engine;
        _c = c;
        SelfAttention = new ParlerAttention<T>(engine, layers, c.ModelDim, c.Heads, c.KeyValueHeads, c.AttentionDropout);
        SelfNorm = ParlerCore<T>.Own(layers, new LayerNormalizationLayer<T>(c.ModelDim));
        CrossAttention = new ParlerAttention<T>(engine, layers, c.ModelDim, c.Heads, c.CrossKeyValueHeads, c.AttentionDropout);
        CrossNorm = ParlerCore<T>.Own(layers, new LayerNormalizationLayer<T>(c.ModelDim));
        FeedForwardIn = ParlerCore<T>.Own(layers, new BiasFreeLinearLayer<T>(c.ModelDim, c.FeedForwardDim));
        FeedForwardOut = ParlerCore<T>.Own(layers, new BiasFreeLinearLayer<T>(c.FeedForwardDim, c.ModelDim));
        FinalNorm = ParlerCore<T>.Own(layers, new LayerNormalizationLayer<T>(c.ModelDim));
    }

    public ParlerAttention<T> SelfAttention { get; }
    public LayerNormalizationLayer<T> SelfNorm { get; }
    public ParlerAttention<T> CrossAttention { get; }
    public LayerNormalizationLayer<T> CrossNorm { get; }
    public BiasFreeLinearLayer<T> FeedForwardIn { get; }
    public BiasFreeLinearLayer<T> FeedForwardOut { get; }
    public LayerNormalizationLayer<T> FinalNorm { get; }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> memory, Tensor<T> causalMask, ParlerRotation<T>? rotation, bool training, Random random)
    {
        var normed = SelfNorm.Forward(x);
        x = _engine.TensorAdd(x, Drop(SelfAttention.Forward(normed, normed, causalMask, rotation, rotateKeys: true, training, random),
            _c.Dropout, training, random));
        // The cross-attention's queries are rotated, its keys (the description) are not, as in the reference.
        x = _engine.TensorAdd(x, Drop(CrossAttention.Forward(CrossNorm.Forward(x), memory, null, rotation, rotateKeys: false, training, random),
            _c.Dropout, training, random));
        var hidden = Drop(ActivationHelper.ExactGelu(FeedForwardIn.Forward(FinalNorm.Forward(x)), _engine), _c.ActivationDropout, training, random);
        return _engine.TensorAdd(x, Drop(FeedForwardOut.Forward(hidden), _c.Dropout, training, random));
    }

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random random) =>
        training && rate > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, rate, random) : x;
}

/// <summary>
/// Parler-TTS's codec language model (Lyth &amp; King 2024, §3; huggingface/parler-tts
/// <c>ParlerTTSForConditionalGeneration</c> without its text and audio encoders): the transcript's token embeddings
/// prepended to the summed per-codebook code embeddings, sinusoidal positions, a pre-norm decoder that cross-attends to
/// the description's encoder states, and one bias-free head per codebook, over codes in MusicGen's delay pattern.
/// </summary>
internal sealed class ParlerCore<T>
{
    private readonly IEngine _engine;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public ParlerCore(IEngine engine, List<LayerBase<T>> layers, ParlerConfiguration configuration)
    {
        _engine = engine;
        var c = Configuration = configuration;
        // The transcript embedding first: the model's input (the transcript's token ids) enters there.
        PromptEmbedding = Own(layers, new TiedEmbeddingLayer<T>(c.PromptVocabulary, c.ModelDim));
        for (int q = 0; q < c.Codebooks; q++)
            Embeddings.Add(Own(layers, new TiedEmbeddingLayer<T>(c.AudioVocabulary + 1, c.ModelDim)));
        for (int i = 0; i < c.Layers; i++) Layers.Add(new ParlerDecoderLayer<T>(engine, layers, c));
        FinalNorm = Own(layers, new LayerNormalizationLayer<T>(c.ModelDim));
        for (int q = 0; q < c.Codebooks; q++)
            Heads.Add(Own(layers, new BiasFreeLinearLayer<T>(c.ModelDim, c.AudioVocabulary)));
        if (c.EncoderDim != c.ModelDim)
            Projection = Own(layers, new DenseLayer<T>(c.ModelDim, new IdentityActivation<T>() as IActivationFunction<T>));
    }

    public ParlerConfiguration Configuration { get; }
    public List<TiedEmbeddingLayer<T>> Embeddings { get; } = new();
    public List<ParlerDecoderLayer<T>> Layers { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public List<BiasFreeLinearLayer<T>> Heads { get; } = new();
    public TiedEmbeddingLayer<T> PromptEmbedding { get; }
    public DenseLayer<T>? Projection { get; }

    internal static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    private static Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    /// <summary>
    /// The reference's sinusoidal table (tensor2tensor's, <c>ParlerTTSSinusoidalPositionalEmbedding</c>): cosines in the
    /// first half of the channels, sines in the second, frequencies <c>10000^(−i/(d/2 − 1))</c>, a zero last channel
    /// when the width is odd.
    /// </summary>
    internal static Tensor<T> Sinusoids(int positions, int dim)
    {
        int half = dim / 2;
        double step = Math.Log(10000.0) / (half - 1);
        var table = new Tensor<T>(new[] { positions, dim });
        for (int p = 0; p < positions; p++)
            for (int i = 0; i < half; i++)
            {
                double angle = p * Math.Exp(-step * i);
                table[p, i] = NumOps.FromDouble(Math.Cos(angle));
                table[p, half + i] = NumOps.FromDouble(Math.Sin(angle));
            }
        return table;
    }

    /// <summary>The description's states as the decoder reads them: projected to its width when they differ, and with
    /// the transcript's positioned embeddings appended under <see cref="ParlerConfiguration.PromptCrossAttention"/>.</summary>
    public Tensor<T> Memory(Tensor<T> descriptionStates, IReadOnlyList<int> transcript)
    {
        var memory = Projection is null ? descriptionStates : Projection.Forward(descriptionStates);
        if (!Configuration.PromptCrossAttention) return memory;
        var prompt = PromptEmbedding.Forward(Ids(transcript));
        prompt = _engine.TensorAdd(prompt, Sinusoids(transcript.Count, Configuration.ModelDim));
        return _engine.TensorConcatenate(new[] { memory, prompt }, 0);
    }

    /// <summary>
    /// The logits <c>[length, vocabulary]</c> of each codebook at each code position, for decoder input codes
    /// <paramref name="codes"/> <c>[codebooks, length]</c> after the transcript <paramref name="transcript"/>, with
    /// <paramref name="memory"/> from <see cref="Memory"/>; position <c>t</c> predicts the codes after
    /// <c>codes[·, t]</c>.
    /// </summary>
    public IReadOnlyList<Tensor<T>> Logits(int[,] codes, IReadOnlyList<int> transcript, Tensor<T> memory, bool training, Random random)
    {
        var c = Configuration;
        int codebooks = codes.GetLength(0), length = codes.GetLength(1), d = c.ModelDim;
        if (codebooks != c.Codebooks) throw new ArgumentException($"Expected {c.Codebooks} codebooks of decoder input.", nameof(codes));
        Tensor<T>? x = null;
        for (int q = 0; q < codebooks; q++)
        {
            var row = new int[length];
            for (int t = 0; t < length; t++) row[t] = codes[q, t];
            var embedded = Embeddings[q].Forward(Ids(row));
            x = x is null ? embedded : _engine.TensorAdd(x, embedded);
        }
        int prepended = 0;
        if (!c.PromptCrossAttention)
        {
            prepended = transcript.Count;
            x = _engine.TensorConcatenate(new[] { PromptEmbedding.Forward(Ids(transcript)), x! }, 0);
        }
        int total = prepended + length;
        ParlerRotation<T>? rotation = null;
        if (c.RotaryEmbeddings) rotation = new ParlerRotation<T>(total, d / c.Heads, c.RopeTheta);
        else x = _engine.TensorAdd(x!, Sinusoids(total, d));
        if (training && c.Dropout > 0) x = T5Seq2Seq<T>.Dropout(_engine, x!, c.Dropout, random);

        var mask = new Tensor<T>(new[] { total, total });
        var blocked = NumOps.FromDouble(-1e9);
        for (int i = 0; i < total; i++)
            for (int j = i + 1; j < total; j++) mask[i, j] = blocked;
        foreach (var layer in Layers)
        {
            if (training && c.LayerDrop > 0 && random.NextDouble() < c.LayerDrop) continue;
            x = layer.Forward(x!, memory, mask, rotation, training, random);
        }
        var hidden = FinalNorm.Forward(x!);
        if (prepended > 0) hidden = _engine.TensorSlice(hidden, new[] { prepended, 0 }, new[] { length, d });
        return Heads.Select(head => head.Forward(hidden)).ToList();
    }

    /// <summary>
    /// The training labels <c>[codebooks, frames + codebooks]</c> of codes <c>[frames, codebooks]</c>, as the reference's
    /// training script builds them: codebook <c>q</c> is delayed by <c>q</c> steps behind start tokens and closed with
    /// end tokens.
    /// </summary>
    public int[,] DelayedLabels(int[,] codes)
    {
        var c = Configuration;
        int frames = codes.GetLength(0), codebooks = c.Codebooks, length = frames + codebooks;
        var labels = new int[codebooks, length];
        for (int q = 0; q < codebooks; q++)
            for (int j = 0; j < length; j++)
                labels[q, j] = j < q ? c.BeginToken : j < q + frames ? codes[j - q, q] : c.EndToken;
        return labels;
    }

    /// <summary>
    /// The reference's loss on codes <c>[frames, codebooks]</c>: the delayed labels, the decoder reading them shifted
    /// right behind the start token, and per codebook the mean cross-entropy over the positions whose label is not a
    /// start token and whose input is not an end token; the total averages the codebooks (or weights them by
    /// <see cref="ParlerConfiguration.CodebookWeights"/>). Returns the total and each codebook's loss.
    /// </summary>
    public (Tensor<T> Loss, IReadOnlyList<Tensor<T>> PerCodebook) Loss(int[,] codes, IReadOnlyList<int> transcript, Tensor<T> memory,
        bool training, Random random)
    {
        var c = Configuration;
        var labels = DelayedLabels(codes);
        int codebooks = c.Codebooks, length = labels.GetLength(1);
        var input = new int[codebooks, length];
        for (int q = 0; q < codebooks; q++)
        {
            input[q, 0] = c.BeginToken;
            for (int j = 1; j < length; j++) input[q, j] = labels[q, j - 1];
        }
        var logits = Logits(input, transcript, memory, training, random);
        var weights = c.CodebookWeights;
        double weightSum = weights?.Sum() ?? codebooks;
        Tensor<T>? total = null;
        var perCodebook = new List<Tensor<T>>();
        for (int q = 0; q < codebooks; q++)
        {
            var targets = new int[length];
            for (int j = 0; j < length; j++)
                targets[j] = labels[q, j] == c.BeginToken || input[q, j] == c.EndToken ? -1 : labels[q, j];
            var loss = MeanCrossEntropy(logits[q], targets);
            perCodebook.Add(loss);
            var weighted = _engine.TensorMultiplyScalar(loss, NumOps.FromDouble((weights?[q] ?? 1.0) / weightSum));
            total = total is null ? weighted : _engine.TensorAdd(total, weighted);
        }
        return (total!, perCodebook);
    }

    // The mean of −log softmax(logits)[target] over the rows whose target is not −1.
    private Tensor<T> MeanCrossEntropy(Tensor<T> logits, int[] targets)
    {
        int count = targets.Count(t => t >= 0);
        if (count == 0) throw new ArgumentException("A codebook has no position to learn from.");
        var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
        var selection = new Tensor<T>(logits._shape);
        var weight = NumOps.FromDouble(-1.0 / count);
        for (int t = 0; t < targets.Length; t++)
            if (targets[t] >= 0) selection[t, targets[t]] = weight;
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, selection), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>
    /// Generates codes as the reference's <c>generate</c> does, recomputing the sequence at each step: every codebook
    /// starts with the start token; the delay pattern forces the start token on codebook <c>q</c>'s first <c>q</c> steps
    /// and the end token on its last; the end token is barred for the first <paramref name="minNewTokens"/> steps and,
    /// on codebook <c>q</c>, until codebook <c>q − 1</c> has emitted it (<c>ParlerTTSLogitsProcessor</c>); a codebook
    /// that has ended pads with the end token; generation stops when every codebook has ended or at
    /// <paramref name="maxLength"/> steps. Returns the codes <c>[frames, codebooks]</c> of the frames whose every
    /// codebook holds a code.
    /// </summary>
    public int[,] Generate(IReadOnlyList<int> transcript, Tensor<T> memory, int maxLength, int minNewTokens, bool sample,
        double temperature, int topK, int codebookSize, Random random)
    {
        var c = Configuration;
        int codebooks = c.Codebooks;
        if (maxLength < 2 * codebooks)
            throw new ArgumentException($"Generation needs at least {2 * codebooks} steps for the delay pattern of {codebooks} codebooks.");
        // The delay pattern for a sequence of length `length`: −1 where the model predicts, else the forced token.
        int Forced(int q, int column, int length) =>
            column <= q ? c.BeginToken : column >= length - codebooks + 1 + q ? c.EndToken : -1;

        var raw = new List<int[]> { Enumerable.Repeat(c.BeginToken, codebooks).ToArray() };                 // columns
        var unfinished = Enumerable.Repeat(true, codebooks).ToArray();
        int firstUnfinished = 0;
        while (raw.Count < maxLength)
        {
            int length = raw.Count;
            var input = new int[codebooks, length];
            for (int q = 0; q < codebooks; q++)
                for (int t = 0; t < length; t++)
                {
                    int forced = Forced(q, t, maxLength);
                    input[q, t] = forced == -1 ? raw[t][q] : forced;
                }
            var logits = Logits(input, transcript, memory, training: false, random);
            // ParlerTTSLogitsProcessor: once the first unfinished codebook has emitted the end token, the next may.
            if (firstUnfinished < codebooks - 1 && raw.Any(column => column[firstUnfinished] == c.EndToken))
                firstUnfinished++;
            var next = new int[codebooks];
            for (int q = 0; q < codebooks; q++)
            {
                var row = new double[c.AudioVocabulary];
                for (int v = 0; v < row.Length; v++) row[v] = NumOps.ToDouble(logits[q][length - 1, v]);
                if (length - 1 < minNewTokens || q > firstUnfinished) row[c.EndToken] = double.NegativeInfinity;
                int token = sample ? VallECore<T>.SampleTopK(row, temperature, topK, random) : ArgMax(row);
                next[q] = unfinished[q] ? token : c.EndToken;
            }
            raw.Add(next);
            for (int q = 0; q < codebooks; q++) unfinished[q] &= next[q] != c.EndToken;
            if (!unfinished.Any(u => u)) break;
        }

        // apply_delay_pattern_mask, then keep the positions the delay pattern of the final length leaves to the model
        // (all of them when that length is too short to hold a delay pattern, as build_delay_pattern_mask returns then).
        int final = raw.Count;
        bool patterned = final >= 2 * codebooks - 1;
        int frames = patterned ? final - codebooks : final;
        var output = new int[codebooks, frames];
        for (int q = 0; q < codebooks; q++)
            for (int f = 0; f < frames; f++)
            {
                int column = patterned ? q + 1 + f : f, forced = Forced(q, column, maxLength);
                output[q, f] = forced == -1 ? raw[column][q] : forced;
            }
        var kept = Enumerable.Range(0, frames).Where(f => Enumerable.Range(0, codebooks).All(q => output[q, f] < codebookSize)).ToList();
        var codes = new int[kept.Count, codebooks];
        for (int i = 0; i < kept.Count; i++)
            for (int q = 0; q < codebooks; q++) codes[i, q] = output[q, kept[i]];
        return codes;
    }

    private static int ArgMax(double[] row)
    {
        int best = 0;
        for (int v = 1; v < row.Length; v++) if (row[v] > row[best]) best = v;
        return best;
    }

    /// <summary>
    /// The reference's initialization (<c>_init_weights</c> with <c>initializer_factor</c> <paramref name="std"/>):
    /// every linear weight and embedding from N(0, std²), biases zero, layer norms at one and zero.
    /// </summary>
    public void InitializeLikeReference(Random random, double std)
    {
        var c = Configuration;
        void Fill(BiasFreeLinearLayer<T> layer) =>
            layer.SetParameters(TorchParameters.Gaussian<T>(random, layer.InputSize * layer.OutputSize, std));
        foreach (var embedding in Embeddings.Append(PromptEmbedding))
            embedding.Reinitialize(() => std * Normal(random));
        foreach (var layer in Layers)
        {
            foreach (var attention in new[] { layer.SelfAttention, layer.CrossAttention })
            {
                Fill(attention.Query);
                Fill(attention.Key);
                Fill(attention.Value);
                Fill(attention.Output);
            }
            Fill(layer.FeedForwardIn);
            Fill(layer.FeedForwardOut);
        }
        foreach (var head in Heads) Fill(head);
        if (Projection is not null)
            TorchParameters.Linear(_engine, Projection, c.EncoderDim, c.ModelDim,
                TorchParameters.Gaussian<double>(random, c.ModelDim * c.EncoderDim, std).ToArray(), new double[c.ModelDim]);
    }

    private static double Normal(Random random) =>
        Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());

    /// <summary>
    /// Loads the decoder, the transcript embedding and the projection from a <c>ParlerTTSForConditionalGeneration</c>
    /// state dict; <paramref name="read"/> returns a named tensor's row-major values after checking its shape. Separate
    /// or fused (<c>use_fused_lm_heads</c>) heads are both read.
    /// </summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, Func<string, bool> contains)
    {
        var c = Configuration;
        int d = c.ModelDim, ff = c.FeedForwardDim, vocabulary = c.AudioVocabulary;
        const string decoder = "decoder.model.decoder";
        for (int q = 0; q < c.Codebooks; q++)
            Embeddings[q].LoadTable(read($"{decoder}.embed_tokens.{q}.weight", new[] { vocabulary + 1, d }));
        void Norm(string name, LayerNormalizationLayer<T> norm) =>
            TorchParameters.LayerNorm(_engine, norm, read(name + ".weight", new[] { d }), read(name + ".bias", new[] { d }));
        void Attention(string name, ParlerAttention<T> attention)
        {
            int kv = attention.KeyValueWidth;
            TorchParameters.Linear(attention.Query, d, d, read(name + ".q_proj.weight", new[] { d, d }));
            TorchParameters.Linear(attention.Key, d, kv, read(name + ".k_proj.weight", new[] { kv, d }));
            TorchParameters.Linear(attention.Value, d, kv, read(name + ".v_proj.weight", new[] { kv, d }));
            TorchParameters.Linear(attention.Output, d, d, read(name + ".out_proj.weight", new[] { d, d }));
        }
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string p = $"{decoder}.layers.{i}";
            Attention(p + ".self_attn", layer.SelfAttention);
            Norm(p + ".self_attn_layer_norm", layer.SelfNorm);
            Attention(p + ".encoder_attn", layer.CrossAttention);
            Norm(p + ".encoder_attn_layer_norm", layer.CrossNorm);
            TorchParameters.Linear(layer.FeedForwardIn, d, ff, read(p + ".fc1.weight", new[] { ff, d }));
            TorchParameters.Linear(layer.FeedForwardOut, ff, d, read(p + ".fc2.weight", new[] { d, ff }));
            Norm(p + ".final_layer_norm", layer.FinalNorm);
        }
        Norm(decoder + ".layer_norm", FinalNorm);
        if (contains("decoder.lm_heads.weight"))
        {
            // One [codebooks · vocabulary, d] projection whose rows are codebook-major.
            var fused = read("decoder.lm_heads.weight", new[] { c.Codebooks * vocabulary, d });
            for (int q = 0; q < c.Codebooks; q++)
                TorchParameters.Linear(Heads[q], d, vocabulary, fused.Skip(q * vocabulary * d).Take(vocabulary * d).ToArray());
        }
        else
        {
            for (int q = 0; q < c.Codebooks; q++)
                TorchParameters.Linear(Heads[q], d, vocabulary, read($"decoder.lm_heads.{q}.weight", new[] { vocabulary, d }));
        }
        PromptEmbedding.LoadTable(read("embed_prompts.weight", new[] { c.PromptVocabulary, d }));
        if (Projection is not null)
            TorchParameters.Linear(_engine, Projection, c.EncoderDim, d, read("enc_to_dec_proj.weight", new[] { d, c.EncoderDim }),
                read("enc_to_dec_proj.bias", new[] { d }));
    }
}
