using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of a T5 encoder–decoder (Raffel et al. 2020; Hugging Face <c>T5Config</c> names).</summary>
internal sealed record T5Configuration(
    int VocabularySize, int ModelDim, int FeedForwardDim, int KeyValueDim, int Heads, int EncoderLayers,
    int DecoderLayers, int RelativeBuckets = 32, int RelativeMaxDistance = 128, double Dropout = 0.1,
    double LayerNormEpsilon = 1e-6, int DecoderStartTokenId = 0, int EndTokenId = 2, bool GatedGelu = false,
    bool TieWordEmbeddings = true, int PadTokenId = 0);

/// <summary>
/// The relative position buckets of T5 (Raffel et al. 2020, §2.1; Hugging Face <c>T5Attention._relative_position_bucket</c>):
/// exact buckets for small offsets and logarithmically wider ones up to <c>maxDistance</c>, split by direction when
/// bidirectional.
/// </summary>
internal static class T5RelativePosition
{
    /// <summary>The bucket of <c>key − query</c>. The logarithm is taken in single precision, as the reference does
    /// (<c>relative_position.float()</c>), so offsets near a bucket boundary land where the reference puts them.</summary>
    public static int Bucket(int relative, bool bidirectional, int buckets, int maxDistance)
    {
        int result = 0;
        if (bidirectional)
        {
            buckets /= 2;
            if (relative > 0) result += buckets;
            relative = Math.Abs(relative);
        }
        else
        {
            relative = -Math.Min(relative, 0);
        }
        int maxExact = buckets / 2;
        if (relative < maxExact) return result + relative;
        float scaled = (float)Math.Log((float)relative / maxExact) / (float)Math.Log((float)maxDistance / maxExact)
                       * (buckets - maxExact);
        int large = maxExact + (int)scaled;
        return result + Math.Min(large, buckets - 1);
    }
}

/// <summary>
/// One T5 attention (Hugging Face <c>T5Attention</c>): bias-free query, key, value and output projections to
/// <c>heads × d_kv</c>, scores <c>q kᵀ</c> without the 1/√d scale (T5 folds it into its initialization), plus a
/// position bias, a softmax with dropout, and the weighted values.
/// </summary>
internal sealed class T5Attention<T>
{
    private readonly IEngine _engine;
    private readonly int _heads;
    private readonly int _keyValueDim;
    private readonly double _dropout;

    public T5Attention(IEngine engine, List<LayerBase<T>> layers, int modelDim, int heads, int keyValueDim, double dropout)
    {
        _engine = engine;
        _heads = heads;
        _keyValueDim = keyValueDim;
        _dropout = dropout;
        int inner = heads * keyValueDim;
        Query = Own(layers, new BiasFreeLinearLayer<T>(modelDim, inner));
        Key = Own(layers, new BiasFreeLinearLayer<T>(modelDim, inner));
        Value = Own(layers, new BiasFreeLinearLayer<T>(modelDim, inner));
        Output = Own(layers, new BiasFreeLinearLayer<T>(inner, modelDim));
    }

    public BiasFreeLinearLayer<T> Query { get; }
    public BiasFreeLinearLayer<T> Key { get; }
    public BiasFreeLinearLayer<T> Value { get; }
    public BiasFreeLinearLayer<T> Output { get; }

    internal static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    /// <summary>Attends from <paramref name="queries"/> <c>[Lq, d]</c> over <paramref name="keyValues"/> <c>[Lk, d]</c>
    /// with <paramref name="bias"/> <c>[heads, Lq, Lk]</c> (position bias and any mask) added to the scores.</summary>
    public Tensor<T> Forward(Tensor<T> queries, Tensor<T> keyValues, Tensor<T>? bias, bool training, Random random)
    {
        int lq = queries.Shape[0], lk = keyValues.Shape[0];
        Tensor<T> Heads(Tensor<T> x, int length) =>
            _engine.TensorPermute(_engine.Reshape(x, new[] { length, _heads, _keyValueDim }), new[] { 1, 0, 2 });   // [H, L, dkv]
        var q = Heads(Query.Forward(queries), lq);
        var k = Heads(Key.Forward(keyValues), lk);
        var v = Heads(Value.Forward(keyValues), lk);
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                       // [H, Lq, Lk]
        if (bias is not null) scores = _engine.TensorAdd(scores, bias);
        var weights = _engine.Softmax(scores, axis: 2);
        if (training && _dropout > 0) weights = Dropout(weights, random);
        var context = _engine.BatchMatMul(weights, v);                                                         // [H, Lq, dkv]
        var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { lq, _heads * _keyValueDim });
        return Output.Forward(merged);
    }

    private Tensor<T> Dropout(Tensor<T> x, Random random) => T5Seq2Seq<T>.Dropout(_engine, x, _dropout, random);
}

/// <summary>A T5 block (Hugging Face <c>T5Block</c>): pre-norm residual self-attention, cross-attention in the decoder,
/// and the bias-free feed-forward, each sublayer's output dropped out before its residual add. The feed-forward is
/// T5 v1.0's <c>wo · dropout(relu(wi · x))</c>, or with <see cref="T5Configuration.GatedGelu"/> T5 v1.1's
/// <c>wo · dropout(gelu(wi_0 · x) ⊙ wi_1 · x)</c> (<c>T5DenseGatedActDense</c>, GELU's tanh form).</summary>
internal sealed class T5Block<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;

    public T5Block(IEngine engine, List<LayerBase<T>> layers, T5Configuration c, bool decoder)
    {
        _engine = engine;
        _dropout = c.Dropout;
        SelfNorm = T5Attention<T>.Own(layers, new RMSNormalizationLayer<T>(c.ModelDim, c.LayerNormEpsilon));
        SelfAttention = new T5Attention<T>(engine, layers, c.ModelDim, c.Heads, c.KeyValueDim, c.Dropout);
        if (decoder)
        {
            CrossNorm = T5Attention<T>.Own(layers, new RMSNormalizationLayer<T>(c.ModelDim, c.LayerNormEpsilon));
            CrossAttention = new T5Attention<T>(engine, layers, c.ModelDim, c.Heads, c.KeyValueDim, c.Dropout);
        }
        FeedForwardNorm = T5Attention<T>.Own(layers, new RMSNormalizationLayer<T>(c.ModelDim, c.LayerNormEpsilon));
        Wi = T5Attention<T>.Own(layers, new BiasFreeLinearLayer<T>(c.ModelDim, c.FeedForwardDim));
        if (c.GatedGelu)
            WiLinear = T5Attention<T>.Own(layers, new BiasFreeLinearLayer<T>(c.ModelDim, c.FeedForwardDim));
        Wo = T5Attention<T>.Own(layers, new BiasFreeLinearLayer<T>(c.FeedForwardDim, c.ModelDim));
    }

    public RMSNormalizationLayer<T> SelfNorm { get; }
    public T5Attention<T> SelfAttention { get; }
    public RMSNormalizationLayer<T>? CrossNorm { get; }
    public T5Attention<T>? CrossAttention { get; }
    public RMSNormalizationLayer<T> FeedForwardNorm { get; }
    /// <summary><c>wi</c>, or under the gated feed-forward <c>wi_0</c> (the GELU branch).</summary>
    public BiasFreeLinearLayer<T> Wi { get; }
    /// <summary><c>wi_1</c>, the gated feed-forward's linear branch; null in T5 v1.0.</summary>
    public BiasFreeLinearLayer<T>? WiLinear { get; }
    public BiasFreeLinearLayer<T> Wo { get; }

    public Tensor<T> Forward(Tensor<T> h, Tensor<T> selfBias, Tensor<T>? memory, Tensor<T>? crossBias, bool training, Random random)
    {
        var normed = SelfNorm.Forward(h);
        h = _engine.TensorAdd(h, Drop(SelfAttention.Forward(normed, normed, selfBias, training, random), training, random));
        if (CrossAttention is not null && CrossNorm is not null && memory is not null)
        {
            var crossNormed = CrossNorm.Forward(h);
            h = _engine.TensorAdd(h, Drop(CrossAttention.Forward(crossNormed, memory, crossBias, training, random), training, random));
        }
        var feedForwardInput = FeedForwardNorm.Forward(h);
        var hidden = WiLinear is null
            ? _engine.ReLU(Wi.Forward(feedForwardInput))
            : _engine.TensorMultiply(_engine.GELU(Wi.Forward(feedForwardInput)), WiLinear.Forward(feedForwardInput));
        hidden = Drop(hidden, training, random);
        return _engine.TensorAdd(h, Drop(Wo.Forward(hidden), training, random));
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && _dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, _dropout, random) : x;
}

/// <summary>
/// T5 v1.0 as a sequence-to-sequence model (Raffel et al. 2020; Hugging Face <c>T5ForConditionalGeneration</c> with
/// ReLU feed-forward and tied embeddings): a shared token embedding; an encoder and a decoder stack whose first block
/// holds the relative position bias table the rest of the stack shares (bidirectional buckets in the encoder,
/// unidirectional with a causal mask in the decoder); RMS layer norms (ε = 1e-6) and a final norm on each stack;
/// logits from the shared embedding applied to the decoder output scaled by <c>d_model^-1/2</c>. With
/// <see cref="T5Configuration.GatedGelu"/> and without <see cref="T5Configuration.TieWordEmbeddings"/> it is T5 v1.1
/// (the gated-GELU feed-forward and an unscaled, untied <c>lm_head</c>), the form ByT5 (Xue et al. 2022) uses.
/// </summary>
internal sealed class T5Seq2Seq<T>
{
    private readonly IEngine _engine;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public T5Seq2Seq(IEngine engine, List<LayerBase<T>> layers, T5Configuration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Shared = T5Attention<T>.Own(layers, new TiedEmbeddingLayer<T>(c.VocabularySize, c.ModelDim));
        for (int i = 0; i < c.EncoderLayers; i++) Encoder.Add(new T5Block<T>(engine, layers, c, decoder: false));
        EncoderBias = T5Attention<T>.Own(layers, new TiedEmbeddingLayer<T>(c.RelativeBuckets, c.Heads));
        EncoderFinalNorm = T5Attention<T>.Own(layers, new RMSNormalizationLayer<T>(c.ModelDim, c.LayerNormEpsilon));
        for (int i = 0; i < c.DecoderLayers; i++) Decoder.Add(new T5Block<T>(engine, layers, c, decoder: true));
        // An encoder-only T5 (T5EncoderModel, as text conditioners use it) owns no decoder parameters.
        if (c.DecoderLayers > 0)
        {
            DecoderBias = T5Attention<T>.Own(layers, new TiedEmbeddingLayer<T>(c.RelativeBuckets, c.Heads));
            DecoderFinalNorm = T5Attention<T>.Own(layers, new RMSNormalizationLayer<T>(c.ModelDim, c.LayerNormEpsilon));
            if (!c.TieWordEmbeddings)
                LanguageModelHead = T5Attention<T>.Own(layers, new BiasFreeLinearLayer<T>(c.ModelDim, c.VocabularySize));
        }
    }

    /// <summary>The untied output projection <c>lm_head</c> (T5 v1.1); null when the logits reuse the shared
    /// embedding.</summary>
    public BiasFreeLinearLayer<T>? LanguageModelHead { get; }

    public T5Configuration Configuration { get; }
    public TiedEmbeddingLayer<T> Shared { get; }
    public List<T5Block<T>> Encoder { get; } = new();
    public List<T5Block<T>> Decoder { get; } = new();
    /// <summary>The encoder's relative attention bias table <c>[buckets, heads]</c> (Hugging Face
    /// <c>encoder.block.0.layer.0.SelfAttention.relative_attention_bias</c>).</summary>
    public TiedEmbeddingLayer<T> EncoderBias { get; }
    public RMSNormalizationLayer<T> EncoderFinalNorm { get; }
    /// <summary>The decoder's relative attention bias table <c>[buckets, heads]</c>; null for an encoder-only T5.</summary>
    public TiedEmbeddingLayer<T>? DecoderBias { get; }
    public RMSNormalizationLayer<T>? DecoderFinalNorm { get; }

    internal static Tensor<T> Dropout(IEngine engine, Tensor<T> x, double rate, Random random)
    {
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return engine.TensorMultiply(x, mask);
    }

    private Tensor<T> PositionBias(TiedEmbeddingLayer<T> table, int queries, int keys, bool bidirectional, bool causal) =>
        RelativeBias(_engine, Configuration, table, queries, keys, bidirectional, causal);

    /// <summary>The attention mask as a score bias <c>[heads, queries, keys]</c>: -1e9 on the keys
    /// <paramref name="keyMask"/> marks false (Hugging Face adds the dtype's minimum; either leaves them no weight).</summary>
    private Tensor<T> MaskBias(int queries, bool[] keyMask)
    {
        int heads = Configuration.Heads, keys = keyMask.Length;
        var bias = new Tensor<T>(new[] { heads, queries, keys });
        var blocked = NumOps.FromDouble(-1e9);
        for (int j = 0; j < keys; j++)
        {
            if (keyMask[j]) continue;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < queries; i++) bias[h, i, j] = blocked;
        }
        return bias;
    }

    /// <summary>The relative position bias <c>[heads, queries, keys]</c> from a bucket table, with −1e9 above the
    /// diagonal when <paramref name="causal"/>.</summary>
    internal static Tensor<T> RelativeBias(IEngine engine, T5Configuration c, TiedEmbeddingLayer<T> table, int queries, int keys,
        bool bidirectional, bool causal)
    {
        var _engine = engine;
        var buckets = new Tensor<T>(new[] { queries * keys });
        for (int i = 0; i < queries; i++)
            for (int j = 0; j < keys; j++)
                buckets[i * keys + j] = NumOps.FromDouble(T5RelativePosition.Bucket(j - i, bidirectional, c.RelativeBuckets, c.RelativeMaxDistance));
        var values = table.Forward(buckets);                                                                  // [Lq·Lk, H]
        var bias = _engine.TensorPermute(_engine.Reshape(values, new[] { queries, keys, c.Heads }), new[] { 2, 0, 1 });
        if (!causal) return bias;
        var mask = new Tensor<T>(new[] { c.Heads, queries, keys });
        var blocked = NumOps.FromDouble(-1e9);
        for (int h = 0; h < c.Heads; h++)
            for (int i = 0; i < queries; i++)
                for (int j = i + 1; j < keys; j++) mask[h, i, j] = blocked;
        return _engine.TensorAdd(bias, mask);
    }

    /// <summary>The encoder states <c>[L, d]</c> of token ids <c>[L]</c>; positions <paramref name="mask"/> marks false
    /// (padding, missing values) receive no attention.</summary>
    public Tensor<T> Encode(Tensor<T> ids, bool training, Random random, bool[]? mask = null)
    {
        int length = ids.Length;
        var h = Drop(Shared.Forward(ids), training, random);
        var bias = PositionBias(EncoderBias, length, length, bidirectional: true, causal: false);
        if (mask is not null) bias = _engine.TensorAdd(bias, MaskBias(length, mask));
        foreach (var block in Encoder) h = block.Forward(h, bias, null, null, training, random);
        return Drop(EncoderFinalNorm.Forward(h), training, random);
    }

    /// <summary>The decoder's token logits <c>[L, vocabulary]</c> for decoder input ids <c>[L]</c> given the encoder
    /// states; the cross-attention skips the encoder positions <paramref name="memoryMask"/> marks false.</summary>
    public Tensor<T> Decode(Tensor<T> decoderIds, Tensor<T> memory, bool training, Random random, bool[]? memoryMask = null)
    {
        int length = decoderIds.Length;
        var h = Drop(Shared.Forward(decoderIds), training, random);
        var decoderBias = DecoderBias ?? throw new InvalidOperationException("This T5 has no decoder.");
        var decoderNorm = DecoderFinalNorm ?? throw new InvalidOperationException("This T5 has no decoder.");
        var bias = PositionBias(decoderBias, length, length, bidirectional: false, causal: true);
        var crossBias = memoryMask is null ? null : MaskBias(length, memoryMask);
        foreach (var block in Decoder) h = block.Forward(h, bias, memory, crossBias, training, random);
        h = Drop(decoderNorm.Forward(h), training, random);
        // Tied embeddings rescale the decoder output by d_model^-1/2 first; an untied lm_head does not.
        if (LanguageModelHead is not null) return LanguageModelHead.Forward(h);
        var scaled = _engine.TensorMultiplyScalar(h, NumOps.FromDouble(1.0 / Math.Sqrt(Configuration.ModelDim)));
        return Shared.Logits(scaled);
    }

    /// <summary>
    /// The teacher-forced cross-entropy of <paramref name="labels"/> given <paramref name="inputIds"/>, as
    /// <c>T5ForConditionalGeneration(labels=...)</c> computes it: the decoder reads the labels shifted right behind the
    /// start token, and the loss averages the token cross-entropies. A negative label (Hugging Face's -100) is
    /// ignored: it adds no term, and the decoder reads the pad token in its place. With
    /// <paramref name="freezeEncoder"/> the encoder runs in inference mode outside the tape, so only the decoder (and
    /// the embedding rows it reads) learn. <paramref name="inputMask"/> masks encoder positions as in
    /// <see cref="Encode"/>.
    /// </summary>
    public Tensor<T> Loss(Tensor<T> inputIds, int[] labels, bool training, Random random, bool freezeEncoder = false,
        bool[]? inputMask = null)
    {
        Tensor<T> memory;
        if (freezeEncoder)
        {
            using (new NoGradScope<T>()) memory = Encode(inputIds, training: false, random, inputMask);
        }
        else
        {
            memory = Encode(inputIds, training, random, inputMask);
        }
        var decoderIds = new Tensor<T>(new[] { labels.Length });
        decoderIds[0] = NumOps.FromDouble(Configuration.DecoderStartTokenId);
        for (int i = 1; i < labels.Length; i++)
            decoderIds[i] = NumOps.FromDouble(labels[i - 1] < 0 ? Configuration.PadTokenId : labels[i - 1]);
        var logits = Decode(decoderIds, memory, training, random, inputMask);                                   // [L, V]
        var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
        var target = new Tensor<T>(logits._shape);
        int counted = 0;
        foreach (int label in labels) if (label >= 0) counted++;
        if (counted == 0) throw new ArgumentException("Every label is ignored.", nameof(labels));
        for (int i = 0; i < labels.Length; i++)
            if (labels[i] >= 0) target[i, labels[i]] = NumOps.FromDouble(-1.0 / counted);
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, target), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>
    /// Samples a continuation of <paramref name="decoderPrefix"/> (Hugging Face <c>generate</c> with
    /// <c>do_sample=True</c>, one beam): each step draws from the softmax of logits / temperature restricted to the
    /// <paramref name="topK"/> largest, stopping at the end token or after <paramref name="maxNewTokens"/>.
    /// Returns the prefix followed by the sampled tokens (without the end token).
    /// </summary>
    /// <param name="allowed">When given, <c>allowed(token, step)</c> decides which tokens step <c>step</c> (0 for the first
    /// new token) may sample; the others are removed before the top-k cut, as a Hugging Face logits processor does.</param>
    public List<int> Generate(Tensor<T> inputIds, IReadOnlyList<int> decoderPrefix, double temperature, int topK,
        int maxNewTokens, Random random, Func<int, int, bool>? allowed = null) =>
        Sample(Encode(inputIds, training: false, random), null, decoderPrefix, temperature, topK, 1.0, maxNewTokens, random,
            allowed);

    /// <summary>
    /// <see cref="Generate"/> from encoder states already computed (so several samples share one encoding), with the
    /// cross-attention mask of <see cref="Decode"/> and nucleus filtering: after the top-k cut, the smallest set of
    /// tokens whose probability reaches <paramref name="topP"/> is kept (Hugging Face <c>TopPLogitsWarper</c>; 1 keeps
    /// all).
    /// </summary>
    public List<int> Sample(Tensor<T> memory, bool[]? memoryMask, IReadOnlyList<int> decoderPrefix, double temperature,
        int topK, double topP, int maxNewTokens, Random random, Func<int, int, bool>? allowed = null)
    {
        var sequence = new List<int> { Configuration.DecoderStartTokenId };
        sequence.AddRange(decoderPrefix);
        for (int step = 0; step < maxNewTokens; step++)
        {
            var ids = new Tensor<T>(new[] { sequence.Count });
            for (int i = 0; i < sequence.Count; i++) ids[i] = NumOps.FromDouble(sequence[i]);
            var logits = Decode(ids, memory, training: false, random, memoryMask);
            int vocabulary = logits.Shape[1], last = sequence.Count - 1;
            var row = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++)
                row[v] = allowed is null || allowed(v, step) ? NumOps.ToDouble(logits[last, v]) / temperature : double.NegativeInfinity;
            int next = SampleTopK(row, topK, topP, random);
            if (next == Configuration.EndTokenId) break;
            sequence.Add(next);
        }
        sequence.RemoveAt(0);
        return sequence;
    }

    // Hugging Face's TopKLogitsWarper keeps every score at least the k-th largest (ties included); TopPLogitsWarper
    // then removes the tokens whose cumulative probability, accumulated from the least likely, stays at or below
    // 1 - topP (always keeping the most likely); the draw is from the softmax of what is left.
    private static int SampleTopK(double[] scores, int topK, double topP, Random random)
    {
        // Masked tokens score -infinity: never among the k drawn from.
        int finite = 0;
        foreach (double score in scores) if (!double.IsNegativeInfinity(score)) finite++;
        if (finite == 0) throw new InvalidOperationException("No token is allowed at this step.");
        var order = new int[scores.Length];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => scores[b].CompareTo(scores[a]));
        int k = topK <= 0 ? finite : Math.Min(topK, finite);
        double threshold = scores[order[k - 1]];
        while (k < finite && scores[order[k]] >= threshold) k++;
        double max = scores[order[0]], total = 0;
        var weights = new double[k];
        for (int i = 0; i < k; i++) total += weights[i] = Math.Exp(scores[order[i]] - max);
        if (topP < 1.0)
        {
            double removed = 0;
            int kept = k;
            for (int i = k - 1; i > 0; i--)
            {
                removed += weights[i] / total;
                if (removed > 1.0 - topP) break;
                kept = i;
            }
            k = kept;
            total = 0;
            for (int i = 0; i < k; i++) total += weights[i];
        }
        double draw = random.NextDouble() * total;
        for (int i = 0; i < k; i++)
        {
            draw -= weights[i];
            if (draw <= 0) return order[i];
        }
        return order[k - 1];
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && Configuration.Dropout > 0 ? Dropout(_engine, x, Configuration.Dropout, random) : x;

    /// <summary>
    /// Initializes the parameters as Hugging Face's <c>T5PreTrainedModel._init_weights</c> does (initializer factor 1):
    /// the shared embedding from N(0, 1); query N(0, (d_model · d_kv)^-1/2), key and value N(0, d_model^-1/2), output
    /// N(0, (heads · d_kv)^-1/2); the relative bias tables N(0, d_model^-1/2); wi N(0, d_model^-1/2), wo
    /// N(0, d_ff^-1/2); layer norms at one. A <paramref name="factor"/> (Hugging Face's <c>initializer_factor</c>)
    /// multiplies every standard deviation and the layer norms' starting value.
    /// </summary>
    public void InitializeLikeHuggingFace(Random random, double factor = 1.0)
    {
        var c = Configuration;
        int inner = c.Heads * c.KeyValueDim;
        double Normal(double std)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        void Fill(BiasFreeLinearLayer<T> layer, int inputs, int outputs, double std)
        {
            var values = new Vector<T>(inputs * outputs);
            for (int i = 0; i < values.Length; i++) values[i] = NumOps.FromDouble(Normal(factor * std));
            layer.SetParameters(values);
        }
        void Attention(T5Attention<T> attention)
        {
            Fill(attention.Query, c.ModelDim, inner, Math.Pow(c.ModelDim * c.KeyValueDim, -0.5));
            Fill(attention.Key, c.ModelDim, inner, Math.Pow(c.ModelDim, -0.5));
            Fill(attention.Value, c.ModelDim, inner, Math.Pow(c.ModelDim, -0.5));
            Fill(attention.Output, inner, c.ModelDim, Math.Pow(inner, -0.5));
        }
        Shared.Reinitialize(() => Normal(factor));
        EncoderBias.Reinitialize(() => Normal(factor * Math.Pow(c.ModelDim, -0.5)));
        DecoderBias?.Reinitialize(() => Normal(factor * Math.Pow(c.ModelDim, -0.5)));
        // T5LayerNorm weights start at the factor.
        var norm = new Vector<T>(Enumerable.Repeat(NumOps.FromDouble(factor), c.ModelDim).ToArray());
        foreach (var block in Encoder.Concat(Decoder))
        {
            block.SelfNorm.SetParameters(norm);
            block.CrossNorm?.SetParameters(norm);
            block.FeedForwardNorm.SetParameters(norm);
        }
        EncoderFinalNorm.SetParameters(norm);
        DecoderFinalNorm?.SetParameters(norm);
        foreach (var block in Encoder.Concat(Decoder))
        {
            Attention(block.SelfAttention);
            if (block.CrossAttention is not null) Attention(block.CrossAttention);
            Fill(block.Wi, c.ModelDim, c.FeedForwardDim, Math.Pow(c.ModelDim, -0.5));
            if (block.WiLinear is not null) Fill(block.WiLinear, c.ModelDim, c.FeedForwardDim, Math.Pow(c.ModelDim, -0.5));
            Fill(block.Wo, c.FeedForwardDim, c.ModelDim, Math.Pow(c.FeedForwardDim, -0.5));
        }
        // An untied lm_head is drawn from N(0, 1).
        if (LanguageModelHead is not null) Fill(LanguageModelHead, c.ModelDim, c.VocabularySize, 1.0);
    }

    /// <summary>
    /// Loads a Hugging Face <c>T5ForConditionalGeneration</c> state dict; <paramref name="read"/> returns the named
    /// tensor's values, row-major, after checking its shape. Linear weights are <c>[out, in]</c> in the checkpoint.
    /// With <paramref name="encoderOnly"/> only the encoder stack is read (not <c>shared</c>, the decoder or
    /// <c>lm_head</c>).
    /// </summary>
    public void LoadHuggingFaceWeights(Func<string, int[], double[]> read, bool encoderOnly = false)
    {
        var c = Configuration;
        int inner = c.Heads * c.KeyValueDim;
        void Linear(string name, BiasFreeLinearLayer<T> layer, int output, int input)
        {
            var values = read(name + ".weight", new[] { output, input });
            var transposed = new Vector<T>(output * input);
            for (int o = 0; o < output; o++)
                for (int i = 0; i < input; i++) transposed[i * output + o] = NumOps.FromDouble(values[o * input + i]);
            layer.SetParameters(transposed);
        }
        void Norm(string name, RMSNormalizationLayer<T> layer) =>
            layer.SetParameters(new Vector<T>(Array.ConvertAll(read(name + ".weight", new[] { c.ModelDim }), NumOps.FromDouble)));
        void Attention(string name, T5Attention<T> attention)
        {
            Linear(name + ".q", attention.Query, inner, c.ModelDim);
            Linear(name + ".k", attention.Key, inner, c.ModelDim);
            Linear(name + ".v", attention.Value, inner, c.ModelDim);
            Linear(name + ".o", attention.Output, c.ModelDim, inner);
        }
        void FeedForward(string name, T5Block<T> block)
        {
            if (block.WiLinear is null)
            {
                Linear(name + ".DenseReluDense.wi", block.Wi, c.FeedForwardDim, c.ModelDim);
            }
            else
            {
                Linear(name + ".DenseReluDense.wi_0", block.Wi, c.FeedForwardDim, c.ModelDim);
                Linear(name + ".DenseReluDense.wi_1", block.WiLinear, c.FeedForwardDim, c.ModelDim);
            }
            Linear(name + ".DenseReluDense.wo", block.Wo, c.ModelDim, c.FeedForwardDim);
            Norm(name + ".layer_norm", block.FeedForwardNorm);
        }

        if (!encoderOnly) Shared.LoadTable(read("shared.weight", new[] { c.VocabularySize, c.ModelDim }));
        for (int i = 0; i < Encoder.Count; i++)
        {
            string block = $"encoder.block.{i}.layer";
            Attention($"{block}.0.SelfAttention", Encoder[i].SelfAttention);
            Norm($"{block}.0.layer_norm", Encoder[i].SelfNorm);
            FeedForward($"{block}.1", Encoder[i]);
        }
        EncoderBias.LoadTable(read("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", new[] { c.RelativeBuckets, c.Heads }));
        Norm("encoder.final_layer_norm", EncoderFinalNorm);
        if (encoderOnly) return;
        for (int i = 0; i < Decoder.Count; i++)
        {
            string block = $"decoder.block.{i}.layer";
            Attention($"{block}.0.SelfAttention", Decoder[i].SelfAttention);
            Norm($"{block}.0.layer_norm", Decoder[i].SelfNorm);
            var cross = Decoder[i].CrossAttention ?? throw new InvalidOperationException("A decoder block has cross-attention.");
            var crossNorm = Decoder[i].CrossNorm ?? throw new InvalidOperationException("A decoder block has cross-attention.");
            Attention($"{block}.1.EncDecAttention", cross);
            Norm($"{block}.1.layer_norm", crossNorm);
            FeedForward($"{block}.2", Decoder[i]);
        }
        if (DecoderBias is not null && DecoderFinalNorm is not null)
        {
            DecoderBias.LoadTable(read("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", new[] { c.RelativeBuckets, c.Heads }));
            Norm("decoder.final_layer_norm", DecoderFinalNorm);
        }
        if (LanguageModelHead is not null) Linear("lm_head", LanguageModelHead, c.VocabularySize, c.ModelDim);
    }

    /// <summary>
    /// Loads only the encoder of a Hugging Face T5 checkpoint whose vocabulary is the first
    /// <paramref name="pretrainedVocabulary"/> rows of this model's: those rows of <c>shared</c>, the encoder blocks,
    /// its relative bias table and final norm. The decoder and the remaining embedding rows keep their values.
    /// </summary>
    public void LoadHuggingFaceEncoderWeights(Func<string, int[], double[]> read, int pretrainedVocabulary)
    {
        var c = Configuration;
        if (pretrainedVocabulary <= 0 || pretrainedVocabulary > c.VocabularySize)
            throw new ArgumentOutOfRangeException(nameof(pretrainedVocabulary));
        Shared.LoadLeadingRows(read("shared.weight", new[] { pretrainedVocabulary, c.ModelDim }));
        LoadHuggingFaceWeights(read, encoderOnly: true);
    }
}

/// <summary>
/// A decoder-only T5 language model: a token embedding, a causal stack of T5 blocks without cross-attention sharing one
/// unidirectional relative position bias table (held by the first block in Hugging Face's layout), a final RMS norm,
/// and an output head. AudioLM's three stages (Borsos et al. 2022, §IV-B: decoder-only Transformers with T5-style
/// relative positional embeddings) are this model.
/// </summary>
internal sealed class T5DecoderOnly<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public T5DecoderOnly(IEngine engine, List<LayerBase<T>> layers, T5Configuration configuration)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        Configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        var c = configuration;
        Embedding = T5Attention<T>.Own(layers, new TiedEmbeddingLayer<T>(c.VocabularySize, c.ModelDim));
        for (int i = 0; i < c.DecoderLayers; i++) Blocks.Add(new T5Block<T>(engine, layers, c, decoder: false));
        Bias = T5Attention<T>.Own(layers, new TiedEmbeddingLayer<T>(c.RelativeBuckets, c.Heads));
        FinalNorm = T5Attention<T>.Own(layers, new RMSNormalizationLayer<T>(c.ModelDim, c.LayerNormEpsilon));
        if (!c.TieWordEmbeddings)
            Head = T5Attention<T>.Own(layers, new BiasFreeLinearLayer<T>(c.ModelDim, c.VocabularySize));
    }

    public T5Configuration Configuration { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public List<T5Block<T>> Blocks { get; } = new();
    public TiedEmbeddingLayer<T> Bias { get; }
    public RMSNormalizationLayer<T> FinalNorm { get; }
    public BiasFreeLinearLayer<T>? Head { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && Configuration.Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, Configuration.Dropout, random) : x;

    /// <summary>The next-token logits <c>[L, vocabulary]</c> of ids <c>[L]</c>: row i predicts token i + 1.</summary>
    public Tensor<T> Logits(IReadOnlyList<int> ids, bool training, Random random)
    {
        int length = ids.Count;
        var tensor = new Tensor<T>(new[] { length });
        for (int i = 0; i < length; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        var h = Drop(Embedding.Forward(tensor), training, random);
        var bias = T5Seq2Seq<T>.RelativeBias(_engine, Configuration, Bias, length, length, bidirectional: false, causal: true);
        foreach (var block in Blocks) h = block.Forward(h, bias, null, null, training, random);
        h = Drop(FinalNorm.Forward(h), training, random);
        if (Head is not null) return Head.Forward(h);
        return Embedding.Logits(_engine.TensorMultiplyScalar(h, NumOps.FromDouble(1.0 / Math.Sqrt(Configuration.ModelDim))));
    }

    /// <summary>The mean cross-entropy of the tokens from position <paramref name="firstTarget"/> on, each predicted
    /// from the tokens before it.</summary>
    public Tensor<T> Loss(IReadOnlyList<int> ids, int firstTarget, bool training, Random random)
    {
        if (firstTarget < 1 || firstTarget >= ids.Count)
            throw new ArgumentOutOfRangeException(nameof(firstTarget), "At least one token must be predicted from a non-empty prefix.");
        var input = ids.Take(ids.Count - 1).ToArray();
        var logits = Logits(input, training, random);                                                          // [L−1, V]
        var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
        var target = new Tensor<T>(logits._shape);
        int count = ids.Count - firstTarget;
        for (int i = firstTarget; i < ids.Count; i++) target[i - 1, ids[i]] = NumOps.FromDouble(-1.0 / count);
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, target), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>
    /// Samples a continuation of <paramref name="prefix"/> at <paramref name="temperature"/> (pure temperature sampling,
    /// greedy at 0). <paramref name="allowed"/>(position, token) restricts each step's choices; generation stops after
    /// <paramref name="maxNewTokens"/> or at <paramref name="stopToken"/>, which is not returned.
    /// </summary>
    public List<int> Generate(IReadOnlyList<int> prefix, double temperature, int maxNewTokens, Func<int, int, bool> allowed,
        int stopToken, Random random)
    {
        var sequence = new List<int>(prefix);
        var generated = new List<int>();
        for (int step = 0; step < maxNewTokens; step++)
        {
            var logits = Logits(sequence, training: false, random);
            int vocabulary = logits.Shape[1], last = sequence.Count - 1;
            var scores = new double[vocabulary];
            double max = double.NegativeInfinity;
            for (int v = 0; v < vocabulary; v++)
            {
                scores[v] = allowed(step, v) ? NumOps.ToDouble(logits[last, v]) : double.NegativeInfinity;
                max = Math.Max(max, scores[v]);
            }
            if (double.IsNegativeInfinity(max)) throw new InvalidOperationException("No token is allowed at this step.");
            int next;
            if (temperature <= 0)
            {
                next = Array.IndexOf(scores, max);
            }
            else
            {
                double total = 0;
                for (int v = 0; v < vocabulary; v++) total += scores[v] = double.IsNegativeInfinity(scores[v]) ? 0 : Math.Exp((scores[v] - max) / temperature);
                double draw = random.NextDouble() * total;
                next = vocabulary - 1;
                for (int v = 0; v < vocabulary; v++)
                {
                    if (scores[v] <= 0) continue;
                    draw -= scores[v];
                    if (draw <= 0) { next = v; break; }
                }
                while (scores[next] <= 0) next--;
            }
            if (next == stopToken) break;
            sequence.Add(next);
            generated.Add(next);
        }
        return generated;
    }

    /// <summary>Initializes as Hugging Face's T5 does (initializer factor 1); an untied head from N(0, 1).</summary>
    public void InitializeLikeHuggingFace(Random random)
    {
        var c = Configuration;
        int inner = c.Heads * c.KeyValueDim;
        double Normal(double std)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        void Fill(BiasFreeLinearLayer<T> layer, int inputs, int outputs, double std)
        {
            var values = new Vector<T>(inputs * outputs);
            for (int i = 0; i < values.Length; i++) values[i] = NumOps.FromDouble(Normal(std));
            layer.SetParameters(values);
        }
        Embedding.Reinitialize(() => Normal(1.0));
        Bias.Reinitialize(() => Normal(Math.Pow(c.ModelDim, -0.5)));
        foreach (var block in Blocks)
        {
            var a = block.SelfAttention;
            Fill(a.Query, c.ModelDim, inner, Math.Pow(c.ModelDim * c.KeyValueDim, -0.5));
            Fill(a.Key, c.ModelDim, inner, Math.Pow(c.ModelDim, -0.5));
            Fill(a.Value, c.ModelDim, inner, Math.Pow(c.ModelDim, -0.5));
            Fill(a.Output, inner, c.ModelDim, Math.Pow(inner, -0.5));
            Fill(block.Wi, c.ModelDim, c.FeedForwardDim, Math.Pow(c.ModelDim, -0.5));
            if (block.WiLinear is not null) Fill(block.WiLinear, c.ModelDim, c.FeedForwardDim, Math.Pow(c.ModelDim, -0.5));
            Fill(block.Wo, c.FeedForwardDim, c.ModelDim, Math.Pow(c.FeedForwardDim, -0.5));
        }
        if (Head is not null) Fill(Head, c.ModelDim, c.VocabularySize, 1.0);
    }
}