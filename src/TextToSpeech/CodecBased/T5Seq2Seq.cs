using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of a T5 encoder–decoder (Raffel et al. 2020; Hugging Face <c>T5Config</c> names).</summary>
internal sealed record T5Configuration(
    int VocabularySize, int ModelDim, int FeedForwardDim, int KeyValueDim, int Heads, int EncoderLayers,
    int DecoderLayers, int RelativeBuckets = 32, int RelativeMaxDistance = 128, double Dropout = 0.1,
    double LayerNormEpsilon = 1e-6, int DecoderStartTokenId = 0, int EndTokenId = 2);

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
/// and the bias-free ReLU feed-forward <c>wo · dropout(relu(wi · x))</c>, each sublayer's output dropped out before its
/// residual add.</summary>
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
        Wo = T5Attention<T>.Own(layers, new BiasFreeLinearLayer<T>(c.FeedForwardDim, c.ModelDim));
    }

    public RMSNormalizationLayer<T> SelfNorm { get; }
    public T5Attention<T> SelfAttention { get; }
    public RMSNormalizationLayer<T>? CrossNorm { get; }
    public T5Attention<T>? CrossAttention { get; }
    public RMSNormalizationLayer<T> FeedForwardNorm { get; }
    public BiasFreeLinearLayer<T> Wi { get; }
    public BiasFreeLinearLayer<T> Wo { get; }

    public Tensor<T> Forward(Tensor<T> h, Tensor<T> selfBias, Tensor<T>? memory, Tensor<T>? crossBias, bool training, Random random)
    {
        var normed = SelfNorm.Forward(h);
        h = _engine.TensorAdd(h, Drop(SelfAttention.Forward(normed, normed, selfBias, training, random), training, random));
        if (CrossAttention is not null && memory is not null)
        {
            var crossNormed = CrossNorm!.Forward(h);
            h = _engine.TensorAdd(h, Drop(CrossAttention.Forward(crossNormed, memory, crossBias, training, random), training, random));
        }
        var hidden = _engine.ReLU(Wi.Forward(FeedForwardNorm.Forward(h)));
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
/// logits from the shared embedding applied to the decoder output scaled by <c>d_model^-1/2</c>.
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
        DecoderBias = T5Attention<T>.Own(layers, new TiedEmbeddingLayer<T>(c.RelativeBuckets, c.Heads));
        DecoderFinalNorm = T5Attention<T>.Own(layers, new RMSNormalizationLayer<T>(c.ModelDim, c.LayerNormEpsilon));
    }

    public T5Configuration Configuration { get; }
    public TiedEmbeddingLayer<T> Shared { get; }
    public List<T5Block<T>> Encoder { get; } = new();
    public List<T5Block<T>> Decoder { get; } = new();
    /// <summary>The encoder's relative attention bias table <c>[buckets, heads]</c> (Hugging Face
    /// <c>encoder.block.0.layer.0.SelfAttention.relative_attention_bias</c>).</summary>
    public TiedEmbeddingLayer<T> EncoderBias { get; }
    public RMSNormalizationLayer<T> EncoderFinalNorm { get; }
    /// <summary>The decoder's relative attention bias table <c>[buckets, heads]</c>.</summary>
    public TiedEmbeddingLayer<T> DecoderBias { get; }
    public RMSNormalizationLayer<T> DecoderFinalNorm { get; }

    internal static Tensor<T> Dropout(IEngine engine, Tensor<T> x, double rate, Random random)
    {
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return engine.TensorMultiply(x, mask);
    }

    private Tensor<T> PositionBias(TiedEmbeddingLayer<T> table, int queries, int keys, bool bidirectional, bool causal)
    {
        var c = Configuration;
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

    /// <summary>The encoder states <c>[L, d]</c> of token ids <c>[L]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> ids, bool training, Random random)
    {
        int length = ids.Length;
        var h = Drop(Shared.Forward(ids), training, random);
        var bias = PositionBias(EncoderBias, length, length, bidirectional: true, causal: false);
        foreach (var block in Encoder) h = block.Forward(h, bias, null, null, training, random);
        return Drop(EncoderFinalNorm.Forward(h), training, random);
    }

    /// <summary>The decoder's token logits <c>[L, vocabulary]</c> for decoder input ids <c>[L]</c> given the encoder
    /// states.</summary>
    public Tensor<T> Decode(Tensor<T> decoderIds, Tensor<T> memory, bool training, Random random)
    {
        int length = decoderIds.Length;
        var h = Drop(Shared.Forward(decoderIds), training, random);
        var bias = PositionBias(DecoderBias, length, length, bidirectional: false, causal: true);
        foreach (var block in Decoder) h = block.Forward(h, bias, memory, null, training, random);
        h = Drop(DecoderFinalNorm.Forward(h), training, random);
        var scaled = _engine.TensorMultiplyScalar(h, NumOps.FromDouble(1.0 / Math.Sqrt(Configuration.ModelDim)));
        return Shared.Logits(scaled);
    }

    /// <summary>
    /// The teacher-forced cross-entropy of <paramref name="labels"/> given <paramref name="inputIds"/>, as
    /// <c>T5ForConditionalGeneration(labels=...)</c> computes it: the decoder reads the labels shifted right behind the
    /// start token, and the loss averages the token cross-entropies.
    /// </summary>
    public Tensor<T> Loss(Tensor<T> inputIds, int[] labels, bool training, Random random)
    {
        var memory = Encode(inputIds, training, random);
        var decoderIds = new Tensor<T>(new[] { labels.Length });
        decoderIds[0] = NumOps.FromDouble(Configuration.DecoderStartTokenId);
        for (int i = 1; i < labels.Length; i++) decoderIds[i] = NumOps.FromDouble(labels[i - 1]);
        var logits = Decode(decoderIds, memory, training, random);                                             // [L, V]
        var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
        var target = new Tensor<T>(logits._shape);
        for (int i = 0; i < labels.Length; i++) target[i, labels[i]] = NumOps.FromDouble(-1.0 / labels.Length);
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, target), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>
    /// Samples a continuation of <paramref name="decoderPrefix"/> (Hugging Face <c>generate</c> with
    /// <c>do_sample=True</c>, one beam): each step draws from the softmax of logits / temperature restricted to the
    /// <paramref name="topK"/> largest, stopping at the end token or after <paramref name="maxNewTokens"/>.
    /// Returns the prefix followed by the sampled tokens (without the end token).
    /// </summary>
    public List<int> Generate(Tensor<T> inputIds, IReadOnlyList<int> decoderPrefix, double temperature, int topK,
        int maxNewTokens, Random random)
    {
        var memory = Encode(inputIds, training: false, random);
        var sequence = new List<int> { Configuration.DecoderStartTokenId };
        sequence.AddRange(decoderPrefix);
        for (int step = 0; step < maxNewTokens; step++)
        {
            var ids = new Tensor<T>(new[] { sequence.Count });
            for (int i = 0; i < sequence.Count; i++) ids[i] = NumOps.FromDouble(sequence[i]);
            var logits = Decode(ids, memory, training: false, random);
            int vocabulary = logits.Shape[1], last = sequence.Count - 1;
            var row = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++) row[v] = NumOps.ToDouble(logits[last, v]) / temperature;
            int next = SampleTopK(row, topK, random);
            if (next == Configuration.EndTokenId) break;
            sequence.Add(next);
        }
        sequence.RemoveAt(0);
        return sequence;
    }

    private static int SampleTopK(double[] scores, int topK, Random random)
    {
        int k = topK <= 0 ? scores.Length : Math.Min(topK, scores.Length);
        var order = new int[scores.Length];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => scores[b].CompareTo(scores[a]));
        double max = scores[order[0]], total = 0;
        var weights = new double[k];
        for (int i = 0; i < k; i++) total += weights[i] = Math.Exp(scores[order[i]] - max);
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
    /// N(0, d_ff^-1/2); layer norms at one.
    /// </summary>
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
        void Attention(T5Attention<T> attention)
        {
            Fill(attention.Query, c.ModelDim, inner, Math.Pow(c.ModelDim * c.KeyValueDim, -0.5));
            Fill(attention.Key, c.ModelDim, inner, Math.Pow(c.ModelDim, -0.5));
            Fill(attention.Value, c.ModelDim, inner, Math.Pow(c.ModelDim, -0.5));
            Fill(attention.Output, inner, c.ModelDim, Math.Pow(inner, -0.5));
        }
        Shared.Reinitialize(() => Normal(1.0));
        EncoderBias.Reinitialize(() => Normal(Math.Pow(c.ModelDim, -0.5)));
        DecoderBias.Reinitialize(() => Normal(Math.Pow(c.ModelDim, -0.5)));
        foreach (var block in Encoder.Concat(Decoder))
        {
            Attention(block.SelfAttention);
            if (block.CrossAttention is not null) Attention(block.CrossAttention);
            Fill(block.Wi, c.ModelDim, c.FeedForwardDim, Math.Pow(c.ModelDim, -0.5));
            Fill(block.Wo, c.FeedForwardDim, c.ModelDim, Math.Pow(c.FeedForwardDim, -0.5));
        }
    }

    /// <summary>
    /// Loads a Hugging Face <c>T5ForConditionalGeneration</c> state dict; <paramref name="read"/> returns the named
    /// tensor's values, row-major, after checking its shape. Linear weights are <c>[out, in]</c> in the checkpoint.
    /// </summary>
    public void LoadHuggingFaceWeights(Func<string, int[], double[]> read)
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
            Linear(name + ".DenseReluDense.wi", block.Wi, c.FeedForwardDim, c.ModelDim);
            Linear(name + ".DenseReluDense.wo", block.Wo, c.ModelDim, c.FeedForwardDim);
            Norm(name + ".layer_norm", block.FeedForwardNorm);
        }

        Shared.LoadTable(read("shared.weight", new[] { c.VocabularySize, c.ModelDim }));
        for (int i = 0; i < Encoder.Count; i++)
        {
            string block = $"encoder.block.{i}.layer";
            Attention($"{block}.0.SelfAttention", Encoder[i].SelfAttention);
            Norm($"{block}.0.layer_norm", Encoder[i].SelfNorm);
            FeedForward($"{block}.1", Encoder[i]);
        }
        EncoderBias.LoadTable(read("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", new[] { c.RelativeBuckets, c.Heads }));
        Norm("encoder.final_layer_norm", EncoderFinalNorm);
        for (int i = 0; i < Decoder.Count; i++)
        {
            string block = $"decoder.block.{i}.layer";
            Attention($"{block}.0.SelfAttention", Decoder[i].SelfAttention);
            Norm($"{block}.0.layer_norm", Decoder[i].SelfNorm);
            Attention($"{block}.1.EncDecAttention", Decoder[i].CrossAttention!);
            Norm($"{block}.1.layer_norm", Decoder[i].CrossNorm!);
            FeedForward($"{block}.2", Decoder[i]);
        }
        DecoderBias.LoadTable(read("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", new[] { c.RelativeBuckets, c.Heads }));
        Norm("decoder.final_layer_norm", DecoderFinalNorm);
    }
}
