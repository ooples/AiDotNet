using AiDotNet.Helpers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.NeuralNetworks;

/// <summary>The shape of an ALBERT encoder; the defaults are ALBERT-base's (Lan et al. 2020).</summary>
internal sealed record AlbertConfiguration
{
    public int VocabSize { get; init; } = 30000;

    /// <summary>Width of the factorized token embedding (128).</summary>
    public int EmbeddingSize { get; init; } = 128;

    public int Hidden { get; init; } = 768;

    public int Layers { get; init; } = 12;

    public int Heads { get; init; } = 12;

    public int Intermediate { get; init; } = 3072;

    public int MaxPositions { get; init; } = 512;

    public int TypeVocabSize { get; init; } = 2;

    public double LayerNormEpsilon { get; init; } = 1e-12;

    /// <summary>Hidden and attention dropout (0, Hugging Face's AlbertConfig default); applied only in training.</summary>
    public double Dropout { get; init; }
}

/// <summary>
/// ALBERT (Lan et al. 2020) without its pooler, as Hugging Face's <c>AlbertModel</c> computes its last hidden state:
/// factorized word, position and token-type embeddings summed and layer-normalized, a linear map to the hidden width,
/// then one post-norm Transformer layer whose parameters every layer shares, with the tanh-approximate GELU
/// (<c>gelu_new</c>) feed-forward.
/// </summary>
/// <remarks>Parameter names follow the Hugging Face checkpoints (<c>embeddings.*</c>,
/// <c>encoder.embedding_hidden_mapping_in.*</c>, <c>encoder.albert_layer_groups.0.albert_layers.0.*</c>). StyleTTS 2's
/// PL-BERT is this network.</remarks>
internal sealed class AlbertNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public AlbertNetwork(IEngine engine, List<LayerBase<T>> layers, AlbertConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Hidden % c.Heads != 0) throw new ArgumentException("The width must divide by the head count.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        WordEmbeddings = Add(new TiedEmbeddingLayer<T>(c.VocabSize, c.EmbeddingSize));
        PositionEmbeddings = Add(new TiedEmbeddingLayer<T>(c.MaxPositions, c.EmbeddingSize));
        TypeEmbeddings = Add(new TiedEmbeddingLayer<T>(c.TypeVocabSize, c.EmbeddingSize));
        EmbeddingNorm = Add(new LayerNormalizationLayer<T>(c.EmbeddingSize, c.LayerNormEpsilon));
        EmbeddingMapping = Linear(c.Hidden);
        Query = Linear(c.Hidden);
        Key = Linear(c.Hidden);
        Value = Linear(c.Hidden);
        AttentionOutput = Linear(c.Hidden);
        AttentionNorm = Add(new LayerNormalizationLayer<T>(c.Hidden, c.LayerNormEpsilon));
        FeedForward = Linear(c.Intermediate);
        FeedForwardOutput = Linear(c.Hidden);
        OutputNorm = Add(new LayerNormalizationLayer<T>(c.Hidden, c.LayerNormEpsilon));
    }

    public AlbertConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> WordEmbeddings { get; }
    public TiedEmbeddingLayer<T> PositionEmbeddings { get; }
    public TiedEmbeddingLayer<T> TypeEmbeddings { get; }
    public LayerNormalizationLayer<T> EmbeddingNorm { get; }
    public DenseLayer<T> EmbeddingMapping { get; }
    public DenseLayer<T> Query { get; }
    public DenseLayer<T> Key { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> AttentionOutput { get; }
    public LayerNormalizationLayer<T> AttentionNorm { get; }
    public DenseLayer<T> FeedForward { get; }
    public DenseLayer<T> FeedForwardOutput { get; }
    public LayerNormalizationLayer<T> OutputNorm { get; }

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var t = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) t[i] = NumOps.FromDouble(ids[i]);
        return t;
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && Configuration.Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, Configuration.Dropout, random) : x;

    /// <summary>The last hidden states <c>[L, Hidden]</c> of token ids <c>[L]</c> (one sequence, every position
    /// attended).</summary>
    public Tensor<T> Encode(IReadOnlyList<int> ids, bool training, Random random)
    {
        var c = Configuration;
        int length = ids.Count, heads = c.Heads, headDim = c.Hidden / heads;
        if (length > c.MaxPositions) throw new ArgumentException($"At most {c.MaxPositions} tokens.", nameof(ids));
        var positions = Enumerable.Range(0, length).ToArray();
        var x = _engine.TensorAdd(_engine.TensorAdd(WordEmbeddings.Forward(Ids(ids)), TypeEmbeddings.Forward(Ids(new int[length]))),
            PositionEmbeddings.Forward(Ids(positions)));
        x = Drop(EmbeddingNorm.Forward(x), training, random);
        x = EmbeddingMapping.Forward(x);                                                                    // [L, H]
        Tensor<T> Heads(Tensor<T> rows) => _engine.TensorPermute(_engine.Reshape(rows, new[] { length, heads, headDim }), new[] { 1, 0, 2 });
        for (int layer = 0; layer < c.Layers; layer++)
        {
            var q = Heads(Query.Forward(x));
            var k = Heads(Key.Forward(x));
            var v = Heads(Value.Forward(x));
            var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));
            var weights = Drop(_engine.Softmax(scores, axis: 2), training, random);
            var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, c.Hidden });
            var attended = AttentionNorm.Forward(_engine.TensorAdd(x, Drop(AttentionOutput.Forward(context), training, random)));
            var hidden = FeedForwardOutput.Forward(_engine.GELU(FeedForward.Forward(attended)));
            x = OutputNorm.Forward(_engine.TensorAdd(Drop(hidden, training, random), attended));
        }
        return x;
    }

    /// <summary>Loads a Hugging Face <c>AlbertModel</c> state dict (names after an optional <paramref name="prefix"/>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "")
    {
        var c = Configuration;
        int h = c.Hidden, e = c.EmbeddingSize;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{prefix}{name}.weight", new[] { outputs, inputs }),
                read($"{prefix}{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name, int width)
            => TorchParameters.LayerNorm(_engine, norm, read($"{prefix}{name}.weight", new[] { width }), read($"{prefix}{name}.bias", new[] { width }));
        WordEmbeddings.LoadTable(read($"{prefix}embeddings.word_embeddings.weight", new[] { c.VocabSize, e }));
        PositionEmbeddings.LoadTable(read($"{prefix}embeddings.position_embeddings.weight", new[] { c.MaxPositions, e }));
        TypeEmbeddings.LoadTable(read($"{prefix}embeddings.token_type_embeddings.weight", new[] { c.TypeVocabSize, e }));
        Norm(EmbeddingNorm, "embeddings.LayerNorm", e);
        Linear(EmbeddingMapping, "encoder.embedding_hidden_mapping_in", e, h);
        const string layer = "encoder.albert_layer_groups.0.albert_layers.0";
        Linear(Query, $"{layer}.attention.query", h, h);
        Linear(Key, $"{layer}.attention.key", h, h);
        Linear(Value, $"{layer}.attention.value", h, h);
        Linear(AttentionOutput, $"{layer}.attention.dense", h, h);
        Norm(AttentionNorm, $"{layer}.attention.LayerNorm", h);
        Linear(FeedForward, $"{layer}.ffn", h, c.Intermediate);
        Linear(FeedForwardOutput, $"{layer}.ffn_output", c.Intermediate, h);
        Norm(OutputNorm, $"{layer}.full_layer_layer_norm", h);
    }
}
