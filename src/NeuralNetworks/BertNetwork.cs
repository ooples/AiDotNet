using AiDotNet.Helpers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.NeuralNetworks;

/// <summary>The shape of a BERT encoder; the defaults are BERT-base (uncased).</summary>
internal sealed record BertConfiguration
{
    public int VocabSize { get; init; } = 30522;

    public int Hidden { get; init; } = 768;

    public int Layers { get; init; } = 12;

    public int Heads { get; init; } = 12;

    public int Intermediate { get; init; } = 3072;

    public int MaxPositions { get; init; } = 512;

    public int TypeVocabSize { get; init; } = 2;

    public double LayerNormEpsilon { get; init; } = 1e-12;

    /// <summary>Hidden-layer dropout (0.1 in BERT); applied only in training mode.</summary>
    public double Dropout { get; init; } = 0.1;
}

/// <summary>
/// BERT (Devlin et al. 2019) without its pooler, as Hugging Face's <c>BertModel(add_pooling_layer=False)</c> computes it:
/// word, position and token-type embeddings summed and layer-normalized, then post-norm Transformer layers with exact
/// GELU feed-forward blocks.
/// </summary>
/// <remarks>
/// <para>Grounding DINO's text backbone is this network. It passes its own position ids (restarting at each phrase) and
/// a three-dimensional attention mask (each phrase attends only within itself), so both are inputs here.</para>
/// <para>Parameter names follow the Hugging Face checkpoints (<c>embeddings.*</c>, <c>encoder.layer.{i}.*</c>).</para>
/// </remarks>
internal sealed class BertNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _owned = new();
    private readonly List<DropoutLayer<T>> _dropouts = new();

    internal sealed record Layer(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> AttentionOutput,
        LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Intermediate, DenseLayer<T> Output, LayerNormalizationLayer<T> OutputNorm,
        DropoutLayer<T>? AttentionDropout, DropoutLayer<T>? OutputDropout);

    public BertNetwork(IEngine engine, List<LayerBase<T>> layers, BertConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Hidden % c.Heads != 0) throw new ArgumentException("The width must divide by the head count.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            _owned.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm() => Add(new LayerNormalizationLayer<T>(c.Hidden, c.LayerNormEpsilon));
        DropoutLayer<T>? Dropout()
        {
            if (c.Dropout <= 0) return null;
            var dropout = Add(new DropoutLayer<T>(c.Dropout));
            _dropouts.Add(dropout);
            return dropout;
        }

        WordEmbeddings = Add(new BiasFreeLinearLayer<T>(1, c.VocabSize * c.Hidden));
        PositionEmbeddings = Add(new BiasFreeLinearLayer<T>(1, c.MaxPositions * c.Hidden));
        TypeEmbeddings = Add(new BiasFreeLinearLayer<T>(1, c.TypeVocabSize * c.Hidden));
        EmbeddingNorm = Norm();
        EmbeddingDropout = Dropout();
        for (int i = 0; i < c.Layers; i++)
            Layers.Add(new Layer(Linear(c.Hidden), Linear(c.Hidden), Linear(c.Hidden), Linear(c.Hidden), Norm(),
                Linear(c.Intermediate), Linear(c.Hidden), Norm(), Dropout(), Dropout()));
        Training = false;
    }

    public BertConfiguration Configuration { get; }

    /// <summary>Whether dropout is active. A new network starts in inference mode.</summary>
    public bool Training
    {
        get => _training;
        set
        {
            _training = value;
            foreach (var dropout in _dropouts) dropout.SetTrainingMode(value);
        }
    }

    private bool _training = true;

    public BiasFreeLinearLayer<T> WordEmbeddings { get; }

    public BiasFreeLinearLayer<T> PositionEmbeddings { get; }

    public BiasFreeLinearLayer<T> TypeEmbeddings { get; }

    public LayerNormalizationLayer<T> EmbeddingNorm { get; }

    private DropoutLayer<T>? EmbeddingDropout { get; }

    public List<Layer> Layers { get; } = new();

    private Tensor<T> Table(BiasFreeLinearLayer<T> holder, int rows)
        => _engine.Reshape(holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }))),
            new[] { rows, Configuration.Hidden });

    private Tensor<T> Lookup(BiasFreeLinearLayer<T> holder, int rows, int[,] ids)
    {
        int b = ids.GetLength(0), l = ids.GetLength(1);
        var index = new Tensor<int>(new[] { b * l });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                int id = ids[i, j];
                if (id < 0 || id >= rows) throw new ArgumentOutOfRangeException(nameof(ids), $"Id {id} is outside [0, {rows}).");
                index[i * l + j] = id;
            }
        return _engine.Reshape(_engine.TensorIndexSelect(Table(holder, rows), index, 0), new[] { b, l, Configuration.Hidden });
    }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    private static Tensor<T> Drop(DropoutLayer<T>? dropout, Tensor<T> x) => dropout is null ? x : dropout.Forward(x);

    /// <summary>
    /// The last hidden states <c>[B, L, Hidden]</c> of token ids <c>[B, L]</c>.
    /// </summary>
    /// <param name="inputIds">Token ids.</param>
    /// <param name="tokenTypeIds">Segment ids, or null for all zeros.</param>
    /// <param name="positionIds">Position ids, or null for 0..L-1.</param>
    /// <param name="attentionMask">1 where a query may attend a key: <c>[B, L]</c> (per key) or <c>[B, L, L]</c> (per
    /// query and key); null attends everywhere.</param>
    /// <param name="prefix">Per-layer key and value prefixes <c>[prefixLength, Hidden]</c> that every query also attends
    /// to, as Hugging Face's <c>past_key_values</c> carry them (P-tuning v2's deep prompts, Liu et al. 2022); the
    /// default positions then start after the prefix, as there.</param>
    /// <param name="layerCount">How many layers to run (null: all): <c>hidden_states[k]</c> of Hugging Face's
    /// <c>output_hidden_states</c> is the output after k layers.</param>
    public Tensor<T> Encode(int[,] inputIds, int[,]? tokenTypeIds = null, int[,]? positionIds = null, bool[,,]? attentionMask = null,
        IReadOnlyList<(Tensor<T> Key, Tensor<T> Value)>? prefix = null, int? layerCount = null)
    {
        var c = Configuration;
        int b = inputIds.GetLength(0), l = inputIds.GetLength(1);
        if (prefix is not null && prefix.Count != c.Layers)
            throw new ArgumentException($"Expected a key and value prefix for each of the {c.Layers} layers.", nameof(prefix));
        int prefixLength = prefix is null ? 0 : prefix[0].Key.Shape[0];
        if (l + prefixLength > c.MaxPositions) throw new ArgumentException($"At most {c.MaxPositions} positions.", nameof(inputIds));
        tokenTypeIds ??= new int[b, l];
        if (positionIds is null)
        {
            positionIds = new int[b, l];
            for (int i = 0; i < b; i++) for (int j = 0; j < l; j++) positionIds[i, j] = prefixLength + j;
        }
        var x = _engine.TensorAdd(_engine.TensorAdd(Lookup(WordEmbeddings, c.VocabSize, inputIds),
            Lookup(TypeEmbeddings, c.TypeVocabSize, tokenTypeIds)), Lookup(PositionEmbeddings, c.MaxPositions, positionIds));
        x = Drop(EmbeddingDropout, Rows(EmbeddingNorm, x));

        // HF's extended mask: (1 - m) * finfo.min added to the scores.
        // The prefix positions are always attended to.
        Tensor<T>? bias = null;
        if (attentionMask is not null)
        {
            bias = new Tensor<T>(new[] { b * c.Heads, l, prefixLength + l });
            var blocked = NumOps.FromDouble(typeof(T) == typeof(float) ? float.MinValue : double.MinValue);   // torch.finfo(dtype).min
            for (int i = 0; i < b; i++)
                for (int h = 0; h < c.Heads; h++)
                    for (int q = 0; q < l; q++)
                        for (int k = 0; k < l; k++)
                            if (!attentionMask[i, q, k]) bias[(i * c.Heads) + h, q, prefixLength + k] = blocked;
        }
        int run = layerCount ?? Layers.Count;
        if (run < 0 || run > Layers.Count) throw new ArgumentOutOfRangeException(nameof(layerCount));
        for (int i = 0; i < run; i++) x = Run(Layers[i], x, bias, prefix?[i]);
        return x;
    }

    /// <summary>A key-padding mask <c>[B, L]</c> expanded to the per-query form <see cref="Encode"/> takes.</summary>
    public static bool[,,] KeyMask(bool[,] keys)
    {
        int b = keys.GetLength(0), l = keys.GetLength(1);
        var mask = new bool[b, l, l];
        for (int i = 0; i < b; i++)
            for (int q = 0; q < l; q++)
                for (int k = 0; k < l; k++) mask[i, q, k] = keys[i, k];
        return mask;
    }

    private Tensor<T> Run(Layer layer, Tensor<T> x, Tensor<T>? bias, (Tensor<T> Key, Tensor<T> Value)? prefix)
    {
        var c = Configuration;
        int b = x.Shape[0], l = x.Shape[1], hd = c.Hidden / c.Heads;
        Tensor<T> Split(Tensor<T> rows, int length) => _engine.Reshape(
            _engine.TensorPermute(_engine.Reshape(rows, new[] { b, length, c.Heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { b * c.Heads, length, hd });
        Tensor<T> Heads(DenseLayer<T> projection) => Split(Rows(projection, x), l);
        // A prefix [P, Hidden] is shared by every batch row and precedes the keys and values.
        Tensor<T> Prefixed(Tensor<T> heads, Tensor<T> prefixRows)
        {
            int p = prefixRows.Shape[0];
            var tiled = _engine.TensorTile(_engine.Reshape(prefixRows, new[] { 1, p, c.Hidden }), new[] { b, 1, 1 });
            return _engine.TensorConcatenate(new[] { Split(tiled, p), heads }, 1);
        }
        var q = Heads(layer.Query);
        var k = Heads(layer.Key);
        var v = Heads(layer.Value);
        if (prefix is { } kv)
        {
            k = Prefixed(k, kv.Key);
            v = Prefixed(v, kv.Value);
        }
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        if (bias is not null) scores = _engine.TensorAdd(scores, bias);
        var probabilities = Drop(layer.AttentionDropout, _engine.Softmax(scores, axis: 2));
        var context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(probabilities, v), new[] { b, c.Heads, l, hd }),
            new[] { 0, 2, 1, 3 }), new[] { b, l, c.Hidden });
        var attended = Rows(layer.AttentionNorm, _engine.TensorAdd(Drop(layer.OutputDropout, Rows(layer.AttentionOutput, context)), x));
        var hidden = ActivationHelper.ExactGelu(Rows(layer.Intermediate, attended), _engine);
        return Rows(layer.OutputNorm, _engine.TensorAdd(Drop(layer.OutputDropout, Rows(layer.Output, hidden)), attended));
    }

    /// <summary>Loads a Hugging Face <c>BertModel</c> state dict (names after an optional <paramref name="prefix"/>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "")
    {
        var c = Configuration;
        int h = c.Hidden;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{prefix}{name}.weight", new[] { outputs, inputs }),
                read($"{prefix}{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name)
            => TorchParameters.LayerNorm(_engine, norm, read($"{prefix}{name}.weight", new[] { h }), read($"{prefix}{name}.bias", new[] { h }));
        TorchParameters.Linear(WordEmbeddings, 1, c.VocabSize * h, read($"{prefix}embeddings.word_embeddings.weight", new[] { c.VocabSize, h }));
        TorchParameters.Linear(PositionEmbeddings, 1, c.MaxPositions * h, read($"{prefix}embeddings.position_embeddings.weight", new[] { c.MaxPositions, h }));
        TorchParameters.Linear(TypeEmbeddings, 1, c.TypeVocabSize * h, read($"{prefix}embeddings.token_type_embeddings.weight", new[] { c.TypeVocabSize, h }));
        Norm(EmbeddingNorm, "embeddings.LayerNorm");
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string n = $"encoder.layer.{i}";
            Linear(layer.Query, $"{n}.attention.self.query", h, h);
            Linear(layer.Key, $"{n}.attention.self.key", h, h);
            Linear(layer.Value, $"{n}.attention.self.value", h, h);
            Linear(layer.AttentionOutput, $"{n}.attention.output.dense", h, h);
            Norm(layer.AttentionNorm, $"{n}.attention.output.LayerNorm");
            Linear(layer.Intermediate, $"{n}.intermediate.dense", h, c.Intermediate);
            Linear(layer.Output, $"{n}.output.dense", c.Intermediate, h);
            Norm(layer.OutputNorm, $"{n}.output.LayerNorm");
        }
    }

    /// <summary>BERT's initialization (Hugging Face <c>_init_weights</c>): N(0, 0.02²) weights and embeddings, zero biases,
    /// unit LayerNorms.</summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            if (name.Contains("LayerNorm"))
            {
                if (name.EndsWith(".weight", StringComparison.Ordinal)) for (int i = 0; i < count; i++) values[i] = 1.0;
                return values;
            }
            if (name.EndsWith(".weight", StringComparison.Ordinal))
                for (int i = 0; i < count; i++) values[i] = 0.02 * Normal();
            return values;
        });
    }
}
