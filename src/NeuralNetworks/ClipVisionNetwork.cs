using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.NeuralNetworks;

/// <summary>The shape of a CLIP vision transformer; the defaults are ViT-L/14 at 336 px.</summary>
internal sealed record ClipVisionConfiguration
{
    public int ImageSize { get; init; } = 336;

    public int PatchSize { get; init; } = 14;

    public int Channels { get; init; } = 3;

    public int Hidden { get; init; } = 1024;

    public int Layers { get; init; } = 24;

    public int Heads { get; init; } = 16;

    public int Intermediate { get; init; } = 4096;

    public double LayerNormEpsilon { get; init; } = 1e-5;
}

/// <summary>
/// CLIP's image encoder (Radford et al. 2021) as Hugging Face's <c>CLIPVisionModel</c> computes it: a bias-free patch
/// embedding, a learned class token and position embeddings, a LayerNorm before the transformer, and pre-norm layers of
/// self-attention and a quick-GELU feed-forward block.
/// </summary>
/// <remarks>
/// <para><see cref="Forward"/> returns every hidden state (the embeddings after the first LayerNorm, then each layer's
/// output), since LLaVA reads the second-to-last layer and GLaMM's region encoder reads every third.</para>
/// <para>Parameter names follow the Hugging Face checkpoints after a prefix (<c>vision_model.</c> by default).</para>
/// </remarks>
internal sealed class ClipVisionNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Layer(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, LayerNormalizationLayer<T> Norm2, DenseLayer<T> Fc1, DenseLayer<T> Fc2);

    public ClipVisionNetwork(IEngine engine, List<LayerBase<T>> layers, ClipVisionConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Hidden % c.Heads != 0) throw new ArgumentException("The width must divide by the head count.", nameof(configuration));
        if (c.ImageSize % c.PatchSize != 0) throw new ArgumentException("The image side must divide by the patch side.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm() => Add(new LayerNormalizationLayer<T>(c.Hidden, c.LayerNormEpsilon));
        PatchEmbedding = Add(new BiasFreeLinearLayer<T>(c.Channels * c.PatchSize * c.PatchSize, c.Hidden));
        ClassEmbedding = Add(new BiasFreeLinearLayer<T>(1, c.Hidden));
        PositionEmbedding = Add(new BiasFreeLinearLayer<T>(1, (Patches + 1) * c.Hidden));
        PreNorm = Norm();
        for (int i = 0; i < c.Layers; i++)
            Layers.Add(new Layer(Norm(), Linear(c.Hidden), Linear(c.Hidden), Linear(c.Hidden), Linear(c.Hidden), Norm(),
                Linear(c.Intermediate), Linear(c.Hidden)));
        PostNorm = Norm();
    }

    public ClipVisionConfiguration Configuration { get; }

    /// <summary>Patches per image: (ImageSize / PatchSize)².</summary>
    public int Patches => (Configuration.ImageSize / Configuration.PatchSize) * (Configuration.ImageSize / Configuration.PatchSize);

    public BiasFreeLinearLayer<T> PatchEmbedding { get; }

    public BiasFreeLinearLayer<T> ClassEmbedding { get; }

    public BiasFreeLinearLayer<T> PositionEmbedding { get; }

    public LayerNormalizationLayer<T> PreNorm { get; }

    public List<Layer> Layers { get; } = new();

    /// <summary>The LayerNorm Hugging Face applies to the pooled class token (not used by LLaVA).</summary>
    public LayerNormalizationLayer<T> PostNorm { get; }

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder) => holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    /// <summary>
    /// Every hidden state <c>[B, 1 + Patches, Hidden]</c> of normalized images <c>[B, Channels, ImageSize, ImageSize]</c>:
    /// entry 0 is the embeddings after the first LayerNorm, entry i the output of layer i.
    /// </summary>
    public List<Tensor<T>> Forward(Tensor<T> images)
    {
        var c = Configuration;
        int b = images.Shape[0], p = c.PatchSize;
        if (images.Shape[1] != c.Channels || images.Shape[2] != c.ImageSize || images.Shape[3] != c.ImageSize)
            throw new ArgumentException($"CLIP reads [B, {c.Channels}, {c.ImageSize}, {c.ImageSize}] images.", nameof(images));
        var columns = _engine.Unfold(images, new[] { p, p }, new[] { p, p }, new[] { 0, 0 });                       // [B, C·p·p, P]
        var patches = Rows(PatchEmbedding, _engine.TensorPermute(columns, new[] { 0, 2, 1 }));                      // [B, P, H]
        var cls = _engine.TensorBroadcastTo(_engine.Reshape(Value(ClassEmbedding), new[] { 1, 1, c.Hidden }), new[] { b, 1, c.Hidden });
        var x = _engine.TensorConcatenate(new[] { cls, patches }, 1);
        var positions = _engine.Reshape(Value(PositionEmbedding), new[] { 1, Patches + 1, c.Hidden });
        x = Rows(PreNorm, _engine.TensorAdd(x, _engine.TensorBroadcastTo(positions, x._shape)));
        var states = new List<Tensor<T>> { x };
        foreach (var layer in Layers)
        {
            x = Run(layer, x);
            states.Add(x);
        }
        return states;
    }

    private Tensor<T> Run(Layer layer, Tensor<T> x)
    {
        var c = Configuration;
        int b = x.Shape[0], n = x.Shape[1], hd = c.Hidden / c.Heads;
        var normed = Rows(layer.Norm1, x);
        Tensor<T> Heads(DenseLayer<T> projection) => _engine.Reshape(
            _engine.TensorPermute(_engine.Reshape(Rows(projection, normed), new[] { b, n, c.Heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { b * c.Heads, n, hd });
        var q = _engine.TensorMultiplyScalar(Heads(layer.Query), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var k = Heads(layer.Key);
        var v = Heads(layer.Value);
        var probabilities = _engine.Softmax(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), axis: 2);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(probabilities, v), new[] { b, c.Heads, n, hd }),
            new[] { 0, 2, 1, 3 }), new[] { b, n, c.Hidden });
        x = _engine.TensorAdd(x, Rows(layer.Output, context));
        var hidden = Rows(layer.Fc1, Rows(layer.Norm2, x));
        // quick_gelu: x · sigmoid(1.702 x).
        hidden = _engine.TensorMultiply(hidden, _engine.Sigmoid(_engine.TensorMultiplyScalar(hidden, NumOps.FromDouble(1.702))));
        return _engine.TensorAdd(x, Rows(layer.Fc2, hidden));
    }

    /// <summary>Loads a Hugging Face <c>CLIPVisionModel</c> state dict (names after <paramref name="prefix"/>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "vision_model.")
    {
        var c = Configuration;
        int h = c.Hidden;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{prefix}{name}.weight", new[] { outputs, inputs }),
                read($"{prefix}{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name)
            => TorchParameters.LayerNorm(_engine, norm, read($"{prefix}{name}.weight", new[] { h }), read($"{prefix}{name}.bias", new[] { h }));
        int patch = c.Channels * c.PatchSize * c.PatchSize;
        TorchParameters.Linear(PatchEmbedding, patch, h, read($"{prefix}embeddings.patch_embedding.weight", new[] { h, c.Channels, c.PatchSize, c.PatchSize }));
        TorchParameters.Linear(ClassEmbedding, 1, h, read($"{prefix}embeddings.class_embedding", new[] { h }));
        TorchParameters.Linear(PositionEmbedding, 1, (Patches + 1) * h, read($"{prefix}embeddings.position_embedding.weight", new[] { Patches + 1, h }));
        Norm(PreNorm, "pre_layrnorm");
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string n = $"encoder.layers.{i}";
            Norm(layer.Norm1, $"{n}.layer_norm1");
            Linear(layer.Query, $"{n}.self_attn.q_proj", h, h);
            Linear(layer.Key, $"{n}.self_attn.k_proj", h, h);
            Linear(layer.Value, $"{n}.self_attn.v_proj", h, h);
            Linear(layer.Output, $"{n}.self_attn.out_proj", h, h);
            Norm(layer.Norm2, $"{n}.layer_norm2");
            Linear(layer.Fc1, $"{n}.mlp.fc1", h, c.Intermediate);
            Linear(layer.Fc2, $"{n}.mlp.fc2", c.Intermediate, h);
        }
        Norm(PostNorm, "post_layernorm");
    }

    /// <summary>
    /// The reference initialization (Hugging Face <c>CLIPPreTrainedModel._init_weights</c>, factor 1): class embedding
    /// N(0, (H^-0.5)²), patch embedding N(0, 0.02²), position embedding N(0, 0.02²); attention projections
    /// N(0, (H^-0.5 · (2L)^-0.5)²) and the output projection N(0, (H^-0.5)²); fc1 N(0, ((2H)^-0.5)²) and fc2
    /// N(0, (H^-0.5 · (2L)^-0.5)²); zero biases and unit LayerNorms.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double inStd = Math.Pow(c.Hidden, -0.5) * Math.Pow(2.0 * c.Layers, -0.5), outStd = Math.Pow(c.Hidden, -0.5);
        double fcStd = Math.Pow(2.0 * c.Hidden, -0.5);
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            if (name.EndsWith(".bias", StringComparison.Ordinal)) return values;
            if (name.Contains("layer_norm") || name.Contains("layrnorm") || name.Contains("layernorm"))
            {
                for (int i = 0; i < count; i++) values[i] = 1.0;
                return values;
            }
            double std = name.EndsWith("class_embedding", StringComparison.Ordinal) ? outStd
                : name.Contains("embeddings.") ? 0.02
                : name.Contains("out_proj") ? outStd
                : name.Contains("_proj") ? inStd
                : name.Contains("fc1") ? fcStd
                : inStd;
            for (int i = 0; i < count; i++) values[i] = std * Normal();
            return values;
        });
    }
}
