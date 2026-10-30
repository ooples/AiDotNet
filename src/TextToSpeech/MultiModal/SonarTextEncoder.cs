using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes of a SONAR text encoder (the released <c>text_sonar_basic_encoder</c>, NLLB-200's 1.3B encoder).</summary>
internal sealed record SonarTextEncoderConfiguration
{
    public int VocabSize { get; init; } = 256206;
    public int Width { get; init; } = 1024;
    public int Layers { get; init; } = 24;
    public int Heads { get; init; } = 16;
    public int FeedForward { get; init; } = 8192;
    public int PaddingId { get; init; } = 1;
}

/// <summary>
/// NLLB-200's tokenizer as SONAR's text encoder reads it (transformers' <c>NllbTokenizer</c>, non-legacy): the source
/// language's code, the SentencePiece pieces (ids shifted by one past <c>&lt;s&gt;</c>, <c>&lt;pad&gt;</c>,
/// <c>&lt;/s&gt;</c>, <c>&lt;unk&gt;</c>; an unknown piece is 3) and <c>&lt;/s&gt;</c>; language code i is
/// <c>pieces + 1 + i</c>.
/// </summary>
internal sealed class SonarTextTokenizer
{
    private readonly SentencePieceModel _model;
    private readonly Dictionary<string, int> _languages;

    public SonarTextTokenizer(SentencePieceModel model, IReadOnlyList<string> languageCodes)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _languages = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < languageCodes.Count; i++) _languages[languageCodes[i]] = model.Count + 1 + i;
    }

    /// <summary>Reads <c>sentencepiece.bpe.model</c> and the language codes (<c>special_tokens_map.json</c>'s
    /// <c>additional_special_tokens</c>) from a tokenizer directory.</summary>
    public static SonarTextTokenizer FromDirectory(string directory)
    {
        var model = new SentencePieceModel(File.ReadAllBytes(Path.Combine(directory, "sentencepiece.bpe.model")));
        var map = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(directory, "special_tokens_map.json")));
        var codes = map["additional_special_tokens"]!.Select(t => t.Type == Newtonsoft.Json.Linq.JTokenType.Object ? (string)t["content"]! : (string)t!).ToList();
        return new SonarTextTokenizer(model, codes);
    }

    public const int Eos = 2, Unknown = 3;

    /// <summary>The ids of <paramref name="text"/> in <paramref name="language"/> (e.g. <c>eng_Latn</c>).</summary>
    public int[] Encode(string text, string language)
    {
        if (!_languages.TryGetValue(language, out int code)) throw new ArgumentException($"Unknown language code '{language}'.", nameof(language));
        var ids = new List<int> { code };
        foreach (int id in _model.Encode(text ?? throw new ArgumentNullException(nameof(text)))) ids.Add(id == _model.UnknownId ? Unknown : id + 1);
        ids.Add(Eos);
        return ids.ToArray();
    }
}

/// <summary>
/// SONAR's text encoder (Duquenne et al. 2023, "SONAR: Sentence-Level Multimodal and Language-Agnostic Representations"):
/// NLLB-200's Transformer encoder whose token states are mean-pooled into one 1024-dimensional sentence embedding.
/// SONAR Expressive's decoders read it as the semantic conditioning.
/// </summary>
/// <remarks>The encoder is M2M-100's (as the released checkpoint's transformers port runs it): token embeddings scaled by
/// √d plus sinusoidal positions starting at the padding id + 1, pre-norm layers (self-attention, then a ReLU feed-forward,
/// each after a layer norm and added back), a final layer norm. Frozen: SONAR Expressive never trains it.</remarks>
internal sealed class SonarTextEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Layer(LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        LayerNormalizationLayer<T> FeedForwardNorm, DenseLayer<T> FeedForwardIn, DenseLayer<T> FeedForwardOut);

    public SonarTextEncoder(IEngine engine, List<LayerBase<T>> layers, SonarTextEncoderConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Width % c.Heads != 0) throw new ArgumentException($"The width {c.Width} must divide into {c.Heads} heads.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm() => Own(new LayerNormalizationLayer<T>(c.Width, 1e-5));
        Embedding = Own(new TiedEmbeddingLayer<T>(c.VocabSize, c.Width));
        for (int i = 0; i < c.Layers; i++)
            Layers.Add(new Layer(Norm(), Linear(c.Width), Linear(c.Width), Linear(c.Width), Linear(c.Width), Norm(), Linear(c.FeedForward), Linear(c.Width)));
        FinalNorm = Norm();
    }

    public SonarTextEncoderConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public List<Layer> Layers { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }

    /// <summary>The sinusoidal position table row <paramref name="position"/> (tensor2tensor's: sines, then cosines).</summary>
    private double[] Position(int position)
    {
        int d = Configuration.Width, half = d / 2;
        double step = Math.Log(10000) / (half - 1);
        var row = new double[d];
        for (int i = 0; i < half; i++)
        {
            double angle = position * Math.Exp(-step * i);
            row[i] = Math.Sin(angle);
            row[half + i] = Math.Cos(angle);
        }
        return row;
    }

    /// <summary>The token states <c>[length, width]</c> of token ids.</summary>
    public Tensor<T> Forward(IReadOnlyList<int> ids)
    {
        var c = Configuration;
        int length = ids.Count, d = c.Width, heads = c.Heads, e = d / heads;
        var idTensor = new Tensor<T>(new[] { length });
        for (int i = 0; i < length; i++)
        {
            if (ids[i] < 0 || ids[i] >= c.VocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Token ids lie in [0, {c.VocabSize}).");
            idTensor[i] = NumOps.FromDouble(ids[i]);
        }
        var positions = new Tensor<T>(new[] { length, d });
        for (int t = 0; t < length; t++)
        {
            var row = Position(c.PaddingId + 1 + t);
            for (int j = 0; j < d; j++) positions[t, j] = NumOps.FromDouble(row[j]);
        }
        var x = _engine.TensorAdd(_engine.TensorMultiplyScalar(Embedding.Forward(idTensor), NumOps.FromDouble(Math.Sqrt(d))), positions);
        foreach (var l in Layers)
        {
            var h = l.AttentionNorm.Forward(x);
            Tensor<T> Heads(DenseLayer<T> projection) =>
                _engine.TensorPermute(_engine.Reshape(projection.Forward(h), new[] { length, heads, e }), new[] { 1, 0, 2 });
            var q = _engine.TensorMultiplyScalar(Heads(l.Query), NumOps.FromDouble(1.0 / Math.Sqrt(e)));
            var k = Heads(l.Key);
            var v = Heads(l.Value);
            var context = _engine.BatchMatMul(_engine.Softmax(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), axis: 2), v);
            x = _engine.TensorAdd(x, l.Output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, d })));
            x = _engine.TensorAdd(x, l.FeedForwardOut.Forward(_engine.ReLU(l.FeedForwardIn.Forward(l.FeedForwardNorm.Forward(x)))));
        }
        return FinalNorm.Forward(x);
    }

    /// <summary>The SONAR embedding <c>[width]</c> of token ids: the mean of the token states.</summary>
    public Tensor<T> Embed(IReadOnlyList<int> ids)
    {
        var states = Forward(ids);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(states, new[] { 0 }, keepDims: false), NumOps.FromDouble(1.0 / ids.Count));
    }

    /// <summary>Loads transformers' <c>M2M100Encoder</c> state (<c>encoder.</c>-prefixed or not; the token table as
    /// <c>embed_tokens.weight</c> or <c>shared.weight</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> take)
    {
        var c = Configuration;
        double[] Read(string name) => take("encoder." + name) ?? take(name) ?? throw new InvalidDataException($"The SONAR encoder checkpoint has no tensor '{name}'.");
        Embedding.LoadTable(take("encoder.embed_tokens.weight") ?? take("embed_tokens.weight") ?? take("shared.weight")
            ?? throw new InvalidDataException("The SONAR encoder checkpoint has no token table."));
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs) => TorchParameters.Linear(_engine, layer, inputs, outputs, Read(name + ".weight"), Read(name + ".bias"));
        void Norm(LayerNormalizationLayer<T> layer, string name) => TorchParameters.LayerNorm(_engine, layer, Read(name + ".weight"), Read(name + ".bias"));
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            string p = $"layers.{i}";
            Norm(l.AttentionNorm, $"{p}.self_attn_layer_norm");
            Lin(l.Query, $"{p}.self_attn.q_proj", c.Width, c.Width);
            Lin(l.Key, $"{p}.self_attn.k_proj", c.Width, c.Width);
            Lin(l.Value, $"{p}.self_attn.v_proj", c.Width, c.Width);
            Lin(l.Output, $"{p}.self_attn.out_proj", c.Width, c.Width);
            Norm(l.FeedForwardNorm, $"{p}.final_layer_norm");
            Lin(l.FeedForwardIn, $"{p}.fc1", c.Width, c.FeedForward);
            Lin(l.FeedForwardOut, $"{p}.fc2", c.FeedForward, c.Width);
        }
        Norm(FinalNorm, "layer_norm");
    }

    /// <summary>The embedding of text without gradients.</summary>
    public Tensor<T> EmbedText(SonarTextTokenizer tokenizer, string text, string language)
    {
        using var _ = new NoGradScope<T>();
        return Embed(tokenizer.Encode(text, language));
    }
}
