using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The kinds of Zonos's conditioners (<c>zonos/conditioning.py</c>).</summary>
public enum ZonosConditionerKind
{
    /// <summary>The text's eSpeak phonemes, embedded one token each.</summary>
    EspeakPhonemes,

    /// <summary>A vector passed through (the speaker embedding), projected to the model's width.</summary>
    Passthrough,

    /// <summary>Real values in [min, max] as random Fourier features.</summary>
    Fourier,

    /// <summary>An integer in [min, max], embedded.</summary>
    Integer,
}

/// <summary>One of Zonos's conditioners as its release config describes it.</summary>
public sealed record ZonosConditioner(string Name, ZonosConditionerKind Kind, int InputDim = 1, double Min = 0, double Max = 1, int CondDim = 0,
    bool LearnedUnconditional = true, bool Projected = false)
{
    /// <summary>Zonos-v0.1-transformer's conditioners.</summary>
    public static IReadOnlyList<ZonosConditioner> Transformer { get; } = new[]
    {
        new ZonosConditioner("espeak", ZonosConditionerKind.EspeakPhonemes, LearnedUnconditional: false),
        new ZonosConditioner("speaker", ZonosConditionerKind.Passthrough, CondDim: 128, Projected: true),
        new ZonosConditioner("emotion", ZonosConditionerKind.Fourier, InputDim: 8),
        new ZonosConditioner("fmax", ZonosConditionerKind.Fourier, Min: 0, Max: 24000),
        new ZonosConditioner("pitch_std", ZonosConditionerKind.Fourier, Min: 0, Max: 400),
        new ZonosConditioner("speaking_rate", ZonosConditionerKind.Fourier, Min: 0, Max: 40),
        new ZonosConditioner("language_id", ZonosConditionerKind.Integer, Min: -1, Max: 126),
    };

    /// <summary>Zonos-v0.1-hybrid's conditioners: the transformer's and four quality targets.</summary>
    public static IReadOnlyList<ZonosConditioner> Hybrid { get; } = Transformer.Concat(new[]
    {
        new ZonosConditioner("vqscore_8", ZonosConditionerKind.Fourier, InputDim: 8, Min: 0.5, Max: 0.8),
        new ZonosConditioner("ctc_loss", ZonosConditionerKind.Fourier, Min: -1, Max: 1000),
        new ZonosConditioner("dnsmos_ovrl", ZonosConditionerKind.Fourier, Min: 1, Max: 5),
        new ZonosConditioner("speaker_noised", ZonosConditionerKind.Integer, Min: 0, Max: 1),
    }).ToArray();
}

/// <summary>
/// Zonos's prefix conditioner (<c>PrefixConditioner</c>): each conditioner's tokens — the phonemes, or one token for each
/// other input, or its learned unconditional vector when the input is absent — concatenated, projected and layer-normalized.
/// </summary>
internal sealed class ZonosPrefixConditioner<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _width;

    /// <summary>The phoneme symbols after the four special tokens (pad, unknown, start, end), as <c>symbols</c> lists them.</summary>
    internal const string Symbols = "\u003b\u003a\u002c\u002e\u0021\u003f\u00a1\u00bf\u2014\u2026\u0022\u00ab\u00bb\u201c\u201d\u0028\u0029\u0020\u002a\u007e\u002d\u002f\u005c\u0026\u0041\u0042\u0043\u0044\u0045\u0046\u0047\u0048\u0049\u004a\u004b\u004c\u004d\u004e\u004f\u0050\u0051\u0052\u0053\u0054\u0055\u0056\u0057\u0058\u0059\u005a\u0061\u0062\u0063\u0064\u0065\u0066\u0067\u0068\u0069\u006a\u006b\u006c\u006d\u006e\u006f\u0070\u0071\u0072\u0073\u0074\u0075\u0076\u0077\u0078\u0079\u007a\u0251\u0250\u0252\u00e6\u0253\u0299\u03b2\u0254\u0255\u00e7\u0257\u0256\u00f0\u02a4\u0259\u0258\u025a\u025b\u025c\u025d\u025e\u025f\u0284\u0261\u0260\u0262\u029b\u0266\u0267\u0127\u0265\u029c\u0268\u026a\u029d\u026d\u026c\u026b\u026e\u029f\u0271\u026f\u0270\u014b\u0273\u0272\u0274\u00f8\u0275\u0278\u03b8\u0153\u0276\u0298\u0279\u027a\u027e\u027b\u0280\u0281\u027d\u0282\u0283\u0288\u02a7\u0289\u028a\u028b\u2c71\u028c\u0263\u0264\u028d\u03c7\u028e\u028f\u0291\u0290\u0292\u0294\u02a1\u0295\u02a2\u01c0\u01c1\u01c2\u01c3\u02c8\u02cc\u02d0\u02d1\u02bc\u02b4\u02b0\u02b1\u02b2\u02b7\u02e0\u02e4\u02de\u2193\u2191\u2192\u2197\u2198\u0027\u0329\u0027\u1d7b";

    internal const int PadId = 0, UnknownId = 1, BeginId = 2, EndId = 3;

    private static readonly Dictionary<char, int> SymbolIds = BuildSymbolIds();

    private static Dictionary<char, int> BuildSymbolIds()
    {
        // A symbol listed twice (the apostrophe) keeps its later index, as the release's dict does.
        var ids = new Dictionary<char, int>();
        for (int i = 0; i < Symbols.Length; i++) ids[Symbols[i]] = i + 4;
        return ids;
    }

    internal sealed record Part(ZonosConditioner Spec, BiasFreeLinearLayer<T>? Table, BiasFreeLinearLayer<T>? ProjectWeight,
        BiasFreeLinearLayer<T>? ProjectBias, BiasFreeLinearLayer<T>? Unconditional)
    {
        /// <summary>The Fourier features' frequencies <c>[width / 2, input]</c>: a buffer, drawn once and never trained.</summary>
        public Tensor<T>? Frequencies { get; set; }
    }

    public ZonosPrefixConditioner(IEngine engine, List<LayerBase<T>> layers, IReadOnlyList<ZonosConditioner> conditioners, int width, Random random)
    {
        _engine = engine;
        _width = width;
        if (width % 2 != 0) throw new ArgumentException("Fourier features need an even width.", nameof(width));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        foreach (var spec in conditioners)
        {
            BiasFreeLinearLayer<T>? table = spec.Kind switch
            {
                ZonosConditionerKind.EspeakPhonemes => Values((4 + Symbols.Length) * width),
                ZonosConditionerKind.Integer => Values(((int)spec.Max - (int)spec.Min + 1) * width),
                _ => null,
            };
            int condDim = spec.CondDim > 0 ? spec.CondDim : width;
            var part = new Part(spec, table, spec.Projected ? Add(new BiasFreeLinearLayer<T>(condDim, width)) : null,
                spec.Projected ? Values(width) : null, spec.LearnedUnconditional ? Values(width) : null);
            if (spec.Kind == ZonosConditionerKind.Fourier)
                part.Frequencies = new Tensor<T>(new[] { width / 2, spec.InputDim }, TorchParameters.Gaussian<T>(random, width / 2 * spec.InputDim, 1.0));
            Parts.Add(part);
        }
        ProjectWeight = Add(new BiasFreeLinearLayer<T>(width, width));
        ProjectBias = Values(width);
        NormWeight = Values(width);
        NormBias = Values(width);
    }

    public List<Part> Parts { get; } = new();
    public BiasFreeLinearLayer<T> ProjectWeight { get; }
    public BiasFreeLinearLayer<T> ProjectBias { get; }
    public BiasFreeLinearLayer<T> NormWeight { get; }
    public BiasFreeLinearLayer<T> NormBias { get; }

    /// <summary>The names of the conditioners that must be given (those without a learned unconditional vector).</summary>
    public IEnumerable<string> RequiredNames => Parts.Where(p => p.Unconditional is null).Select(p => p.Spec.Name);

    /// <summary>The phoneme ids of eSpeak's phonemes (<c>tokenize_phonemes</c>): start, each character's id (unknown
    /// characters 1), end.</summary>
    public static int[] PhonemeIds(string phonemes)
    {
        var ids = new List<int>(phonemes.Length + 2) { BeginId };
        foreach (char c in phonemes) ids.Add(SymbolIds.TryGetValue(c, out int id) ? id : UnknownId);
        ids.Add(EndId);
        return ids.ToArray();
    }

    private static Tensor<T> Value(BiasFreeLinearLayer<T> holder) => LlamaNetwork<T>.Paged(holder.Weights);

    private Tensor<T> Linear(BiasFreeLinearLayer<T> weight, BiasFreeLinearLayer<T> bias, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = weight.Forward(_engine.Reshape(x, new[] { rows, last }));
        y = _engine.TensorAdd(y, _engine.TensorBroadcastTo(_engine.Reshape(Value(bias), new[] { 1, _width }), new[] { rows, _width }));
        shape[^1] = _width;
        return _engine.Reshape(y, shape);
    }

    /// <summary>
    /// The prefix <c>[1, tokens, width]</c> of one set of inputs: each conditioner in order — its tokens for a given
    /// input, its unconditional vector for a missing one — then the projection and the LayerNorm.
    /// </summary>
    /// <param name="inputs">By conditioner name: the phoneme ids (<c>int[]</c>) for the phoneme conditioner, the vector
    /// (<c>double[]</c>) for the others (one value for a scalar, the integer as its value); absent for unconditional.</param>
    public Tensor<T> Forward(IReadOnlyDictionary<string, object> inputs, double epsilon = 1e-5)
    {
        var tokens = new List<Tensor<T>>();
        foreach (var part in Parts)
        {
            var spec = part.Spec;
            if (!inputs.TryGetValue(spec.Name, out var input) || input is null)
            {
                if (part.Unconditional is null) throw new ArgumentException($"Zonos needs the '{spec.Name}' input.", nameof(inputs));
                tokens.Add(_engine.Reshape(Value(part.Unconditional), new[] { 1, 1, _width }));
                continue;
            }
            switch (spec.Kind)
            {
                case ZonosConditionerKind.EspeakPhonemes:
                {
                    var ids = (int[])input;
                    var index = new Tensor<int>(new[] { ids.Length }, new Vector<int>(ids));
                    var table = _engine.Reshape(Value(part.Table!), new[] { 4 + Symbols.Length, _width });
                    tokens.Add(_engine.Reshape(_engine.TensorIndexSelect(table, index, 0), new[] { 1, ids.Length, _width }));
                    break;
                }
                case ZonosConditionerKind.Integer:
                {
                    int value = (int)Math.Round(((double[])input)[0]);
                    int count = (int)spec.Max - (int)spec.Min + 1, row = value - (int)spec.Min;
                    if (row < 0 || row >= count) throw new ArgumentOutOfRangeException(nameof(inputs), value, $"'{spec.Name}' lies in [{spec.Min}, {spec.Max}].");
                    var table = _engine.Reshape(Value(part.Table!), new[] { count, _width });
                    tokens.Add(_engine.Reshape(_engine.TensorIndexSelect(table, new Tensor<int>(new[] { 1 }, new Vector<int>(new[] { row })), 0), new[] { 1, 1, _width }));
                    break;
                }
                case ZonosConditionerKind.Fourier:
                {
                    var values = (double[])input;
                    if (values.Length != spec.InputDim) throw new ArgumentException($"'{spec.Name}' has {spec.InputDim} values.", nameof(inputs));
                    var scaled = new Tensor<T>(new[] { 1, spec.InputDim },
                        new Vector<T>(values.Select(v => NumOps.FromDouble(2 * Math.PI * (v - spec.Min) / (spec.Max - spec.Min))).ToArray()));
                    var f = _engine.TensorMatMul(scaled, _engine.TensorTranspose(part.Frequencies!));                    // [1, width/2]
                    tokens.Add(_engine.Reshape(_engine.TensorConcatenate(new[] { _engine.TensorCos(f), _engine.TensorSin(f) }, 1), new[] { 1, 1, _width }));
                    break;
                }
                default:
                {
                    var values = (double[])input;
                    int condDim = spec.CondDim > 0 ? spec.CondDim : _width;
                    if (values.Length != condDim) throw new ArgumentException($"'{spec.Name}' has {condDim} values.", nameof(inputs));
                    var x = new Tensor<T>(new[] { 1, 1, condDim }, new Vector<T>(values.Select(v => NumOps.FromDouble(v)).ToArray()));
                    tokens.Add(spec.Projected ? Linear(part.ProjectWeight!, part.ProjectBias!, x) : x);
                    break;
                }
            }
        }
        var joined = tokens.Count == 1 ? tokens[0] : _engine.TensorConcatenate(tokens.ToArray(), 1);
        var projected = Linear(ProjectWeight, ProjectBias, joined);
        // LayerNorm(width).
        var shape = projected._shape;
        var centered = _engine.TensorSubtract(projected, _engine.TensorBroadcastTo(_engine.ReduceMean(projected, new[] { 2 }, true), shape));
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centered, centered), new[] { 2 }, true);
        var normed = _engine.TensorMultiply(centered, _engine.TensorBroadcastTo(
            _engine.TensorPower(_engine.TensorAddScalar(variance, NumOps.FromDouble(epsilon)), NumOps.FromDouble(-0.5)), shape));
        Tensor<T> Row(BiasFreeLinearLayer<T> holder) => _engine.TensorBroadcastTo(_engine.Reshape(Value(holder), new[] { 1, 1, _width }), shape);
        return _engine.TensorAdd(_engine.TensorMultiply(normed, Row(NormWeight)), Row(NormBias));
    }

    /// <summary>PyTorch's defaults, as the release builds the modules: embeddings N(0, 1), linear weights and biases
    /// U(±1/√fan-in), the unconditional vectors 0, LayerNorm 1 and 0 (the Fourier frequencies were drawn at construction).</summary>
    public void InitializeLikeReference(Random random)
    {
        double[] Uniform(int count, int fanIn) => Enumerable.Range(0, count).Select(_ => (2 * random.NextDouble() - 1) / Math.Sqrt(fanIn)).ToArray();
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            if (name.EndsWith("embedder.weight", StringComparison.Ordinal)) return TorchParameters.Gaussian<double>(random, count, 1.0).ToArray();
            if (name.EndsWith("norm.weight", StringComparison.Ordinal)) return Enumerable.Repeat(1.0, count).ToArray();
            if (name.EndsWith("norm.bias", StringComparison.Ordinal) || name.EndsWith("uncond_vector", StringComparison.Ordinal)) return new double[count];
            var part = name.StartsWith("prefix_conditioner.conditioners.", StringComparison.Ordinal)
                ? Parts[int.Parse(name.Split('.')[2], System.Globalization.CultureInfo.InvariantCulture)] : null;
            if (part?.Spec.Kind == ZonosConditionerKind.Fourier && !name.Contains(".project.")) return part.Frequencies!.ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
            int fanIn = part is null ? _width : (part.Spec.CondDim > 0 ? part.Spec.CondDim : _width);
            return Uniform(count, fanIn);
        });
    }

    /// <summary>Loads the release's <c>prefix_conditioner.*</c> tensors; returns the names read.</summary>
    public HashSet<string> LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[] Read(string name, int[] shape)
        {
            used.Add(name);
            return read(name, shape);
        }
        void Vector(BiasFreeLinearLayer<T> holder, string name, int[] shape) =>
            TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, s) => a * s), Read(name, shape));
        for (int i = 0; i < Parts.Count; i++)
        {
            var part = Parts[i];
            string n = $"prefix_conditioner.conditioners.{i}";
            switch (part.Spec.Kind)
            {
                case ZonosConditionerKind.EspeakPhonemes:
                    Vector(part.Table!, $"{n}.phoneme_embedder.weight", new[] { 4 + Symbols.Length, _width });
                    break;
                case ZonosConditionerKind.Integer:
                    Vector(part.Table!, $"{n}.int_embedder.weight", new[] { (int)part.Spec.Max - (int)part.Spec.Min + 1, _width });
                    break;
                case ZonosConditionerKind.Fourier:
                {
                    var shape = new[] { _width / 2, part.Spec.InputDim };
                    part.Frequencies = new Tensor<T>(shape, new Vector<T>(Read($"{n}.weight", shape).Select(v => NumOps.FromDouble(v)).ToArray()));
                    break;
                }
            }
            if (part.ProjectWeight is not null)
            {
                int condDim = part.Spec.CondDim > 0 ? part.Spec.CondDim : _width;
                TorchParameters.Linear(part.ProjectWeight, condDim, _width, Read($"{n}.project.weight", new[] { _width, condDim }));
                Vector(part.ProjectBias!, $"{n}.project.bias", new[] { _width });
            }
            if (part.Unconditional is not null) Vector(part.Unconditional, $"{n}.uncond_vector", new[] { _width });
        }
        TorchParameters.Linear(ProjectWeight, _width, _width, Read("prefix_conditioner.project.weight", new[] { _width, _width }));
        Vector(ProjectBias, "prefix_conditioner.project.bias", new[] { _width });
        Vector(NormWeight, "prefix_conditioner.norm.weight", new[] { _width });
        Vector(NormBias, "prefix_conditioner.norm.bias", new[] { _width });
        return used;
    }
}
