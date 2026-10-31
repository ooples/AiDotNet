using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes of a Moshi transformer (the release's <c>StreamingTransformer</c> with RMSNorm, SiLU gating, interleaved
/// rotary positions or none, bias-free projections and, for the Depth Transformer, weights per step).</summary>
internal sealed record MoshiTransformerConfiguration
{
    public int Dim { get; init; } = 4096;
    public int Heads { get; init; } = 32;
    public int Layers { get; init; } = 32;

    /// <summary>The nominal feed-forward width (<c>hidden_scale · dim</c>, 4.125 · 4096 = 16896); the gate's hidden width is
    /// 2/3 of it (11264), or 21·dim/8 when it is exactly 4·dim.</summary>
    public int FeedForward { get; init; } = 16896;

    /// <summary>How many past positions (the current one included) each attends to; 0 for all (3000 for Helium in Moshi).</summary>
    public int Context { get; init; } = 3000;

    /// <summary>Whether positions are rotary (interleaved pairs, period 10000); the Depth Transformer has none.</summary>
    public bool Rotary { get; init; } = true;

    public double RopeMaxPeriod { get; init; } = 10000;

    /// <summary>Separate projections and feed-forward per position (the Depth Transformer's <c>weights_per_step</c>); 1 for
    /// shared weights.</summary>
    public int WeightsPerStep { get; init; } = 1;

    /// <summary>The RMSNorm ε (<c>rms_norm_f32</c>: 1e-8).</summary>
    public double NormEpsilon { get; init; } = 1e-8;

    public int GateHidden => FeedForward == 4 * Dim ? 21 * Dim / 8 : 2 * FeedForward / 3;
}

/// <summary>
/// A Moshi transformer (Défossez et al. 2024 §3.2, §3.4.1; the release's <c>StreamingTransformer</c>): pre-norm layers of
/// RMSNorm, causal self-attention from one bias-free projection laid out (part, head, dim) with interleaved rotary
/// positions, and a SiLU-gated feed-forward block. <see cref="MoshiTransformerConfiguration.WeightsPerStep"/> gives each
/// position its own projections and feed-forward (the Depth Transformer's per-codebook parameters).
/// </summary>
internal sealed class MoshiTransformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Layer(BiasFreeLinearLayer<T> Norm1, BiasFreeLinearLayer<T>[] InProj, BiasFreeLinearLayer<T>[] OutProj,
        BiasFreeLinearLayer<T> Norm2, BiasFreeLinearLayer<T>[] GateIn, BiasFreeLinearLayer<T>[] GateOut);

    /// <summary>A streaming cache: each layer's keys and values <c>[heads, positions, head dim]</c> and the position count.</summary>
    internal sealed class Cache
    {
        public readonly List<(Tensor<T> Keys, Tensor<T> Values)?> Layers = new();
        public int Offset;
    }

    public MoshiTransformer(IEngine engine, List<LayerBase<T>> layers, MoshiTransformerConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.Heads != 0 || (c.Dim / c.Heads) % 2 != 0) throw new ArgumentException("The width must divide into heads of even width.");
        BiasFreeLinearLayer<T> Add(BiasFreeLinearLayer<T> layer) { layers.Add(layer); return layer; }
        BiasFreeLinearLayer<T>[] Many(int inputs, int outputs) => Enumerable.Range(0, c.WeightsPerStep).Select(_ => Add(new BiasFreeLinearLayer<T>(inputs, outputs))).ToArray();
        for (int i = 0; i < c.Layers; i++)
        {
            var norm1 = Add(new BiasFreeLinearLayer<T>(1, c.Dim));
            var inProj = Many(c.Dim, 3 * c.Dim);
            var outProj = Many(c.Dim, c.Dim);
            var norm2 = Add(new BiasFreeLinearLayer<T>(1, c.Dim));
            Layers.Add(new Layer(norm1, inProj, outProj, norm2, Many(c.Dim, 2 * c.GateHidden), Many(c.GateHidden, c.Dim)));
        }
        foreach (var l in Layers)
        {
            Fill(l.Norm1, 1.0);
            Fill(l.Norm2, 1.0);
        }
    }

    public MoshiTransformerConfiguration Configuration { get; }
    public List<Layer> Layers { get; } = new();

    /// <summary>Every layer the Transformer owns.</summary>
    public IEnumerable<LayerBase<T>> AllLayers => Layers.SelectMany(l => new LayerBase<T>[] { l.Norm1, l.Norm2 }
        .Concat(l.InProj).Concat(l.OutProj).Concat(l.GateIn).Concat(l.GateOut));

    private void Fill(BiasFreeLinearLayer<T> holder, double value)
    {
        var w = holder.Weights;
        for (int i = 0; i < w.Length; i++) w[i] = NumOps.FromDouble(value);
        _engine.InvalidatePersistentTensor(w);
    }

    internal static Tensor<T> Value(BiasFreeLinearLayer<T> holder) => holder.Weights;

    /// <summary>RMSNorm (<c>x · α / √(mean(x²) + ε)</c>) of rows <c>[L, D]</c>.</summary>
    internal Tensor<T> RmsNorm(BiasFreeLinearLayer<T> alpha, Tensor<T> x)
    {
        var meanSquare = _engine.ReduceMean(_engine.TensorMultiply(x, x), new[] { 1 }, true);
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(Configuration.NormEpsilon)), NumOps.FromDouble(-0.5));
        var shape = x.Shape.ToArray();
        return _engine.TensorMultiply(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(inverse, shape)),
            _engine.TensorBroadcastTo(_engine.Reshape(Value(alpha), new[] { 1, shape[1] }), shape));
    }

    // Rows through per-step linears: row t uses linears[min(offset + t, steps − 1)] (one linear when shared).
    private Tensor<T> PerStep(BiasFreeLinearLayer<T>[] linears, Tensor<T> x, int offset)
    {
        if (linears.Length == 1) return linears[0].Forward(x);
        int rows = x.Shape[0];
        var parts = new Tensor<T>[rows];
        for (int t = 0; t < rows; t++)
        {
            int index = offset + t;
            if (index >= linears.Length) throw new InvalidOperationException($"Position {index} has no weights (the transformer holds {linears.Length} steps).");
            parts[t] = linears[index].Forward(_engine.TensorSlice(x, new[] { t, 0 }, new[] { 1, x.Shape[1] }));
        }
        return rows == 1 ? parts[0] : _engine.TensorConcatenate(parts, 0);
    }

    // Interleaved rotary embedding of heads [H, L, hd] at positions offset … offset + L − 1.
    private Tensor<T> Rotate(Tensor<T> x, int offset)
    {
        int h = x.Shape[0], l = x.Shape[1], hd = x.Shape[2], half = hd / 2;
        var cos = new Tensor<T>(new[] { 1, l, half, 1 });
        var sin = new Tensor<T>(new[] { 1, l, half, 1 });
        for (int t = 0; t < l; t++)
            for (int d = 0; d < half; d++)
            {
                double angle = (offset + t) * Math.Exp(-Math.Log(Configuration.RopeMaxPeriod) * 2.0 * d / hd);
                cos[0, t, d, 0] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, t, d, 0] = NumOps.FromDouble(Math.Sin(angle));
            }
        var pairs = _engine.Reshape(x, new[] { h, l, half, 2 });
        var real = _engine.TensorSlice(pairs, new[] { 0, 0, 0, 0 }, new[] { h, l, half, 1 });
        var imaginary = _engine.TensorSlice(pairs, new[] { 0, 0, 0, 1 }, new[] { h, l, half, 1 });
        var shape = new[] { h, l, half, 1 };
        var c = _engine.TensorBroadcastTo(cos, shape);
        var s = _engine.TensorBroadcastTo(sin, shape);
        var outReal = _engine.TensorSubtract(_engine.TensorMultiply(real, c), _engine.TensorMultiply(imaginary, s));
        var outImaginary = _engine.TensorAdd(_engine.TensorMultiply(real, s), _engine.TensorMultiply(imaginary, c));
        return _engine.Reshape(_engine.TensorConcatenate(new[] { outReal, outImaginary }, 3), new[] { h, l, hd });
    }

    /// <summary>
    /// The transformer on rows <c>[L, D]</c> at positions <c>offset …</c> (the cache's, or <paramref name="offset"/>
    /// without one): each attends causally to itself and the positions before it within the context.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, Cache? cache = null, int offset = 0)
    {
        var c = Configuration;
        int l = x.Shape[0], heads = c.Heads, hd = c.Dim / heads;
        if (cache is not null)
        {
            offset = cache.Offset;
            while (cache.Layers.Count < Layers.Count) cache.Layers.Add(null);
        }
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var h = RmsNorm(layer.Norm1, x);
            var projected = PerStep(layer.InProj, h, offset);                                                   // [L, 3D]
            Tensor<T> Part(int p) => _engine.TensorPermute(_engine.Reshape(_engine.TensorSlice(projected, new[] { 0, p * c.Dim }, new[] { l, c.Dim }),
                new[] { l, heads, hd }), new[] { 1, 0, 2 });
            var q = Part(0);
            var k = Part(1);
            var v = Part(2);
            if (c.Rotary)
            {
                q = Rotate(q, offset);
                k = Rotate(k, offset);
            }
            int past = 0;
            if (cache is not null)
            {
                if (cache.Layers[i] is { } held)
                {
                    past = held.Keys.Shape[1];
                    k = _engine.TensorConcatenate(new[] { held.Keys, k }, 1);
                    v = _engine.TensorConcatenate(new[] { held.Values, v }, 1);
                }
                // Keys older than the context are never read again.
                int keep = c.Context > 0 ? Math.Min(k.Shape[1], c.Context) : k.Shape[1];
                int drop = k.Shape[1] - keep;
                cache.Layers[i] = (drop > 0 ? _engine.TensorSlice(k, new[] { 0, drop, 0 }, new[] { heads, keep, hd }) : k,
                    drop > 0 ? _engine.TensorSlice(v, new[] { 0, drop, 0 }, new[] { heads, keep, hd }) : v);
            }
            int keys = k.Shape[1];
            var bias = new Tensor<T>(new[] { 1, l, keys });
            var blocked = NumOps.FromDouble(double.NegativeInfinity);
            for (int qi = 0; qi < l; qi++)
                for (int ki = 0; ki < keys; ki++)
                {
                    int delta = (past + qi) - ki;
                    if (delta < 0 || (c.Context > 0 && delta >= c.Context)) bias[0, qi, ki] = blocked;
                }
            var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(bias, new[] { heads, l, keys }));
            var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v), new[] { 1, 0, 2 }), new[] { l, c.Dim });
            x = _engine.TensorAdd(x, PerStep(layer.OutProj, context, offset));
            var g = PerStep(layer.GateIn, RmsNorm(layer.Norm2, x), offset);                                    // [L, 2H]
            int hidden = c.GateHidden;
            var gated = _engine.TensorMultiply(_engine.Swish(_engine.TensorSlice(g, new[] { 0, 0 }, new[] { l, hidden })),
                _engine.TensorSlice(g, new[] { 0, hidden }, new[] { l, hidden }));
            x = _engine.TensorAdd(x, PerStep(layer.GateOut, gated, offset));
        }
        if (cache is not null) cache.Offset += l;
        return x;
    }

    /// <summary>Loads the release's names under <paramref name="prefix"/> (<c>layers.{i}.norm1.alpha</c>,
    /// <c>self_attn.in_projs.{s}.weight</c>, <c>self_attn.out_projs.{s}.weight</c>, <c>norm2.alpha</c>,
    /// <c>gating.{s}.linear_in/linear_out.weight</c>; shared weights without the step index).</summary>
    public void LoadTorchWeights(Func<string, double[]?> take, string prefix)
    {
        var c = Configuration;
        double[] Read(string name) => take(prefix + name) ?? throw new InvalidDataException($"The Moshi checkpoint has no tensor '{prefix}{name}'.");
        void Load(BiasFreeLinearLayer<T> layer, int inputs, int outputs, string name) => TorchParameters.Linear(layer, inputs, outputs, Read(name));
        void LoadValues(BiasFreeLinearLayer<T> holder, string name)
        {
            var values = Read(name);
            var w = holder.Weights;
            for (int i = 0; i < values.Length; i++) w[i] = NumOps.FromDouble(values[i]);
            _engine.InvalidatePersistentTensor(w);
        }
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            string p = $"layers.{i}.";
            LoadValues(l.Norm1, p + "norm1.alpha");
            LoadValues(l.Norm2, p + "norm2.alpha");
            for (int s = 0; s < c.WeightsPerStep; s++)
            {
                Load(l.InProj[s], c.Dim, 3 * c.Dim, p + $"self_attn.in_projs.{s}.weight");
                Load(l.OutProj[s], c.Dim, c.Dim, p + $"self_attn.out_projs.{s}.weight");
                string gate = c.WeightsPerStep == 1 ? "gating." : $"gating.{s}.";
                Load(l.GateIn[s], c.Dim, 2 * c.GateHidden, p + gate + "linear_in.weight");
                Load(l.GateOut[s], c.GateHidden, c.Dim, p + gate + "linear_out.weight");
            }
        }
    }

    /// <summary>The release's initialization: every linear truncated-normal with std 1/√fan_in at ±3 std (the Temporal
    /// Transformer's <c>_init_layer</c>), or torch's default U(±1/√fan_in) (<paramref name="truncatedNormal"/> false, the
    /// Depth Transformer, which the release leaves at its default).</summary>
    public void Initialize(Random random, bool truncatedNormal)
    {
        var c = Configuration;
        void Init(BiasFreeLinearLayer<T> layer, int inputs, int outputs)
        {
            double std = 1.0 / Math.Sqrt(inputs);
            var w = new double[inputs * outputs];
            for (int i = 0; i < w.Length; i++) w[i] = truncatedNormal ? MoshiLanguageModel<T>.TruncatedNormal(random, std) : (2 * random.NextDouble() - 1) * std;
            TorchParameters.Linear(layer, inputs, outputs, w);
        }
        foreach (var l in Layers)
            for (int s = 0; s < c.WeightsPerStep; s++)
            {
                Init(l.InProj[s], c.Dim, 3 * c.Dim);
                Init(l.OutProj[s], c.Dim, c.Dim);
                Init(l.GateIn[s], c.Dim, 2 * c.GateHidden);
                Init(l.GateOut[s], c.GateHidden, c.Dim);
            }
    }
}
