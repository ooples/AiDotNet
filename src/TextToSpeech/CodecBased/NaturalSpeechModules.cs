using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>How NaturalSpeech's learnable upsampling layer is built.</summary>
public enum NaturalSpeechUpsampling
{
    /// <summary>The paper's (App. B, after Parallel Tacotron 2): a linear projection of the phoneme hidden states, a
    /// convolution to 8 channels with layer normalization and Swish, two-layer Swish MLPs 10 → q and 10 → p, and
    /// O = Proj(WH) + Proj(Einsum(W, C)) followed by the mean and variance linear layers.</summary>
    Paper,

    /// <summary>heatz123/naturalspeech's: no input projection, a convolution with batch normalization, SiLU and layer
    /// normalization, three-layer Swish MLPs, layer normalization of O and one linear layer to the mean and variance.</summary>
    Heatz123,
}

/// <summary>NaturalSpeech's phoneme encoder.</summary>
public enum NaturalSpeechPhonemeEncoder
{
    /// <summary>The paper's six feed-forward Transformer blocks (FastSpeech, absolute sinusoidal positions).</summary>
    FeedForwardTransformer,

    /// <summary>heatz123's VITS encoder (relative-position attention).</summary>
    Vits,
}

/// <summary>
/// NaturalSpeech's duration predictor (App. B): convolutions with ReLU, layer normalization and dropout, then a projection
/// to the log duration, its input detached (the gradient stop of App. F). The paper uses 3 layers of 192 filters;
/// heatz123 uses 2 of 256.
/// </summary>
internal sealed class NaturalSpeechDurationPredictor<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;
    private readonly List<(NormedConv1DLayer<T> Conv, LayerNormalizationLayer<T> Norm)> _layers = new();

    public NaturalSpeechDurationPredictor(IEngine engine, List<LayerBase<T>> layers, int inChannels, int filter, int kernel, int count, double dropout)
    {
        _engine = engine;
        _dropout = dropout;
        for (int i = 0; i < count; i++)
            _layers.Add((MeloOps.Conv(layers, i == 0 ? inChannels : filter, filter, kernel, 1, kernel / 2),
                IndexTtsOps.LayerNorm(layers, filter, 1e-5)));
        Projection = MeloOps.Conv(layers, filter, 1, 1);
    }

    public NormedConv1DLayer<T> Projection { get; }

    /// <summary>The log durations <c>[1, 1, n]</c> of the phoneme hidden states <c>[1, h, n]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> hidden, bool training, Random random)
    {
        var x = MeloOps.Detach(hidden);
        foreach (var (conv, norm) in _layers)
        {
            x = _engine.ReLU(conv.Forward(x));
            x = NaturalSpeechOps.ChannelNorm(_engine, norm, x);
            if (training && _dropout > 0) x = T5Seq2Seq<T>.Dropout(_engine, x, _dropout, random);
        }
        return Projection.Forward(x);
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        for (int i = 0; i < _layers.Count; i++)
        {
            MeloOps.LoadConv(read, _layers[i].Conv, $"{prefix}conv_{i + 1}");
            AiDotNet.TextToSpeech.CodecBased.TorchParameters.LayerNorm(_engine, _layers[i].Norm, read($"{prefix}norm_{i + 1}.gamma"), read($"{prefix}norm_{i + 1}.beta"));
        }
        MeloOps.LoadConv(read, Projection, prefix + "proj");
    }
}

/// <summary>
/// NaturalSpeech's learnable upsampling layer (App. B, Eqs. 6-9, after Parallel Tacotron 2): from the durations d, the
/// start and end matrices S (frame i less the phonemes' start) and E (end less i); attention weights W = softmax over the
/// phonemes of an MLP of [S, E, conv features] and auxiliary context C; the frame-level hidden states
/// O = Proj(W H) + Proj(Einsum(W, C)); the prior's mean and log-scale from O.
/// </summary>
internal sealed class NaturalSpeechUpsampler<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly NaturalSpeechUpsampling _variant;
    private readonly int _hidden, _convOut, _q, _p, _out;

    public NaturalSpeechUpsampler(IEngine engine, List<LayerBase<T>> layers, Random random, NaturalSpeechUpsampling variant, int hidden, int kernel,
        int convOut, int q, int p, int outChannels)
    {
        _engine = engine;
        _variant = variant;
        _hidden = hidden;
        _convOut = convOut;
        _q = q;
        _p = p;
        _out = outChannels;
        bool paper = variant == NaturalSpeechUpsampling.Paper;
        if (paper) InputProjection = IndexTtsOps.Linear(engine, layers, random, hidden, hidden);
        ConvW = MeloOps.Conv(layers, hidden, convOut, kernel, 1, (kernel - 1) / 2);
        ConvC = MeloOps.Conv(layers, hidden, convOut, kernel, 1, (kernel - 1) / 2);
        NormW = IndexTtsOps.LayerNorm(layers, convOut, 1e-5);
        NormC = IndexTtsOps.LayerNorm(layers, convOut, 1e-5);
        if (!paper)
        {
            BatchNormW = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, 4 * convOut).ToArray());
            BatchNormC = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, 4 * convOut).ToArray());
        }
        // The MLPs: [S, E, conv] (2 + convOut) → q or p; the paper's has two layers, heatz123's three.
        MlpW = Mlp(layers, random, 2 + convOut, q, q, paper ? 2 : 3);
        MlpC = Mlp(layers, random, 2 + convOut, p, p, paper ? 2 : 3);
        ProjectionW = IndexTtsOps.Linear(engine, layers, random, q * hidden, hidden);
        ProjectionC = IndexTtsOps.Linear(engine, layers, random, q * p, hidden);
        if (paper)
        {
            Mean = IndexTtsOps.Linear(engine, layers, random, hidden, outChannels);
            LogScale = IndexTtsOps.Linear(engine, layers, random, hidden, outChannels);
        }
        else
        {
            OutputNorm = IndexTtsOps.LayerNorm(layers, hidden, 1e-5);
            Stats = IndexTtsOps.Linear(engine, layers, random, hidden, 2 * outChannels);
        }
    }

    private List<DenseLayer<T>> Mlp(List<LayerBase<T>> layers, Random random, int inputs, int hidden, int outputs, int count)
    {
        var mlp = new List<DenseLayer<T>>();
        for (int i = 0; i < count; i++)
            mlp.Add(IndexTtsOps.Linear(_engine, layers, random, i == 0 ? inputs : hidden, i == count - 1 ? outputs : hidden));
        return mlp;
    }

    public DenseLayer<T>? InputProjection { get; }
    public NormedConv1DLayer<T> ConvW { get; }
    public NormedConv1DLayer<T> ConvC { get; }
    public LayerNormalizationLayer<T> NormW { get; }
    public LayerNormalizationLayer<T> NormC { get; }

    /// <summary>heatz123's batch normalizations (running statistics in evaluation): weight, bias, running mean and
    /// variance, each <c>convOut</c> long.</summary>
    public BiasFreeLinearLayer<T>? BatchNormW { get; }
    public BiasFreeLinearLayer<T>? BatchNormC { get; }
    public List<DenseLayer<T>> MlpW { get; }
    public List<DenseLayer<T>> MlpC { get; }
    public DenseLayer<T> ProjectionW { get; }
    public DenseLayer<T> ProjectionC { get; }
    public DenseLayer<T>? Mean { get; }
    public DenseLayer<T>? LogScale { get; }
    public LayerNormalizationLayer<T>? OutputNorm { get; }
    public DenseLayer<T>? Stats { get; }

    private Tensor<T> Swish(Tensor<T> x) => _engine.Swish(x);

    // [1, h, n] through the convolution block: rows [n, convOut].
    private Tensor<T> ConvFeatures(Tensor<T> hiddenRows, NormedConv1DLayer<T> conv, LayerNormalizationLayer<T> norm, BiasFreeLinearLayer<T>? batchNorm)
    {
        int n = hiddenRows.Shape[0];
        var x = _engine.Reshape(_engine.TensorTranspose(hiddenRows), new[] { 1, _hidden, n });
        var y = conv.Forward(x);                                                                        // [1, convOut, n]
        var rows = _engine.TensorTranspose(_engine.Reshape(y, new[] { _convOut, n }));                   // [n, convOut]
        if (batchNorm is not null)
        {
            // BatchNorm1d in evaluation (the release's checkpoint statistics), then SiLU.
            var bn = _engine.Reshape(IndexTtsOps.Value(batchNorm), new[] { 4 * _convOut });
            var gamma = _engine.TensorNarrow(bn, 0, 0, _convOut);
            var beta = _engine.TensorNarrow(bn, 0, _convOut, _convOut);
            var mean = MeloOps.Detach(_engine.TensorNarrow(bn, 0, 2 * _convOut, _convOut));
            var variance = MeloOps.Detach(_engine.TensorNarrow(bn, 0, 3 * _convOut, _convOut));
            var scale = _engine.TensorDivide(gamma, _engine.TensorSqrt(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-5))));
            var shape = new[] { n, _convOut };
            rows = _engine.TensorAdd(_engine.TensorMultiply(_engine.TensorSubtract(rows, _engine.TensorBroadcastTo(_engine.Reshape(mean, new[] { 1, _convOut }), shape)),
                _engine.TensorBroadcastTo(_engine.Reshape(scale, new[] { 1, _convOut }), shape)), _engine.TensorBroadcastTo(_engine.Reshape(beta, new[] { 1, _convOut }), shape));
            return norm.Forward(Swish(rows));
        }
        // The paper: layer normalization, then Swish.
        return Swish(norm.Forward(rows));
    }

    private Tensor<T> RunMlp(List<DenseLayer<T>> mlp, Tensor<T> x)
    {
        for (int i = 0; i < mlp.Count; i++)
        {
            x = mlp[i].Forward(x);
            if (i < mlp.Count - 1) x = Swish(x);
        }
        return x;
    }

    /// <summary>
    /// The frame-level prior <c>(mean, log-scale)</c>, each <c>[1, out, m]</c> with m = round(Σd), and the attention
    /// weights, of phoneme hidden states <c>[1, h, n]</c> and durations <c>[n]</c> (on the tape).
    /// </summary>
    public (Tensor<T> Mean, Tensor<T> LogScale, int Frames) Forward(Tensor<T> hidden, Tensor<T> durations, bool training, Random random)
    {
        int n = hidden.Shape[2];
        double total = 0;
        for (int i = 0; i < n; i++) total += NumOps.ToDouble(durations.GetFlat(i));
        int m = Math.Max(1, (int)Math.Round(total));
        var rows = _engine.TensorTranspose(_engine.Reshape(hidden, new[] { _hidden, n }));              // H [n, h]
        var values = InputProjection is not null ? InputProjection.Forward(rows) : rows;
        // Token boundary grid: e_k = cumsum(d), s_k = e_k − d; S[t, k] = t − s_k, E[t, k] = e_k − t for t = 1..m.
        var ones = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) ones[i, j] = NumOps.One;
        var d = _engine.Reshape(durations, new[] { 1, n });
        var e = _engine.TensorMatMul(d, _engine.TensorTranspose(ones));                                // [1, n] cumulative
        var s = _engine.TensorSubtract(e, d);
        var t = new Tensor<T>(new[] { m, 1 });
        for (int i = 0; i < m; i++) t[i, 0] = NumOps.FromDouble(i + 1);
        var grid = new[] { m, n };
        var S = _engine.TensorSubtract(_engine.TensorBroadcastTo(t, grid), _engine.TensorBroadcastTo(s, grid));
        var E = _engine.TensorSubtract(_engine.TensorBroadcastTo(e, grid), _engine.TensorBroadcastTo(t, grid));
        Tensor<T> Inputs(Tensor<T> conv) => _engine.TensorConcatenate(new[]
        {
            _engine.Reshape(S, new[] { m, n, 1 }), _engine.Reshape(E, new[] { m, n, 1 }),
            _engine.TensorBroadcastTo(_engine.Reshape(conv, new[] { 1, n, _convOut }), new[] { m, n, _convOut }),
        }, 2);                                                                                         // [m, n, 2 + convOut]
        var inputW = _engine.Reshape(Inputs(ConvFeatures(values, ConvW, NormW, BatchNormW)), new[] { m * n, 2 + _convOut });
        var inputC = _engine.Reshape(Inputs(ConvFeatures(values, ConvC, NormC, BatchNormC)), new[] { m * n, 2 + _convOut });
        var w = _engine.Reshape(RunMlp(MlpW, inputW), new[] { m, n, _q });
        var c = _engine.Reshape(RunMlp(MlpC, inputC), new[] { m, n, _p });
        w = _engine.Softmax(w, axis: 1);                                                               // over the phonemes
        // W H: per q, [m, n] · [n, h] → [m, q·h]; Einsum(W, C): per q, Σ_n W[m,n,q] C[m,n,p] → [m, q·p].
        var wq = _engine.TensorPermute(w, new[] { 2, 0, 1 });                                         // [q, m, n]
        var wh = _engine.BatchMatMul(wq, _engine.TensorBroadcastTo(_engine.Reshape(rows, new[] { 1, n, _hidden }), new[] { _q, n, _hidden }));
        var whRows = _engine.Reshape(_engine.TensorPermute(wh, new[] { 1, 0, 2 }), new[] { m, _q * _hidden });
        var wPerFrame = _engine.TensorPermute(w, new[] { 0, 2, 1 });                                  // [m, q, n]
        var wc = _engine.BatchMatMul(wPerFrame, c);                                                   // [m, q, p]
        var wcRows = _engine.Reshape(wc, new[] { m, _q * _p });
        var o = _engine.TensorAdd(ProjectionW.Forward(whRows), ProjectionC.Forward(wcRows));           // [m, h]
        Tensor<T> mean, logs;
        if (Stats is not null)
        {
            var stats = Stats.Forward(OutputNorm!.Forward(o));
            mean = _engine.TensorNarrow(stats, 1, 0, _out);
            logs = _engine.TensorNarrow(stats, 1, _out, _out);
        }
        else
        {
            mean = Mean!.Forward(o);
            logs = LogScale!.Forward(o);
        }
        Tensor<T> Channels(Tensor<T> r) => _engine.Reshape(_engine.TensorTranspose(r), new[] { 1, _out, m });
        return (Channels(mean), Channels(logs), m);
    }

    /// <summary>Loads heatz123/naturalspeech's <c>LearnableUpsampling</c> (names after <paramref name="prefix"/>).</summary>
    public void LoadHeatz123(Func<string, double[]?> read, string prefix)
    {
        double[] Require(string n) => read(n) ?? throw new KeyNotFoundException($"The checkpoint has no '{n}'.");
        if (BatchNormW is null) throw new InvalidOperationException("heatz123's checkpoint has its learnable upsampling; build with NaturalSpeechUpsampling.Heatz123.");
        void ConvBlock(string name, NormedConv1DLayer<T> conv, BiasFreeLinearLayer<T> bn, LayerNormalizationLayer<T> norm)
        {
            MeloOps.LoadConv(Require, conv, name + "conv_layer.0.conv");
            var stats = Require(name + "conv_layer.1.weight").Concat(Require(name + "conv_layer.1.bias"))
                .Concat(Require(name + "conv_layer.1.running_mean")).Concat(Require(name + "conv_layer.1.running_var")).ToArray();
            TorchParameters.Linear(bn, 1, stats.Length, stats);
            TorchParameters.LayerNorm(_engine, norm, Require(name + "layer_norm.weight"), Require(name + "layer_norm.bias"));
        }
        ConvBlock(prefix + "conv_w.", ConvW, BatchNormW, NormW);
        ConvBlock(prefix + "conv_c.", ConvC, BatchNormC!, NormC);
        void Mlp(string name, List<DenseLayer<T>> mlp)
        {
            for (int i = 0; i < mlp.Count; i++)
            {
                var w = Require($"{name}layer.{2 * i}.linear.weight");
                int outputs = Require($"{name}layer.{2 * i}.linear.bias").Length;
                TorchParameters.Linear(_engine, mlp[i], w.Length / outputs, outputs, w, Require($"{name}layer.{2 * i}.linear.bias"));
            }
        }
        Mlp(prefix + "swish_w.", MlpW);
        Mlp(prefix + "swish_c.", MlpC);
        TorchParameters.Linear(_engine, ProjectionW, _q * _hidden, _hidden, Require(prefix + "linear_w.linear.weight"),
            Require(prefix + "linear_w.linear.bias"));
        TorchParameters.Linear(_engine, ProjectionC, _q * _p, _hidden, Require(prefix + "linear_einsum.linear.weight"),
            read(prefix + "linear_einsum.linear.bias"));
        TorchParameters.LayerNorm(_engine, OutputNorm!, Require(prefix + "layer_norm.weight"), Require(prefix + "layer_norm.bias"));
        TorchParameters.Linear(_engine, Stats!, _hidden, 2 * _out, Require(prefix + "proj_o.linear.weight"),
            read(prefix + "proj_o.linear.bias"));
    }
}


/// <summary>
/// NaturalSpeech's VAE memory (§3.5): the posterior z as the query of multi-head attention over a learned memory bank
/// M ∈ ℝ^{L×h} (keys and values), whose output the waveform decoder reads.
/// </summary>
internal sealed class NaturalSpeechMemoryBank<T>
{
    private readonly IEngine _engine;
    private readonly int _hidden, _size, _heads;

    public NaturalSpeechMemoryBank(IEngine engine, List<LayerBase<T>> layers, Random random, int hidden, int size, int heads)
    {
        _engine = engine;
        _hidden = hidden;
        _size = size;
        _heads = heads;
        Bank = IndexTtsOps.Values(layers, Enumerable.Range(0, hidden * size).Select(_ => IndexTtsOps.Gaussian(random)).ToArray());
        Query = MeloOps.Conv(layers, hidden, hidden, 1);
        Key = MeloOps.Conv(layers, hidden, hidden, 1);
        Value = MeloOps.Conv(layers, hidden, hidden, 1);
        Output = MeloOps.Conv(layers, hidden, hidden, 1);
    }

    /// <summary>The memory bank, stored <c>[h, L]</c> as the release's parameter.</summary>
    public BiasFreeLinearLayer<T> Bank { get; }
    public NormedConv1DLayer<T> Query { get; }
    public NormedConv1DLayer<T> Key { get; }
    public NormedConv1DLayer<T> Value { get; }
    public NormedConv1DLayer<T> Output { get; }

    /// <summary>Attention(z, M, M) for z <c>[1, h, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> z)
    {
        int frames = z.Shape[2], dk = _hidden / _heads;
        var bank = _engine.Reshape(IndexTtsOps.Value(Bank), new[] { 1, _hidden, _size });
        Tensor<T> Heads(Tensor<T> x, int length) =>
            _engine.TensorPermute(_engine.Reshape(x, new[] { _heads, dk, length }), new[] { 0, 2, 1 });      // [heads, len, dk]
        var q = Heads(Query.Forward(z), frames);
        var k = Heads(Key.Forward(bank), _size);
        var v = Heads(Value.Forward(bank), _size);
        var attended = IndexTtsOps.Attend(_engine, q, k, v, 1.0 / Math.Sqrt(dk), null);               // [heads, T, dk]
        var merged = _engine.Reshape(_engine.TensorPermute(attended, new[] { 0, 2, 1 }), new[] { 1, _hidden, frames });
        return Output.Forward(merged);
    }

    /// <summary>Sets the bank to cluster centres <c>[L][h]</c> (the paper's K-means initialization).</summary>
    public void SetBank(double[][] centres)
    {
        if (centres.Length != _size || centres.Any(c => c.Length != _hidden)) throw new ArgumentException($"The bank is {_size} rows of {_hidden}.", nameof(centres));
        var stored = new double[_hidden * _size];
        for (int l = 0; l < _size; l++)
            for (int c = 0; c < _hidden; c++) stored[c * _size + l] = centres[l][c];
        TorchParameters.Linear(Bank, 1, stored.Length, stored);
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        TorchParameters.Linear(Bank, 1, _hidden * _size, read(prefix + "memory_bank"));
        MeloOps.LoadConv(read, Query, prefix + "encoder.conv_q");
        MeloOps.LoadConv(read, Key, prefix + "encoder.conv_k");
        MeloOps.LoadConv(read, Value, prefix + "encoder.conv_v");
        MeloOps.LoadConv(read, Output, prefix + "encoder.conv_o");
    }
}

/// <summary>NaturalSpeech's loss helpers.</summary>
internal static class NaturalSpeechOps
{
    /// <summary>A channel LayerNorm (VITS's <c>modules.LayerNorm</c>) over <c>[1, C, T]</c>.</summary>
    public static Tensor<T> ChannelNorm<T>(IEngine engine, LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var rows = engine.TensorTranspose(engine.Reshape(x, new[] { c, t }));
        return engine.Reshape(engine.TensorTranspose(norm.Forward(rows)), new[] { 1, c, t });
    }

    /// <summary>
    /// The pairwise KL matrix <c>[m, n]</c> of sample-based Gaussian KLs (VITS's <c>kl_loss</c> per frame pair, summed over
    /// channels): <c>Σ_c logs_p[j] − logs_q[i] − ½ + ½ (z[i] − m_p[j])² e^{−2 logs_p[j]}</c> for samples z <c>[1, C, m]</c>
    /// with their log-scales and a distribution <c>[1, C, n]</c>.
    /// </summary>
    public static Tensor<T> KlMatrix<T>(IEngine engine, Tensor<T> z, Tensor<T> logsQ, Tensor<T> meanP, Tensor<T> logsP)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int c = z.Shape[1], m = z.Shape[2], n = meanP.Shape[2];
        Tensor<T> Rows(Tensor<T> x, int length) => engine.TensorTranspose(engine.Reshape(x, new[] { c, length }));   // [len, C]
        var zr = Rows(z, m);
        var lq = Rows(logsQ, m);
        var mp = Rows(meanP, n);
        var lp = Rows(logsP, n);
        var inv = engine.TensorExp(engine.TensorMultiplyScalar(lp, ops.FromDouble(-2.0)));                            // [n, C]
        // ½ Σ_c (z² − 2 z m + m²) e^{−2 logs_p} = ½ [z² · invᵀ − 2 z · (m inv)ᵀ + 1 · (m² inv)ᵀ]
        var zz = engine.TensorMatMul(engine.TensorMultiply(zr, zr), engine.TensorTranspose(inv));
        var zm = engine.TensorMatMul(zr, engine.TensorTranspose(engine.TensorMultiply(mp, inv)));
        var mm = engine.ReduceSum(engine.TensorMultiply(engine.TensorMultiply(mp, mp), inv), new[] { 1 }, keepDims: false);  // [n]
        var quadratic = engine.TensorMultiplyScalar(engine.TensorAdd(engine.TensorSubtract(zz, engine.TensorMultiplyScalar(zm, ops.FromDouble(2))),
            engine.TensorBroadcastTo(engine.Reshape(mm, new[] { 1, n }), new[] { m, n })), ops.FromDouble(0.5));
        var sumP = engine.Reshape(engine.ReduceSum(lp, new[] { 1 }, keepDims: false), new[] { 1, n });
        var sumQ = engine.Reshape(engine.ReduceSum(lq, new[] { 1 }, keepDims: false), new[] { m, 1 });
        var linear = engine.TensorAddScalar(engine.TensorSubtract(engine.TensorBroadcastTo(sumP, new[] { m, n }), engine.TensorBroadcastTo(sumQ, new[] { m, n })),
            ops.FromDouble(-0.5 * c));
        return engine.TensorAdd(linear, quadratic);
    }

    /// <summary>
    /// Soft dynamic time warping (App. D) of a cost matrix <c>[m, n]</c> on the tape: <c>R[i,j] = D[i,j] + min_γ(R[i−1,j−1],
    /// R[i−1,j] + warp, R[i,j−1] + warp)</c> with the soft minimum <c>−γ log Σ e^{−a/γ}</c>, computed along anti-diagonals;
    /// returns <c>R[m−1, n−1]</c>.
    /// </summary>
    public static Tensor<T> SoftDtw<T>(IEngine engine, Tensor<T> cost, double gamma, double warp)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int m = cost.Shape[0], n = cost.Shape[1];
        var flat = engine.Reshape(cost, new[] { m * n, 1 });
        const double Infinity = 1e10;
        // diagonals[d] holds R at (i, d − i) for i in [lo(d), hi(d)].
        var diagonals = new List<(int Lo, Tensor<T> Values)>();
        Tensor<T> Neighbour(int d, int[] rows, double edge)
        {
            // R on diagonal d at the given rows (the boundary, +∞ or 0, outside it).
            if (d < 0)
            {
                var b = new Tensor<T>(new[] { rows.Length, 1 });
                for (int r = 0; r < rows.Length; r++) b[r, 0] = ops.FromDouble(d == -2 && rows[r] == -1 ? 0.0 : Infinity);
                return b;
            }
            var (lo, values) = diagonals[d];
            int count = values.Shape[0];
            // Pad with +∞ on both ends so out-of-range rows read the boundary.
            var padded = engine.TensorConcatenate(new[] { Const(1, Infinity), values, Const(1, Infinity) }, 0);
            var index = new Tensor<int>(new[] { rows.Length });
            for (int r = 0; r < rows.Length; r++)
            {
                int k = rows[r] - lo;
                index[r] = k < 0 ? 0 : k >= count ? count + 1 : k + 1;
            }
            var picked = engine.TensorIndexSelect(padded, index, 0);
            if (edge == 0.0 && d == -2) return picked;
            return picked;
        }
        Tensor<T> Const(int count, double value)
        {
            var t = new Tensor<T>(new[] { count, 1 });
            for (int i = 0; i < count; i++) t[i, 0] = ops.FromDouble(value);
            return t;
        }
        for (int d = 0; d <= m + n - 2; d++)
        {
            int lo = Math.Max(0, d - (n - 1)), hi = Math.Min(m - 1, d);
            int count = hi - lo + 1;
            var rows = Enumerable.Range(lo, count).ToArray();
            var index = new Tensor<int>(new[] { count });
            for (int r = 0; r < count; r++) index[r] = rows[r] * n + (d - rows[r]);
            var dCost = engine.TensorIndexSelect(flat, index, 0);                                      // [count, 1]
            Tensor<T> diag, up, left;
            if (d == 0)
            {
                diag = Const(1, 0.0);
                up = Const(1, Infinity);
                left = Const(1, Infinity);
            }
            else
            {
                diag = d >= 2 ? Neighbour(d - 2, rows.Select(r => r - 1).ToArray(), Infinity) : Const(count, Infinity);
                // (i−1, j): diagonal d−1 at row i−1; (i, j−1): diagonal d−1 at row i.
                up = engine.TensorAddScalar(Neighbour(d - 1, rows.Select(r => r - 1).ToArray(), Infinity), ops.FromDouble(warp));
                left = engine.TensorAddScalar(Neighbour(d - 1, rows, Infinity), ops.FromDouble(warp));
            }
            var stacked = engine.TensorConcatenate(new[] { diag, up, left }, 1);                           // [count, 3]
            // −γ log Σ e^{−a/γ}, stabilized by the (detached) minimum.
            var scaled = engine.TensorMultiplyScalar(stacked, ops.FromDouble(-1.0 / gamma));
            var max = new Tensor<T>(new[] { count, 1 });
            for (int r = 0; r < count; r++)
            {
                double best = double.NegativeInfinity;
                for (int k = 0; k < 3; k++) best = Math.Max(best, ops.ToDouble(scaled[r, k]));
                max[r, 0] = ops.FromDouble(best);
            }
            var shifted = engine.TensorSubtract(scaled, engine.TensorBroadcastTo(max, new[] { count, 3 }));
            var lse = engine.TensorAdd(engine.TensorLog(engine.ReduceSum(engine.TensorExp(shifted), new[] { 1 }, keepDims: true)), max);
            var softmin = engine.TensorMultiplyScalar(lse, ops.FromDouble(-gamma));
            diagonals.Add((lo, engine.TensorAdd(dCost, softmin)));
        }
        var last = diagonals[diagonals.Count - 1].Values;
        return engine.Reshape(last, new[] { 1 });
    }
}

/// <summary>
/// Sup-phonemes (Mixed-Phoneme BERT, Zhang et al. 2022, as NaturalSpeech's phoneme pre-training uses them): byte-pair
/// merges of adjacent phonemes inside words, learned from a phoneme corpus; each phoneme is mapped to the sup-phoneme
/// containing it.
/// </summary>
public sealed class NaturalSpeechSupPhonemes
{
    private readonly Dictionary<(int, int), int> _merges = new();
    private readonly int _phonemes;

    private NaturalSpeechSupPhonemes(int phonemes) => _phonemes = phonemes;

    /// <summary>The sup-phoneme vocabulary: the phonemes, then one id per merge.</summary>
    public int VocabSize => _phonemes + _merges.Count;

    /// <summary>The merges in order (each pair of ids and its new id).</summary>
    public IReadOnlyList<(int Left, int Right, int Id)> Merges => _merges.OrderBy(kv => kv.Value).Select(kv => (kv.Key.Item1, kv.Key.Item2, kv.Value)).ToList();

    /// <summary>Learns BPE merges until the vocabulary reaches <paramref name="vocabSize"/> (the paper: 30,088) from words
    /// given as phoneme-id sequences.</summary>
    public static NaturalSpeechSupPhonemes Learn(IEnumerable<int[]> words, int phonemes, int vocabSize)
    {
        var result = new NaturalSpeechSupPhonemes(phonemes);
        var corpus = words.Select(w => w.ToList()).Where(w => w.Count > 0).ToList();
        while (result.VocabSize < vocabSize)
        {
            var counts = new Dictionary<(int, int), int>();
            foreach (var w in corpus)
                for (int i = 0; i + 1 < w.Count; i++)
                {
                    var pair = (w[i], w[i + 1]);
                    counts[pair] = counts.TryGetValue(pair, out int c) ? c + 1 : 1;
                }
            if (counts.Count == 0) break;
            var best = counts.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key.Item1).ThenBy(kv => kv.Key.Item2).First().Key;
            int id = result.VocabSize;
            result._merges[best] = id;
            for (int k = 0; k < corpus.Count; k++) corpus[k] = Merge(corpus[k], best, id);
        }
        return result;
    }

    /// <summary>Loads merges (each pair and its id, ids consecutive from the phoneme count).</summary>
    public static NaturalSpeechSupPhonemes FromMerges(int phonemes, IEnumerable<(int Left, int Right)> merges)
    {
        var result = new NaturalSpeechSupPhonemes(phonemes);
        foreach (var m in merges) result._merges[(m.Left, m.Right)] = result.VocabSize;
        return result;
    }

    private static List<int> Merge(List<int> ids, (int, int) pair, int id)
    {
        var merged = new List<int>(ids.Count);
        for (int i = 0; i < ids.Count;)
        {
            if (i + 1 < ids.Count && ids[i] == pair.Item1 && ids[i + 1] == pair.Item2) { merged.Add(id); i += 2; }
            else merged.Add(ids[i++]);
        }
        return merged;
    }

    /// <summary>The sup-phoneme id of each phoneme of a word, applying the merges in order.</summary>
    public int[] Assign(int[] word)
    {
        // Track spans: each element covers a run of phonemes.
        var units = word.Select((p, i) => (Id: p, Start: i, Length: 1)).ToList();
        foreach (var (pair, id) in _merges.OrderBy(kv => kv.Value).Select(kv => (kv.Key, kv.Value)))
        {
            var next = new List<(int Id, int Start, int Length)>(units.Count);
            for (int i = 0; i < units.Count;)
            {
                if (i + 1 < units.Count && units[i].Id == pair.Item1 && units[i + 1].Id == pair.Item2)
                {
                    next.Add((id, units[i].Start, units[i].Length + units[i + 1].Length));
                    i += 2;
                }
                else next.Add(units[i++]);
            }
            units = next;
        }
        var result = new int[word.Length];
        foreach (var u in units)
            for (int k = 0; k < u.Length; k++) result[u.Start + k] = u.Id;
        return result;
    }
}
