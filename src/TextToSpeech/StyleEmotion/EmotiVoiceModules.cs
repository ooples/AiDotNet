using AiDotNet.Audio.Features;
using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// EmotiVoice's Transformer encoder (netease-youdao/EmotiVoice <c>models/prompt_tts_modified/modules/encoder.py</c>,
/// ESPnet's): a scaled positional encoding (the input plus a learned α times the sinusoidal table, no √d scaling),
/// pre-norm blocks of multi-head self-attention and a two-convolution feed-forward with exact GELU (kernel 3), and a final
/// LayerNorm (all ε 1e-12). The acoustic model's encoder and decoder are both this network.
/// </summary>
internal sealed class EmotiVoiceEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, LayerNormalizationLayer<T> Norm2, NormedConv1DLayer<T> FeedForward1, NormedConv1DLayer<T> FeedForward2);

    public EmotiVoiceEncoder(IEngine engine, List<LayerBase<T>> layers, int dim, int heads, int linearUnits, int blocks, int kernel, double dropout)
    {
        if (dim % heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(heads));
        _engine = engine;
        Dim = dim;
        Heads = heads;
        LinearUnits = linearUnits;
        Kernel = kernel;
        Dropout = dropout;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> => StyleTtsOps.Own(layers, layer);
        LayerNormalizationLayer<T> Norm() => Own(new LayerNormalizationLayer<T>(dim, 1e-12));
        DenseLayer<T> Linear() => Own(StyleTtsOps.Linear<T>(dim));
        NormedConv1DLayer<T> Conv(int input, int output) =>
            Own(new NormedConv1DLayer<T>(input, output, kernel, 1, 1, 1, (kernel - 1) / 2, false, ConvolutionNormalization.None));
        Alpha = Own(new TiedEmbeddingLayer<T>(1, 1));
        Alpha.Reinitialize(() => 1.0);
        for (int i = 0; i < blocks; i++)
            Blocks.Add(new Block(Norm(), Linear(), Linear(), Linear(), Linear(), Norm(), Conv(dim, linearUnits), Conv(linearUnits, dim)));
        AfterNorm = Norm();
    }

    public int Dim { get; }
    public int Heads { get; }
    public int LinearUnits { get; }
    public int Kernel { get; }
    public double Dropout { get; }

    /// <summary>The positional encoding's learned scale α (<c>ScaledPositionalEncoding.alpha</c>, initially 1).</summary>
    public TiedEmbeddingLayer<T> Alpha { get; }

    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> AfterNorm { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, Dropout, random) : x;

    /// <summary>ESPnet's sinusoidal table <c>[L, d]</c>: sin at even columns, cos at odd, frequencies 10000^(−2k/d).</summary>
    private Tensor<T> Table(int length)
    {
        // A fixed table of constants, built once per length in double and cast.
        var table = new double[length, Dim];
        for (int p = 0; p < length; p++)
            for (int k = 0; k < Dim; k += 2)
            {
                double frequency = Math.Exp(k * -(Math.Log(10000.0) / Dim));
                table[p, k] = Math.Sin(p * frequency);
                if (k + 1 < Dim) table[p, k + 1] = Math.Cos(p * frequency);
            }
        return EngineStft<T>.Table(table);
    }

    private Tensor<T> FeedForward(Block b, Tensor<T> x, bool training, Random random)
    {
        int length = x.Shape[0];
        var channels = _engine.Reshape(_engine.TensorTranspose(x), new[] { 1, Dim, length });
        var hidden = ActivationHelper.ExactGelu(b.FeedForward1.Forward(channels), _engine);
        var output = b.FeedForward2.Forward(Drop(hidden, training, random));
        return _engine.TensorTranspose(_engine.Reshape(output, new[] { Dim, length }));
    }

    /// <summary>The encoded rows <c>[L, d]</c> of input rows <c>[L, d]</c> (one sequence, every position attended).</summary>
    public Tensor<T> Forward(Tensor<T> input, bool training, Random random)
    {
        int length = input.Shape[0], heads = Heads, dk = Dim / heads;
        var alpha = _engine.TensorTile(_engine.Reshape(Alpha.Forward(new Tensor<T>(new[] { 1 })), new[] { 1, 1 }), new[] { length, Dim });
        var x = Drop(_engine.TensorAdd(input, _engine.TensorMultiply(alpha, Table(length))), training, random);
        Tensor<T> Split(Tensor<T> t) => _engine.TensorPermute(_engine.Reshape(t, new[] { length, heads, dk }), new[] { 1, 0, 2 });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(dk));
        foreach (var b in Blocks)
        {
            var normed = b.Norm1.Forward(x);
            var q = Split(b.Query.Forward(normed));
            var k = Split(b.Key.Forward(normed));
            var v = Split(b.Value.Forward(normed));
            var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), scale);
            var weights = Drop(_engine.Softmax(scores, axis: 2), training, random);
            var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, Dim });
            x = _engine.TensorAdd(x, Drop(b.Output.Forward(context), training, random));
            x = _engine.TensorAdd(x, Drop(FeedForward(b, b.Norm2.Forward(x), training, random), training, random));
        }
        return AfterNorm.Forward(x);
    }

    public void Load(TorchTensorSource source, string name)
    {
        int d = Dim;
        void Norm(LayerNormalizationLayer<T> norm, string n) =>
            TorchParameters.LayerNorm(_engine, norm, source, n + ".weight", n + ".bias");
        TorchParameters.Parameters(Alpha, source, name + ".embed.0.alpha", new[] { 1 });
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{name}.encoders.{i}";
            Norm(b.Norm1, p + ".norm1");
            StyleTtsOps.LoadLinear(_engine, b.Query, source, p + ".self_attn.linear_q", d, d);
            StyleTtsOps.LoadLinear(_engine, b.Key, source, p + ".self_attn.linear_k", d, d);
            StyleTtsOps.LoadLinear(_engine, b.Value, source, p + ".self_attn.linear_v", d, d);
            StyleTtsOps.LoadLinear(_engine, b.Output, source, p + ".self_attn.linear_out", d, d);
            Norm(b.Norm2, p + ".norm2");
            StyleTtsOps.LoadPlain(b.FeedForward1, source, p + ".feed_forward.w_1");
            StyleTtsOps.LoadPlain(b.FeedForward2, source, p + ".feed_forward.w_2");
        }
        Norm(AfterNorm, name + ".after_norm");
    }
}

/// <summary>
/// EmotiVoice's duration and variance predictors (<c>modules/variance.py</c>, ESPnet's): convolutions with ReLU, a
/// LayerNorm over channels (ε 1e-12) and dropout, then a linear map to one value per phoneme.
/// </summary>
internal sealed class EmotiVoiceVariancePredictor<T>
{
    private readonly IEngine _engine;

    public EmotiVoiceVariancePredictor(IEngine engine, List<LayerBase<T>> layers, int input, int layersCount, int channels, int kernel, double dropout)
    {
        _engine = engine;
        Channels = channels;
        Dropout = dropout;
        for (int i = 0; i < layersCount; i++)
            Convs.Add((StyleTtsOps.Own(layers, new NormedConv1DLayer<T>(i == 0 ? input : channels, channels, kernel, 1, 1, 1, (kernel - 1) / 2, false,
                ConvolutionNormalization.None)), StyleTtsOps.Own(layers, new LayerNormalizationLayer<T>(channels, 1e-12))));
        Linear = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(1));
    }

    public int Channels { get; }
    public double Dropout { get; }
    public List<(NormedConv1DLayer<T> Conv, LayerNormalizationLayer<T> Norm)> Convs { get; } = new();
    public DenseLayer<T> Linear { get; }

    /// <summary>One value per row <c>[L]</c> of rows <c>[L, d]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, bool training, Random random)
    {
        int length = x.Shape[0];
        var h = _engine.Reshape(_engine.TensorTranspose(x), new[] { 1, x.Shape[1], length });
        foreach (var (conv, norm) in Convs)
        {
            var y = _engine.ReLU(conv.Forward(h));
            var rows = norm.Forward(_engine.TensorTranspose(_engine.Reshape(y, new[] { Channels, length })));
            rows = training && Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, rows, Dropout, random) : rows;
            h = _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, Channels, length });
        }
        var output = Linear.Forward(_engine.TensorTranspose(_engine.Reshape(h, new[] { Channels, length })));
        return _engine.Reshape(output, new[] { length });
    }

    public void Load(TorchTensorSource source, string name, int input)
    {
        for (int i = 0; i < Convs.Count; i++)
        {
            StyleTtsOps.LoadPlain(Convs[i].Conv, source, $"{name}.conv.{i}.0");
            TorchParameters.LayerNorm(_engine, Convs[i].Norm, source, $"{name}.conv.{i}.2.weight", $"{name}.conv.{i}.2.bias");
        }
        StyleTtsOps.LoadLinear(_engine, Linear, source, name + ".linear", Channels, 1);
    }
}

/// <summary>
/// JETS's alignment module (Lim et al. 2022; EmotiVoice <c>modules/alignment.py</c>): text and mel frames encoded by small
/// convolution stacks, the negative L2 distance of every frame to every phoneme, a log-softmax over phonemes, plus the
/// beta-binomial alignment prior; with its hard alignment (monotonic alignment search), the binarization loss and the
/// forward-sum (CTC) loss.
/// </summary>
internal sealed class EmotiVoiceAlignment<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public EmotiVoiceAlignment(IEngine engine, List<LayerBase<T>> layers, int textDim, int mels)
    {
        _engine = engine;
        NormedConv1DLayer<T> Conv(int input, int output, int kernel) =>
            StyleTtsOps.Own(layers, new NormedConv1DLayer<T>(input, output, kernel, 1, 1, 1, (kernel - 1) / 2, false, ConvolutionNormalization.None));
        TextConv1 = Conv(textDim, textDim, 3);
        TextConv2 = Conv(textDim, textDim, 1);
        FeatConv1 = Conv(mels, textDim, 3);
        FeatConv2 = Conv(textDim, textDim, 3);
        FeatConv3 = Conv(textDim, textDim, 1);
        TextDim = textDim;
        Mels = mels;
    }

    public int TextDim { get; }
    public int Mels { get; }
    public NormedConv1DLayer<T> TextConv1 { get; }
    public NormedConv1DLayer<T> TextConv2 { get; }
    public NormedConv1DLayer<T> FeatConv1 { get; }
    public NormedConv1DLayer<T> FeatConv2 { get; }
    public NormedConv1DLayer<T> FeatConv3 { get; }

    /// <summary>Lanczos ln Γ (g = 7, 9 terms; relative error ~1e-15).</summary>
    internal static double LogGamma(double x)
    {
        if (x < 0.5) return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * x))) - LogGamma(1 - x);
        double[] g =
        {
            0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
        };
        x -= 1;
        double a = g[0], t = x + 7.5;
        for (int i = 1; i < 9; i++) a += g[i] / (x + i);
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(a);
    }

    private static double LogBeta(double a, double b) => LogGamma(a) + LogGamma(b) - LogGamma(a + b);

    /// <summary>The beta-binomial prior <c>[T, N]</c>: for frame t (1-based) the log-pmf of phoneme k ∈ [0, N) under
    /// BetaBinomial(N, t, T − t + 1) (scipy's <c>betabinom.logpmf</c>).</summary>
    internal static double[,] Prior(int frames, int phonemes)
    {
        var prior = new double[frames, phonemes];
        for (int t = 1; t <= frames; t++)
        {
            double a = t, b = frames - t + 1;
            for (int k = 0; k < phonemes; k++)
                prior[t - 1, k] = LogGamma(phonemes + 1) - LogGamma(k + 1) - LogGamma(phonemes - k + 1)
                    + LogBeta(k + a, phonemes - k + b) - LogBeta(a, b);
        }
        return prior;
    }

    /// <summary>The alignment log-probabilities <c>[T, N]</c> of text rows <c>[N, d]</c> and mel frames <c>[T, mels]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> text, Tensor<T> feats)
    {
        int n = text.Shape[0], frames = feats.Shape[0], d = TextDim;
        var t = _engine.Reshape(_engine.TensorTranspose(text), new[] { 1, d, n });
        t = TextConv2.Forward(_engine.ReLU(TextConv1.Forward(t)));
        var f = _engine.Reshape(_engine.TensorTranspose(feats), new[] { 1, Mels, frames });
        f = FeatConv3.Forward(_engine.ReLU(FeatConv2.Forward(_engine.ReLU(FeatConv1.Forward(f)))));
        var textRows = _engine.TensorTranspose(_engine.Reshape(t, new[] { d, n }));                    // [N, d]
        var frameRows = _engine.TensorTranspose(_engine.Reshape(f, new[] { d, frames }));              // [T, d]
        var diff = _engine.TensorSubtract(_engine.TensorTile(_engine.Reshape(frameRows, new[] { frames, 1, d }), new[] { 1, n, 1 }),
            _engine.TensorTile(_engine.Reshape(textRows, new[] { 1, n, d }), new[] { frames, 1, 1 }));
        var distance = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(diff, diff), new[] { 2 }, keepDims: false));
        var logP = _engine.TensorLogSoftmax(_engine.TensorNegate(distance), axis: 1);
        return _engine.TensorAdd(logP, EngineStft<T>.Table(Prior(frames, n)));
    }

    /// <summary>JETS's monotonic alignment search (<c>_monotonic_alignment_search</c>) of log-probabilities
    /// <c>[T, N]</c>: each frame's phoneme.</summary>
    /// <remarks>The recurrence Q[i, j] = max(Q[i − 1, j − 1], Q[i, j − 1]) + log p[j, i] (Q[0, j] the running sum of
    /// column 0) runs frame by frame as one engine step over all phonemes: the phonemes past j stay −∞ by themselves.
    /// The scores are scaled by 2<sup>−⌈log₂ T⌉</sup> first — exact, and it leaves every comparison unchanged — so a
    /// half-precision path sum cannot overflow. The backtrack reads the comparisons Q[i − 1, j] ≥ Q[i, j] as integers.</remarks>
    internal static int[] Viterbi(Tensor<T> logP)
    {
        if (logP is null) throw new ArgumentNullException(nameof(logP));
        var engine = AiDotNetEngine.Current;
        int frames = logP.Shape[0], n = logP.Shape[1];
        T negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);
        var none = Tensor<T>.CreateDefault(new[] { 1 }, negativeInfinity);
        int[] takeLower;
        using (new NoGradScope<T>())
        {
            int exponent = 0;
            while ((1 << exponent) < frames && exponent < 30) exponent++;
            var scaled = engine.TensorMultiplyScalar(logP, NumOps.FromDouble(Math.Pow(2, -exponent)));
            Tensor<T> Row(int j) => engine.Reshape(engine.TensorSlice(scaled, new[] { j, 0 }, new[] { 1, n }), new[] { n });
            Tensor<T> ShiftDown(Tensor<T> column) => n == 1 ? none
                : engine.TensorConcatenate(new[] { none, engine.TensorSlice(column, new[] { 0 }, new[] { n - 1 }) }, 0);
            var columns = new Tensor<T>[frames];
            // Q[:, 0] = [log p[0, 0], −∞, …].
            var q = engine.TensorAdd(Row(0), ShiftDown(Tensor<T>.CreateDefault(new[] { n }, NumOps.Zero)));
            columns[0] = q;
            for (int j = 1; j < frames; j++)
            {
                q = engine.TensorAdd(engine.TensorMax(ShiftDown(q), q), Row(j));
                columns[j] = q;
            }
            var all = engine.TensorStack(columns, 0);                                                      // [T, N]
            var lower = n == 1 ? Tensor<T>.CreateDefault(new[] { frames, 1 }, negativeInfinity)
                : engine.TensorConcatenate(new[] { Tensor<T>.CreateDefault(new[] { frames, 1 }, negativeInfinity),
                    engine.TensorSlice(all, new[] { 0, 0 }, new[] { frames, n - 1 }) }, 1);
            // Q[i − 1, j] ≥ Q[i, j] ⇔ ¬(Q[i, j] > Q[i − 1, j]).
            takeLower = engine.TensorAddScalar(engine.TensorNegate(engine.TensorGreaterThan(all, lower)), NumOps.One).Cast<int>().ToArray();
        }
        var path = new int[frames];
        path[frames - 1] = n - 1;
        for (int j = frames - 2; j >= 0; j--)
        {
            int ib = path[j + 1];
            path[j] = ib == 0 ? 0 : takeLower[j * n + ib] != 0 ? ib - 1 : ib;
        }
        return path;
    }

    /// <summary>The durations (frames per phoneme) of a monotonic path (<c>np.bincount</c>, padded to N).</summary>
    internal static int[] Durations(int[] path, int phonemes)
    {
        var d = new int[phonemes];
        foreach (int p in path) d[p]++;
        return d;
    }

    /// <summary>The binarization loss: minus the mean log-probability along the path.</summary>
    public Tensor<T> BinarizationLoss(Tensor<T> logP, int[] path)
    {
        int frames = logP.Shape[0], n = logP.Shape[1];
        var frameIndex = new int[frames];
        for (int t = 0; t < frames; t++) frameIndex[t] = t;
        var mask = _engine.TensorIndexPut(new Tensor<T>(new[] { frames, n }),
            new[] { new Tensor<int>(frameIndex, new[] { frames }), new Tensor<int>(path, new[] { frames }) },
            Tensor<T>.CreateDefault(new[] { frames }, NumOps.FromDouble(-1.0 / frames)), false);
        return _engine.ReduceSum(_engine.TensorMultiply(logP, mask), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>
    /// The forward-sum loss (<c>ForwardSumLoss</c>): a blank column of log(e⁻¹) prepended, a log-softmax over phonemes,
    /// and the CTC loss of the phoneme sequence 1…N over the frames (<c>F.ctc_loss</c>, mean reduction: divided by N),
    /// computed by the forward algorithm on the tape.
    /// </summary>
    public Tensor<T> ForwardSumLoss(Tensor<T> logP)
    {
        int frames = logP.Shape[0], n = logP.Shape[1], states = 2 * n + 1;
        var blank = Tensor<T>.CreateDefault(new[] { frames, 1 }, NumOps.FromDouble(-1.0));
        var padded = _engine.TensorLogSoftmax(_engine.TensorConcatenate(new[] { blank, logP }, 1), axis: 1);   // [T, N + 1]
        // Extended labels: blank, 1, blank, 2, …, N, blank; the label states (odd, from 3) may skip from s − 2.
        var labelIds = new int[states];
        var initial = new double[states];
        var skip = new double[states];
        double floor = UnreachableFloor();
        for (int s = 0; s < states; s++)
        {
            labelIds[s] = s % 2 == 0 ? 0 : (s + 1) / 2;
            initial[s] = s < 2 ? 0.0 : floor;
            skip[s] = s % 2 == 1 && s >= 2 ? 0.0 : floor;
        }
        var emissions = _engine.TensorIndexSelect(padded, new Tensor<int>(labelIds, new[] { states }), 1);      // [T, S]
        T floorValue = NumOps.FromDouble(floor);
        var floorRow = Tensor<T>.CreateDefault(new[] { 1 }, floorValue);
        Tensor<T> Emission(int t) => _engine.Reshape(_engine.TensorSlice(emissions, new[] { t, 0 }, new[] { 1, states }), new[] { states });
        var alpha = _engine.TensorAdd(Emission(0), new Tensor<double>(initial, new[] { states }).Cast<T>());
        var skipMask = new Tensor<double>(skip, new[] { states }).Cast<T>();
        for (int t = 1; t < frames; t++)
        {
            var stay = alpha;
            var one = _engine.TensorConcatenate(new[] { floorRow, _engine.TensorSlice(alpha, new[] { 0 }, new[] { states - 1 }) }, 0);
            var two = _engine.TensorAdd(_engine.TensorConcatenate(new[] { floorRow, floorRow, _engine.TensorSlice(alpha, new[] { 0 }, new[] { states - 2 }) }, 0), skipMask);
            // Unreachable states are held at the floor so a 16-bit T never reaches −∞ − (−∞).
            alpha = _engine.TensorMax(_engine.TensorAdd(LogSumExp(stay, one, two), Emission(t)), floorValue);
        }
        var last = LogSumExp(_engine.TensorSlice(alpha, new[] { states - 1 }, new[] { 1 }), _engine.TensorSlice(alpha, new[] { states - 2 }, new[] { 1 }));
        return _engine.TensorMultiplyScalar(_engine.TensorNegate(last), NumOps.FromDouble(1.0 / n));
    }

    // The log-space "unreachable" value: −1e30, or a quarter of the most negative finite value where −1e30 does not fit T.
    private static double UnreachableFloor()
    {
        double floor = NumOps.ToDouble(NumOps.FromDouble(-1e30));
        return double.IsInfinity(floor) ? -0.25 * NumOps.ToDouble(NumOps.MaxValue) : -1e30;
    }

    // log(Σ exp) of same-shaped tensors, stabilized by their (detached) elementwise maximum.
    private Tensor<T> LogSumExp(params Tensor<T>[] terms)
    {
        Tensor<T> max;
        using (new NoGradScope<T>())
        {
            max = terms[0];
            for (int i = 1; i < terms.Length; i++) max = _engine.TensorMax(max, terms[i]);
            max = max.Clone();
        }
        Tensor<T>? sum = null;
        foreach (var x in terms)
        {
            var e = _engine.TensorExp(_engine.TensorSubtract(x, max));
            sum = sum is null ? e : _engine.TensorAdd(sum, e);
        }
        return _engine.TensorAdd(_engine.TensorLog(sum!), max);
    }

    public void Load(TorchTensorSource source, string name)
    {
        StyleTtsOps.LoadPlain(TextConv1, source, name + ".t_conv1");
        StyleTtsOps.LoadPlain(TextConv2, source, name + ".t_conv2");
        StyleTtsOps.LoadPlain(FeatConv1, source, name + ".f_conv1");
        StyleTtsOps.LoadPlain(FeatConv2, source, name + ".f_conv2");
        StyleTtsOps.LoadPlain(FeatConv3, source, name + ".f_conv3");
    }
}

/// <summary>JETS's Gaussian upsampling (<c>GaussianUpsampling</c>, δ = 0.1): frame t attends to phoneme n with
/// softmax over n of −δ (t − c_n)², c = cumsum(d) − d/2.</summary>
internal static class EmotiVoiceUpsampling
{
    /// <summary>The upsampled rows <c>[frames, d]</c> of rows <c>[N, d]</c> with (possibly fractional) durations
    /// <c>[N]</c>.</summary>
    public static Tensor<T> Upsample<T>(IEngine engine, Tensor<T> rows, Tensor<T> durations, int frames, double delta = 0.1)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = rows.Shape[0];
        var d = engine.Reshape(durations, new[] { n, 1 });
        var centres = engine.TensorSubtract(engine.TensorCumSum(d, 0), engine.TensorMultiplyScalar(d, ops.FromDouble(0.5)));       // [N, 1]
        var t = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, frames, 1), new[] { frames, 1 }), new[] { frames, n });
        var diff = engine.TensorSubtract(t, engine.TensorTile(engine.TensorTranspose(centres), new[] { frames, 1 }));
        var energy = engine.TensorMultiplyScalar(engine.TensorMultiply(diff, diff), ops.FromDouble(-delta));
        return engine.TensorMatMul(engine.Softmax(energy, axis: 1), rows);
    }
}
