using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Shared construction and loading helpers of the IndexTTS modules, with PyTorch's default initializations.</summary>
internal static class IndexTtsOps
{
    /// <summary>An <c>nn.Linear</c>: weights and biases from U(±1/√fan_in).</summary>
    public static DenseLayer<T> Linear<T>(IEngine engine, List<LayerBase<T>> layers, Random random, int inputs, int outputs)
    {
        var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.Add(layer);
        TorchParameters.Linear(engine, layer, inputs, outputs, Uniform(random, inputs * outputs, inputs), Uniform(random, outputs, inputs));
        return layer;
    }

    /// <summary>A bias-free <c>nn.Linear</c> (U(±1/√fan_in)).</summary>
    public static BiasFreeLinearLayer<T> BiasFree<T>(List<LayerBase<T>> layers, Random random, int inputs, int outputs)
    {
        var layer = new BiasFreeLinearLayer<T>(inputs, outputs);
        layers.Add(layer);
        TorchParameters.Linear(layer, inputs, outputs, Uniform(random, inputs * outputs, inputs));
        return layer;
    }

    /// <summary>A parameter table of <paramref name="count"/> values held as a <c>[1, count]</c> bias-free layer.</summary>
    public static BiasFreeLinearLayer<T> Values<T>(List<LayerBase<T>> layers, double[] values)
    {
        var holder = new BiasFreeLinearLayer<T>(1, values.Length);
        layers.Add(holder);
        TorchParameters.Linear(holder, 1, values.Length, values);
        return holder;
    }

    /// <summary>The held table (through the layer's forward, so it stays on the tape and pages in a streamed weight).</summary>
    public static Tensor<T> Value<T>(BiasFreeLinearLayer<T> holder) =>
        holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { MathHelper.GetNumericOperations<T>().One })));

    public static TiedEmbeddingLayer<T> Embedding<T>(List<LayerBase<T>> layers, Random random, int count, int dim, double std)
    {
        var table = new TiedEmbeddingLayer<T>(count, dim);
        layers.Add(table);
        table.Reinitialize(() => std * Gaussian(random));
        return table;
    }

    public static LayerNormalizationLayer<T> LayerNorm<T>(List<LayerBase<T>> layers, int dim, double epsilon = 1e-5)
    {
        var norm = new LayerNormalizationLayer<T>(dim, epsilon);
        layers.Add(norm);
        return norm;
    }

    public static double[] Uniform(Random random, int count, int fanIn)
    {
        var values = new double[count];
        double bound = 1.0 / Math.Sqrt(fanIn);
        for (int i = 0; i < count; i++) values[i] = bound * (2 * random.NextDouble() - 1);
        return values;
    }

    public static double Gaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    public static double[] Normal(Random random, int count, double std)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = std * Gaussian(random);
        return values;
    }

    /// <summary>Token ids as the float index tensor the embedding lookup takes.</summary>
    public static Tensor<T> Ids<T>(IReadOnlyList<int> ids)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) t[i] = ops.FromDouble(ids[i]);
        return t;
    }

    /// <summary>GPT-2's <c>gelu_new</c>: <c>0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))</c>.</summary>
    public static Tensor<T> NewGelu<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var cube = engine.TensorMultiply(engine.TensorMultiply(x, x), x);
        var inner = engine.TensorMultiplyScalar(engine.TensorAdd(x, engine.TensorMultiplyScalar(cube, ops.FromDouble(0.044715))),
            ops.FromDouble(Math.Sqrt(2.0 / Math.PI)));
        var gate = engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.Tanh(inner), ops.One), ops.FromDouble(0.5));
        return engine.TensorMultiply(x, gate);
    }

    /// <summary>Scaled dot-product attention of <c>[n, Lq, d]</c> queries over <c>[n, Lk, d]</c> keys and values, plus an
    /// optional additive bias <c>[Lq, Lk]</c>.</summary>
    public static Tensor<T> Attend<T>(IEngine engine, Tensor<T> q, Tensor<T> k, Tensor<T> v, double scale, Tensor<T>? bias)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var scores = engine.TensorMultiplyScalar(engine.BatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 })), ops.FromDouble(scale));
        if (bias is not null)
            scores = engine.TensorAdd(scores, engine.TensorBroadcastTo(engine.Reshape(bias, new[] { 1, bias.Shape[0], bias.Shape[1] }), scores.Shape.ToArray()));
        return engine.BatchMatMul(engine.Softmax(scores, axis: 2), v);
    }

    /// <summary><c>[L, heads·dk]</c> → <c>[heads, L, dk]</c>.</summary>
    public static Tensor<T> SplitHeads<T>(IEngine engine, Tensor<T> x, int heads)
    {
        int length = x.Shape[0], dk = x.Shape[1] / heads;
        return engine.TensorPermute(engine.Reshape(x, new[] { length, heads, dk }), new[] { 1, 0, 2 });
    }

    /// <summary><c>[heads, L, dk]</c> → <c>[L, heads·dk]</c>.</summary>
    public static Tensor<T> MergeHeads<T>(IEngine engine, Tensor<T> x)
    {
        int heads = x.Shape[0], length = x.Shape[1], dk = x.Shape[2];
        return engine.Reshape(engine.TensorPermute(x, new[] { 1, 0, 2 }), new[] { length, heads * dk });
    }

    public static void LoadLinear<T>(IEngine engine, Func<string, int[], double[]> read, DenseLayer<T> layer, string name, int inputs, int outputs) =>
        TorchParameters.Linear(engine, layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }), read(name + ".bias", new[] { outputs }));

    public static void LoadLinear<T>(Func<string, int[], double[]> read, BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs) =>
        TorchParameters.Linear(layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }));

    public static void LoadNorm<T>(IEngine engine, Func<string, int[], double[]> read, LayerNormalizationLayer<T> norm, string name, int dim) =>
        TorchParameters.LayerNorm(engine, norm, read(name + ".weight", new[] { dim }), read(name + ".bias", new[] { dim }));

    public static void LoadConv<T>(Func<string, int[], double[]> read, NormedConv1DLayer<T> conv, string name, int outputs) =>
        conv.LoadTorchWeights(read(name + ".weight", conv.TorchWeightShape), null, read(name + ".bias", new[] { outputs }));

    /// <summary>Loads an <c>nn.Conv2d</c>'s weight <c>[out, in, kh, kw]</c> and bias.</summary>
    public static void LoadConv2D<T>(IEngine engine, Func<string, int[], double[]> read, Conv2DLayer<T> conv, string name)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var kernel = conv.Kernel;
        var weight = read(name + ".weight", kernel.Shape.ToArray());
        if (weight.Length != kernel.Length) throw new ArgumentException($"'{name}.weight' has {weight.Length} values, expected {kernel.Length}.");
        for (int i = 0; i < weight.Length; i++) kernel[i] = ops.FromDouble(weight[i]);
        var bias = read(name + ".bias", new[] { conv.Bias.Length });
        for (int i = 0; i < bias.Length; i++) conv.Bias[i] = ops.FromDouble(bias[i]);
        engine.InvalidatePersistentTensor(kernel);
        engine.InvalidatePersistentTensor(conv.Bias);
    }
}

/// <summary>The sizes of an IndexTTS conformer conditioning encoder (the release's <c>condition_module</c>).</summary>
internal sealed record IndexTtsConformerConfiguration(int InputSize, int OutputSize, int LinearUnits, int Heads, int Blocks,
    int Kernel = 15, double Dropout = 0.0);

/// <summary>
/// The conformer that IndexTTS conditions on (the release's <c>ConformerEncoder</c>, after wenet): a 3×3, stride-2
/// convolution over (time, feature) with ReLU and a linear map (<c>conv2d2</c>, halving the frames), the states scaled by
/// √d with a sinusoidal position table, then pre-norm blocks of relative-position self-attention, the convolution module
/// (pointwise convolution, GLU, depthwise convolution, LayerNorm, Swish, pointwise convolution) and a Swish feed-forward,
/// each block closed by a LayerNorm, and a final LayerNorm. All norms use ε = 1e-5.
/// </summary>
/// <remarks>
/// The attention is wenet's early relative-position form: scores <c>((q + u) kᵀ + (q + v) pⱼᵀ) / √d_k</c> where
/// <c>pⱼ</c> projects the absolute sinusoid of key j, with no relative shift.
/// </remarks>
internal sealed class IndexTtsConformerEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> NormAttention, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, BiasFreeLinearLayer<T> Position, BiasFreeLinearLayer<T> BiasU, BiasFreeLinearLayer<T> BiasV,
        LayerNormalizationLayer<T> NormConvolution, NormedConv1DLayer<T> Pointwise1, NormedConv1DLayer<T> Depthwise,
        LayerNormalizationLayer<T> ConvolutionNorm, NormedConv1DLayer<T> Pointwise2, LayerNormalizationLayer<T> NormFeedForward,
        DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2, LayerNormalizationLayer<T> NormFinal);

    public IndexTtsConformerEncoder(IEngine engine, List<LayerBase<T>> layers, IndexTtsConformerConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int d = c.OutputSize, dk = d / c.Heads;
        if (d % c.Heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(configuration));
        if (c.Kernel % 2 != 1) throw new ArgumentException("The convolution kernel must be odd.", nameof(configuration));
        if (c.InputSize < 3) throw new ArgumentException("The input needs at least three features.", nameof(configuration));

        Subsample = new Conv2DLayer<T>(1, d, 3, 3, 2, 2, 0, 0);
        layers.Add(Subsample);
        var subsampleKernel = IndexTtsOps.Uniform(random, Subsample.Kernel.Length, 9);
        var subsampleBias = IndexTtsOps.Uniform(random, d, 9);
        IndexTtsOps.LoadConv2D(engine, (name, _) => name.EndsWith(".weight", StringComparison.Ordinal) ? subsampleKernel : subsampleBias, Subsample, "init");
        SubsampledFeatures = (c.InputSize - 1) / 2;
        SubsampleOut = IndexTtsOps.Linear(engine, layers, random, d * SubsampledFeatures, d);

        double xavier = Math.Sqrt(6.0 / (c.Heads + dk));
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int groups)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, 1, 1, groups, (kernel - 1) / 2, false, ConvolutionNormalization.None);
            layers.Add(conv);
            int fanIn = inputs / groups * kernel;
            var weights = IndexTtsOps.Uniform(random, conv.TorchWeightShape.Aggregate(1, (a, s) => a * s), fanIn);
            conv.LoadTorchWeights(weights, null, IndexTtsOps.Uniform(random, outputs, fanIn));
            return conv;
        }
        for (int i = 0; i < c.Blocks; i++)
        {
            var normAttention = IndexTtsOps.LayerNorm(layers, d);
            var query = IndexTtsOps.Linear(engine, layers, random, d, d);
            var key = IndexTtsOps.Linear(engine, layers, random, d, d);
            var value = IndexTtsOps.Linear(engine, layers, random, d, d);
            var output = IndexTtsOps.Linear(engine, layers, random, d, d);
            var position = IndexTtsOps.BiasFree(layers, random, d, d);
            var biasU = IndexTtsOps.Values(layers, Enumerable.Range(0, d).Select(_ => xavier * (2 * random.NextDouble() - 1)).ToArray());
            var biasV = IndexTtsOps.Values(layers, Enumerable.Range(0, d).Select(_ => xavier * (2 * random.NextDouble() - 1)).ToArray());
            var normConvolution = IndexTtsOps.LayerNorm(layers, d);
            var pointwise1 = Conv(d, 2 * d, 1, 1);
            var depthwise = Conv(d, d, c.Kernel, d);
            var convolutionNorm = IndexTtsOps.LayerNorm(layers, d);
            var pointwise2 = Conv(d, d, 1, 1);
            var normFeedForward = IndexTtsOps.LayerNorm(layers, d);
            var ff1 = IndexTtsOps.Linear(engine, layers, random, d, c.LinearUnits);
            var ff2 = IndexTtsOps.Linear(engine, layers, random, c.LinearUnits, d);
            var normFinal = IndexTtsOps.LayerNorm(layers, d);
            Blocks.Add(new Block(normAttention, query, key, value, output, position, biasU, biasV, normConvolution, pointwise1, depthwise,
                convolutionNorm, pointwise2, normFeedForward, ff1, ff2, normFinal));
        }
        AfterNorm = IndexTtsOps.LayerNorm(layers, d);
    }

    public IndexTtsConformerConfiguration Configuration { get; }
    public Conv2DLayer<T> Subsample { get; }
    public DenseLayer<T> SubsampleOut { get; }
    public int SubsampledFeatures { get; }
    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> AfterNorm { get; }

    /// <summary>Frames after the stride-2 subsampling of <paramref name="frames"/> input frames.</summary>
    public static int OutputFrames(int frames) => (frames - 1) / 2;

    private Tensor<T> Dropout(Tensor<T> x, bool training, Random? random) =>
        training && Configuration.Dropout > 0 && random is not null ? T5Seq2Seq<T>.Dropout(_engine, x, Configuration.Dropout, random) : x;

    private double[]? _storedPositions;

    /// <summary>
    /// Uses a checkpoint's position table (<c>embed.pos_enc.pe</c>, <c>[1, max_len, d]</c>) in place of the computed one.
    /// The release registers the table as a persistent buffer, so loading a checkpoint replaces the computed sinusoids with
    /// the stored (float32) ones, and the release runs on those.
    /// </summary>
    public void LoadPositionTable(double[] table)
    {
        if (table is null) throw new ArgumentNullException(nameof(table));
        if (table.Length % Configuration.OutputSize != 0) throw new ArgumentException("The table holds whole rows of the encoder's width.", nameof(table));
        _storedPositions = table;
    }

    /// <summary>The sinusoid table <c>[L, d]</c> of absolute positions 0 … L−1 (sin at even columns, cos at odd), or the
    /// first L rows of a loaded table.</summary>
    internal Tensor<T> PositionTable(int length)
    {
        int d = Configuration.OutputSize;
        var table = new Tensor<T>(new[] { length, d });
        if (_storedPositions is { } stored)
        {
            if (length * d > stored.Length) throw new ArgumentException($"The loaded position table holds {stored.Length / d} positions.", nameof(length));
            for (int i = 0; i < length * d; i++) table[i / d, i % d] = NumOps.FromDouble(stored[i]);
            return table;
        }
        for (int p = 0; p < length; p++)
            for (int k = 0; k < d; k += 2)
            {
                double frequency = Math.Exp(k * -(Math.Log(10000.0) / d));
                table[p, k] = NumOps.FromDouble(Math.Sin(p * frequency));
                if (k + 1 < d) table[p, k + 1] = NumOps.FromDouble(Math.Cos(p * frequency));
            }
        return table;
    }

    /// <summary>The encoded states <c>[⌊(T−1)/2⌋, d]</c> of features <c>[T, InputSize]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> input, bool training = false, Random? random = null)
    {
        var c = Configuration;
        int frames = input.Shape[0], d = c.OutputSize, heads = c.Heads, dk = d / heads;
        if (frames < 3) throw new ArgumentException("The conformer needs at least three frames.", nameof(input));
        if (input.Shape[1] != c.InputSize) throw new ArgumentException($"Expected {c.InputSize} features a frame.", nameof(input));
        var conv = _engine.ReLU(Subsample.Forward(_engine.Reshape(input, new[] { 1, 1, frames, c.InputSize })));   // [1, d, T', F']
        int length = conv.Shape[2], features = conv.Shape[3];
        var flat = _engine.Reshape(_engine.TensorPermute(conv, new[] { 0, 2, 1, 3 }), new[] { length, d * features });
        var x = _engine.TensorMultiplyScalar(SubsampleOut.Forward(flat), NumOps.FromDouble(Math.Sqrt(d)));
        x = Dropout(x, training, random);
        var positions = Dropout(PositionTable(length), training, random);

        foreach (var block in Blocks)
        {
            var normed = block.NormAttention.Forward(x);
            var q = IndexTtsOps.SplitHeads(_engine, block.Query.Forward(normed), heads);
            var k = IndexTtsOps.SplitHeads(_engine, block.Key.Forward(normed), heads);
            var v = IndexTtsOps.SplitHeads(_engine, block.Value.Forward(normed), heads);
            var p = IndexTtsOps.SplitHeads(_engine, block.Position.Forward(positions), heads);
            var u = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(block.BiasU), new[] { heads, 1, dk }), new[] { heads, length, dk });
            var w = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(block.BiasV), new[] { heads, 1, dk }), new[] { heads, length, dk });
            var ac = _engine.BatchMatMul(_engine.TensorAdd(q, u), _engine.TensorPermute(k, new[] { 0, 2, 1 }));
            var bd = _engine.BatchMatMul(_engine.TensorAdd(q, w), _engine.TensorPermute(p, new[] { 0, 2, 1 }));
            var scores = _engine.TensorMultiplyScalar(_engine.TensorAdd(ac, bd), NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
            var context = IndexTtsOps.MergeHeads(_engine, _engine.BatchMatMul(Dropout(_engine.Softmax(scores, axis: 2), training, random), v));
            x = _engine.TensorAdd(x, Dropout(block.Output.Forward(context), training, random));

            var h = _engine.Reshape(_engine.TensorTranspose(block.NormConvolution.Forward(x)), new[] { 1, d, length });
            h = block.Depthwise.Forward(_engine.GLU(block.Pointwise1.Forward(h), 1));
            h = _engine.Swish(block.ConvolutionNorm.Forward(_engine.TensorTranspose(_engine.Reshape(h, new[] { d, length }))));
            h = block.Pointwise2.Forward(_engine.Reshape(_engine.TensorTranspose(h), new[] { 1, d, length }));
            x = _engine.TensorAdd(x, Dropout(_engine.TensorTranspose(_engine.Reshape(h, new[] { d, length })), training, random));

            var hidden = _engine.Swish(block.FeedForward1.Forward(block.NormFeedForward.Forward(x)));
            hidden = block.FeedForward2.Forward(Dropout(hidden, training, random));
            x = block.NormFinal.Forward(_engine.TensorAdd(x, Dropout(hidden, training, random)));
        }
        return AfterNorm.Forward(x);
    }

    /// <summary>Loads the release's <c>{prefix}embed.*</c>, <c>{prefix}encoders.{i}.*</c> and <c>{prefix}after_norm</c>.</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        int d = c.OutputSize;
        IndexTtsOps.LoadConv2D(_engine, read, Subsample, prefix + "embed.conv.0");
        IndexTtsOps.LoadLinear(_engine, read, SubsampleOut, prefix + "embed.out.0", d * SubsampledFeatures, d);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}encoders.{i}.";
            IndexTtsOps.LoadNorm(_engine, read, b.NormAttention, p + "norm_mha", d);
            IndexTtsOps.LoadLinear(_engine, read, b.Query, p + "self_attn.linear_q", d, d);
            IndexTtsOps.LoadLinear(_engine, read, b.Key, p + "self_attn.linear_k", d, d);
            IndexTtsOps.LoadLinear(_engine, read, b.Value, p + "self_attn.linear_v", d, d);
            IndexTtsOps.LoadLinear(_engine, read, b.Output, p + "self_attn.linear_out", d, d);
            IndexTtsOps.LoadLinear(read, b.Position, p + "self_attn.linear_pos", d, d);
            TorchParameters.Linear(b.BiasU, 1, d, read(p + "self_attn.pos_bias_u", new[] { c.Heads, d / c.Heads }));
            TorchParameters.Linear(b.BiasV, 1, d, read(p + "self_attn.pos_bias_v", new[] { c.Heads, d / c.Heads }));
            IndexTtsOps.LoadNorm(_engine, read, b.NormConvolution, p + "norm_conv", d);
            IndexTtsOps.LoadConv(read, b.Pointwise1, p + "conv_module.pointwise_conv1", 2 * d);
            IndexTtsOps.LoadConv(read, b.Depthwise, p + "conv_module.depthwise_conv", d);
            IndexTtsOps.LoadNorm(_engine, read, b.ConvolutionNorm, p + "conv_module.norm", d);
            IndexTtsOps.LoadConv(read, b.Pointwise2, p + "conv_module.pointwise_conv2", d);
            IndexTtsOps.LoadNorm(_engine, read, b.NormFeedForward, p + "norm_ff", d);
            IndexTtsOps.LoadLinear(_engine, read, b.FeedForward1, p + "feed_forward.w_1", d, c.LinearUnits);
            IndexTtsOps.LoadLinear(_engine, read, b.FeedForward2, p + "feed_forward.w_2", c.LinearUnits, d);
            IndexTtsOps.LoadNorm(_engine, read, b.NormFinal, p + "norm_final", d);
        }
        IndexTtsOps.LoadNorm(_engine, read, AfterNorm, prefix + "after_norm", d);
    }
}

/// <summary>
/// The perceiver resampler IndexTTS pools a conditioning sequence with (the release's <c>PerceiverResampler</c>, after
/// lucidrains): learned latents attend over themselves and the projected context in two layers of bias-free multi-head
/// attention (heads of 64) and a GEGLU feed-forward, each residual, then an RMS norm
/// (<c>x / max(‖x‖, 1e-12) · √dim · γ</c>).
/// </summary>
internal sealed class IndexTtsPerceiver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int HeadDim = 64;
    private readonly IEngine _engine;
    private readonly int _dim, _contextDim, _latents, _heads, _feedForward;

    internal sealed record Layer(BiasFreeLinearLayer<T> ToQuery, BiasFreeLinearLayer<T> ToKeyValue, BiasFreeLinearLayer<T> ToOutput,
        DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    public IndexTtsPerceiver(IEngine engine, List<LayerBase<T>> layers, Random random, int dim, int contextDim, int latents, int heads,
        int feedForwardMultiple, int depth = 2)
    {
        _engine = engine;
        _dim = dim;
        _contextDim = contextDim;
        _latents = latents;
        _heads = heads;
        _feedForward = (int)(dim * feedForwardMultiple * 2 / 3.0);
        int inner = HeadDim * heads;
        ProjectContext = contextDim != dim ? IndexTtsOps.Linear(engine, layers, random, contextDim, dim) : null;
        Latents = IndexTtsOps.Values(layers, IndexTtsOps.Normal(random, latents * dim, 0.02));
        for (int i = 0; i < depth; i++)
            Layers.Add(new Layer(IndexTtsOps.BiasFree(layers, random, dim, inner), IndexTtsOps.BiasFree(layers, random, dim, 2 * inner),
                IndexTtsOps.BiasFree(layers, random, inner, dim), IndexTtsOps.Linear(engine, layers, random, dim, 2 * _feedForward),
                IndexTtsOps.Linear(engine, layers, random, _feedForward, dim)));
        Gamma = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, dim).ToArray());
    }

    public DenseLayer<T>? ProjectContext { get; }
    public BiasFreeLinearLayer<T> Latents { get; }
    public List<Layer> Layers { get; } = new();
    public BiasFreeLinearLayer<T> Gamma { get; }

    /// <summary>The pooled latents <c>[latents, dim]</c> of a context <c>[T, contextDim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> context)
    {
        var x = ProjectContext is null ? context : ProjectContext.Forward(context);
        var latents = _engine.Reshape(IndexTtsOps.Value(Latents), new[] { _latents, _dim });
        foreach (var layer in Layers)
        {
            var all = _engine.TensorConcatenate(new[] { latents, x }, 0);
            var q = IndexTtsOps.SplitHeads(_engine, layer.ToQuery.Forward(latents), _heads);
            var kv = layer.ToKeyValue.Forward(all);
            int inner = HeadDim * _heads;
            var k = IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(kv, 1, 0, inner), _heads);
            var v = IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(kv, 1, inner, inner), _heads);
            var attended = IndexTtsOps.MergeHeads(_engine, IndexTtsOps.Attend(_engine, q, k, v, 1.0 / Math.Sqrt(HeadDim), null));
            latents = _engine.TensorAdd(layer.ToOutput.Forward(attended), latents);
            var h = layer.FeedForward1.Forward(latents);
            var gated = _engine.TensorMultiply(_engine.TensorNarrow(h, 1, 0, _feedForward),
                ActivationHelper.ExactGelu(_engine.TensorNarrow(h, 1, _feedForward, _feedForward), _engine));
            latents = _engine.TensorAdd(layer.FeedForward2.Forward(gated), latents);
        }
        var norm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(latents, latents), new[] { 1 }, true));
        norm = _engine.TensorMax(norm, NumOps.FromDouble(1e-12));
        var normalized = _engine.TensorDivide(latents, _engine.TensorBroadcastTo(norm, new[] { _latents, _dim }));
        var gamma = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(Gamma), new[] { 1, _dim }), new[] { _latents, _dim });
        return _engine.TensorMultiply(_engine.TensorMultiplyScalar(normalized, NumOps.FromDouble(Math.Sqrt(_dim))), gamma);
    }

    /// <summary>Loads the release's <c>{prefix}proj_context</c>, <c>latents</c>, <c>layers.{i}.*</c> and <c>norm.gamma</c>.</summary>
    public void Load(IEngine engine, Func<string, int[], double[]> read, string prefix)
    {
        int inner = HeadDim * _heads;
        if (ProjectContext is not null) IndexTtsOps.LoadLinear(engine, read, ProjectContext, prefix + "proj_context", _contextDim, _dim);
        TorchParameters.Linear(Latents, 1, _latents * _dim, read(prefix + "latents", new[] { _latents, _dim }));
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            string p = $"{prefix}layers.{i}.";
            IndexTtsOps.LoadLinear(read, l.ToQuery, p + "0.to_q", _dim, inner);
            IndexTtsOps.LoadLinear(read, l.ToKeyValue, p + "0.to_kv", _dim, 2 * inner);
            IndexTtsOps.LoadLinear(read, l.ToOutput, p + "0.to_out", inner, _dim);
            IndexTtsOps.LoadLinear(engine, read, l.FeedForward1, p + "1.0", _dim, 2 * _feedForward);
            IndexTtsOps.LoadLinear(engine, read, l.FeedForward2, p + "1.2", _feedForward, _dim);
        }
        TorchParameters.Linear(Gamma, 1, _dim, read(prefix + "norm.gamma", new[] { _dim }));
    }
}
