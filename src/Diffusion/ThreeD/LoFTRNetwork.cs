using AiDotNet.NeuralRadianceFields.NeuS;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// LoFTR's tensors under the reference state-dict names (<c>backbone.layer1.0.conv1.weight</c>,
/// <c>loftr_coarse.layers.0.q_proj.weight</c>, …), in module registration order, with the batch-norm running
/// statistics marked as buffers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class LoFTRWeights<T>
{
    private readonly List<string> _names = new List<string>();
    private readonly Dictionary<string, Tensor<T>> _values = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
    private readonly HashSet<string> _buffers = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>The names in registration order.</summary>
    public IReadOnlyList<string> Names => _names;

    /// <summary>The tensor named <paramref name="name"/>.</summary>
    public Tensor<T> this[string name] => _values[name];

    /// <summary>Whether a tensor of that name exists.</summary>
    public bool Contains(string name) => _values.ContainsKey(name);

    /// <summary>Whether the tensor is a buffer (running statistics) rather than a trainable parameter.</summary>
    public bool IsBuffer(string name) => _buffers.Contains(name);

    /// <summary>The trainable element count (PyTorch's <c>sum(p.numel() for p in parameters())</c>).</summary>
    public int ParameterCount => _names.Where(n => !_buffers.Contains(n)).Sum(n => _values[n].Length);

    /// <summary>Registers a tensor.</summary>
    public void Add(string name, Tensor<T> value, bool buffer = false)
    {
        _names.Add(name);
        _values.Add(name, value);
        if (buffer) _buffers.Add(name);
    }

    /// <summary>Replaces a tensor; the element count must match and the stored shape is kept.</summary>
    public void Set(string name, Tensor<T> value)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));
        var current = _values[name];
        if (value.Length != current.Length)
            throw new ArgumentException($"'{name}' expects {current.Length} values, got {value.Length}.", nameof(value));
        _values[name] = value.Reshape(current.Shape.ToArray());
    }
}

/// <summary>
/// The LoFTR network (Sun et al. 2021, zju3dv/LoFTR, as vendored in One-2-3-45 <c>elevation_estimate/loftr/</c>):
/// the ResNet-FPN 8/2 backbone, the sine positional encoding, the coarse linear-attention transformer, dual-softmax
/// coarse matching, the fine window crop and fine transformer, and expectation-based fine refinement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Inference only: batch normalization uses its running statistics, as the reference runs in <c>eval()</c> mode.
/// Initialization follows the reference: Kaiming-normal (fan-out, ReLU gain) convolutions and fine-preprocess
/// weights, γ = 1 / β = 0 normalizations, and Xavier-uniform transformer weights.
/// </remarks>
internal sealed class LoFTRNetwork<T>
{
    private const double BatchNormEpsilon = 1e-5;
    private const double LayerNormEpsilon = 1e-5;
    private const double LeakySlope = 0.01;
    private const double AttentionEpsilon = 1e-6;

    private readonly IEngine _e;

    /// <summary>Builds and initializes the network.</summary>
    public LoFTRNetwork(LoFTROptions options, Random random, IEngine engine)
    {
        Options = options;
        _e = engine;
        int i0 = options.InitialDim, b0 = options.BlockDims[0], b1 = options.BlockDims[1], b2 = options.BlockDims[2];
        AddConv("backbone.conv1", 1, i0, 7, random);
        AddBatchNorm("backbone.bn1", i0);
        AddStage("backbone.layer1", i0, b0, 1, random);
        AddStage("backbone.layer2", b0, b1, 2, random);
        AddStage("backbone.layer3", b1, b2, 2, random);
        AddConv("backbone.layer3_outconv", b2, b2, 1, random);
        AddConv("backbone.layer2_outconv", b1, b2, 1, random);
        AddConv("backbone.layer2_outconv2.0", b2, b2, 3, random);
        AddBatchNorm("backbone.layer2_outconv2.1", b2);
        AddConv("backbone.layer2_outconv2.3", b2, b1, 3, random);
        AddConv("backbone.layer1_outconv", b0, b1, 1, random);
        AddConv("backbone.layer1_outconv2.0", b1, b1, 3, random);
        AddBatchNorm("backbone.layer1_outconv2.1", b1);
        AddConv("backbone.layer1_outconv2.3", b1, b0, 3, random);
        for (int l = 0; l < 2 * options.CoarseLayerPairs; l++) AddEncoderLayer($"loftr_coarse.layers.{l}", b2, random);
        if (options.FineConcatCoarseFeature)
        {
            AddLinear("fine_preprocess.down_proj", b2, b0, true, kaiming: true, random);
            AddLinear("fine_preprocess.merge_feat", 2 * b0, b0, true, kaiming: true, random);
        }

        for (int l = 0; l < 2 * options.FineLayerPairs; l++) AddEncoderLayer($"loftr_fine.layers.{l}", b0, random);
    }

    /// <summary>The configuration.</summary>
    public LoFTROptions Options { get; }

    /// <summary>The named tensors.</summary>
    public LoFTRWeights<T> Weights { get; } = new LoFTRWeights<T>();

    #region Construction

    private void AddConv(string name, int inChannels, int outChannels, int kernel, Random random)
    {
        // kaiming_normal_(mode='fan_out', nonlinearity='relu'): std = sqrt(2 / (out · k²)).
        double std = Math.Sqrt(2.0 / (outChannels * kernel * kernel));
        var w = new double[outChannels * inChannels * kernel * kernel];
        for (int i = 0; i < w.Length; i++) w[i] = std * NeuSHost<T>.NextGaussian(random);
        Weights.Add(name + ".weight", NeuSHost<T>.FromDoubles(w, outChannels, inChannels, kernel, kernel));
    }

    private void AddBatchNorm(string name, int channels)
    {
        Weights.Add(name + ".weight", NeuSHost<T>.Filled(1.0, channels));
        Weights.Add(name + ".bias", NeuSHost<T>.Filled(0.0, channels));
        Weights.Add(name + ".running_mean", NeuSHost<T>.Filled(0.0, channels), buffer: true);
        Weights.Add(name + ".running_var", NeuSHost<T>.Filled(1.0, channels), buffer: true);
    }

    private void AddStage(string prefix, int inChannels, int channels, int stride, Random random)
    {
        AddBlock(prefix + ".0", inChannels, channels, stride, random);
        AddBlock(prefix + ".1", channels, channels, 1, random);
    }

    private void AddBlock(string prefix, int inChannels, int channels, int stride, Random random)
    {
        // BasicBlock registers conv1, conv2, bn1, bn2, then the projection shortcut.
        AddConv(prefix + ".conv1", inChannels, channels, 3, random);
        AddConv(prefix + ".conv2", channels, channels, 3, random);
        AddBatchNorm(prefix + ".bn1", channels);
        AddBatchNorm(prefix + ".bn2", channels);
        if (stride != 1)
        {
            AddConv(prefix + ".downsample.0", inChannels, channels, 1, random);
            AddBatchNorm(prefix + ".downsample.1", channels);
        }
    }

    private void AddLinear(string name, int inputs, int outputs, bool bias, bool kaiming, Random random)
    {
        var w = new double[outputs * inputs];
        if (kaiming)
        {
            double std = Math.Sqrt(2.0 / outputs); // kaiming_normal_(mode='fan_out') on [out, in]
            for (int i = 0; i < w.Length; i++) w[i] = std * NeuSHost<T>.NextGaussian(random);
        }
        else
        {
            double bound = Math.Sqrt(6.0 / (inputs + outputs)); // xavier_uniform_
            for (int i = 0; i < w.Length; i++) w[i] = (random.NextDouble() * 2.0 - 1.0) * bound;
        }

        Weights.Add(name + ".weight", NeuSHost<T>.FromDoubles(w, outputs, inputs));
        if (!bias) return;
        double biasBound = 1.0 / Math.Sqrt(inputs); // nn.Linear's default bias initialization
        var b = new double[outputs];
        for (int i = 0; i < b.Length; i++) b[i] = (random.NextDouble() * 2.0 - 1.0) * biasBound;
        Weights.Add(name + ".bias", NeuSHost<T>.FromDoubles(b, outputs));
    }

    private void AddEncoderLayer(string prefix, int d, Random random)
    {
        AddLinear(prefix + ".q_proj", d, d, false, kaiming: false, random);
        AddLinear(prefix + ".k_proj", d, d, false, kaiming: false, random);
        AddLinear(prefix + ".v_proj", d, d, false, kaiming: false, random);
        AddLinear(prefix + ".merge", d, d, false, kaiming: false, random);
        AddLinear(prefix + ".mlp.0", 2 * d, 2 * d, false, kaiming: false, random);
        AddLinear(prefix + ".mlp.2", 2 * d, d, false, kaiming: false, random);
        Weights.Add(prefix + ".norm1.weight", NeuSHost<T>.Filled(1.0, d));
        Weights.Add(prefix + ".norm1.bias", NeuSHost<T>.Filled(0.0, d));
        Weights.Add(prefix + ".norm2.weight", NeuSHost<T>.Filled(1.0, d));
        Weights.Add(prefix + ".norm2.bias", NeuSHost<T>.Filled(0.0, d));
    }

    #endregion

    #region Backbone

    /// <summary><c>ResNetFPN_8_2.forward</c> on <c>[N, 1, H, W]</c>: the 1/8 coarse and 1/2 fine maps.</summary>
    public (Tensor<T> Coarse, Tensor<T> Fine) Backbone(Tensor<T> x)
    {
        var x0 = _e.TensorReLU(BatchNorm(Conv(x, "backbone.conv1", 2, 3), "backbone.bn1"));
        var x1 = Stage(x0, "backbone.layer1", 1);
        var x2 = Stage(x1, "backbone.layer2", 2);
        var x3 = Stage(x2, "backbone.layer3", 2);

        var x3Out = Conv(x3, "backbone.layer3_outconv", 1, 0);
        var x2Out = Conv(x2, "backbone.layer2_outconv", 1, 0);
        x2Out = OutConv2(_e.TensorAdd(x2Out, Upsample(x3Out)), "backbone.layer2_outconv2");
        var x1Out = Conv(x1, "backbone.layer1_outconv", 1, 0);
        x1Out = OutConv2(_e.TensorAdd(x1Out, Upsample(x2Out)), "backbone.layer1_outconv2");
        return (x3Out, x1Out);
    }

    private Tensor<T> Stage(Tensor<T> x, string prefix, int stride)
        => Block(Block(x, prefix + ".0", stride), prefix + ".1", 1);

    private Tensor<T> Block(Tensor<T> x, string prefix, int stride)
    {
        var y = _e.TensorReLU(BatchNorm(Conv(x, prefix + ".conv1", stride, 1), prefix + ".bn1"));
        y = BatchNorm(Conv(y, prefix + ".conv2", 1, 1), prefix + ".bn2");
        if (stride != 1) x = BatchNorm(Conv(x, prefix + ".downsample.0", stride, 0), prefix + ".downsample.1");
        return _e.TensorReLU(_e.TensorAdd(x, y));
    }

    private Tensor<T> OutConv2(Tensor<T> x, string prefix)
    {
        var y = _e.TensorLeakyReLU(BatchNorm(Conv(x, prefix + ".0", 1, 1), prefix + ".1"), NeuSHost<T>.Scalar(LeakySlope));
        return Conv(y, prefix + ".3", 1, 1);
    }

    // F.interpolate(scale_factor=2, mode='bilinear', align_corners=True).
    private Tensor<T> Upsample(Tensor<T> x)
        => _e.Interpolate(x, new[] { x.Shape[2] * 2, x.Shape[3] * 2 }, InterpolateMode.Bilinear, true);

    private Tensor<T> Conv(Tensor<T> x, string name, int stride, int padding)
        => _e.Conv2D(x, Weights[name + ".weight"], stride, padding, 1);

    private Tensor<T> BatchNorm(Tensor<T> x, string name)
    {
        int c = x.Shape[1];
        var shape = new[] { 1, c, 1, 1 };
        var scale = _e.TensorDivide(Weights[name + ".weight"],
            _e.TensorSqrt(_e.TensorAddScalar(Weights[name + ".running_var"], NeuSHost<T>.Epsilon(BatchNormEpsilon))));
        var shift = _e.TensorSubtract(Weights[name + ".bias"], _e.TensorMultiply(Weights[name + ".running_mean"], scale));
        return _e.TensorAdd(_e.TensorMultiply(x, _e.Reshape(scale, shape)), _e.Reshape(shift, shape));
    }

    #endregion

    #region Positional encoding

    /// <summary>
    /// <c>PositionEncodingSine</c> for a <c>[C, h, w]</c> map: channel <c>4k</c> is <c>sin(x·f_k)</c>, <c>4k+1</c>
    /// <c>cos(x·f_k)</c>, <c>4k+2</c> <c>sin(y·f_k)</c> and <c>4k+3</c> <c>cos(y·f_k)</c>, with 1-based pixel
    /// coordinates and <c>f_k = exp(2k · (−ln 10⁴ / (C/2)))</c> (the original release's floor-divided exponent
    /// without the bug fix).
    /// </summary>
    internal static double[] PositionEncoding(int channels, int height, int width, bool bugFix)
    {
        double coefficient = bugFix
            ? -Math.Log(10000.0) / (channels / 2)
            : Math.Floor(-Math.Log(10000.0) / channels / 2.0);
        var pe = new double[channels * height * width];
        for (int k = 0; k < channels / 4; k++)
        {
            double frequency = Math.Exp(2 * k * coefficient);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int pixel = y * width + x;
                    pe[(4 * k) * height * width + pixel] = Math.Sin((x + 1) * frequency);
                    pe[(4 * k + 1) * height * width + pixel] = Math.Cos((x + 1) * frequency);
                    pe[(4 * k + 2) * height * width + pixel] = Math.Sin((y + 1) * frequency);
                    pe[(4 * k + 3) * height * width + pixel] = Math.Cos((y + 1) * frequency);
                }
        }

        return pe;
    }

    #endregion

    #region Transformer

    /// <summary><c>LocalFeatureTransformer.forward</c> on <c>[N, L, C]</c> and <c>[N, S, C]</c> features.</summary>
    public (Tensor<T> Feat0, Tensor<T> Feat1) Transformer(Tensor<T> feat0, Tensor<T> feat1, string prefix, int pairs, int heads)
    {
        for (int p = 0; p < pairs; p++)
        {
            string self = $"{prefix}.layers.{2 * p}", cross = $"{prefix}.layers.{2 * p + 1}";
            feat0 = EncoderLayer(feat0, feat0, self, heads);
            feat1 = EncoderLayer(feat1, feat1, self, heads);
            feat0 = EncoderLayer(feat0, feat1, cross, heads);
            feat1 = EncoderLayer(feat1, feat0, cross, heads);
        }

        return (feat0, feat1);
    }

    /// <summary><c>LoFTREncoderLayer.forward</c>: linear multi-head attention, merge, norm, then the MLP on
    /// <c>[x, message]</c>, norm, and the residual.</summary>
    private Tensor<T> EncoderLayer(Tensor<T> x, Tensor<T> source, string prefix, int heads)
    {
        int n = x.Shape[0], l = x.Shape[1], s = source.Shape[1], c = x.Shape[2], d = c / heads;
        var query = _e.Reshape(Linear(x, prefix + ".q_proj"), new[] { n, l, heads, d });
        var key = _e.Reshape(Linear(source, prefix + ".k_proj"), new[] { n, s, heads, d });
        var value = _e.Reshape(Linear(source, prefix + ".v_proj"), new[] { n, s, heads, d });
        var message = _e.Reshape(LinearAttention(_e, query, key, value), new[] { n, l, c });
        message = LayerNorm(Linear(message, prefix + ".merge"), prefix + ".norm1");
        message = Linear(_e.TensorReLU(Linear(_e.TensorConcatenate(new[] { x, message }, 2), prefix + ".mlp.0")), prefix + ".mlp.2");
        message = LayerNorm(message, prefix + ".norm2");
        return _e.TensorAdd(x, message);
    }

    /// <summary>
    /// <c>LinearAttention.forward</c> (Katharopoulos et al. 2020) on <c>[N, L, H, D]</c> queries and <c>[N, S, H, D]</c>
    /// keys and values: <c>φ(x) = elu(x) + 1</c>, values divided by S, <c>KV = φ(K)ᵀV</c>,
    /// <c>Z = 1 / (φ(Q)·Σ_s φ(K_s) + 10⁻⁶)</c>, and the output <c>φ(Q)·KV·Z·S</c>.
    /// </summary>
    internal static Tensor<T> LinearAttention(IEngine e, Tensor<T> queries, Tensor<T> keys, Tensor<T> values)
    {
        int n = queries.Shape[0], l = queries.Shape[1], h = queries.Shape[2], d = queries.Shape[3];
        int s = keys.Shape[1], dv = values.Shape[3];
        var one = NeuSHost<T>.Scalar(1.0);
        var q = e.Reshape(e.TensorPermute(e.TensorAddScalar(e.ELU(queries, 1.0), one), new[] { 0, 2, 1, 3 }), new[] { n * h, l, d });
        var k = e.Reshape(e.TensorPermute(e.TensorAddScalar(e.ELU(keys, 1.0), one), new[] { 0, 2, 1, 3 }), new[] { n * h, s, d });
        var length = NeuSHost<T>.Scalar(s);
        var v = e.Reshape(e.TensorPermute(e.TensorDivideScalar(values, length), new[] { 0, 2, 1, 3 }), new[] { n * h, s, dv });
        var kv = e.BatchMatMul(e.TensorPermute(k, new[] { 0, 2, 1 }), v);                         // [NH, D, Dv]
        var keySum = e.TensorPermute(e.ReduceSum(k, new[] { 1 }, true), new[] { 0, 2, 1 });       // [NH, D, 1]
        var z = e.TensorReciprocal(e.TensorAddScalar(e.BatchMatMul(q, keySum), NeuSHost<T>.Epsilon(AttentionEpsilon))); // [NH, L, 1]
        var output = e.TensorMultiplyScalar(e.TensorMultiply(e.BatchMatMul(q, kv), z), length);   // [NH, L, Dv]
        return e.TensorPermute(e.Reshape(output, new[] { n, h, l, dv }), new[] { 0, 2, 1, 3 });
    }

    /// <summary><c>nn.Linear</c> on the last axis; the bias is optional.</summary>
    public Tensor<T> Linear(Tensor<T> x, string name)
    {
        var weight = Weights[name + ".weight"];
        int outputs = weight.Shape[0], inputs = weight.Shape[1];
        var shape = x.Shape.ToArray();
        var flat = _e.Reshape(x, new[] { x.Length / inputs, inputs });
        var y = _e.TensorMatMul(flat, _e.TensorTranspose(weight));
        if (Weights.Contains(name + ".bias")) y = _e.TensorAdd(y, _e.Reshape(Weights[name + ".bias"], new[] { 1, outputs }));
        shape[shape.Length - 1] = outputs;
        return _e.Reshape(y, shape);
    }

    private Tensor<T> LayerNorm(Tensor<T> x, string name)
        => _e.LayerNorm(x, Weights[name + ".weight"], Weights[name + ".bias"], LayerNormEpsilon, out _, out _);

    #endregion

    #region Matching

    /// <summary>
    /// The dual-softmax confidence of <c>CoarseMatching.forward</c> for <c>[L, C]</c> and <c>[S, C]</c> features:
    /// both scaled by <c>1/√C</c>, <c>sim = f₀f₁ᵀ / τ</c>, and <c>softmax(sim, 0) · softmax(sim, 1)</c>.
    /// </summary>
    internal static Tensor<T> DualSoftmax(IEngine e, Tensor<T> feat0, Tensor<T> feat1, double temperature)
    {
        int c = feat0.Shape[1];
        // (f₀/√C)(f₁/√C)ᵀ/τ = f₀f₁ᵀ / (C·τ).
        var sim = e.TensorMultiplyScalar(e.TensorMatMul(feat0, e.TensorTranspose(feat1)), NeuSHost<T>.Scalar(1.0 / (c * temperature)));
        return e.TensorMultiply(e.TensorSoftmax(sim, 0), e.TensorSoftmax(sim, 1));
    }

    /// <summary>
    /// <c>get_coarse_match</c> on a host copy of the <c>[L, S]</c> confidence: above the threshold, outside the
    /// border band of both grids, and the maximum of both its row and its column (mutual nearest); per row the
    /// first such column, as <c>mask.max(dim=2)</c> picks.
    /// </summary>
    internal static List<(int I, int J, double Confidence)> CoarseMatches(double[] confidence, int h0, int w0, int h1, int w1,
        double threshold, int border)
    {
        int l = h0 * w0, s = h1 * w1;
        if (confidence.Length != l * s) throw new ArgumentException("The confidence has the wrong size.", nameof(confidence));
        var columnMax = new double[s];
        for (int j = 0; j < s; j++) columnMax[j] = double.NegativeInfinity;
        for (int i = 0; i < l; i++)
            for (int j = 0; j < s; j++)
                columnMax[j] = Math.Max(columnMax[j], confidence[i * s + j]);

        var matches = new List<(int, int, double)>();
        for (int i = 0; i < l; i++)
        {
            if (!Inside(i, h0, w0, border)) continue;
            double rowMax = double.NegativeInfinity;
            for (int j = 0; j < s; j++) rowMax = Math.Max(rowMax, confidence[i * s + j]);
            for (int j = 0; j < s; j++)
            {
                double v = confidence[i * s + j];
                if (v != rowMax || !(v > threshold) || v != columnMax[j] || !Inside(j, h1, w1, border)) continue;
                matches.Add((i, j, v));
                break;
            }
        }

        return matches;
    }

    private static bool Inside(int index, int height, int width, int border)
    {
        if (border <= 0) return true;
        int row = index / width, column = index % width;
        return row >= border && row < height - border && column >= border && column < width - border;
    }

    /// <summary>
    /// The <c>F.unfold</c> window rows of <c>FinePreprocess</c>: for coarse cell <c>id</c> on a grid of width
    /// <paramref name="coarseWidth"/>, the <c>W × W</c> fine pixels centered on <c>(row·stride, col·stride)</c>,
    /// row-major, with <paramref name="padRow"/> standing for the zero padding.
    /// </summary>
    internal static int[] WindowRows(IReadOnlyList<int> ids, int coarseWidth, int stride, int window, int fineHeight, int fineWidth, int padRow)
    {
        int half = window / 2, ww = window * window;
        var rows = new int[ids.Count * ww];
        for (int m = 0; m < ids.Count; m++)
        {
            int cy = ids[m] / coarseWidth * stride, cx = ids[m] % coarseWidth * stride;
            for (int dy = 0; dy < window; dy++)
                for (int dx = 0; dx < window; dx++)
                {
                    int y = cy - half + dy, x = cx - half + dx;
                    rows[m * ww + dy * window + dx] = y >= 0 && y < fineHeight && x >= 0 && x < fineWidth ? y * fineWidth + x : padRow;
                }
        }

        return rows;
    }

    /// <summary>
    /// <c>FineMatching.forward</c>'s expectation on <c>[M, W², C]</c> windows: the first window's center feature
    /// against every second-window feature, <c>softmax(sim / √C)</c> as a <c>W × W</c> heatmap, and its spatial
    /// expectation over the normalized grid in [−1, 1] (<c>dsnt.spatial_expectation2d</c>), as <c>[M, 2]</c> (x, y).
    /// </summary>
    internal static Tensor<T> FineExpectation(IEngine e, Tensor<T> feat0, Tensor<T> feat1, int window)
    {
        int m = feat0.Shape[0], ww = feat0.Shape[1], c = feat0.Shape[2];
        var center = e.Reshape(e.TensorSlice(feat0, new[] { 0, ww / 2, 0 }, new[] { m, 1, c }), new[] { m, c, 1 });
        var sim = e.Reshape(e.BatchMatMul(feat1, center), new[] { m, ww });
        var heatmap = e.TensorSoftmax(e.TensorMultiplyScalar(sim, NeuSHost<T>.Scalar(1.0 / Math.Sqrt(c))), 1);
        var grid = new double[ww * 2];
        for (int dy = 0; dy < window; dy++)
            for (int dx = 0; dx < window; dx++)
            {
                grid[(dy * window + dx) * 2] = -1.0 + 2.0 * dx / (window - 1);
                grid[(dy * window + dx) * 2 + 1] = -1.0 + 2.0 * dy / (window - 1);
            }

        return e.TensorMatMul(heatmap, NeuSHost<T>.FromDoubles(grid, ww, 2));
    }

    #endregion
}
