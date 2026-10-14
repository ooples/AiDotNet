using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// The sizes of a Vision Mamba (the official <c>VisionMamba</c>; defaults are Vim-Tiny at 224 pixels).
/// </summary>
internal sealed record VimConfiguration
{
    public int ImageHeight { get; init; } = 224;
    public int ImageWidth { get; init; } = 224;
    public int Channels { get; init; } = 3;
    public int PatchSize { get; init; } = 16;

    /// <summary>The patch stride (16 for the standard models, 8 for the overlapping "stride8" ones).</summary>
    public int Stride { get; init; } = 16;

    public int Dim { get; init; } = 192;
    public int Depth { get; init; } = 24;
    public int StateDim { get; init; } = 16;
    public int ConvKernel { get; init; } = 4;
    public int Expand { get; init; } = 2;
    public double NormEpsilon { get; init; } = 1e-5;

    /// <summary>The stochastic-depth rate: layer i drops its mixer output with probability rate·(i−1)/(depth−1)
    /// (0 for the first layer), and the final residual add with the full rate.</summary>
    public double DropPathRate { get; init; } = 0.1;

    public int InnerDim => Expand * Dim;
    public int DtRank => (int)Math.Ceiling(Dim / 16.0);
    public int GridHeight => (ImageHeight - PatchSize) / Stride + 1;
    public int GridWidth => (ImageWidth - PatchSize) / Stride + 1;
    public int Patches => GridHeight * GridWidth;

    /// <summary>A class token in the middle of the sequence (the classifier); the segmentation backbone has none.</summary>
    public bool ClassToken { get; init; } = true;

    /// <summary>2D rotary position embedding on the hidden states before every layer (the official
    /// <c>VisionRotaryEmbeddingFast</c>; the Vim-Small segmentation configuration uses it).</summary>
    public bool Rope { get; init; }

    /// <summary>With <see cref="Rope"/>, rotate the residual stream as well (<c>if_rope_residual</c>).</summary>
    public bool RopeResidual { get; init; }

    /// <summary>The pre-training grid side that RoPE positions are scaled to (<c>pt_hw_seq_len</c>).</summary>
    public int RopePretrainGrid { get; init; } = 14;

    /// <summary>The patches and, with <see cref="ClassToken"/>, the class token.</summary>
    public int Tokens => Patches + (ClassToken ? 1 : 0);

    /// <summary>Where the class token sits: the middle of the patch sequence.</summary>
    public int ClassTokenPosition => Patches / 2;
}

/// <summary>
/// Vision Mamba (Zhu et al. 2024) as the official code (hustvl/Vim, <c>models_mamba.VisionMamba</c>) builds it.
/// </summary>
/// <remarks>
/// <para>The image is cut into patches by a strided convolution; a learned class token is inserted in the middle of the
/// patch sequence and a learned position table added. Each of the layers keeps a residual stream: it adds the previous
/// layer's output to the residual, RMS-normalizes the sum, and runs a bidirectional Mamba mixer on it ("bimamba v2"):
/// one input projection, then two complete Mamba branches (causal depthwise convolution, SiLU, the input-dependent
/// Δ, B and C projections, the selective scan with its skip D, the SiLU gate) over the sequence and over its reverse,
/// averaged, then one output projection. A final add and RMSNorm give the token features; the classifier reads the
/// class token.</para>
/// <para>The selective scan follows the reference implementation (<c>selective_scan_ref</c>):
/// <c>h_t = exp(Δ_t·A)·h_{t−1} + Δ_t·B_t·u_t</c>, <c>y_t = C_t·h_t + D·u_t</c>, with <c>Δ = softplus(dt_proj(·))</c>
/// and <c>A = −exp(A_log)</c>.</para>
/// </remarks>
internal sealed class VimNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _owned = new();

    public VimNetwork(IEngine engine, List<LayerBase<T>> layers, VimConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.GridHeight <= 0 || c.GridWidth <= 0)
            throw new ArgumentException($"A {c.ImageHeight}x{c.ImageWidth} image has no {c.PatchSize}-pixel patch.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            _owned.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Plain(int inputs, int outputs) => Own(new BiasFreeLinearLayer<T>(inputs, outputs));
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        // A learned tensor of n values: a bias-free projection of the constant 1.
        BiasFreeLinearLayer<T> Values(int count) => Plain(1, count);

        PatchProjection = Own(ConvolutionalLayer<T>.WithInputDepth(c.Channels, c.Dim, c.PatchSize, c.Stride, padding: 0));
        ClassToken = c.ClassToken ? Values(c.Dim) : null;
        PositionTable = Values(c.Tokens * c.Dim);
        int inner = c.InnerDim, projected = c.DtRank + 2 * c.StateDim;
        for (int i = 0; i < c.Depth; i++)
        {
            Branch Direction() => new(Values(inner * c.ConvKernel), Values(inner), Plain(inner, projected), Linear(inner),
                Values(inner * c.StateDim), Values(inner));
            Layers.Add(new Layer(Own(new RMSNormalizationLayer<T>(c.Dim, c.NormEpsilon)), Plain(c.Dim, 2 * inner),
                Direction(), Direction(), Plain(inner, c.Dim)));
        }
        FinalNorm = Own(new RMSNormalizationLayer<T>(c.Dim, c.NormEpsilon));
        if (c.Rope)
        {
            if (c.ClassToken || c.GridHeight != c.GridWidth || c.Dim % 4 != 0)
                throw new ArgumentException("RoPE is defined here for a square patch grid without a class token and a width divisible by 4.");
            (_ropeCos, _ropeSin, _rotateHalf) = RopeTables(c);
        }
    }

    private readonly Tensor<T>? _ropeCos, _ropeSin, _rotateHalf;

    /// <summary>The cos and sin tables <c>[Patches, Dim]</c> of VisionRotaryEmbeddingFast and the constant matrix of
    /// <c>rotate_half</c> (pairs (a, b) → (−b, a)).</summary>
    private static (Tensor<T> Cos, Tensor<T> Sin, Tensor<T> RotateHalf) RopeTables(VimConfiguration c)
    {
        int d = c.Dim, half = d / 2, side = c.GridHeight;
        var frequencies = new double[half / 2];
        for (int k = 0; k < frequencies.Length; k++) frequencies[k] = 1.0 / Math.Pow(10000.0, 2.0 * k / half);
        double[] Axis(int position)
        {
            double t = (double)position / side * c.RopePretrainGrid;
            var v = new double[half];
            for (int k = 0; k < frequencies.Length; k++) v[2 * k] = v[2 * k + 1] = t * frequencies[k];
            return v;
        }
        var cos = new Tensor<T>(new[] { c.Patches, d });
        var sin = new Tensor<T>(new[] { c.Patches, d });
        for (int i = 0; i < side; i++)
            for (int j = 0; j < side; j++)
            {
                var row = Axis(i);
                var column = Axis(j);
                for (int k = 0; k < d; k++)
                {
                    double angle = k < half ? row[k] : column[k - half];
                    cos[i * side + j, k] = NumOps.FromDouble(Math.Cos(angle));
                    sin[i * side + j, k] = NumOps.FromDouble(Math.Sin(angle));
                }
            }
        var rotate = new Tensor<T>(new[] { d, d });
        for (int k = 0; k < d; k += 2)
        {
            rotate[k + 1, k] = NumOps.FromDouble(-1.0);
            rotate[k, k + 1] = NumOps.One;
        }
        return (cos, sin, rotate);
    }

    private Tensor<T> ApplyRope(Tensor<T> x)
    {
        int batch = x.Shape[0], tokens = x.Shape[1], d = x.Shape[2];
        var shape = new[] { batch, tokens, d };
        var rotated = _engine.Reshape(_engine.TensorMatMul(_engine.Reshape(x, new[] { batch * tokens, d }), _rotateHalf!), shape);
        return _engine.TensorAdd(
            _engine.TensorMultiply(x, Broadcast(_ropeCos!, new[] { 1, tokens, d }, shape)),
            _engine.TensorMultiply(rotated, Broadcast(_ropeSin!, new[] { 1, tokens, d }, shape)));
    }

    /// <summary>One direction of a bidirectional Mamba mixer: the depthwise convolution's kernel <c>[inner, K]</c> and
    /// bias, the Δ/B/C projection, the Δ projection (its bias is the scan's Δ bias), <c>A_log</c> <c>[inner, N]</c>
    /// and the skip D.</summary>
    internal sealed record Branch(BiasFreeLinearLayer<T> ConvKernel, BiasFreeLinearLayer<T> ConvBias,
        BiasFreeLinearLayer<T> XProjection, DenseLayer<T> DtProjection, BiasFreeLinearLayer<T> ALog,
        BiasFreeLinearLayer<T> Skip);

    /// <summary>A layer: its RMSNorm, the mixer's input projection, the forward and backward branches, and the output
    /// projection.</summary>
    internal sealed record Layer(RMSNormalizationLayer<T> Norm, BiasFreeLinearLayer<T> InputProjection, Branch Forward,
        Branch Backward, BiasFreeLinearLayer<T> OutputProjection);

    public VimConfiguration Configuration { get; }
    public ConvolutionalLayer<T> PatchProjection { get; }
    public BiasFreeLinearLayer<T>? ClassToken { get; }
    public BiasFreeLinearLayer<T> PositionTable { get; }
    public List<Layer> Layers { get; } = new();
    public RMSNormalizationLayer<T> FinalNorm { get; }

    /// <summary>The layers this network added to its owner's list, in order.</summary>
    public IReadOnlyList<LayerBase<T>> OwnedLayers => _owned;

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder)
        => holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));

    private Tensor<T> Broadcast(Tensor<T> x, int[] shape, int[] target)
        => _engine.TensorBroadcastTo(_engine.Reshape(x, shape), target);

    /// <summary>
    /// The normalized token features <c>[B, Tokens, Dim]</c> of an image batch <c>[B, C, H, W]</c>, the class token at
    /// <see cref="VimConfiguration.ClassTokenPosition"/>. Training draws stochastic-depth decisions from
    /// <paramref name="random"/>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> images, bool training, Random random)
    {
        var (hidden, residual) = Run(images, training, random, null);
        residual = _engine.TensorAdd(residual, DropPath(hidden, Configuration.DropPathRate, training, random));
        return Normalize(FinalNorm, residual);
    }

    /// <summary>
    /// The mixer outputs of the layers in <paramref name="indices"/>, each as a map <c>[B, Dim, GridHeight, GridWidth]</c>
    /// (the segmentation backbone's features; it needs no class token).
    /// </summary>
    public List<Tensor<T>> LayerOutputs(Tensor<T> images, IReadOnlyList<int> indices, bool training, Random random)
    {
        var c = Configuration;
        if (c.ClassToken) throw new InvalidOperationException("Layer maps need a network without a class token.");
        var maps = new List<Tensor<T>>();
        Run(images, training, random, (i, hidden) =>
        {
            if (!indices.Contains(i)) return;
            maps.Add(_engine.Reshape(_engine.TensorPermute(hidden, new[] { 0, 2, 1 }),
                new[] { hidden.Shape[0], c.Dim, c.GridHeight, c.GridWidth }));
        });
        return maps;
    }

    private (Tensor<T> Hidden, Tensor<T> Residual) Run(Tensor<T> images, bool training, Random random, Action<int, Tensor<T>>? tap)
    {
        var c = Configuration;
        int batch = images.Shape[0];
        var patches = PatchProjection.Forward(images);                                            // [B, D, h, w]
        var x = _engine.TensorPermute(_engine.Reshape(patches, new[] { batch, c.Dim, c.Patches }), new[] { 0, 2, 1 });
        if (ClassToken is not null)
        {
            var cls = Broadcast(Value(ClassToken), new[] { 1, 1, c.Dim }, new[] { batch, 1, c.Dim });
            int at = c.ClassTokenPosition;
            var parts = new List<Tensor<T>>();
            if (at > 0) parts.Add(_engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, at, c.Dim }));
            parts.Add(cls);
            if (at < c.Patches) parts.Add(_engine.TensorSlice(x, new[] { 0, at, 0 }, new[] { batch, c.Patches - at, c.Dim }));
            x = _engine.TensorConcatenate(parts.ToArray(), 1);
        }
        x = _engine.TensorAdd(x, Broadcast(Value(PositionTable), new[] { 1, c.Tokens, c.Dim }, new[] { batch, c.Tokens, c.Dim }));

        Tensor<T>? residual = null;
        var hidden = x;
        for (int i = 0; i < Layers.Count; i++)
        {
            // timm's stochastic-depth rule, shifted by one: layer i drops with the rate of layer i − 1.
            double rate = i == 0 || c.Depth < 2 ? 0.0 : c.DropPathRate * (i - 1) / (c.Depth - 1);
            if (c.Rope)
            {
                hidden = ApplyRope(hidden);
                if (residual is not null && c.RopeResidual) residual = ApplyRope(residual);
            }
            residual = residual is null ? hidden : _engine.TensorAdd(residual, DropPath(hidden, rate, training, random));
            hidden = Mixer(Layers[i], Normalize(Layers[i].Norm, residual));
            tap?.Invoke(i, hidden);
        }
        return (hidden, residual ?? hidden);
    }

    /// <summary>The class token's features <c>[B, Dim]</c> from <see cref="Forward"/>'s output.</summary>
    public Tensor<T> ClassFeatures(Tensor<T> tokens)
    {
        var c = Configuration;
        return _engine.Reshape(_engine.TensorSlice(tokens, new[] { 0, c.ClassTokenPosition, 0 }, new[] { tokens.Shape[0], 1, c.Dim }),
            new[] { tokens.Shape[0], c.Dim });
    }

    private Tensor<T> Normalize(RMSNormalizationLayer<T> norm, Tensor<T> x)
    {
        int batch = x.Shape[0], tokens = x.Shape[1];
        return _engine.Reshape(norm.Forward(_engine.Reshape(x, new[] { batch * tokens, x.Shape[2] })), x.Shape.ToArray());
    }

    private Tensor<T> DropPath(Tensor<T> x, double rate, bool training, Random random)
    {
        if (!training || rate <= 0) return x;
        var shape = x.Shape.ToArray();
        var mask = new Tensor<T>(new[] { shape[0], 1, 1 });
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int b = 0; b < shape[0]; b++) mask[b] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(mask, shape));
    }

    private Tensor<T> Project(LayerBase<T> projection, Tensor<T> x)
    {
        int batch = x.Shape[0], tokens = x.Shape[1];
        var y = projection.Forward(_engine.Reshape(x, new[] { batch * tokens, x.Shape[2] }));
        return _engine.Reshape(y, new[] { batch, tokens, y.Shape[1] });
    }

    /// <summary>The bidirectional ("v2") Mamba mixer on <c>[B, L, Dim]</c>.</summary>
    private Tensor<T> Mixer(Layer layer, Tensor<T> x)
    {
        var xz = Project(layer.InputProjection, x);                                               // [B, L, 2·inner]
        var forward = Direction(layer.Forward, xz);
        var backward = _engine.TensorFlip(Direction(layer.Backward, _engine.TensorFlip(xz, new[] { 1 })), new[] { 1 });
        var mixed = _engine.TensorMultiplyScalar(_engine.TensorAdd(forward, backward), NumOps.FromDouble(0.5));
        return Project(layer.OutputProjection, mixed);
    }

    /// <summary>One Mamba branch (<c>mamba_inner_ref</c> without its output projection) on <c>xz</c> <c>[B, L, 2·inner]</c>.</summary>
    private Tensor<T> Direction(Branch branch, Tensor<T> xz)
    {
        var c = Configuration;
        int batch = xz.Shape[0], length = xz.Shape[1], inner = c.InnerDim, n = c.StateDim, k = c.ConvKernel, rank = c.DtRank;
        var u = _engine.TensorSlice(xz, new[] { 0, 0, 0 }, new[] { batch, length, inner });
        var z = _engine.TensorSlice(xz, new[] { 0, 0, inner }, new[] { batch, length, inner });

        // The causal depthwise convolution: out_t = b + Σ_j w_j · u_{t−(K−1)+j}, zeros before the sequence.
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { batch, k - 1, inner }), u }, 1);
        var kernel = _engine.Reshape(Value(branch.ConvKernel), new[] { inner, k });
        var shape = new[] { batch, length, inner };
        var conv = Broadcast(Value(branch.ConvBias), new[] { 1, 1, inner }, shape);
        for (int j = 0; j < k; j++)
        {
            var tap = _engine.Reshape(_engine.TensorSlice(kernel, new[] { 0, j }, new[] { inner, 1 }), new[] { 1, 1, inner });
            conv = _engine.TensorAdd(conv, _engine.TensorMultiply(
                _engine.TensorSlice(padded, new[] { 0, j, 0 }, new[] { batch, length, inner }), _engine.TensorBroadcastTo(tap, shape)));
        }
        u = _engine.TensorSiLU(conv);

        var projected = Project(branch.XProjection, u);                                          // [B, L, R + 2N]
        var dt = _engine.TensorSlice(projected, new[] { 0, 0, 0 }, new[] { batch, length, rank });
        var bInput = _engine.TensorSlice(projected, new[] { 0, 0, rank }, new[] { batch, length, n });
        var cInput = _engine.TensorSlice(projected, new[] { 0, 0, rank + n }, new[] { batch, length, n });
        var delta = _engine.Softplus(Project(branch.DtProjection, dt));                          // [B, L, inner]
        // The selective scan with its skip, as one differentiable op (the reference's selective_scan_ref):
        // h_t = exp(Δ_t·A)·h_{t−1} + Δ_t·B_t·u_t, y_t = C_t·h_t + D·u_t, A = −exp(A_log).
        var y = _engine.MambaSelectiveScanForward(u, delta, _engine.Reshape(Value(branch.ALog), new[] { inner, n }), bInput, cInput,
            _engine.Reshape(Value(branch.Skip), new[] { inner }));
        return _engine.TensorMultiply(y, _engine.TensorSiLU(z));
    }

    // ------------------------------------------------------------------ parameters

    private static void LoadValues(BiasFreeLinearLayer<T> holder, double[] values)
        => TorchParameters.Linear(holder, 1, values.Length, values);

    /// <summary>Loads an official VisionMamba state dict (its <c>head</c> entries are the owner's), given a reader of a
    /// named tensor of a given shape.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "")
    {
        var inner0 = read;
        read = (name, shape) => inner0(prefix + name, shape);
        var c = Configuration;
        int inner = c.InnerDim, n = c.StateDim, k = c.ConvKernel, rank = c.DtRank, projected = rank + 2 * n;
        var kernel = read("patch_embed.proj.weight", new[] { c.Dim, c.Channels, c.PatchSize, c.PatchSize });
        var bias = read("patch_embed.proj.bias", new[] { c.Dim });
        var patch = new Vector<T>(kernel.Length + bias.Length);
        for (int i = 0; i < kernel.Length; i++) patch[i] = NumOps.FromDouble(kernel[i]);
        for (int i = 0; i < bias.Length; i++) patch[kernel.Length + i] = NumOps.FromDouble(bias[i]);
        PatchProjection.SetParameters(patch);
        if (ClassToken is not null) LoadValues(ClassToken, read("cls_token", new[] { 1, 1, c.Dim }));
        LoadValues(PositionTable, read("pos_embed", new[] { 1, c.Tokens, c.Dim }));
        void Norm(RMSNormalizationLayer<T> norm, string name)
            => norm.SetParameters(new Vector<T>(Array.ConvertAll(read(name, new[] { c.Dim }), NumOps.FromDouble)));
        void Direction(Branch branch, string p, string suffix)
        {
            LoadValues(branch.ConvKernel, read($"{p}.conv1d{suffix}.weight", new[] { inner, 1, k }));
            LoadValues(branch.ConvBias, read($"{p}.conv1d{suffix}.bias", new[] { inner }));
            TorchParameters.Linear(branch.XProjection, inner, projected, read($"{p}.x_proj{suffix}.weight", new[] { projected, inner }));
            TorchParameters.Linear(_engine, branch.DtProjection, rank, inner, read($"{p}.dt_proj{suffix}.weight", new[] { inner, rank }),
                read($"{p}.dt_proj{suffix}.bias", new[] { inner }));
            string a = suffix.Length == 0 ? "A_log" : "A_b_log", d = suffix.Length == 0 ? "D" : "D_b";
            LoadValues(branch.ALog, read($"{p}.{a}", new[] { inner, n }));
            LoadValues(branch.Skip, read($"{p}.{d}", new[] { inner }));
        }
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string p = $"layers.{i}.mixer";
            Norm(layer.Norm, $"layers.{i}.norm.weight");
            TorchParameters.Linear(layer.InputProjection, c.Dim, 2 * inner, read($"{p}.in_proj.weight", new[] { 2 * inner, c.Dim }));
            Direction(layer.Forward, p, "");
            Direction(layer.Backward, p, "_b");
            TorchParameters.Linear(layer.OutputProjection, inner, c.Dim, read($"{p}.out_proj.weight", new[] { c.Dim, inner }));
        }
        Norm(FinalNorm, "norm_f.weight");
    }

    /// <summary>
    /// The official initialization: LeCun-normal patch convolution with a zero bias, class token and position table
    /// N(0, 0.02²); Mamba's own (U(±1/√fan-in) projections and convolution, Δ projection U(±R^−½), Δ bias the inverse
    /// softplus of a log-uniform Δ in [0.001, 0.1] floored at 1e-4, <c>A_log = log(1..N)</c>, D = 1), and the output
    /// projection rescaled by 1/√depth.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        int inner = c.InnerDim, n = c.StateDim, k = c.ConvKernel, rank = c.DtRank, projected = rank + 2 * n;
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double Truncated(double std)
        {
            while (true)
            {
                double v = Normal();
                if (Math.Abs(v) <= 2.0) return v * std;
            }
        }
        double[] Uniform(int count, double bound)
        {
            var a = new double[count];
            for (int i = 0; i < count; i++) a[i] = (2 * random.NextDouble() - 1) * bound;
            return a;
        }
        double[] Repeat(int count, Func<int, double> value) => Enumerable.Range(0, count).Select(value).ToArray();

        // timm's lecun_normal_: a normal truncated at ±2σ, scaled so its variance is 1/fan-in.
        int fanIn = c.Channels * c.PatchSize * c.PatchSize;
        var patch = new Vector<T>(c.Dim * fanIn + c.Dim);
        for (int i = 0; i < c.Dim * fanIn; i++) patch[i] = NumOps.FromDouble(Truncated(Math.Sqrt(1.0 / fanIn) / 0.87962566103423978));
        PatchProjection.SetParameters(patch);
        if (ClassToken is not null) LoadValues(ClassToken, Repeat(c.Dim, _ => Truncated(0.02)));
        LoadValues(PositionTable, Repeat(c.Tokens * c.Dim, _ => Truncated(0.02)));
        void Direction(Branch branch)
        {
            LoadValues(branch.ConvKernel, Uniform(inner * k, 1.0 / Math.Sqrt(k)));
            LoadValues(branch.ConvBias, Uniform(inner, 1.0 / Math.Sqrt(k)));
            TorchParameters.Linear(branch.XProjection, inner, projected, Uniform(projected * inner, 1.0 / Math.Sqrt(inner)));
            var dtBias = Repeat(inner, _ =>
            {
                double dt = Math.Max(1e-4, Math.Exp(random.NextDouble() * (Math.Log(0.1) - Math.Log(0.001)) + Math.Log(0.001)));
                // The inverse of softplus: dt + log(1 − exp(−dt)), with 1 − exp(−dt) by its series for tiny dt.
                double oneMinus = dt < 1e-5 ? dt - dt * dt / 2 + dt * dt * dt / 6 : 1.0 - Math.Exp(-dt);
                return dt + Math.Log(oneMinus);
            });
            TorchParameters.Linear(_engine, branch.DtProjection, rank, inner, Uniform(inner * rank, Math.Pow(rank, -0.5)), dtBias);
            LoadValues(branch.ALog, Repeat(inner * n, i => Math.Log(i % n + 1)));
            LoadValues(branch.Skip, Repeat(inner, _ => 1.0));
        }
        foreach (var layer in Layers)
        {
            layer.Norm.SetParameters(new Vector<T>(Repeat(c.Dim, _ => 1.0).Select(NumOps.FromDouble).ToArray()));
            TorchParameters.Linear(layer.InputProjection, c.Dim, 2 * inner, Uniform(2 * inner * c.Dim, 1.0 / Math.Sqrt(c.Dim)));
            Direction(layer.Forward);
            Direction(layer.Backward);
            TorchParameters.Linear(layer.OutputProjection, inner, c.Dim,
                Uniform(c.Dim * inner, 1.0 / Math.Sqrt(inner)).Select(v => v / Math.Sqrt(c.Depth)).ToArray());
        }
        FinalNorm.SetParameters(new Vector<T>(Repeat(c.Dim, _ => 1.0).Select(NumOps.FromDouble).ToArray()));
    }

    /// <summary>
    /// The segmentation backbone's initialization (VisionMambaSeg.init_weights): <see cref="InitializeLikeReference"/>,
    /// then every linear layer re-drawn from N(0, 0.02²) truncated at ±2 with a zero bias (the Δ projection's bias
    /// included).
    /// </summary>
    public void InitializeForSegmentation(Random random)
    {
        InitializeLikeReference(random);
        var c = Configuration;
        int inner = c.InnerDim, rank = c.DtRank, projected = rank + 2 * c.StateDim;
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double[] Truncated(int count)
        {
            var a = new double[count];
            for (int i = 0; i < count; i++)
            {
                double v;
                do v = Normal(); while (Math.Abs(v * 0.02) > 2.0);
                a[i] = v * 0.02;
            }
            return a;
        }
        foreach (var layer in Layers)
        {
            TorchParameters.Linear(layer.InputProjection, c.Dim, 2 * inner, Truncated(2 * inner * c.Dim));
            foreach (var branch in new[] { layer.Forward, layer.Backward })
            {
                TorchParameters.Linear(branch.XProjection, inner, projected, Truncated(projected * inner));
                TorchParameters.Linear(_engine, branch.DtProjection, rank, inner, Truncated(inner * rank), new double[inner]);
            }
            TorchParameters.Linear(layer.OutputProjection, inner, c.Dim, Truncated(c.Dim * inner));
        }
    }
}
