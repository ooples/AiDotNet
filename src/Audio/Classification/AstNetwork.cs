using AiDotNet.Helpers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Classification;

/// <summary>The shape of an Audio Spectrogram Transformer; the defaults are AST-Base on 10-second AudioSet clips.</summary>
internal sealed record AstConfiguration
{
    /// <summary>Mel bins of the input filterbank (128).</summary>
    public int MelBins { get; init; } = 128;

    /// <summary>Filterbank frames the network takes (1024, about 10 s at a 10 ms shift).</summary>
    public int Frames { get; init; } = 1024;

    public int PatchSize { get; init; } = 16;

    /// <summary>Patch stride along frequency; 10 overlaps 16×16 patches by 6 (§2.1).</summary>
    public int FrequencyStride { get; init; } = 10;

    /// <summary>Patch stride along time.</summary>
    public int TimeStride { get; init; } = 10;

    public int Dim { get; init; } = 768;

    public int Depth { get; init; } = 12;

    public int Heads { get; init; } = 12;

    public double MlpRatio { get; init; } = 4.0;

    /// <summary>Output labels (527 AudioSet classes).</summary>
    public int Classes { get; init; } = 527;

    public int FrequencyPatches => (MelBins - PatchSize) / FrequencyStride + 1;

    public int TimePatches => (Frames - PatchSize) / TimeStride + 1;

    public int Patches => FrequencyPatches * TimePatches;
}

/// <summary>
/// The Audio Spectrogram Transformer (Gong, Chung and Glass 2021) as the official code builds it (YuanGongND/ast,
/// <c>ASTModel</c> over timm's DeiT): overlapping 16×16 patches of the <c>[frames, mel]</c> filterbank (a convolution with
/// stride 10), the DeiT class and distillation tokens and a learned position embedding, pre-norm Transformer blocks,
/// and a head on the average of the two tokens' outputs.
/// </summary>
/// <remarks>
/// <para>Both AST models run this network: <see cref="AST{T}"/> as an audio event classifier, and
/// <see cref="Fingerprinting.ASTModel{T}"/> for embeddings. Its parameter names follow the official checkpoints
/// (<c>v.*</c> for the DeiT body, <c>mlp_head.*</c>), so <see cref="LoadTorchWeights"/> reads a released model, and
/// <see cref="LoadImageNetDeiT"/> applies the paper's ImageNet adaptation (§2.2) to a DeiT checkpoint.</para>
/// </remarks>
internal sealed class AstNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Qkv, DenseLayer<T> Projection,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> Fc1, DenseLayer<T> Fc2);

    public AstNetwork(IEngine engine, List<LayerBase<T>> layers, AstConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.Heads != 0) throw new ArgumentException("The width must divide by the head count.", nameof(configuration));
        if (c.MelBins < c.PatchSize || c.Frames < c.PatchSize)
            throw new ArgumentException($"The input must be at least one {c.PatchSize}×{c.PatchSize} patch.", nameof(configuration));
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        // timm's ViT LayerNorms use ε = 1e-6; the head's is nn.LayerNorm with PyTorch's default 1e-5.
        LayerNormalizationLayer<T> Norm(double epsilon) => Add(new LayerNormalizationLayer<T>(c.Dim, epsilon));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }

        PatchProjection = Linear(c.Dim);
        ClassToken = Values(c.Dim);
        DistillationToken = Values(c.Dim);
        PositionTable = Values((c.Patches + 2) * c.Dim);
        int hidden = (int)(c.Dim * c.MlpRatio);
        for (int i = 0; i < c.Depth; i++)
            Blocks.Add(new Block(Norm(1e-6), Linear(3 * c.Dim), Linear(c.Dim), Norm(1e-6), Linear(hidden), Linear(c.Dim)));
        FinalNorm = Norm(1e-6);
        HeadNorm = Norm(1e-5);
        Head = Linear(c.Classes);
    }

    public AstConfiguration Configuration { get; }

    public DenseLayer<T> PatchProjection { get; }

    public BiasFreeLinearLayer<T> ClassToken { get; }

    public BiasFreeLinearLayer<T> DistillationToken { get; }

    public BiasFreeLinearLayer<T> PositionTable { get; }

    public List<Block> Blocks { get; } = new();

    public LayerNormalizationLayer<T> FinalNorm { get; }

    public LayerNormalizationLayer<T> HeadNorm { get; }

    public DenseLayer<T> Head { get; }

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder)
        => holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    /// <summary>
    /// The audio representation <c>[B, Dim]</c> of filterbanks <c>[B, Frames, MelBins]</c>: the average of the class and
    /// distillation tokens after the last block's norm (the official <c>ASTModel.forward</c> before <c>mlp_head</c>).
    /// </summary>
    public Tensor<T> Encode(Tensor<T> fbank)
    {
        var c = Configuration;
        if (fbank.Rank != 3 || fbank.Shape[1] != c.Frames || fbank.Shape[2] != c.MelBins)
            throw new ArgumentException($"The filterbank must be [batch, {c.Frames}, {c.MelBins}], got [{string.Join(", ", fbank.Shape.ToArray())}].", nameof(fbank));
        int b = fbank.Shape[0], d = c.Dim, n = c.Patches + 2;
        // x.unsqueeze(1).transpose(2, 3): [B, 1, mel, frames]; the patches run frequency-major, as the convolution's flatten.
        var image = _engine.Reshape(_engine.TensorPermute(fbank, new[] { 0, 2, 1 }), new[] { b, 1, c.MelBins, c.Frames });
        var columns = _engine.Unfold(image, new[] { c.PatchSize, c.PatchSize }, new[] { c.FrequencyStride, c.TimeStride }, new[] { 0, 0 });
        var patches = Rows(PatchProjection, _engine.TensorPermute(columns, new[] { 0, 2, 1 }));                 // [B, P, D]
        var cls = _engine.TensorBroadcastTo(_engine.Reshape(Value(ClassToken), new[] { 1, 1, d }), new[] { b, 1, d });
        var dist = _engine.TensorBroadcastTo(_engine.Reshape(Value(DistillationToken), new[] { 1, 1, d }), new[] { b, 1, d });
        var x = _engine.TensorConcatenate(new[] { cls, dist, patches }, 1);
        x = _engine.TensorAdd(x, _engine.TensorBroadcastTo(_engine.Reshape(Value(PositionTable), new[] { 1, n, d }), new[] { b, n, d }));
        foreach (var block in Blocks) x = Run(block, x);
        x = Rows(FinalNorm, x);
        var first = _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { b, 1, d });
        var second = _engine.TensorSlice(x, new[] { 0, 1, 0 }, new[] { b, 1, d });
        return _engine.Reshape(_engine.TensorMultiplyScalar(_engine.TensorAdd(first, second), NumOps.FromDouble(0.5)), new[] { b, d });
    }

    /// <summary>The label logits <c>[B, Classes]</c> of representations <c>[B, Dim]</c> (the official <c>mlp_head</c>:
    /// LayerNorm, then a linear layer).</summary>
    public Tensor<T> Classify(Tensor<T> representation) => Head.Forward(HeadNorm.Forward(representation));

    /// <summary>The label logits <c>[B, Classes]</c> of filterbanks <c>[B, Frames, MelBins]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> fbank) => Classify(Encode(fbank));

    private Tensor<T> Run(Block block, Tensor<T> x)
    {
        x = _engine.TensorAdd(x, Attend(block, Rows(block.Norm1, x)));
        var h = Rows(block.Fc2, ActivationHelper.ExactGelu(Rows(block.Fc1, Rows(block.Norm2, x)), _engine));
        return _engine.TensorAdd(x, h);
    }

    /// <summary>timm's <c>Attention</c>: one fused qkv projection, scaled dot products, softmax and the output projection.</summary>
    private Tensor<T> Attend(Block block, Tensor<T> x)
    {
        int b = x.Shape[0], n = x.Shape[1], d = x.Shape[2], heads = Configuration.Heads, hd = d / heads;
        var qkv = _engine.Reshape(Rows(block.Qkv, x), new[] { b, n, 3, heads, hd });
        qkv = _engine.Reshape(_engine.TensorPermute(qkv, new[] { 2, 0, 3, 1, 4 }), new[] { 3, b * heads, n, hd });
        Tensor<T> Part(int i) => _engine.Reshape(_engine.TensorSlice(qkv, new[] { i, 0, 0, 0 }, new[] { 1, b * heads, n, hd }),
            new[] { b * heads, n, hd });
        var q = Part(0);
        var k = Part(1);
        var v = Part(2);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);                                // [B·h, N, hd]
        context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { b, heads, n, hd }), new[] { 0, 2, 1, 3 }), new[] { b, n, d });
        return Rows(block.Projection, context);
    }

    /// <summary>Loads an official AST state dict (as <c>ASTModel</c> names it, without a DataParallel <c>module.</c>
    /// prefix), given a reader of a named tensor of a given shape.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.Dim, hidden = (int)(c.Dim * c.MlpRatio), p = c.PatchSize;
        TorchParameters.Linear(_engine, PatchProjection, p * p, d, read("v.patch_embed.proj.weight", new[] { d, 1, p, p }),
            read("v.patch_embed.proj.bias", new[] { d }));
        TorchParameters.Linear(ClassToken, 1, d, read("v.cls_token", new[] { 1, 1, d }));
        TorchParameters.Linear(DistillationToken, 1, d, read("v.dist_token", new[] { 1, 1, d }));
        TorchParameters.Linear(PositionTable, 1, (c.Patches + 2) * d, read("v.pos_embed", new[] { 1, c.Patches + 2, d }));
        LoadBlocks(read, "v.", hidden);
        Norm(FinalNorm, "v.norm", read);
        Norm(HeadNorm, "mlp_head.0", read);
        Linear(Head, "mlp_head.1", d, c.Classes, read);
    }

    /// <summary>
    /// Initializes from a timm DeiT (distilled) checkpoint as the paper adapts ImageNet weights (§2.2): the patch
    /// projection's three input channels are averaged into one, the patch position embeddings are cut (or bilinearly
    /// interpolated) from the DeiT grid to the AST grid, the class and distillation tokens and their positions are kept,
    /// and the head is newly initialized by <paramref name="random"/>.
    /// </summary>
    /// <remarks>The official code sums the three channels; the paper averages them, and the paper is followed here
    /// (with inputs normalized to a standard deviation of 0.5, the two differ by a factor of 3 in the first layer).</remarks>
    public void LoadImageNetDeiT(Func<string, int[], double[]> read, int imageGrid, Random random)
    {
        var c = Configuration;
        int d = c.Dim, hidden = (int)(c.Dim * c.MlpRatio), p = c.PatchSize, original = imageGrid * imageGrid;
        var rgb = read("patch_embed.proj.weight", new[] { d, 3, p, p });
        var mono = new double[d * p * p];
        for (int o = 0; o < d; o++)
            for (int i = 0; i < p * p; i++)
                mono[o * p * p + i] = (rgb[(o * 3 + 0) * p * p + i] + rgb[(o * 3 + 1) * p * p + i] + rgb[(o * 3 + 2) * p * p + i]) / 3.0;
        TorchParameters.Linear(_engine, PatchProjection, p * p, d, mono, read("patch_embed.proj.bias", new[] { d }));
        TorchParameters.Linear(ClassToken, 1, d, read("cls_token", new[] { 1, 1, d }));
        TorchParameters.Linear(DistillationToken, 1, d, read("dist_token", new[] { 1, 1, d }));
        var table = read("pos_embed", new[] { 1, original + 2, d });
        var grid = new double[d, imageGrid, imageGrid];
        for (int t = 0; t < original; t++)
            for (int k = 0; k < d; k++) grid[k, t / imageGrid, t % imageGrid] = table[(t + 2) * d + k];
        // The official order: time (width) first, then frequency (height); each is a centred cut when the AST side is no
        // larger, otherwise a bilinear interpolation.
        var widened = Resize(grid, imageGrid, imageGrid, imageGrid, c.TimePatches);
        var resized = Resize(widened, imageGrid, c.TimePatches, c.FrequencyPatches, c.TimePatches);
        var position = new double[(c.Patches + 2) * d];
        Array.Copy(table, position, 2 * d);
        for (int f = 0; f < c.FrequencyPatches; f++)
            for (int t = 0; t < c.TimePatches; t++)
                for (int k = 0; k < d; k++) position[(2 + f * c.TimePatches + t) * d + k] = resized[k, f, t];
        TorchParameters.Linear(PositionTable, 1, (c.Patches + 2) * d, position);
        LoadBlocks(read, "", hidden);
        Norm(FinalNorm, "norm", read);
        // mlp_head is new: nn.LayerNorm's defaults and nn.Linear's.
        TorchParameters.LayerNorm(_engine, HeadNorm, Enumerable.Repeat(1.0, d).ToArray(), new double[d]);
        TorchParameters.Linear(_engine, Head, d, c.Classes, Uniform(random, c.Classes * d, d), Uniform(random, c.Classes, d));
    }

    /// <summary>The official cut-or-interpolate of a <c>[D, h, w]</c> grid to <c>[D, toH, toW]</c>, where only one of the
    /// two sides changes per call (width when <paramref name="toH"/> equals <paramref name="h"/>).</summary>
    private static double[,,] Resize(double[,,] grid, int h, int w, int toH, int toW)
    {
        int d = grid.GetLength(0);
        var result = new double[d, toH, toW];
        bool cutRows = toH <= h, cutCols = toW <= w;
        // The cut takes the centred slice [h/2 − toH/2, h/2 − toH/2 + toH) (integer halves, as the official code).
        int rowStart = h / 2 - toH / 2, colStart = w / 2 - toW / 2;
        for (int k = 0; k < d; k++)
            for (int y = 0; y < toH; y++)
                for (int x = 0; x < toW; x++)
                {
                    double sy = cutRows ? rowStart + y : Source(y, h, toH);
                    double sx = cutCols ? colStart + x : Source(x, w, toW);
                    result[k, y, x] = Bilinear(grid, k, sy, sx, h, w);
                }
        return result;
    }

    // F.interpolate(mode='bilinear', align_corners=False): source = (i + 0.5)·in/out − 0.5, clamped at 0.
    private static double Source(int i, int input, int output) => Math.Max((i + 0.5) * input / output - 0.5, 0.0);

    private static double Bilinear(double[,,] grid, int k, double y, double x, int h, int w)
    {
        int y0 = (int)Math.Floor(y), x0 = (int)Math.Floor(x);
        int y1 = Math.Min(y0 + 1, h - 1), x1 = Math.Min(x0 + 1, w - 1);
        double fy = y - y0, fx = x - x0;
        return (1 - fy) * ((1 - fx) * grid[k, y0, x0] + fx * grid[k, y0, x1]) + fy * ((1 - fx) * grid[k, y1, x0] + fx * grid[k, y1, x1]);
    }

    /// <summary>
    /// The official initialization when training from scratch (<c>imagenet_pretrain=False</c>): timm's ViT
    /// initialization (linear weights N(0, 0.02²) truncated at ±2, zero biases, unit LayerNorms, tokens N(0, 0.02²)),
    /// a new position embedding N(0, 0.02²), and PyTorch's defaults for the new patch convolution and the head.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        int d = c.Dim, hidden = (int)(c.Dim * c.MlpRatio), p = c.PatchSize;
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double[] Truncated(int count)
        {
            var v = new double[count];
            for (int i = 0; i < count; i++)
            {
                double z;
                do z = 0.02 * Normal(); while (z < -2 || z > 2);
                v[i] = z;
            }
            return v;
        }
        TorchParameters.Linear(_engine, PatchProjection, p * p, d, Uniform(random, d * p * p, p * p), Uniform(random, d, p * p));
        TorchParameters.Linear(ClassToken, 1, d, Truncated(d));
        TorchParameters.Linear(DistillationToken, 1, d, Truncated(d));
        TorchParameters.Linear(PositionTable, 1, (c.Patches + 2) * d, Truncated((c.Patches + 2) * d));
        LoadBlocks((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, b) => a * b);
            if (name.Contains(".norm")) return name.EndsWith(".weight", StringComparison.Ordinal) ? Enumerable.Repeat(1.0, count).ToArray() : new double[count];
            return name.EndsWith(".weight", StringComparison.Ordinal) ? Truncated(count) : new double[count];
        }, "v.", hidden);
        TorchParameters.LayerNorm(_engine, FinalNorm, Enumerable.Repeat(1.0, d).ToArray(), new double[d]);
        TorchParameters.LayerNorm(_engine, HeadNorm, Enumerable.Repeat(1.0, d).ToArray(), new double[d]);
        TorchParameters.Linear(_engine, Head, d, c.Classes, Uniform(random, c.Classes * d, d), Uniform(random, c.Classes, d));
    }

    // nn.Linear / nn.Conv2d defaults: kaiming_uniform(a=√5) gives U(±1/√fanIn) for the weight, and the bias is U(±1/√fanIn).
    private static double[] Uniform(Random random, int count, int fanIn)
    {
        double bound = 1.0 / Math.Sqrt(fanIn);
        var v = new double[count];
        for (int i = 0; i < count; i++) v[i] = (2 * random.NextDouble() - 1) * bound;
        return v;
    }

    private void LoadBlocks(Func<string, int[], double[]> read, string prefix, int hidden)
    {
        int d = Configuration.Dim;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string n = $"{prefix}blocks.{i}";
            Norm(b.Norm1, $"{n}.norm1", read);
            Linear(b.Qkv, $"{n}.attn.qkv", d, 3 * d, read);
            Linear(b.Projection, $"{n}.attn.proj", d, d, read);
            Norm(b.Norm2, $"{n}.norm2", read);
            Linear(b.Fc1, $"{n}.mlp.fc1", d, hidden, read);
            Linear(b.Fc2, $"{n}.mlp.fc2", hidden, d, read);
        }
    }

    private void Linear(DenseLayer<T> layer, string name, int inputs, int outputs, Func<string, int[], double[]> read)
        => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));

    private void Norm(LayerNormalizationLayer<T> norm, string name, Func<string, int[], double[]> read)
    {
        int d = Configuration.Dim;
        TorchParameters.LayerNorm(_engine, norm, read($"{name}.weight", new[] { d }), read($"{name}.bias", new[] { d }));
    }
}
