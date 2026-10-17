using AiDotNet.ComputerVision;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of GLaMM's region encoder; the defaults are the released model's (CLIP-L/14-336 levels, Vicuna-7B).</summary>
internal sealed record GlammRegionConfiguration
{
    /// <summary>Channels of each CLIP level (1024).</summary>
    public int Embed { get; init; } = 1024;

    /// <summary>Side of the CLIP patch grid (24).</summary>
    public int Grid { get; init; } = 24;

    public int Levels { get; init; } = 4;

    public int FuseRounds { get; init; } = 5;

    public int GroupNormGroups { get; init; } = 64;

    public int RoiSize { get; init; } = 14;

    public int SamplingRatio { get; init; } = 2;

    public int PositionHidden { get; init; } = 256;

    /// <summary>Width of the region and position features (1024 in the official code).</summary>
    public int RegionWidth { get; init; } = 1024;

    /// <summary>Output width, the language model's (4096).</summary>
    public int OutDims { get; init; } = 4096;

    /// <summary>The official code scales the normalized boxes by 224.</summary>
    public double BoxScale { get; init; } = 224;

    /// <summary>Per level, the image pixels per feature cell: 14/8, 14/4, 14/2, 14.</summary>
    public double[] Strides { get; init; } = { 14.0 / 8, 14.0 / 4, 14.0 / 2, 14.0 };
}

/// <summary>
/// GLaMM's region encoder (Rasheed et al. 2024, §3.1; the official MLVLROIQueryModule): four CLIP levels upsampled to
/// 8x, 4x, 2x and 1x the patch grid, fused by five rounds of channel shuffling between neighbouring levels, then each box
/// pooled from every level by RoIAlign, merged, and projected with a position embedding of the box into one language-model
/// token.
/// </summary>
/// <remarks>As the official code: the input convolutions see each cell's (x, y) in [-1, 1]; a round keeps half the
/// channels and takes a quarter from the coarser and a quarter from the finer level (bilinear, align_corners=True), then a
/// bias-free 3x3 convolution, GroupNorm and ReLU shared by every level; RoIAlign (14x14, two samples, aligned) reads boxes
/// scaled by 224. Parameter names follow the official module.</remarks>
internal sealed class GlammRegionEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record FuseConv(BiasFreeLinearLayer<T> Conv, GroupNormalizationLayer<T> Norm);

    public GlammRegionEncoder(IEngine engine, List<LayerBase<T>> layers, GlammRegionConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Embed % 4 != 0 || c.Embed % c.GroupNormGroups != 0)
            throw new ArgumentException("The level width must divide by 4 and by the GroupNorm groups.", nameof(configuration));
        if (c.Strides.Length != c.Levels) throw new ArgumentException("One stride per level.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        for (int l = 0; l < c.Levels; l++) InputConvs.Add(Linear(c.Embed));
        for (int r = 0; r < c.FuseRounds; r++)
            FuseConvs.Add(new FuseConv(Add(new BiasFreeLinearLayer<T>(c.Embed * 9, c.Embed)), Add(new GroupNormalizationLayer<T>(c.GroupNormGroups, c.Embed, 1e-5))));
        for (int l = 0; l < c.Levels; l++) PConvs.Add(Linear(c.Embed));
        PositionIn = Linear(c.PositionHidden);
        PositionNorm1 = Add(new LayerNormalizationLayer<T>(c.PositionHidden, 1e-5));
        PositionOut = Linear(c.RegionWidth);
        PositionNorm2 = Add(new LayerNormalizationLayer<T>(c.RegionWidth, 1e-5));
        Flatten = Linear(c.RegionWidth);
        UpDims = Linear(c.OutDims);
    }

    public GlammRegionConfiguration Configuration { get; }

    public List<DenseLayer<T>> InputConvs { get; } = new();

    public List<FuseConv> FuseConvs { get; } = new();

    public List<DenseLayer<T>> PConvs { get; } = new();

    public DenseLayer<T> PositionIn { get; }

    public LayerNormalizationLayer<T> PositionNorm1 { get; }

    public DenseLayer<T> PositionOut { get; }

    public LayerNormalizationLayer<T> PositionNorm2 { get; }

    public DenseLayer<T> Flatten { get; }

    public DenseLayer<T> UpDims { get; }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    /// <summary>A convolution over <c>[B, C, H, W]</c> as a linear layer over its k x k patches (stride 1, same padding).</summary>
    private Tensor<T> Convolve(LayerBase<T> layer, Tensor<T> x, int kernel)
    {
        int b = x.Shape[0], h = x.Shape[2], w = x.Shape[3], pad = kernel / 2;
        var columns = _engine.Unfold(x, new[] { kernel, kernel }, new[] { 1, 1 }, new[] { pad, pad });             // [B, C·k·k, h·w]
        var y = Rows(layer, _engine.TensorPermute(columns, new[] { 0, 2, 1 }));                                    // [B, h·w, O]
        return _engine.Reshape(_engine.TensorPermute(y, new[] { 0, 2, 1 }), new[] { b, y.Shape[2], h, w });
    }

    /// <summary>The bilinear interpolation matrix <c>[outSize, inSize]</c> with align_corners=True.</summary>
    private static Tensor<T> Interpolation(int outSize, int inSize)
    {
        var m = new Tensor<T>(new[] { outSize, inSize });
        for (int i = 0; i < outSize; i++)
        {
            double source = outSize > 1 ? i * (inSize - 1.0) / (outSize - 1.0) : 0.0;
            int low = Math.Min((int)Math.Floor(source), inSize - 1), high = Math.Min(low + 1, inSize - 1);
            double frac = source - low;
            m[i, low] = NumOps.Add(m[i, low], NumOps.FromDouble(1 - frac));
            m[i, high] = NumOps.Add(m[i, high], NumOps.FromDouble(frac));
        }
        return m;
    }

    /// <summary><c>F.interpolate(x, size, mode="bilinear", align_corners=True)</c> on <c>[B, C, H, W]</c>, as two matrix products.</summary>
    private Tensor<T> Resize(Tensor<T> x, int height, int width)
    {
        int b = x.Shape[0], ch = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        if (h == height && w == width) return x;
        var cols = _engine.TensorMatMul(_engine.Reshape(x, new[] { b * ch * h, w }), _engine.TensorTranspose(Interpolation(width, w)));   // [B·C·H, W']
        var swapped = _engine.TensorPermute(_engine.Reshape(cols, new[] { b * ch, h, width }), new[] { 0, 2, 1 });                      // [B·C, W', H]
        var rows = _engine.TensorMatMul(_engine.Reshape(swapped, new[] { b * ch * width, h }), _engine.TensorTranspose(Interpolation(height, h)));
        return _engine.TensorPermute(_engine.Reshape(rows, new[] { b, ch, width, height }), new[] { 0, 1, 3, 2 });
    }

    private Tensor<T> Channels(Tensor<T> x, int start, int count)
        => _engine.TensorSlice(x, new[] { 0, start, 0, 0 }, new[] { x.Shape[0], count, x.Shape[2], x.Shape[3] });

    /// <summary>
    /// The region tokens <c>[boxes, OutDims]</c> for CLIP levels <c>[B, Grid², Embed]</c> (class token removed; finest-read
    /// first, as the official code orders them) and per image its boxes as normalized (x1, y1, x2, y2).
    /// </summary>
    public Tensor<T> Forward(IReadOnlyList<Tensor<T>> levels, IReadOnlyList<double[][]> boxes)
    {
        var c = Configuration;
        if (levels.Count != c.Levels) throw new ArgumentException($"Expected {c.Levels} levels.", nameof(levels));
        int b = levels[0].Shape[0], g = c.Grid;
        if (boxes.Count != b) throw new ArgumentException("One box list per image.", nameof(boxes));

        // Levels as maps, upsampled so the first is 2^(levels-1) times the grid and the last is the grid.
        var maps = new List<Tensor<T>>();
        for (int l = 0; l < c.Levels; l++)
        {
            var map = _engine.TensorPermute(_engine.Reshape(levels[l], new[] { b, g, g, c.Embed }), new[] { 0, 3, 1, 2 });
            int side = g << (c.Levels - 1 - l);
            maps.Add(Resize(map, side, side));
        }

        // Coordinate channels (x then y, linspace(-1, 1)) and the per-level input convolution.
        for (int l = 0; l < c.Levels; l++)
        {
            int h = maps[l].Shape[2], w = maps[l].Shape[3];
            var coord = new Tensor<T>(new[] { b, 2, h, w });
            for (int i = 0; i < b; i++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        coord[i, 0, y, x] = NumOps.FromDouble(Linspace(x, w));
                        coord[i, 1, y, x] = NumOps.FromDouble(Linspace(y, h));
                    }
            maps[l] = Convolve(InputConvs[l], _engine.TensorConcatenate(new[] { maps[l], coord }, 1), 1);
        }

        // Five rounds of shuffling between neighbouring levels, each followed by the round's shared convolution.
        int quarter = c.Embed / 4, remain = c.Embed - 2 * quarter;
        foreach (var round in FuseConvs)
        {
            var fused = new List<Tensor<T>>();
            for (int l = 0; l < c.Levels; l++)
            {
                var target = maps[l];
                var top = maps[Math.Min(l + 1, c.Levels - 1)];
                var down = maps[Math.Max(l - 1, 0)];
                int h = target.Shape[2], w = target.Shape[3];
                var mixed = _engine.TensorConcatenate(new[]
                {
                    Channels(target, 0, remain),
                    Resize(Channels(top, remain + quarter, quarter), h, w),
                    Resize(Channels(down, remain, quarter), h, w),
                }, 1);
                fused.Add(_engine.ReLU(round.Norm.Forward(Convolve(round.Conv, mixed, 3))));
            }
            maps = fused;
        }

        // Every box pooled from every level, merged by the per-level convolutions, flattened and projected.
        var flatBoxes = new List<double>();
        var owners = new List<int>();
        var normalized = new List<T>();
        for (int i = 0; i < b; i++)
            foreach (var box in boxes[i])
            {
                if (box.Length != 4) throw new ArgumentException("Boxes are (x1, y1, x2, y2).", nameof(boxes));
                flatBoxes.AddRange(box.Select(v => v * c.BoxScale));
                normalized.AddRange(box.Select(v => NumOps.FromDouble(v)));
                owners.Add(i);
            }
        int count = owners.Count;
        if (count == 0) return new Tensor<T>(new[] { 0, c.OutDims });
        Tensor<T>? merged = null;
        for (int l = 0; l < c.Levels; l++)
        {
            var pooled = CvTensorOps<T>.RoIAlignExact(maps[l], flatBoxes.ToArray(), owners.ToArray(), 1.0 / c.Strides[l], c.RoiSize, c.SamplingRatio, aligned: true);
            var conv = Convolve(PConvs[l], pooled, 3);
            merged = merged is null ? conv : _engine.TensorAdd(merged, conv);
        }
        var region = Rows(Flatten, _engine.Reshape(_engine.ReLU(merged!), new[] { count, c.Embed * c.RoiSize * c.RoiSize }));
        var boxTensor = new Tensor<T>(new[] { count, 4 }, new Vector<T>(normalized.ToArray()));
        var position = Rows(PositionNorm2, _engine.ReLU(Rows(PositionOut, Rows(PositionNorm1, _engine.ReLU(Rows(PositionIn, boxTensor))))));
        return Rows(UpDims, _engine.TensorAdd(region, position));
    }

    /// <summary>torch.linspace(-1, 1, n)[i]: the first half counts up from -1, the second down from 1.</summary>
    private static double Linspace(int i, int n)
    {
        if (n == 1) return -1;
        double step = 2.0 / (n - 1);
        return i < n / 2 ? -1 + i * step : 1 - (n - 1 - i) * step;
    }

    /// <summary>Loads the official module's state dict (names after an optional <paramref name="prefix"/>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "")
    {
        var c = Configuration;
        int e = c.Embed;
        void Linear(DenseLayer<T> layer, string name, int[] weightShape, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{prefix}{name}.weight", weightShape), read($"{prefix}{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name, int dim)
            => TorchParameters.LayerNorm(_engine, norm, read($"{prefix}{name}.weight", new[] { dim }), read($"{prefix}{name}.bias", new[] { dim }));
        for (int l = 0; l < c.Levels; l++)
            Linear(InputConvs[l], $"mlvl_fuse.input_conv.{l}", new[] { e, e + 2, 1, 1 }, e + 2, e);
        for (int r = 0; r < c.FuseRounds; r++)
        {
            TorchParameters.Linear(FuseConvs[r].Conv, e * 9, e, read($"{prefix}mlvl_fuse.fuse_convs.{r}.0.weight", new[] { e, e, 3, 3 }));
            var gamma = read($"{prefix}mlvl_fuse.fuse_convs.{r}.1.weight", new[] { e });
            var beta = read($"{prefix}mlvl_fuse.fuse_convs.{r}.1.bias", new[] { e });
            var g = FuseConvs[r].Norm.GetGammaTensor();
            var bt = FuseConvs[r].Norm.GetBetaTensor();
            for (int i = 0; i < e; i++) { g[i] = NumOps.FromDouble(gamma[i]); bt[i] = NumOps.FromDouble(beta[i]); }
            _engine.InvalidatePersistentTensor(g);
            _engine.InvalidatePersistentTensor(bt);
        }
        for (int l = 0; l < c.Levels; l++)
            Linear(PConvs[l], $"roi_align.pconvs.{l}", new[] { e, e, 3, 3 }, e * 9, e);
        Linear(PositionIn, "roi_align.pos_embedd.0", new[] { c.PositionHidden, 4 }, 4, c.PositionHidden);
        Norm(PositionNorm1, "roi_align.pos_embedd.2", c.PositionHidden);
        Linear(PositionOut, "roi_align.pos_embedd.3", new[] { c.RegionWidth, c.PositionHidden }, c.PositionHidden, c.RegionWidth);
        Norm(PositionNorm2, "roi_align.pos_embedd.5", c.RegionWidth);
        Linear(Flatten, "roi_align.flatten_linear", new[] { c.RegionWidth, e * c.RoiSize * c.RoiSize }, e * c.RoiSize * c.RoiSize, c.RegionWidth);
        Linear(UpDims, "roi_align.updims", new[] { c.OutDims, c.RegionWidth }, c.RegionWidth, c.OutDims);
    }

    /// <summary>
    /// The official initialization: PyTorch's defaults (weights and biases uniform in ±1/√fan_in for convolutions and
    /// linear layers; unit norms), with the RoI extractor's convolutions N(0, 0.01²) and zero bias (its norm_init_weights).
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        var fanIn = new Dictionary<string, int>();
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            bool weight = name.EndsWith(".weight", StringComparison.Ordinal);
            string module = name.Substring(0, name.LastIndexOf('.'));
            bool norm = name.StartsWith("roi_align.pos_embedd.2", StringComparison.Ordinal) || name.StartsWith("roi_align.pos_embedd.5", StringComparison.Ordinal)
                || (name.StartsWith("mlvl_fuse.fuse_convs.", StringComparison.Ordinal) && module.EndsWith(".1", StringComparison.Ordinal));
            if (norm)
            {
                if (weight) for (int i = 0; i < count; i++) values[i] = 1.0;
                return values;
            }
            if (weight) fanIn[module] = count / shape[0];
            if (name.StartsWith("roi_align.pconvs.", StringComparison.Ordinal))
            {
                if (weight) for (int i = 0; i < count; i++) values[i] = 0.01 * Normal();
                return values;
            }
            double bound = 1.0 / Math.Sqrt(fanIn[module]);
            for (int i = 0; i < count; i++) values[i] = bound * (2 * random.NextDouble() - 1);
            return values;
        });
    }
}
