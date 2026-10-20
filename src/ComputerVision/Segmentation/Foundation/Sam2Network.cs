using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>The shape of a SAM 2 image model; the defaults are SAM 2 Hiera-T at 1024 pixels.</summary>
internal sealed record Sam2Configuration
{
    public int ImageSize { get; init; } = 1024;

    public int Channels { get; init; } = 3;

    public int PatchKernel { get; init; } = 7;

    public int PatchStride { get; init; } = 4;

    public int PatchPadding { get; init; } = 3;

    public int[] BlocksPerStage { get; init; } = { 1, 2, 7, 2 };

    public int[] EmbedDims { get; init; } = { 96, 192, 384, 768 };

    public int[] HeadsPerStage { get; init; } = { 1, 2, 4, 8 };

    public int[] WindowSizes { get; init; } = { 8, 4, 14, 7 };

    /// <summary>Blocks (0-based over all stages) that attend globally.</summary>
    public int[] GlobalAttentionBlocks { get; init; } = { 5, 7, 9 };

    /// <summary>Stages after the first that halve the resolution by pooling their first block's queries (3).</summary>
    public int QueryPoolStages { get; init; } = 3;

    /// <summary>Side of the interpolated background position embedding (7).</summary>
    public int BackgroundPositionSize { get; init; } = 7;

    public double MlpRatio { get; init; } = 4.0;

    public double LayerNormEpsilon { get; init; } = 1e-6;

    /// <summary>The FPN width, also the prompt and mask-decoder width (256).</summary>
    public int Dim { get; init; } = 256;

    /// <summary>Backbone levels (0 = finest) that receive the coarser level's top-down features.</summary>
    public int[] TopDownLevels { get; init; } = { 2, 3 };

    public int MaskInChannels { get; init; } = 16;

    public int DecoderDepth { get; init; } = 2;

    public int DecoderHeads { get; init; } = 8;

    public int DecoderMlp { get; init; } = 2048;

    public int AttentionDownsample { get; init; } = 2;

    public int MultimaskOutputs { get; init; } = 3;

    public int IouHidden { get; init; } = 256;

    public int IouDepth { get; init; } = 3;

    public double StabilityDelta { get; init; } = 0.05;

    public double StabilityThreshold { get; init; } = 0.98;

    public double[] PixelMean { get; init; } = { 0.485, 0.456, 0.406 };

    public double[] PixelStd { get; init; } = { 0.229, 0.224, 0.225 };

    /// <summary>The largest stochastic-depth rate of the Hiera blocks while training, reached linearly by the last block
    /// (0 disables it).</summary>
    public double DropPath { get; init; }

    /// <summary>The video modules (memory encoder, memory attention and object pointers); null for the image model.</summary>
    public Sam2MemoryConfiguration? Memory { get; init; }

    /// <summary>
    /// The released Hiera image encoders at 1024 pixels: Tiny, Small, Base+ and Large, as the official configs define
    /// them, with the stochastic depth of the paper's training recipe (Ravi et al. 2024, Table 12: 0.1, 0.1, 0.2, 0.3).
    /// </summary>
    public static Sam2Configuration Hiera(int size) => size switch
    {
        0 => new Sam2Configuration { DropPath = 0.1 },
        1 => new Sam2Configuration { BlocksPerStage = new[] { 1, 2, 11, 2 }, GlobalAttentionBlocks = new[] { 7, 10, 13 }, DropPath = 0.1 },
        2 => new Sam2Configuration
        {
            BlocksPerStage = new[] { 2, 3, 16, 3 }, EmbedDims = new[] { 112, 224, 448, 896 }, HeadsPerStage = new[] { 2, 4, 8, 16 },
            GlobalAttentionBlocks = new[] { 12, 16, 20 }, BackgroundPositionSize = 14, DropPath = 0.2,
        },
        3 => new Sam2Configuration
        {
            BlocksPerStage = new[] { 2, 6, 36, 4 }, EmbedDims = new[] { 144, 288, 576, 1152 }, HeadsPerStage = new[] { 2, 4, 8, 16 },
            WindowSizes = new[] { 8, 4, 16, 8 }, GlobalAttentionBlocks = new[] { 23, 33, 43 }, DropPath = 0.3,
        },
        _ => throw new ArgumentOutOfRangeException(nameof(size)),
    };

    public int MaskTokens => MultimaskOutputs + 1;

    /// <summary>Side of the image embedding (1/16 of the image).</summary>
    public int Grid => ImageSize / (PatchStride * 4);
}

/// <summary>
/// SAM 2 (Ravi et al. 2024) for images, as the official network (facebookresearch/sam2; checked against Hugging Face's
/// port): the hierarchical Hiera image encoder with an FPN neck, SAM's prompt encoder, and a mask decoder that adds an
/// object-score token and skip connections from the 1/4 and 1/8 features. For a single image the "no memory"
/// embedding is added to the image embedding, as the image predictor does.
/// </summary>
internal sealed partial class Sam2Network<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Qkv, DenseLayer<T> Projection, DenseLayer<T>? Shortcut,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> Mlp1, DenseLayer<T> Mlp2, int Dim, int DimOut, int Heads, int Window, bool PoolQuery);

    internal sealed record Attention(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output, int Heads);

    internal sealed record TwoWayBlock(Attention SelfAttention, LayerNormalizationLayer<T> Norm1, Attention TokenToImage,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> Mlp1, DenseLayer<T> Mlp2, LayerNormalizationLayer<T> Norm3,
        LayerNormalizationLayer<T> Norm4, Attention ImageToToken, bool SkipFirstPositional);

    public Sam2Network(IEngine engine, List<LayerBase<T>> layers, Sam2Configuration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm(int dim, double eps) => Add(new LayerNormalizationLayer<T>(dim, eps));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        Attention Attend(int downsample) => new(Linear(c.Dim / downsample), Linear(c.Dim / downsample), Linear(c.Dim / downsample), Linear(c.Dim), c.DecoderHeads);

        PatchProjection = Linear(c.EmbedDims[0]);
        BackgroundPosition = Values(c.EmbedDims[0] * c.BackgroundPositionSize * c.BackgroundPositionSize);
        WindowPosition = Values(c.EmbedDims[0] * c.WindowSizes[0] * c.WindowSizes[0]);
        int total = 0;
        for (int stage = 0; stage < c.BlocksPerStage.Length; stage++)
        {
            for (int i = 0; i < c.BlocksPerStage[stage]; i++, total++)
            {
                bool first = stage > 0 && i == 0;
                int dim = first ? c.EmbedDims[stage - 1] : c.EmbedDims[stage], dimOut = c.EmbedDims[stage];
                int window = c.GlobalAttentionBlocks.Contains(total) ? 0 : first ? c.WindowSizes[stage - 1] : c.WindowSizes[stage];
                bool pool = stage > 0 && stage <= c.QueryPoolStages && i == 0;
                Blocks.Add(new Block(Norm(dim, c.LayerNormEpsilon), Linear(3 * dimOut), Linear(dimOut), dim != dimOut ? Linear(dimOut) : null,
                    Norm(dimOut, c.LayerNormEpsilon), Linear((int)(dimOut * c.MlpRatio)), Linear(dimOut), dim, dimOut, c.HeadsPerStage[stage], window, pool));
            }
            StageEnds.Add(total - 1);
        }
        // The neck's 1x1 convolutions, coarsest backbone level first (convs[n - i] serves level i).
        for (int s = c.EmbedDims.Length - 1; s >= 0; s--) NeckConvs.Add(Linear(c.Dim));
        HighRes0 = Linear(c.Dim / 8);
        HighRes1 = Linear(c.Dim / 4);
        NoMemory = Values(c.Dim);

        int p = c.Dim;
        for (int i = 0; i < 4; i++) PointEmbeddings.Add(Values(p));
        NotAPoint = Values(p);
        NoMask = Values(p);
        MaskConv1 = Linear(c.MaskInChannels / 4);
        MaskNorm1 = Norm(c.MaskInChannels / 4, 1e-6);
        MaskConv2 = Linear(c.MaskInChannels);
        MaskNorm2 = Norm(c.MaskInChannels, 1e-6);
        MaskConv3 = Linear(p);
        ObjectToken = Values(p);
        IouToken = Values(p);
        MaskTokenTable = Values(c.MaskTokens * p);
        for (int i = 0; i < c.DecoderDepth; i++)
            TwoWayBlocks.Add(new TwoWayBlock(Attend(1), Norm(p, 1e-5), Attend(c.AttentionDownsample), Norm(p, 1e-5),
                Linear(c.DecoderMlp), Linear(p), Norm(p, 1e-5), Norm(p, 1e-5), Attend(c.AttentionDownsample), SkipFirstPositional: i == 0));
        FinalAttention = Attend(c.AttentionDownsample);
        FinalNorm = Norm(p, 1e-5);
        Upscale1 = new TransposedConvolution2x2<T>(engine, l => Add(l), p, p / 4);
        UpscaleNorm = Norm(p / 4, 1e-6);
        Upscale2 = new TransposedConvolution2x2<T>(engine, l => Add(l), p / 4, p / 8);
        for (int i = 0; i < c.MaskTokens; i++) Hypernetworks.Add(new[] { Linear(p), Linear(p), Linear(p / 8) });
        var iou = new List<DenseLayer<T>>();
        for (int i = 0; i < c.IouDepth - 1; i++) iou.Add(Linear(c.IouHidden));
        iou.Add(Linear(c.MaskTokens));
        IouHead = iou.ToArray();
        ObjectHead = new[] { Linear(p), Linear(p), Linear(1) };
        _gaussian = new double[2, p / 2];
        _bicubicRows = new Dictionary<(int, int), Tensor<T>>();
        if (c.Memory is { } memory) BuildMemory(layers, memory);
    }

    public Sam2Configuration Configuration { get; }

    public DenseLayer<T> PatchProjection { get; }

    public BiasFreeLinearLayer<T> BackgroundPosition { get; }

    public BiasFreeLinearLayer<T> WindowPosition { get; }

    public List<Block> Blocks { get; } = new();

    public List<int> StageEnds { get; } = new();

    public List<DenseLayer<T>> NeckConvs { get; } = new();

    public DenseLayer<T> HighRes0 { get; }

    public DenseLayer<T> HighRes1 { get; }

    public BiasFreeLinearLayer<T> NoMemory { get; }

    public List<BiasFreeLinearLayer<T>> PointEmbeddings { get; } = new();

    public BiasFreeLinearLayer<T> NotAPoint { get; }

    public BiasFreeLinearLayer<T> NoMask { get; }

    public DenseLayer<T> MaskConv1 { get; }

    public LayerNormalizationLayer<T> MaskNorm1 { get; }

    public DenseLayer<T> MaskConv2 { get; }

    public LayerNormalizationLayer<T> MaskNorm2 { get; }

    public DenseLayer<T> MaskConv3 { get; }

    public BiasFreeLinearLayer<T> ObjectToken { get; }

    public BiasFreeLinearLayer<T> IouToken { get; }

    public BiasFreeLinearLayer<T> MaskTokenTable { get; }

    public List<TwoWayBlock> TwoWayBlocks { get; } = new();

    public Attention FinalAttention { get; }

    public LayerNormalizationLayer<T> FinalNorm { get; }

    public TransposedConvolution2x2<T> Upscale1 { get; }

    public LayerNormalizationLayer<T> UpscaleNorm { get; }

    public TransposedConvolution2x2<T> Upscale2 { get; }

    public List<DenseLayer<T>[]> Hypernetworks { get; } = new();

    public DenseLayer<T>[] IouHead { get; }

    public DenseLayer<T>[] ObjectHead { get; }

    /// <summary>The random Fourier matrix <c>[2, Dim/2]</c> of the prompt position encoding: a fixed buffer.</summary>
    private readonly double[,] _gaussian;

    private readonly Dictionary<(int, int), Tensor<T>> _bicubicRows;

    /// <summary>Whether the network is training: the single-mask output then skips the stability selection, and drop
    /// path and the memory attention's dropout are active.</summary>
    public bool Training { get; set; }

    /// <summary>The generator of the training-time drop path and dropout masks.</summary>
    public Random TrainingRandom { get; set; } = new Random(0);

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

    private Tensor<T> Gelu(Tensor<T> x) => ActivationHelper.ExactGelu(x, _engine);

    /// <summary>A LayerNorm over the channel axis of <c>[B, C, H, W]</c>.</summary>
    private Tensor<T> ChannelNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
        => _engine.TensorPermute(Rows(norm, _engine.TensorPermute(x, new[] { 0, 2, 3, 1 })), new[] { 0, 3, 1, 2 });

    /// <summary>A strided convolution of <c>[B, C, H, W]</c> as its patches through a linear layer.</summary>
    private Tensor<T> Convolve(DenseLayer<T> layer, Tensor<T> x, int kernel, int stride, int padding)
    {
        int b = x.Shape[0], h = (x.Shape[2] + 2 * padding - kernel) / stride + 1, w = (x.Shape[3] + 2 * padding - kernel) / stride + 1;
        var columns = _engine.Unfold(x, new[] { kernel, kernel }, new[] { stride, stride }, new[] { padding, padding });
        var y = Rows(layer, _engine.TensorPermute(columns, new[] { 0, 2, 1 }));                                  // [B, h·w, out]
        return _engine.Reshape(_engine.TensorPermute(y, new[] { 0, 2, 1 }), new[] { b, y.Shape[2], h, w });
    }

    /// <summary>
    /// The 1/4, 1/8 and 1/16 image features of normalized images <c>[B, 3, ImageSize, ImageSize]</c>: the 1/16 one is
    /// the image embedding <c>[B, Dim, Grid, Grid]</c> (with the no-memory embedding), the others the high-resolution
    /// skips through the decoder's 1x1 convolutions (<c>Dim/8</c> and <c>Dim/4</c> channels).
    /// </summary>
    public (Tensor<T> Embedding, Tensor<T> HighRes0, Tensor<T> HighRes1) EncodeImage(Tensor<T> images)
    {
        var (features, s0, s1) = EncodeFeatures(images);
        return (AddNoMemory(features), s0, s1);
    }

    /// <summary>1/16 features <c>[B, Dim, Grid, Grid]</c> plus the no-memory embedding, as an image (or a prompted video
    /// frame) reaches the mask decoder.</summary>
    public Tensor<T> AddNoMemory(Tensor<T> features)
        => _engine.TensorAdd(features, _engine.TensorBroadcastTo(_engine.Reshape(Value(NoMemory), new[] { 1, Configuration.Dim, 1, 1 }), features.Shape.ToArray()));

    /// <summary>As <see cref="EncodeImage"/>, but the 1/16 features without the no-memory embedding: the frame features
    /// the video model conditions on its memory and encodes into it.</summary>
    public (Tensor<T> Features, Tensor<T> HighRes0, Tensor<T> HighRes1) EncodeFeatures(Tensor<T> images)
    {
        var c = Configuration;
        int b = images.Shape[0];
        var patches = Convolve(PatchProjection, images, c.PatchKernel, c.PatchStride, c.PatchPadding);       // [B, C, H, W]
        int h = patches.Shape[2], w = patches.Shape[3], d0 = c.EmbedDims[0];
        var x = _engine.TensorPermute(patches, new[] { 0, 2, 3, 1 });                                         // [B, H, W, C]
        x = _engine.TensorAdd(x, _engine.TensorBroadcastTo(PositionEmbedding(h, w), new[] { b, h, w, d0 }));
        var stages = new List<Tensor<T>>();
        for (int i = 0; i < Blocks.Count; i++)
        {
            // Stochastic depth rising linearly from 0 at the first block to DropPath at the last (torch.linspace).
            double drop = Training && c.DropPath > 0 && Blocks.Count > 1 ? c.DropPath * i / (Blocks.Count - 1) : 0.0;
            x = Run(Blocks[i], x, drop);
            if (StageEnds.Contains(i)) stages.Add(x);
        }

        // FPN neck, coarse to fine: each level's 1x1 projection, plus the coarser output upsampled (nearest, x2) on
        // the top-down levels.
        int n = stages.Count - 1;
        var outputs = new Tensor<T>[stages.Count];
        Tensor<T>? previous = null;
        for (int i = n; i >= 0; i--)
        {
            var lateral = _engine.TensorPermute(Rows(NeckConvs[n - i], stages[i]), new[] { 0, 3, 1, 2 });      // [B, Dim, h, w]
            previous = !c.TopDownLevels.Contains(i) || i == n || previous is null ? lateral : _engine.TensorAdd(lateral, Nearest2x(previous));
            outputs[i] = previous;
        }
        return (outputs[2], Convolve(HighRes0, outputs[0], 1, 1, 0), Convolve(HighRes1, outputs[1], 1, 1, 0));
    }

    private Tensor<T> Nearest2x(Tensor<T> x)
    {
        int b = x.Shape[0], ch = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var expanded = _engine.TensorBroadcastTo(_engine.Reshape(x, new[] { b, ch, h, 1, w, 1 }), new[] { b, ch, h, 2, w, 2 });
        return _engine.Reshape(expanded, new[] { b, ch, 2 * h, 2 * w });
    }

    /// <summary>Hiera's position embedding <c>[1, H, W, C]</c>: the background table bicubically interpolated to the grid
    /// plus the first window's table tiled over it.</summary>
    private Tensor<T> PositionEmbedding(int h, int w)
    {
        var c = Configuration;
        int d = c.EmbedDims[0], s = c.BackgroundPositionSize, ws = c.WindowSizes[0];
        var background = _engine.Reshape(Value(BackgroundPosition), new[] { d, s, s });
        // out[c] = Mh · P[c] · Mwᵀ.
        var rows = _engine.BatchMatMul(_engine.TensorBroadcastTo(_engine.Reshape(Bicubic(h, s), new[] { 1, h, s }), new[] { d, h, s }), background);
        var full = _engine.BatchMatMul(rows, _engine.TensorBroadcastTo(_engine.Reshape(_engine.TensorTranspose(Bicubic(w, s)), new[] { 1, s, w }), new[] { d, s, w }));
        var window = _engine.Reshape(Value(WindowPosition), new[] { d, ws, ws });
        var tiled = _engine.TensorTile(window, new[] { 1, h / ws, w / ws });
        return _engine.Reshape(_engine.TensorPermute(_engine.TensorAdd(full, tiled), new[] { 1, 2, 0 }), new[] { 1, h, w, d });
    }

    /// <summary>PyTorch's bicubic interpolation (align_corners false, A = −0.75, border taps clamped) from
    /// <paramref name="input"/> to <paramref name="output"/> samples as a matrix <c>[output, input]</c>.</summary>
    private Tensor<T> Bicubic(int output, int input)
    {
        if (_bicubicRows.TryGetValue((output, input), out var cached)) return cached;
        const double a = -0.75;
        double Near(double x) => ((a + 2) * x - (a + 3)) * x * x + 1;
        double Far(double x) => ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
        var matrix = new Tensor<T>(new[] { output, input });
        double scale = (double)input / output;
        for (int o = 0; o < output; o++)
        {
            double real = (o + 0.5) * scale - 0.5;
            int floor = (int)Math.Floor(real);
            double t = real - floor;
            var weights = new[] { Far(t + 1), Near(t), Near(1 - t), Far(2 - t) };
            for (int k = 0; k < 4; k++)
            {
                int index = Math.Min(Math.Max(floor - 1 + k, 0), input - 1);
                matrix[o, index] = NumOps.Add(matrix[o, index], NumOps.FromDouble(weights[k]));
            }
        }
        _bicubicRows[(output, input)] = matrix;
        return matrix;
    }

    /// <summary>A Hiera block on <c>[B, H, W, C]</c>: windowed (or global) attention, query pooling on a stage's first
    /// block (the shortcut projected and pooled to match), and the MLP, each branch dropped per sample with probability
    /// <paramref name="drop"/> (and the kept ones scaled by 1/(1 − drop)).</summary>
    private Tensor<T> Run(Block block, Tensor<T> x, double drop = 0.0)
    {
        int b = x.Shape[0], h = x.Shape[1], w = x.Shape[2];
        var residual = x;
        var normed = Rows(block.Norm1, x);
        if (block.Shortcut is not null)
        {
            residual = Rows(block.Shortcut, normed);
            if (block.PoolQuery) residual = MaxPool2(residual);
        }
        int ws = block.Window;
        Tensor<T> windows;
        int ph = h, pw = w;
        if (ws > 0)
        {
            ph = (h + ws - 1) / ws * ws;
            pw = (w + ws - 1) / ws * ws;
            var padded = normed;
            if (pw != w) padded = _engine.TensorConcatenate(new[] { padded, new Tensor<T>(new[] { b, h, pw - w, block.Dim }) }, 2);
            if (ph != h) padded = _engine.TensorConcatenate(new[] { padded, new Tensor<T>(new[] { b, ph - h, pw, block.Dim }) }, 1);
            windows = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(padded, new[] { b, ph / ws, ws, pw / ws, ws, block.Dim }), new[] { 0, 1, 3, 2, 4, 5 }),
                new[] { b * (ph / ws) * (pw / ws), ws, ws, block.Dim });
        }
        else
        {
            windows = normed;
        }
        var attended = Attend(block, windows);                                                                // [N, h', w', DimOut]
        if (ws > 0)
        {
            int wo = attended.Shape[1];
            var (rh, rw) = (residual.Shape[1], residual.Shape[2]);
            int qh = (rh + wo - 1) / wo * wo, qw = (rw + wo - 1) / wo * wo;
            var grid = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(attended, new[] { b, qh / wo, qw / wo, wo, wo, block.DimOut }), new[] { 0, 1, 3, 2, 4, 5 }),
                new[] { b, qh, qw, block.DimOut });
            attended = qh == rh && qw == rw ? grid : _engine.TensorSlice(grid, new[] { 0, 0, 0, 0 }, new[] { b, rh, rw, block.DimOut });
        }
        x = _engine.TensorAdd(residual, DropPathOf(attended, drop));
        return _engine.TensorAdd(x, DropPathOf(Rows(block.Mlp2, Gelu(Rows(block.Mlp1, Rows(block.Norm2, x)))), drop));
    }

    /// <summary>timm's DropPath: each sample's branch is zeroed with probability <paramref name="drop"/>, else scaled by
    /// 1/(1 − drop).</summary>
    private Tensor<T> DropPathOf(Tensor<T> branch, double drop)
    {
        if (drop <= 0) return branch;
        int b = branch.Shape[0];
        var shape = new int[branch.Rank];
        shape[0] = b;
        for (int i = 1; i < shape.Length; i++) shape[i] = 1;
        var keep = new Tensor<T>(shape);
        for (int i = 0; i < b; i++) keep[i] = TrainingRandom.NextDouble() < drop ? NumOps.Zero : NumOps.FromDouble(1.0 / (1.0 - drop));
        return _engine.TensorMultiply(branch, _engine.TensorBroadcastTo(keep, branch.Shape.ToArray()));
    }

    private Tensor<T> MaxPool2(Tensor<T> x)
    {
        int b = x.Shape[0], h = x.Shape[1], w = x.Shape[2], d = x.Shape[3];
        if (h % 2 != 0 || w % 2 != 0) x = _engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { b, h / 2 * 2, w / 2 * 2, d });   // ceil_mode false
        var split = _engine.Reshape(x, new[] { b, h / 2, 2, w / 2, 2, d });
        return _engine.ReduceMax(_engine.ReduceMax(split, new[] { 4 }, keepDims: false, out _), new[] { 2 }, keepDims: false, out _);
    }

    /// <summary>Hiera's multi-scale attention on <c>[N, h, w, Dim]</c>: fused qkv, the queries max-pooled 2x2 on a pooling
    /// block, softmax attention, the output projection.</summary>
    private Tensor<T> Attend(Block block, Tensor<T> x)
    {
        int n = x.Shape[0], h = x.Shape[1], w = x.Shape[2], heads = block.Heads, hd = block.DimOut / heads, tokens = h * w;
        var qkv = _engine.Reshape(Rows(block.Qkv, x), new[] { n, tokens, 3, heads, hd });
        Tensor<T> Part(int i) => _engine.Reshape(_engine.TensorSlice(qkv, new[] { 0, 0, i, 0, 0 }, new[] { n, tokens, 1, heads, hd }), new[] { n, tokens, heads, hd });
        var q = Part(0);
        int qh = h, qw = w;
        if (block.PoolQuery)
        {
            var pooled = MaxPool2(_engine.Reshape(q, new[] { n, h, w, heads * hd }));
            qh = pooled.Shape[1];
            qw = pooled.Shape[2];
            q = _engine.Reshape(pooled, new[] { n, qh * qw, heads, hd });
        }
        Tensor<T> Heads(Tensor<T> t, int len) => _engine.Reshape(_engine.TensorPermute(t, new[] { 0, 2, 1, 3 }), new[] { n * heads, len, hd });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(q, qh * qw), _engine.TensorPermute(Heads(Part(1), tokens), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(Part(2), tokens));
        var merged = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { n, heads, qh * qw, hd }), new[] { 0, 2, 1, 3 }), new[] { n, qh, qw, block.DimOut });
        return Rows(block.Projection, merged);
    }
}
