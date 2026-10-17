using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>The sizes of a Segment Anything model (the official <c>build_sam</c>; defaults are ViT-B).</summary>
internal sealed record SamConfiguration
{
    public int ImageSize { get; init; } = 1024;
    public int PatchSize { get; init; } = 16;
    public int Channels { get; init; } = 3;
    public int EncoderDim { get; init; } = 768;
    public int EncoderDepth { get; init; } = 12;
    public int EncoderHeads { get; init; } = 12;
    public double MlpRatio { get; init; } = 4.0;
    public int WindowSize { get; init; } = 14;

    /// <summary>Decomposed relative-position terms in the encoder's attention (the released models use them).</summary>
    public bool RelativePositions { get; init; } = true;

    /// <summary>The blocks that attend globally; the others attend within windows.</summary>
    public int[] GlobalAttention { get; init; } = { 2, 5, 8, 11 };

    public int OutChannels { get; init; } = 256;
    public int MaskInChannels { get; init; } = 16;
    public int DecoderDepth { get; init; } = 2;
    public int DecoderHeads { get; init; } = 8;
    public int DecoderMlp { get; init; } = 2048;
    public int AttentionDownsample { get; init; } = 2;
    public int MultimaskOutputs { get; init; } = 3;
    public int IouHidden { get; init; } = 256;
    public int IouDepth { get; init; } = 3;
    public double[] PixelMean { get; init; } = { 123.675, 116.28, 103.53 };
    public double[] PixelStd { get; init; } = { 58.395, 57.12, 57.375 };

    /// <summary>The image embedding's side (the patch grid).</summary>
    public int Grid => ImageSize / PatchSize;

    /// <summary>The prompt and decoder width (the image embedding's channels).</summary>
    public int PromptDim => OutChannels;

    public int MaskTokens => MultimaskOutputs + 1;

    /// <summary>ViT-B, ViT-L and ViT-H as <c>sam_model_registry</c> builds them.</summary>
    public static SamConfiguration For(SAMModelSize size) => size switch
    {
        SAMModelSize.ViTBase => new SamConfiguration(),
        SAMModelSize.ViTLarge => new SamConfiguration { EncoderDim = 1024, EncoderDepth = 24, EncoderHeads = 16, GlobalAttention = new[] { 5, 11, 17, 23 } },
        SAMModelSize.ViTHuge => new SamConfiguration { EncoderDim = 1280, EncoderDepth = 32, EncoderHeads = 16, GlobalAttention = new[] { 7, 15, 23, 31 } },
        _ => throw new NotSupportedException($"SAM size {size} is not supported."),
    };
}

/// <summary>
/// Segment Anything (Kirillov et al. 2023) as the official code (facebookresearch/segment-anything) builds it.
/// </summary>
/// <remarks>
/// <para><b>Image encoder:</b> an MAE ViT on 16-pixel patches with an absolute position table; most blocks attend
/// within 14×14 windows (zero-padding the grid to whole windows) and a few attend globally, each adding decomposed
/// relative-position terms; a neck of a 1×1 and a 3×3 convolution, each followed by a channel LayerNorm, gives the
/// 256-channel embedding.</para>
/// <para><b>Prompt encoder:</b> points and box corners are encoded by random Fourier features of their normalized
/// coordinates plus a learned embedding per kind; a mask prompt is downscaled by two stride-2 convolutions; without
/// one, a learned "no mask" embedding fills the grid.</para>
/// <para><b>Mask decoder:</b> an IoU token and four mask tokens join the prompt tokens in a two-way transformer
/// (token self-attention, token-to-image and image-to-token cross-attention at half width); the image embedding is
/// upscaled 4× by two transposed convolutions, and each mask token's hypernetwork MLP gives the weights of one mask;
/// an MLP on the IoU token predicts each mask's quality.</para>
/// </remarks>
internal sealed class SamNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _owned = new();

    public SamNetwork(IEngine engine, List<LayerBase<T>> layers, SamConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.EncoderDim % c.EncoderHeads != 0 || c.PromptDim % c.DecoderHeads != 0)
            throw new ArgumentException("The widths must divide by the head counts.");
        LayerBase<T> Own(LayerBase<T> layer)
        {
            layers.Add(layer);
            _owned.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => (DenseLayer<T>)Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        BiasFreeLinearLayer<T> Values(int count) => (BiasFreeLinearLayer<T>)Own(new BiasFreeLinearLayer<T>(1, count));
        // The encoder's LayerNorms and every channel LayerNorm (LayerNorm2d) use ε = 1e-6; the two-way transformer's are
        // nn.LayerNorm with PyTorch's default 1e-5.
        LayerNormalizationLayer<T> Norm(int dim) => (LayerNormalizationLayer<T>)Own(new LayerNormalizationLayer<T>(dim, 1e-6));
        LayerNormalizationLayer<T> DecoderNorm(int dim) => (LayerNormalizationLayer<T>)Own(new LayerNormalizationLayer<T>(dim, 1e-5));
        ConvolutionalLayer<T> Conv(int inputs, int outputs, int kernel, int stride, bool bias) => (ConvolutionalLayer<T>)Own(
            ConvolutionalLayer<T>.WithInputDepth(inputs, outputs, kernel, stride, padding: kernel == 3 ? 1 : 0,
                initializationStrategy: new Initialization.ZeroInitializationStrategy<T>(),
                biasMode: bias ? Enums.BiasMode.Auto : Enums.BiasMode.Never));
        Attention Attend(int dim, int heads, int downsample)
            => new(Linear(dim / downsample), Linear(dim / downsample), Linear(dim / downsample), Linear(dim), heads);

        int d = c.EncoderDim, grid = c.Grid, headDim = d / c.EncoderHeads;
        PatchProjection = Conv(c.Channels, d, c.PatchSize, c.PatchSize, bias: true);
        PositionTable = Values(grid * grid * d);
        for (int i = 0; i < c.EncoderDepth; i++)
        {
            int window = c.GlobalAttention.Contains(i) ? 0 : c.WindowSize;
            int span = window == 0 ? grid : window;
            Blocks.Add(new EncoderBlock(Norm(d), Linear(3 * d), Linear(d),
                c.RelativePositions ? Values((2 * span - 1) * headDim) : null, c.RelativePositions ? Values((2 * span - 1) * headDim) : null,
                Norm(d), Linear((int)(d * c.MlpRatio)), Linear(d), window, span));
        }
        NeckConv1 = Conv(d, c.OutChannels, 1, 1, bias: false);
        NeckNorm1 = Norm(c.OutChannels);
        NeckConv2 = Conv(c.OutChannels, c.OutChannels, 3, 1, bias: false);
        NeckNorm2 = Norm(c.OutChannels);

        int p = c.PromptDim;
        for (int i = 0; i < 4; i++) PointEmbeddings.Add(Values(p));
        NotAPoint = Values(p);
        NoMask = Values(p);
        MaskConv1 = Conv(1, c.MaskInChannels / 4, 2, 2, bias: true);
        MaskNorm1 = Norm(c.MaskInChannels / 4);
        MaskConv2 = Conv(c.MaskInChannels / 4, c.MaskInChannels, 2, 2, bias: true);
        MaskNorm2 = Norm(c.MaskInChannels);
        MaskConv3 = Conv(c.MaskInChannels, p, 1, 1, bias: true);
        _gaussian = new double[2, p / 2];

        IouToken = Values(p);
        MaskTokenTable = Values(c.MaskTokens * p);
        for (int i = 0; i < c.DecoderDepth; i++)
            TwoWayBlocks.Add(new TwoWayBlock(Attend(p, c.DecoderHeads, 1), DecoderNorm(p), Attend(p, c.DecoderHeads, c.AttentionDownsample), DecoderNorm(p),
                Linear(c.DecoderMlp), Linear(p), DecoderNorm(p), DecoderNorm(p), Attend(p, c.DecoderHeads, c.AttentionDownsample), SkipFirstPositional: i == 0));
        FinalAttention = Attend(p, c.DecoderHeads, c.AttentionDownsample);
        FinalNorm = DecoderNorm(p);
        Upscale1 = new TransposedConvolution2x2<T>(engine, Own, p, p / 4);
        UpscaleNorm = Norm(p / 4);
        Upscale2 = new TransposedConvolution2x2<T>(engine, Own, p / 4, p / 8);
        for (int i = 0; i < c.MaskTokens; i++) Hypernetworks.Add(new[] { Linear(p), Linear(p), Linear(p / 8) });
        var iou = new List<DenseLayer<T>>();
        for (int i = 0; i < c.IouDepth - 1; i++) iou.Add(Linear(c.IouHidden));
        iou.Add(Linear(c.MaskTokens));
        IouHead = iou.ToArray();
    }

    /// <summary>An attention block's projections (the cross-attentions project to half width) and head count.</summary>
    internal sealed record Attention(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output, int Heads);

    /// <summary>A ViT block: norms, attention (with its relative-position tables) and MLP, and its window (0: global).</summary>
    internal sealed record EncoderBlock(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Qkv, DenseLayer<T> Projection,
        BiasFreeLinearLayer<T>? RelativeHeight, BiasFreeLinearLayer<T>? RelativeWidth, LayerNormalizationLayer<T> Norm2,
        DenseLayer<T> Mlp1, DenseLayer<T> Mlp2, int Window, int Span);

    /// <summary>A two-way attention block.</summary>
    internal sealed record TwoWayBlock(Attention SelfAttention, LayerNormalizationLayer<T> Norm1, Attention TokenToImage,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> Mlp1, DenseLayer<T> Mlp2, LayerNormalizationLayer<T> Norm3,
        LayerNormalizationLayer<T> Norm4, Attention ImageToToken, bool SkipFirstPositional);

    public SamConfiguration Configuration { get; }
    public ConvolutionalLayer<T> PatchProjection { get; }
    public BiasFreeLinearLayer<T> PositionTable { get; }
    public List<EncoderBlock> Blocks { get; } = new();
    public ConvolutionalLayer<T> NeckConv1 { get; }
    public LayerNormalizationLayer<T> NeckNorm1 { get; }
    public ConvolutionalLayer<T> NeckConv2 { get; }
    public LayerNormalizationLayer<T> NeckNorm2 { get; }
    public List<BiasFreeLinearLayer<T>> PointEmbeddings { get; } = new();
    public BiasFreeLinearLayer<T> NotAPoint { get; }
    public BiasFreeLinearLayer<T> NoMask { get; }
    public ConvolutionalLayer<T> MaskConv1 { get; }
    public LayerNormalizationLayer<T> MaskNorm1 { get; }
    public ConvolutionalLayer<T> MaskConv2 { get; }
    public LayerNormalizationLayer<T> MaskNorm2 { get; }
    public ConvolutionalLayer<T> MaskConv3 { get; }
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

    /// <summary>The random Fourier matrix <c>[2, PromptDim/2]</c> of the position encoding: a fixed buffer.</summary>
    private readonly double[,] _gaussian;

    /// <summary>The layers this network added to its owner's list, in order.</summary>
    public IReadOnlyList<LayerBase<T>> OwnedLayers => _owned;

    // ------------------------------------------------------------------ helpers

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

    /// <summary>A LayerNorm over the channel axis of <c>[B, C, H, W]</c> (the official LayerNorm2d).</summary>
    private Tensor<T> ChannelNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
        => _engine.TensorPermute(Rows(norm, _engine.TensorPermute(x, new[] { 0, 2, 3, 1 })), new[] { 0, 3, 1, 2 });

    private Tensor<T> Gelu(Tensor<T> x) => ActivationHelper.ExactGelu(x, _engine);

    // ------------------------------------------------------------------ image encoder

    /// <summary>The official <c>Sam.preprocess</c>: normalize by the pixel mean and std, then zero-pad the bottom and
    /// right to the square encoder size. <paramref name="image"/> is <c>[B, C, h, w]</c> with h, w ≤ ImageSize.</summary>
    public Tensor<T> Preprocess(Tensor<T> image)
    {
        var c = Configuration;
        int b = image.Shape[0], ch = image.Shape[1], h = image.Shape[2], w = image.Shape[3];
        if (h > c.ImageSize || w > c.ImageSize)
            throw new ArgumentException($"SAM encodes images up to {c.ImageSize} pixels a side; resize the longest side to {c.ImageSize} first.");
        var mean = new Tensor<T>(new[] { 1, ch, 1, 1 });
        var scale = new Tensor<T>(new[] { 1, ch, 1, 1 });
        for (int i = 0; i < ch; i++)
        {
            mean[i] = NumOps.FromDouble(c.PixelMean[i % c.PixelMean.Length]);
            scale[i] = NumOps.FromDouble(1.0 / c.PixelStd[i % c.PixelStd.Length]);
        }
        var shape = image.Shape.ToArray();
        var x = _engine.TensorMultiply(_engine.TensorSubtract(image, _engine.TensorBroadcastTo(mean, shape)), _engine.TensorBroadcastTo(scale, shape));
        if (w < c.ImageSize) x = _engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { b, ch, h, c.ImageSize - w }) }, 3);
        if (h < c.ImageSize) x = _engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { b, ch, c.ImageSize - h, c.ImageSize }) }, 2);
        return x;
    }

    /// <summary>The image embedding <c>[B, OutChannels, Grid, Grid]</c> of a preprocessed batch.</summary>
    public Tensor<T> EncodeImage(Tensor<T> preprocessed)
    {
        var c = Configuration;
        int b = preprocessed.Shape[0], g = c.Grid, d = c.EncoderDim;
        var x = _engine.TensorPermute(PatchProjection.Forward(preprocessed), new[] { 0, 2, 3, 1 });                // [B, g, g, D]
        x = _engine.TensorAdd(x, _engine.TensorBroadcastTo(_engine.Reshape(Value(PositionTable), new[] { 1, g, g, d }), new[] { b, g, g, d }));
        foreach (var block in Blocks) x = Run(block, x);
        var y = _engine.TensorPermute(x, new[] { 0, 3, 1, 2 });
        y = ChannelNorm(NeckNorm1, NeckConv1.Forward(y));
        return ChannelNorm(NeckNorm2, NeckConv2.Forward(y));
    }

    private Tensor<T> Run(EncoderBlock block, Tensor<T> x)
    {
        int b = x.Shape[0], g = x.Shape[1], d = x.Shape[3];
        var h = Rows(block.Norm1, x);
        int padded = g, windows = 1, size = g;
        if (block.Window > 0)
        {
            size = block.Window;
            padded = (g + size - 1) / size * size;
            if (padded > g)
            {
                h = _engine.TensorConcatenate(new[] { h, new Tensor<T>(new[] { b, g, padded - g, d }) }, 2);
                h = _engine.TensorConcatenate(new[] { h, new Tensor<T>(new[] { b, padded - g, padded, d }) }, 1);
            }
            int n = padded / size;
            windows = n * n;
            h = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(h, new[] { b, n, size, n, size, d }), new[] { 0, 1, 3, 2, 4, 5 }),
                new[] { b * windows, size, size, d });
        }
        h = Attend(block, h);
        if (block.Window > 0)
        {
            int n = padded / size;
            h = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(h, new[] { b, n, n, size, size, d }), new[] { 0, 1, 3, 2, 4, 5 }),
                new[] { b, padded, padded, d });
            if (padded > g) h = _engine.TensorSlice(h, new[] { 0, 0, 0, 0 }, new[] { b, g, g, d });
        }
        x = _engine.TensorAdd(x, h);
        return _engine.TensorAdd(x, Rows(block.Mlp2, Gelu(Rows(block.Mlp1, Rows(block.Norm2, x)))));
    }

    /// <summary>Multi-head attention over a <c>[N, S, S, D]</c> grid with decomposed relative positions.</summary>
    private Tensor<T> Attend(EncoderBlock block, Tensor<T> x)
    {
        int n = x.Shape[0], s = x.Shape[1], d = x.Shape[3], heads = Configuration.EncoderHeads, hd = d / heads, tokens = s * s;
        var qkv = _engine.Reshape(Rows(block.Qkv, x), new[] { n, tokens, 3, heads, hd });
        qkv = _engine.Reshape(_engine.TensorPermute(qkv, new[] { 2, 0, 3, 1, 4 }), new[] { 3, n * heads, tokens, hd });
        Tensor<T> Part(int i) => _engine.Reshape(_engine.TensorSlice(qkv, new[] { i, 0, 0, 0 }, new[] { 1, n * heads, tokens, hd }),
            new[] { n * heads, tokens, hd });
        var q = Part(0);
        var k = Part(1);
        var v = Part(2);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        if (block.RelativeHeight is not null && block.RelativeWidth is not null)
            scores = _engine.TensorAdd(scores, RelativeBias(block.RelativeHeight, block.RelativeWidth, q, n * heads, s, hd));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);                               // [N·h, T, hd]
        context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { n, heads, s, s, hd }), new[] { 0, 2, 3, 1, 4 }),
            new[] { n, s, s, d });
        return Rows(block.Projection, context);
    }

    /// <summary>The decomposed relative-position terms (MViTv2) <c>[N·h, T, T]</c>: rel_h[b, i, j, k] = q[b, i, j]·R_h[i, k],
    /// rel_w likewise.</summary>
    private Tensor<T> RelativeBias(BiasFreeLinearLayer<T> heightTable, BiasFreeLinearLayer<T> widthTable, Tensor<T> q, int batch, int s, int hd)
    {
        int tokens = s * s;
        var rh = RelativeTable(heightTable, s, hd);                                                          // [S, S, hd]
        var rw = RelativeTable(widthTable, s, hd);
        var qGrid = _engine.Reshape(q, new[] { batch, s, s, hd });
        var relH = _engine.BatchMatMul(
            _engine.Reshape(_engine.TensorPermute(qGrid, new[] { 1, 0, 2, 3 }), new[] { s, batch * s, hd }),
            _engine.TensorPermute(rh, new[] { 0, 2, 1 }));                                                    // [i, b·j, k_h]
        relH = _engine.TensorPermute(_engine.Reshape(relH, new[] { s, batch, s, s }), new[] { 1, 0, 2, 3 }); // [b, i, j, k_h]
        var relW = _engine.BatchMatMul(
            _engine.Reshape(_engine.TensorPermute(qGrid, new[] { 2, 0, 1, 3 }), new[] { s, batch * s, hd }),
            _engine.TensorPermute(rw, new[] { 0, 2, 1 }));                                                    // [w, b·i, k_w]
        relW = _engine.TensorPermute(_engine.Reshape(relW, new[] { s, batch, s, s }), new[] { 1, 2, 0, 3 }); // [b, i, j, k_w]
        var full = new[] { batch, s, s, s, s };
        var bias = _engine.TensorAdd(
            _engine.TensorBroadcastTo(_engine.Reshape(relH, new[] { batch, s, s, s, 1 }), full),
            _engine.TensorBroadcastTo(_engine.Reshape(relW, new[] { batch, s, s, 1, s }), full));
        return _engine.Reshape(bias, new[] { batch, tokens, tokens });
    }

    /// <summary>The official <c>get_rel_pos</c> for equal query and key sizes: row <c>q − k + S − 1</c> of the table, as
    /// <c>[S, S, hd]</c> (a constant selection of the learned rows, so gradients reach the table).</summary>
    private Tensor<T> RelativeTable(BiasFreeLinearLayer<T> holder, int s, int hd)
    {
        var table = _engine.Reshape(Value(holder), new[] { 2 * s - 1, hd });
        var select = new Tensor<T>(new[] { s * s, 2 * s - 1 });
        for (int q = 0; q < s; q++)
            for (int k = 0; k < s; k++) select[q * s + k, q - k + s - 1] = NumOps.One;
        return _engine.Reshape(_engine.TensorMatMul(select, table), new[] { s, s, hd });
    }

    // ------------------------------------------------------------------ prompt encoder

    /// <summary>The random Fourier features <c>[sin, cos](2π·(2c − 1)·G)</c> of normalized coordinates (x, y).</summary>
    private double[] Fourier(double x, double y)
    {
        int half = Configuration.PromptDim / 2;
        var f = new double[2 * half];
        double cx = 2 * x - 1, cy = 2 * y - 1;
        for (int j = 0; j < half; j++)
        {
            double a = 2 * Math.PI * (cx * _gaussian[0, j] + cy * _gaussian[1, j]);
            f[j] = Math.Sin(a);
            f[half + j] = Math.Cos(a);
        }
        return f;
    }

    /// <summary>The dense position encoding <c>[1, PromptDim, Grid, Grid]</c> of the embedding grid (pixel centres).</summary>
    public Tensor<T> DensePositional()
    {
        int g = Configuration.Grid, p = Configuration.PromptDim;
        var pe = new Tensor<T>(new[] { 1, p, g, g });
        for (int i = 0; i < g; i++)
            for (int j = 0; j < g; j++)
            {
                var f = Fourier((j + 0.5) / g, (i + 0.5) / g);
                for (int k = 0; k < p; k++) pe[(k * g + i) * g + j] = NumOps.FromDouble(f[k]);
            }
        return pe;
    }

    /// <summary>
    /// The sparse prompt embeddings <c>[N, tokens, PromptDim]</c> and the dense ones <c>[N, PromptDim, Grid, Grid]</c>
    /// for N prompts: points <c>[N, k, 2]</c> (x, y in the encoder frame) with labels <c>[N, k]</c> (1 positive,
    /// 0 negative, −1 padding), boxes <c>[N, 4]</c> (x0, y0, x1, y1), and low-resolution mask logits
    /// <c>[N, 1, 4·Grid, 4·Grid]</c>.
    /// </summary>
    public (Tensor<T> Sparse, Tensor<T> Dense) EncodePrompts(Tensor<T>? points, Tensor<T>? labels, Tensor<T>? boxes, Tensor<T>? masks)
    {
        var c = Configuration;
        int p = c.PromptDim, g = c.Grid;
        int n = points?.Shape[0] ?? boxes?.Shape[0] ?? masks?.Shape[0] ?? 1;
        var parts = new List<Tensor<T>>();
        if (points is not null && labels is not null)
        {
            int k = points.Shape[1];
            bool pad = boxes is null;
            int count = k + (pad ? 1 : 0);
            var fourier = new Tensor<T>(new[] { n, count, p });
            var kinds = new Tensor<T>[3];
            for (int i = 0; i < 3; i++) kinds[i] = new Tensor<T>(new[] { n, count, 1 });
            for (int b = 0; b < n; b++)
                for (int i = 0; i < count; i++)
                {
                    int label = i < k ? (int)Math.Round(NumOps.ToDouble(labels[b, i])) : -1;
                    if (label == -1) { kinds[0][b, i, 0] = NumOps.One; continue; }
                    double x = (NumOps.ToDouble(points[b, i, 0]) + 0.5) / c.ImageSize, y = (NumOps.ToDouble(points[b, i, 1]) + 0.5) / c.ImageSize;
                    var f = Fourier(x, y);
                    for (int j = 0; j < p; j++) fourier[(b * count + i) * p + j] = NumOps.FromDouble(f[j]);
                    kinds[label == 1 ? 2 : 1][b, i, 0] = NumOps.One;
                }
            var shape = new[] { n, count, p };
            Tensor<T> Kind(Tensor<T> mask, BiasFreeLinearLayer<T> embedding) => _engine.TensorMultiply(
                _engine.TensorBroadcastTo(mask, shape), _engine.TensorBroadcastTo(_engine.Reshape(Value(embedding), new[] { 1, 1, p }), shape));
            var embedded = _engine.TensorAdd(fourier, Kind(kinds[0], NotAPoint));
            embedded = _engine.TensorAdd(embedded, Kind(kinds[1], PointEmbeddings[0]));
            parts.Add(_engine.TensorAdd(embedded, Kind(kinds[2], PointEmbeddings[1])));
        }
        if (boxes is not null)
        {
            var corners = new Tensor<T>(new[] { n, 2, p });
            for (int b = 0; b < n; b++)
                for (int corner = 0; corner < 2; corner++)
                {
                    var f = Fourier((NumOps.ToDouble(boxes[b, 2 * corner]) + 0.5) / c.ImageSize, (NumOps.ToDouble(boxes[b, 2 * corner + 1]) + 0.5) / c.ImageSize);
                    for (int j = 0; j < p; j++) corners[(b * 2 + corner) * p + j] = NumOps.FromDouble(f[j]);
                }
            var cornerEmbeddings = _engine.TensorConcatenate(new[]
            {
                _engine.Reshape(Value(PointEmbeddings[2]), new[] { 1, 1, p }), _engine.Reshape(Value(PointEmbeddings[3]), new[] { 1, 1, p }),
            }, 1);
            parts.Add(_engine.TensorAdd(corners, _engine.TensorBroadcastTo(cornerEmbeddings, new[] { n, 2, p })));
        }
        var sparse = parts.Count == 0 ? new Tensor<T>(new[] { n, 0, p }) : parts.Count == 1 ? parts[0] : _engine.TensorConcatenate(parts.ToArray(), 1);
        Tensor<T> dense;
        if (masks is not null)
        {
            var m = Gelu(ChannelNorm(MaskNorm1, MaskConv1.Forward(masks)));
            m = Gelu(ChannelNorm(MaskNorm2, MaskConv2.Forward(m)));
            dense = MaskConv3.Forward(m);
        }
        else
        {
            dense = _engine.TensorBroadcastTo(_engine.Reshape(Value(NoMask), new[] { 1, p, 1, 1 }), new[] { n, p, g, g });
        }
        return (sparse, dense);
    }

    // ------------------------------------------------------------------ mask decoder

    private Tensor<T> Attend(Attention attention, Tensor<T> q, Tensor<T> k, Tensor<T> v)
    {
        int n = q.Shape[0], tq = q.Shape[1], tk = k.Shape[1];
        var qp = Rows(attention.Query, q);
        int inner = qp.Shape[2], heads = attention.Heads, hd = inner / heads;
        Tensor<T> Heads(Tensor<T> x, int t) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(x, new[] { n, t, heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { n * heads, t, hd });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(qp, tq), _engine.TensorPermute(Heads(Rows(attention.Key, k), tk), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(Rows(attention.Value, v), tk));
        context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { n, heads, tq, hd }), new[] { 0, 2, 1, 3 }), new[] { n, tq, inner });
        return Rows(attention.Output, context);
    }

    /// <summary>
    /// The mask logits <c>[N, masks, 4·Grid, 4·Grid]</c> and IoU predictions <c>[N, masks]</c> for N prompts on one image
    /// embedding <c>[1, OutChannels, Grid, Grid]</c>: the three ambiguity-aware masks with
    /// <paramref name="multimask"/>, else the single one.
    /// </summary>
    public (Tensor<T> Logits, Tensor<T> Iou) DecodeMasks(Tensor<T> embedding, Tensor<T> positional, Tensor<T> sparse, Tensor<T> dense, bool multimask)
    {
        var c = Configuration;
        int n = sparse.Shape[0], p = c.PromptDim, g = c.Grid, hw = g * g, m = c.MaskTokens;
        var output = _engine.TensorConcatenate(new[] { _engine.Reshape(Value(IouToken), new[] { 1, 1, p }), _engine.Reshape(Value(MaskTokenTable), new[] { 1, m, p }) }, 1);
        var tokens = _engine.TensorBroadcastTo(output, new[] { n, 1 + m, p });
        if (sparse.Shape[1] > 0) tokens = _engine.TensorConcatenate(new[] { tokens, sparse }, 1);
        var src = _engine.TensorAdd(_engine.TensorBroadcastTo(embedding, new[] { n, p, g, g }), dense);
        Tensor<T> Sequence(Tensor<T> x) => _engine.TensorPermute(_engine.Reshape(x, new[] { x.Shape[0], p, hw }), new[] { 0, 2, 1 });
        var keys = Sequence(src);
        var keyPositional = Sequence(_engine.TensorBroadcastTo(positional, new[] { n, p, g, g }));
        var queries = tokens;
        var queryPositional = tokens;
        foreach (var block in TwoWayBlocks)
        {
            if (block.SkipFirstPositional)
            {
                queries = Attend(block.SelfAttention, queries, queries, queries);
            }
            else
            {
                var qp = _engine.TensorAdd(queries, queryPositional);
                queries = _engine.TensorAdd(queries, Attend(block.SelfAttention, qp, qp, queries));
            }
            queries = Rows(block.Norm1, queries);
            queries = Rows(block.Norm2, _engine.TensorAdd(queries,
                Attend(block.TokenToImage, _engine.TensorAdd(queries, queryPositional), _engine.TensorAdd(keys, keyPositional), keys)));
            queries = Rows(block.Norm3, _engine.TensorAdd(queries, Rows(block.Mlp2, _engine.ReLU(Rows(block.Mlp1, queries)))));
            keys = Rows(block.Norm4, _engine.TensorAdd(keys,
                Attend(block.ImageToToken, _engine.TensorAdd(keys, keyPositional), _engine.TensorAdd(queries, queryPositional), queries)));
        }
        queries = Rows(FinalNorm, _engine.TensorAdd(queries,
            Attend(FinalAttention, _engine.TensorAdd(queries, queryPositional), _engine.TensorAdd(keys, keyPositional), keys)));

        var image = _engine.Reshape(_engine.TensorPermute(keys, new[] { 0, 2, 1 }), new[] { n, p, g, g });
        var up = Gelu(ChannelNorm(UpscaleNorm, Upscale1.Forward(image)));
        up = Gelu(Upscale2.Forward(up));                                                                    // [N, p/8, 4g, 4g]
        int side = 4 * g, low = p / 8;
        var weights = new Tensor<T>[m];
        for (int i = 0; i < m; i++)
        {
            var t = _engine.Reshape(_engine.TensorSlice(queries, new[] { 0, 1 + i, 0 }, new[] { n, 1, p }), new[] { n, p });
            var layers = Hypernetworks[i];
            t = _engine.ReLU(layers[0].Forward(t));
            t = _engine.ReLU(layers[1].Forward(t));
            weights[i] = _engine.Reshape(layers[2].Forward(t), new[] { n, 1, low });
        }
        var masks = _engine.Reshape(_engine.BatchMatMul(_engine.TensorConcatenate(weights, 1), _engine.Reshape(up, new[] { n, low, side * side })),
            new[] { n, m, side, side });
        var iou = _engine.Reshape(_engine.TensorSlice(queries, new[] { 0, 0, 0 }, new[] { n, 1, p }), new[] { n, p });
        for (int i = 0; i < IouHead.Length; i++)
        {
            iou = IouHead[i].Forward(iou);
            if (i < IouHead.Length - 1) iou = _engine.ReLU(iou);
        }
        int first = multimask ? 1 : 0, count = multimask ? m - 1 : 1;
        return (_engine.TensorSlice(masks, new[] { 0, first, 0, 0 }, new[] { n, count, side, side }),
            _engine.TensorSlice(iou, new[] { 0, first }, new[] { n, count }));
    }

    /// <summary>The official <c>Sam.postprocess_masks</c>: bilinear to the encoder size, crop to the unpadded input,
    /// bilinear to the original size.</summary>
    public Tensor<T> Postprocess(Tensor<T> logits, int inputHeight, int inputWidth, int originalHeight, int originalWidth)
    {
        int s = Configuration.ImageSize;
        var x = SpatialResampling.Resize(_engine, logits, s, s);
        x = _engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], inputHeight, inputWidth });
        return SpatialResampling.Resize(_engine, x, originalHeight, originalWidth);
    }

    // ------------------------------------------------------------------ parameters

    private static void LoadValues(BiasFreeLinearLayer<T> holder, double[] values) => TorchParameters.Linear(holder, 1, values.Length, values);

    private static void LoadConv(ConvolutionalLayer<T> conv, double[] weight, double[]? bias)
        => conv.SetParameters(new Vector<T>((bias is null ? weight : weight.Concat(bias).ToArray()).Select(NumOps.FromDouble).ToArray()));

    /// <summary>Loads an official Sam state dict, given a reader of a named tensor of a given shape.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.EncoderDim, g = c.Grid, hd = d / c.EncoderHeads, p = c.PromptDim;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name, int dim)
            => TorchParameters.LayerNorm(_engine, norm, read($"{name}.weight", new[] { dim }), read($"{name}.bias", new[] { dim }));
        void Attention(Attention a, string name, int downsample)
        {
            Linear(a.Query, $"{name}.q_proj", p, p / downsample);
            Linear(a.Key, $"{name}.k_proj", p, p / downsample);
            Linear(a.Value, $"{name}.v_proj", p, p / downsample);
            Linear(a.Output, $"{name}.out_proj", p / downsample, p);
        }
        LoadConv(PatchProjection, read("image_encoder.patch_embed.proj.weight", new[] { d, c.Channels, c.PatchSize, c.PatchSize }),
            read("image_encoder.patch_embed.proj.bias", new[] { d }));
        LoadValues(PositionTable, read("image_encoder.pos_embed", new[] { 1, g, g, d }));
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string n = $"image_encoder.blocks.{i}";
            Norm(b.Norm1, $"{n}.norm1", d);
            Linear(b.Qkv, $"{n}.attn.qkv", d, 3 * d);
            Linear(b.Projection, $"{n}.attn.proj", d, d);
            if (b.RelativeHeight is not null) LoadValues(b.RelativeHeight, read($"{n}.attn.rel_pos_h", new[] { 2 * b.Span - 1, hd }));
            if (b.RelativeWidth is not null) LoadValues(b.RelativeWidth, read($"{n}.attn.rel_pos_w", new[] { 2 * b.Span - 1, hd }));
            Norm(b.Norm2, $"{n}.norm2", d);
            int mlp = (int)(d * c.MlpRatio);
            Linear(b.Mlp1, $"{n}.mlp.lin1", d, mlp);
            Linear(b.Mlp2, $"{n}.mlp.lin2", mlp, d);
        }
        LoadConv(NeckConv1, read("image_encoder.neck.0.weight", new[] { c.OutChannels, d, 1, 1 }), null);
        Norm(NeckNorm1, "image_encoder.neck.1", c.OutChannels);
        LoadConv(NeckConv2, read("image_encoder.neck.2.weight", new[] { c.OutChannels, c.OutChannels, 3, 3 }), null);
        Norm(NeckNorm2, "image_encoder.neck.3", c.OutChannels);

        var gaussian = read("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", new[] { 2, p / 2 });
        for (int r = 0; r < 2; r++)
            for (int j = 0; j < p / 2; j++) _gaussian[r, j] = gaussian[r * (p / 2) + j];
        for (int i = 0; i < 4; i++) LoadValues(PointEmbeddings[i], read($"prompt_encoder.point_embeddings.{i}.weight", new[] { 1, p }));
        LoadValues(NotAPoint, read("prompt_encoder.not_a_point_embed.weight", new[] { 1, p }));
        LoadValues(NoMask, read("prompt_encoder.no_mask_embed.weight", new[] { 1, p }));
        int quarter = c.MaskInChannels / 4;
        LoadConv(MaskConv1, read("prompt_encoder.mask_downscaling.0.weight", new[] { quarter, 1, 2, 2 }), read("prompt_encoder.mask_downscaling.0.bias", new[] { quarter }));
        Norm(MaskNorm1, "prompt_encoder.mask_downscaling.1", quarter);
        LoadConv(MaskConv2, read("prompt_encoder.mask_downscaling.3.weight", new[] { c.MaskInChannels, quarter, 2, 2 }),
            read("prompt_encoder.mask_downscaling.3.bias", new[] { c.MaskInChannels }));
        Norm(MaskNorm2, "prompt_encoder.mask_downscaling.4", c.MaskInChannels);
        LoadConv(MaskConv3, read("prompt_encoder.mask_downscaling.6.weight", new[] { p, c.MaskInChannels, 1, 1 }), read("prompt_encoder.mask_downscaling.6.bias", new[] { p }));

        LoadValues(IouToken, read("mask_decoder.iou_token.weight", new[] { 1, p }));
        LoadValues(MaskTokenTable, read("mask_decoder.mask_tokens.weight", new[] { c.MaskTokens, p }));
        for (int i = 0; i < TwoWayBlocks.Count; i++)
        {
            var b = TwoWayBlocks[i];
            string n = $"mask_decoder.transformer.layers.{i}";
            Attention(b.SelfAttention, $"{n}.self_attn", 1);
            Norm(b.Norm1, $"{n}.norm1", p);
            Attention(b.TokenToImage, $"{n}.cross_attn_token_to_image", c.AttentionDownsample);
            Norm(b.Norm2, $"{n}.norm2", p);
            Linear(b.Mlp1, $"{n}.mlp.lin1", p, c.DecoderMlp);
            Linear(b.Mlp2, $"{n}.mlp.lin2", c.DecoderMlp, p);
            Norm(b.Norm3, $"{n}.norm3", p);
            Norm(b.Norm4, $"{n}.norm4", p);
            Attention(b.ImageToToken, $"{n}.cross_attn_image_to_token", c.AttentionDownsample);
        }
        Attention(FinalAttention, "mask_decoder.transformer.final_attn_token_to_image", c.AttentionDownsample);
        Norm(FinalNorm, "mask_decoder.transformer.norm_final_attn", p);
        Upscale1.Load(read("mask_decoder.output_upscaling.0.weight", new[] { p, p / 4, 2, 2 }), read("mask_decoder.output_upscaling.0.bias", new[] { p / 4 }));
        Norm(UpscaleNorm, "mask_decoder.output_upscaling.1", p / 4);
        Upscale2.Load(read("mask_decoder.output_upscaling.3.weight", new[] { p / 4, p / 8, 2, 2 }), read("mask_decoder.output_upscaling.3.bias", new[] { p / 8 }));
        for (int i = 0; i < Hypernetworks.Count; i++)
        {
            string n = $"mask_decoder.output_hypernetworks_mlps.{i}.layers";
            Linear(Hypernetworks[i][0], $"{n}.0", p, p);
            Linear(Hypernetworks[i][1], $"{n}.1", p, p);
            Linear(Hypernetworks[i][2], $"{n}.2", p, p / 8);
        }
        for (int i = 0; i < IouHead.Length; i++)
        {
            int inputs = i == 0 ? p : c.IouHidden, outputs = i == IouHead.Length - 1 ? c.MaskTokens : c.IouHidden;
            Linear(IouHead[i], $"mask_decoder.iou_prediction_head.layers.{i}", inputs, outputs);
        }
    }

    /// <summary>
    /// The official initialization: PyTorch's defaults for every convolution and linear layer (U(±1/√fan-in)), LayerNorm
    /// scale 1 and shift 0, zero position and relative-position tables (as <c>build_sam</c> constructs them before loading
    /// a checkpoint), N(0, 1) embeddings, and an N(0, 1) Fourier matrix.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        int fanIn = 1;
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (x, y) => x * y);
            var values = new double[count];
            bool layerNorm = name.Contains(".norm") || name.Contains("norm_final_attn") || name.EndsWith("neck.1.weight") || name.EndsWith("neck.1.bias")
                || name.EndsWith("neck.3.weight") || name.EndsWith("neck.3.bias") || name.Contains("mask_downscaling.1.") || name.Contains("mask_downscaling.4.")
                || name.Contains("output_upscaling.1.");
            if (layerNorm)
            {
                if (name.EndsWith(".weight", StringComparison.Ordinal)) for (int i = 0; i < count; i++) values[i] = 1.0;
                return values;
            }
            if (name.Contains("pos_embed") || name.Contains("rel_pos")) return values;
            if (name.Contains("embed.weight") || name.Contains("embeddings.") || name.Contains(".iou_token.") || name.Contains(".mask_tokens.") || name.Contains("gaussian_matrix"))
            {
                for (int i = 0; i < count; i++) values[i] = Normal();
                return values;
            }
            if (name.EndsWith(".weight", StringComparison.Ordinal))
            {
                // PyTorch's fan-in: in·k·k for a convolution and a linear layer's in; a transposed convolution's weight is
                // [in, out, k, k], so its fan-in is out·k·k.
                fanIn = shape.Length == 2 ? shape[1]
                    : name.Contains("output_upscaling") ? shape[1] * shape[2] * shape[3]
                    : shape[1] * shape[2] * shape[3];
            }
            double bound = 1.0 / Math.Sqrt(fanIn);
            for (int i = 0; i < count; i++) values[i] = (2 * random.NextDouble() - 1) * bound;
            return values;
        });
    }
}
