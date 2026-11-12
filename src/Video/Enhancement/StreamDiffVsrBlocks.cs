using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>Stream-DiffVSR's flow warp.</summary>
internal static class StreamDiffVsrOps
{
    /// <summary>
    /// The release's <c>flow_warp</c> (util/flow_utils.py): image [N, C, H, W] sampled at pixel + flow [N, 2, H, W],
    /// normalized by 2p / (extent − 1) − 1 but sampled with <c>align_corners=False</c>, bilinear, zero padding.
    /// </summary>
    internal static Tensor<T> FlowWarp<T>(IEngine engine, Tensor<T> image, Tensor<T> flow)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = image.Shape[0], h = image.Shape[2], w = image.Shape[3];
        var grid = new Tensor<T>(new[] { n, h, w, 2 });
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double px = x + ops.ToDouble(flow[b, 0, y, x]), py = y + ops.ToDouble(flow[b, 1, y, x]);
                    grid[b, y, x, 0] = ops.FromDouble(2.0 * px / Math.Max(w - 1, 1) - 1.0);
                    grid[b, y, x, 1] = ops.FromDouble(2.0 * py / Math.Max(h - 1, 1) - 1.0);
                }
        return engine.GridSample(image, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, false);
    }
}

/// <summary>
/// The encoder half diffusers' <c>UNet2DConditionModel</c> and <c>ControlNetModel</c> share, in the Stable Diffusion x4
/// upscaler's topology: a 3x3 conv in, the sinusoidal time embedding ([cos, sin] of the first width, flip_sin_to_cos, no
/// shift) through Linear, SiLU and Linear to four times it, a plain <c>DownBlock2D</c> then <c>CrossAttnDownBlock2D</c>s
/// (residual blocks, each followed by a linear-projection transformer at the attention levels, a 3x3 stride-2 conv
/// between levels), and the <c>UNetMidBlock2DCrossAttn</c> (residual, transformer, residual). Heads are counted per
/// block (the legacy <c>attention_head_dim</c>), so each is width / heads wide.
/// </summary>
internal sealed class StreamDiffVsrDownPath<T>
{
    private readonly int _timeChannels;
    private readonly MgldLinear<T> _time1, _time2;
    private readonly MgldConv2d<T> _convIn;
    private readonly List<(MgldResBlock<T>? Res, MgldSpatialTransformer<T>? Attn, MgldConv2d<T>? Down)> _blocks = new();
    private readonly MgldResBlock<T> _mid1, _mid2;
    private readonly MgldSpatialTransformer<T> _midAttn;

    public StreamDiffVsrDownPath(LazyModuleGroup<T> group, int inChannels, IReadOnlyList<int> widths, int layers,
        IReadOnlyList<bool> onlyCrossAttention, int heads, int contextDim)
    {
        var parts = new MgldParts<T>();
        group.Add(parts.Frozen);
        var g = parts.Frozen;
        _timeChannels = widths[0];
        int emb = 4 * widths[0];
        _time1 = g.Add(new MgldLinear<T>(widths[0], emb));
        _time2 = g.Add(new MgldLinear<T>(emb, emb));
        _convIn = g.Add(new MgldConv2d<T>(inChannels, widths[0], 3, padding: 1));
        var residuals = new List<int> { widths[0] };
        int ch = widths[0];
        for (int level = 0; level < widths.Count; level++)
        {
            for (int i = 0; i < layers; i++)
            {
                var res = new MgldResBlock<T>(parts, false, ch, emb, widths[level]);
                ch = widths[level];
                var attn = level > 0 ? new MgldSpatialTransformer<T>(g, ch, heads, ch / heads, contextDim, onlyCrossAttention[level]) : null;
                _blocks.Add((res, attn, null));
                residuals.Add(ch);
            }
            if (level < widths.Count - 1)
            {
                _blocks.Add((null, null, g.Add(new MgldConv2d<T>(ch, ch, 3, stride: 2, padding: 1))));
                residuals.Add(ch);
            }
        }
        _mid1 = new MgldResBlock<T>(parts, false, ch, emb, ch);
        _midAttn = new MgldSpatialTransformer<T>(g, ch, heads, ch / heads, contextDim);
        _mid2 = new MgldResBlock<T>(parts, false, ch, emb, ch);
        ResidualChannels = residuals;
        Group = parts.Frozen;
    }

    /// <summary>The widths of the residuals the path hands the decoder half, in order.</summary>
    public IReadOnlyList<int> ResidualChannels { get; }

    /// <summary>The path's weights, in construction order, so a ControlNet can start as a copy of the UNet's.</summary>
    public LazyModuleGroup<T> Group { get; }

    /// <summary>The time embedding of one timestep per row.</summary>
    public Tensor<T> Embed(IEngine engine, IReadOnlyList<double> timesteps)
        => _time2.Forward(engine.Swish(_time1.Forward(MgldOps.TimestepEmbedding<T>(timesteps, _timeChannels))));

    /// <summary>
    /// The middle block's output and every residual for x [N, C, h, w], with <paramref name="added"/> (the ControlNet's
    /// condition embedding) summed onto the conv in's output.
    /// </summary>
    public (Tensor<T> Hidden, List<Tensor<T>> Residuals) Forward(IEngine engine, Tensor<T> x, Tensor<T> emb, Tensor<T> context,
        Tensor<T>? added = null)
    {
        var h = _convIn.Forward(x);
        if (added is not null) h = engine.TensorAdd(h, added);
        var residuals = new List<Tensor<T>> { h };
        foreach (var (res, attn, down) in _blocks)
        {
            if (down is not null) h = down.Forward(h);
            else if (res is not null)
            {
                h = res.Forward(engine, h, emb);
                if (attn is not null) h = attn.Forward(engine, h, context);
            }
            residuals.Add(h);
        }
        h = _mid2.Forward(engine, _midAttn.Forward(engine, _mid1.Forward(engine, h, emb), context), emb);
        return (h, residuals);
    }
}

/// <summary>
/// The released denoiser, diffusers' <c>UNet2DConditionModel</c> as configured by Stream-DiffVSR (the x4 upscaler's
/// layout without its noise-level class embedding): seven input channels (the four latent channels and the LR frame),
/// widths 256, 512, 512 and 1024, two residual blocks a level, eight heads, cross-attention on 1024-wide text and
/// <c>only_cross_attention</c> at the first three levels. The decoder half takes each level's residuals back in reverse
/// (three residual blocks a level, nearest 2x upsampling and a 3x3 conv between levels), then GroupNorm, SiLU and a 3x3
/// conv out to the v-prediction. A ControlNet's residuals add onto the encoder's.
/// </summary>
internal sealed class StreamDiffVsrUNet<T>
{
    private readonly StreamDiffVsrDownPath<T> _down;
    private readonly List<(MgldResBlock<T> Res, MgldSpatialTransformer<T>? Attn, MgldConv2d<T>? Up)> _up = new();
    private readonly MgldGroupNorm<T> _outNorm;
    private readonly MgldConv2d<T> _convOut;

    public StreamDiffVsrUNet(LazyModuleGroup<T> group, int inChannels, int outChannels, IReadOnlyList<int> widths, int layers,
        IReadOnlyList<bool> onlyCrossAttention, int heads, int contextDim)
    {
        _down = new StreamDiffVsrDownPath<T>(group, inChannels, widths, layers, onlyCrossAttention, heads, contextDim);
        var parts = new MgldParts<T>();
        group.Add(parts.Frozen);
        var g = parts.Frozen;
        int emb = 4 * widths[0], levels = widths.Count;
        var residuals = _down.ResidualChannels.ToList();
        int ch = widths[levels - 1];
        for (int level = levels - 1; level >= 0; level--)
        {
            for (int i = 0; i <= layers; i++)
            {
                int skip = residuals[residuals.Count - 1];
                residuals.RemoveAt(residuals.Count - 1);
                var res = new MgldResBlock<T>(parts, false, ch + skip, emb, widths[level]);
                ch = widths[level];
                var attn = level > 0 ? new MgldSpatialTransformer<T>(g, ch, heads, ch / heads, contextDim, onlyCrossAttention[level]) : null;
                var up = level > 0 && i == layers ? g.Add(new MgldConv2d<T>(ch, ch, 3, padding: 1)) : null;
                _up.Add((res, attn, up));
            }
        }
        _outNorm = g.Add(new MgldGroupNorm<T>(ch, 1e-5));
        _convOut = g.Add(new MgldConv2d<T>(ch, outChannels, 3, padding: 1));
    }

    /// <summary>The encoder half, which a ControlNet copies.</summary>
    public StreamDiffVsrDownPath<T> Down => _down;

    /// <summary>The model output for x [N, 7, h, w] at one timestep per row, with optional ControlNet residuals.</summary>
    public Tensor<T> Forward(IEngine engine, Tensor<T> x, IReadOnlyList<double> timesteps, Tensor<T> context,
        IReadOnlyList<Tensor<T>>? control = null, Tensor<T>? controlMid = null)
    {
        var emb = _down.Embed(engine, timesteps);
        var (h, residuals) = _down.Forward(engine, x, emb, context);
        if (control is not null)
            for (int i = 0; i < residuals.Count; i++) residuals[i] = engine.TensorAdd(residuals[i], control[i]);
        if (controlMid is not null) h = engine.TensorAdd(h, controlMid);
        foreach (var (res, attn, up) in _up)
        {
            var skip = residuals[residuals.Count - 1];
            residuals.RemoveAt(residuals.Count - 1);
            h = res.Forward(engine, engine.TensorConcatenate(new[] { h, skip }, 1), emb);
            if (attn is not null) h = attn.Forward(engine, h, context);
            if (up is not null) h = up.Forward(engine.Upsample(h, 2, 2));
        }
        return _convOut.Forward(engine.Swish(_outNorm.Forward(h)));
    }
}

/// <summary>
/// The Auto-regressive Temporal Guidance module, diffusers' <c>ControlNetModel.from_unet</c> with conditioning widths 64,
/// 128 and 256: the UNet's encoder half (initialized as a copy of it), the condition embedding (a 3x3 conv, then per
/// width a 3x3 conv and a 3x3 stride-2 conv, SiLU after each, and a zero-initialized 3x3 conv out) added after the conv
/// in, and a zero-initialized 1x1 conv on every residual and on the middle block's output, all scaled by the
/// conditioning scale.
/// </summary>
internal sealed class StreamDiffVsrControlNet<T>
{
    private readonly StreamDiffVsrDownPath<T> _down;
    private readonly MgldConv2d<T> _condIn, _condOut;
    private readonly List<MgldConv2d<T>> _condBlocks = new();
    private readonly List<MgldConv2d<T>> _zero = new();
    private readonly MgldConv2d<T> _zeroMid;

    public StreamDiffVsrControlNet(LazyModuleGroup<T> group, int inChannels, IReadOnlyList<int> widths, int layers,
        IReadOnlyList<bool> onlyCrossAttention, int heads, int contextDim, IReadOnlyList<int> conditioningWidths)
    {
        _down = new StreamDiffVsrDownPath<T>(group, inChannels, widths, layers, onlyCrossAttention, heads, contextDim);
        _condIn = group.Add(new MgldConv2d<T>(3, conditioningWidths[0], 3, padding: 1));
        for (int i = 0; i < conditioningWidths.Count - 1; i++)
        {
            _condBlocks.Add(group.Add(new MgldConv2d<T>(conditioningWidths[i], conditioningWidths[i], 3, padding: 1)));
            _condBlocks.Add(group.Add(new MgldConv2d<T>(conditioningWidths[i], conditioningWidths[i + 1], 3, stride: 2, padding: 1)));
        }
        _condOut = group.Add(new MgldConv2d<T>(conditioningWidths[conditioningWidths.Count - 1], widths[0], 3, padding: 1, init: MgldInit.Zero));
        foreach (int channels in _down.ResidualChannels)
            _zero.Add(group.Add(new MgldConv2d<T>(channels, channels, 1, init: MgldInit.Zero)));
        _zeroMid = group.Add(new MgldConv2d<T>(widths[widths.Count - 1], widths[widths.Count - 1], 1, init: MgldInit.Zero));
    }

    /// <summary>The encoder half's weights, which start as the UNet's.</summary>
    public LazyModuleGroup<T> DownGroup => _down.Group;

    /// <summary>The middle block's zero convolution, the layer the adaptive GAN weight is measured at.</summary>
    public Tensor<T> MidKernel => _zeroMid.Kernel;

    /// <summary>The scaled residuals and middle residual for x [N, 7, h, w] and the condition [N, 3, 4h, 4w] in [−1, 1].</summary>
    public (List<Tensor<T>> Residuals, Tensor<T> Mid) Forward(IEngine engine, Tensor<T> x, IReadOnlyList<double> timesteps,
        Tensor<T> context, Tensor<T> condition, double scale)
    {
        var c = engine.Swish(_condIn.Forward(condition));
        foreach (var conv in _condBlocks) c = engine.Swish(conv.Forward(c));
        var emb = _down.Embed(engine, timesteps);
        var (h, residuals) = _down.Forward(engine, x, emb, context, _condOut.Forward(c));
        var ops = MathHelper.GetNumericOperations<T>();
        var outputs = new List<Tensor<T>>();
        for (int i = 0; i < residuals.Count; i++) outputs.Add(engine.TensorMultiplyScalar(_zero[i].Forward(residuals[i]), ops.FromDouble(scale)));
        return (outputs, engine.TensorMultiplyScalar(_zeroMid.Forward(h), ops.FromDouble(scale)));
    }
}

/// <summary>
/// The release's <c>TemporalAutoencoderTinyBlock</c>: three 3x3 convs with ReLU between, the input added back and a ReLU.
/// In the decoder, given the matching encoder block's features of the warped previous output, both maps are 4x4
/// average-pooled, laid end to end and passed through two 1D convs (kernel 3, ReLU between); the two halves are blended
/// by the learned α (0.5 at init), bilinearly resized back and added at 0.1 before the residual.
/// </summary>
internal sealed class StreamDiffVsrTinyBlock<T>
{
    private const int Pool = 4;
    private const double TemporalScale = 0.1;
    private readonly MgldConv2d<T> _conv1, _conv2, _conv3;
    private readonly MgldConv2d<T>? _temporal1, _temporal2;
    private readonly LazyWeight<T>? _alpha;

    public StreamDiffVsrTinyBlock(LazyModuleGroup<T> group, int channels, bool temporal)
    {
        _conv1 = group.Add(new MgldConv2d<T>(channels, channels, 3, padding: 1));
        _conv2 = group.Add(new MgldConv2d<T>(channels, channels, 3, padding: 1));
        _conv3 = group.Add(new MgldConv2d<T>(channels, channels, 3, padding: 1));
        if (!temporal) return;
        _temporal1 = group.Add(new MgldConv2d<T>(channels, channels, 1, 3, 0, 1));
        _temporal2 = group.Add(new MgldConv2d<T>(channels, channels, 1, 3, 0, 1));
        _alpha = LazyWeight<T>.Constant(new[] { 1 }, 0.5);
        group.Add(new MgldWeights<T>(_alpha));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x, Tensor<T>? previous = null)
    {
        var current = _conv3.Forward(engine.ReLU(_conv2.Forward(engine.ReLU(_conv1.Forward(x)))));
        if (previous is not null && _temporal1 is not null && _temporal2 is not null && _alpha is not null)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            int n = current.Shape[0], c = current.Shape[1], h = current.Shape[2], w = current.Shape[3];
            var pooled = engine.AvgPool2D(current, Pool, Pool, 0, false);
            var pooledPrevious = engine.AvgPool2D(previous, Pool, Pool, 0, false);
            int ph = pooled.Shape[2], pw = pooled.Shape[3], p = ph * pw;
            var sequence = engine.TensorConcatenate(new[]
            {
                engine.Reshape(pooled, new[] { n, c, 1, p }), engine.Reshape(pooledPrevious, new[] { n, c, 1, p }),
            }, 3);
            var output = _temporal2.Forward(engine.ReLU(_temporal1.Forward(sequence)));
            var shape = new[] { n, c, ph, pw };
            var alpha = engine.TensorBroadcastTo(engine.Reshape(_alpha.Value, new[] { 1, 1, 1, 1 }), shape);
            var keep = engine.TensorAddScalar(engine.TensorNegate(alpha), ops.One);
            var fused = engine.TensorAdd(engine.TensorMultiply(alpha, engine.Reshape(engine.TensorNarrow(output, 3, 0, p), shape)),
                engine.TensorMultiply(keep, engine.Reshape(engine.TensorNarrow(output, 3, p, p), shape)));
            var resized = engine.Interpolate(fused, new[] { h, w }, InterpolateMode.Bilinear, false);
            current = engine.TensorAdd(current, engine.TensorMultiplyScalar(resized, ops.FromDouble(TemporalScale)));
        }
        return engine.ReLU(engine.TensorAdd(current, x));
    }
}

/// <summary>
/// The temporal autoencoder's encoder (TAESD-x4's <c>EncoderTiny</c>): frames in [−1, 1] brought to [0, 1], a 3x3 conv
/// and one block, then per further level a bias-free 3x3 stride-2 conv and its blocks (1, 3, 3), and a 3x3 conv to the
/// four latent channels. It keeps every block's output, the decoder's temporal features.
/// </summary>
internal sealed class StreamDiffVsrTinyEncoder<T>
{
    private readonly List<(MgldConv2d<T>? Conv, StreamDiffVsrTinyBlock<T>? Block)> _layers = new();

    public StreamDiffVsrTinyEncoder(LazyModuleGroup<T> group, int channels, IReadOnlyList<int> blocks, int latentChannels)
    {
        for (int level = 0; level < blocks.Count; level++)
        {
            _layers.Add((group.Add(level == 0
                ? new MgldConv2d<T>(3, channels, 3, padding: 1)
                : new MgldConv2d<T>(channels, channels, 3, stride: 2, padding: 1, bias: false)), null));
            for (int i = 0; i < blocks[level]; i++) _layers.Add((null, new StreamDiffVsrTinyBlock<T>(group, channels, temporal: false)));
        }
        _layers.Add((group.Add(new MgldConv2d<T>(channels, latentChannels, 3, padding: 1)), null));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> frames, List<Tensor<T>>? features = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var h = engine.TensorMultiplyScalar(engine.TensorAddScalar(frames, ops.One), ops.FromDouble(0.5));
        foreach (var (conv, block) in _layers)
        {
            if (conv is not null) h = conv.Forward(h);
            else if (block is not null)
            {
                h = block.Forward(engine, h);
                features?.Add(h);
            }
        }
        return h;
    }
}

/// <summary>
/// The temporal autoencoder's decoder (<c>TemporalDecoderTiny</c>): latents soft-clamped by 3·tanh(z / 3), a 3x3 conv and
/// ReLU, then per level its temporal blocks (3, 3, 1), nearest 2x upsampling and a bias-free 3x3 conv between levels and
/// a biased 3x3 conv to RGB at the end, brought from [0, 1] to [−1, 1]. Block k reads the encoder's features of the
/// warped previous output in reverse order.
/// </summary>
internal sealed class StreamDiffVsrTemporalDecoder<T>
{
    private readonly MgldConv2d<T> _convIn;
    private readonly List<(StreamDiffVsrTinyBlock<T>? Block, MgldConv2d<T>? Conv, bool Upsample)> _layers = new();

    public StreamDiffVsrTemporalDecoder(LazyModuleGroup<T> group, int channels, IReadOnlyList<int> blocks, int latentChannels)
    {
        _convIn = group.Add(new MgldConv2d<T>(latentChannels, channels, 3, padding: 1));
        for (int level = 0; level < blocks.Count; level++)
        {
            bool last = level == blocks.Count - 1;
            for (int i = 0; i < blocks[level]; i++) _layers.Add((new StreamDiffVsrTinyBlock<T>(group, channels, temporal: true), null, false));
            _layers.Add((null, group.Add(new MgldConv2d<T>(channels, last ? 3 : channels, 3, padding: 1, bias: last)), !last));
        }
    }

    /// <summary>Frames in [−1, 1] from latents, with the encoder's features of the warped previous output when there is one.</summary>
    public Tensor<T> Forward(IEngine engine, Tensor<T> latents, IReadOnlyList<Tensor<T>>? previous = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var h = engine.TensorMultiplyScalar(engine.Tanh(engine.TensorMultiplyScalar(latents, ops.FromDouble(1.0 / 3.0))), ops.FromDouble(3.0));
        h = engine.ReLU(_convIn.Forward(h));
        int k = 0;
        foreach (var (block, conv, upsample) in _layers)
        {
            if (block is not null)
            {
                h = block.Forward(engine, h, previous?[previous.Count - 1 - k]);
                k++;
            }
            else if (conv is not null)
            {
                if (upsample) h = engine.Upsample(h, 2, 2);
                h = conv.Forward(h);
            }
        }
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(h, ops.FromDouble(2.0)), ops.FromDouble(-1.0));
    }
}

/// <summary>
/// LPIPS (Zhang et al. 2018) on AlexNet, torchmetrics' default: the scaling layer, the five ReLU outputs of AlexNet's
/// features (11x11 stride-4 conv to 64; 3x3 stride-2 max pool and 5x5 conv to 192; max pool and 3x3 convs to 384, 256
/// and 256), each unit-normalized over channels, squared differences weighted per channel, averaged over space and
/// summed. The linear weights come from the released checkpoint; untrained they are one, LPIPS's uncalibrated baseline.
/// </summary>
internal sealed class StreamDiffVsrAlexLpips<T> : LazyWeightModule<T>
{
    private static readonly int[] Widths = { 64, 192, 384, 256, 256 };
    private static readonly double[] Shift = { -0.030, -0.088, -0.188 };
    private static readonly double[] Scale = { 0.458, 0.448, 0.450 };
    private readonly MgldConv2d<T>[] _convs;
    private readonly LazyWeight<T>[] _lin;

    public StreamDiffVsrAlexLpips()
    {
        _convs = new[]
        {
            new MgldConv2d<T>(3, 64, 11, stride: 4, padding: 2), new MgldConv2d<T>(64, 192, 5, padding: 2),
            new MgldConv2d<T>(192, 384, 3, padding: 1), new MgldConv2d<T>(384, 256, 3, padding: 1),
            new MgldConv2d<T>(256, 256, 3, padding: 1),
        };
        _lin = Widths.Select(w => LazyWeight<T>.Constant(new[] { w }, 1.0)).ToArray();
    }

    private List<Tensor<T>> Features(Tensor<T> images)
    {
        var channels = new Tensor<T>[3];
        for (int c = 0; c < 3; c++)
            channels[c] = Engine.TensorMultiplyScalar(Engine.TensorAddScalar(Engine.TensorNarrow(images, 1, c, 1), NumOps.FromDouble(-Shift[c])),
                NumOps.FromDouble(1.0 / Scale[c]));
        var x = Engine.TensorConcatenate(channels, 1);
        var features = new List<Tensor<T>>();
        for (int i = 0; i < _convs.Length; i++)
        {
            if (i == 1 || i == 2) x = Engine.MaxPool2D(x, 3, 2);
            x = Engine.ReLU(_convs[i].Forward(x));
            features.Add(x);
        }
        return features;
    }

    private static Tensor<T> Unit(Tensor<T> x)
    {
        var norm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(x), new[] { 1 }, keepDims: true));
        return Engine.TensorDivide(x, Engine.TensorBroadcastTo(Engine.TensorAddScalar(norm, NumOps.FromDouble(1e-10)), x.Shape.ToArray()));
    }

    /// <summary>The LPIPS distance averaged over images [N, 3, H, W] in [−1, 1].</summary>
    public Tensor<T> Loss(Tensor<T> output, Tensor<T> target)
    {
        List<Tensor<T>> targets;
        using (new NoGradScope<T>()) targets = Features(target);
        var outputs = Features(output);
        Tensor<T>? total = null;
        for (int k = 0; k < outputs.Count; k++)
        {
            var diff = Engine.TensorSquare(Engine.TensorSubtract(Unit(outputs[k]), Unit(targets[k])));
            var lin = Engine.TensorBroadcastTo(Engine.Reshape(_lin[k].Value, new[] { 1, diff.Shape[1], 1, 1 }), diff.Shape.ToArray());
            var term = Engine.ReduceMean(Engine.ReduceSum(Engine.TensorMultiply(diff, lin), new[] { 1 }, keepDims: false), new[] { 0, 1, 2 }, false);
            total = total is null ? term : Engine.TensorAdd(total, term);
        }
        return total ?? throw new InvalidOperationException("LPIPS has no stages.");
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => _lin;

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}
