using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// BasicSR's <c>ResidualDenseBlock</c>: five 3x3 convolutions, each reading the input and every earlier output
/// (growth 32, LeakyReLU 0.2 after the first four), the fifth scaled by 0.2 and added to the input; Kaiming-normal at
/// scale 0.1.
/// </summary>
internal sealed class MgldResidualDenseBlock<T>
{
    private const int Growth = 32;
    private readonly MgldConv2d<T>[] _convs = new MgldConv2d<T>[5];

    public MgldResidualDenseBlock(LazyModuleGroup<T> group, int channels)
    {
        for (int i = 0; i < 5; i++)
            _convs[i] = group.Add(new MgldConv2d<T>(channels + i * Growth, i == 4 ? channels : Growth, 3, padding: 1, init: MgldInit.ScaledKaiming));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var features = new List<Tensor<T>> { x };
        for (int i = 0; i < 4; i++)
            features.Add(engine.LeakyReLU(_convs[i].Forward(engine.TensorConcatenate(features.ToArray(), 1)), ops.FromDouble(0.2)));
        var last = _convs[4].Forward(engine.TensorConcatenate(features.ToArray(), 1));
        return engine.TensorAdd(x, engine.TensorMultiplyScalar(last, ops.FromDouble(0.2)));
    }
}

/// <summary>
/// The controllable feature wrapping module (<c>Fuse_sft_block_ResidualDenseBlock</c>, StableSR's CFW): the encoder's
/// and decoder's features concatenated through a residual block, two residual dense blocks and a residual block, added
/// to the decoder's features at weight w.
/// </summary>
internal sealed class MgldFuseBlock<T>
{
    private readonly KlResnetBlock<T> _first, _last;
    private readonly MgldResidualDenseBlock<T>[] _dense;

    public MgldFuseBlock(LazyModuleGroup<T> group, int channels, int denseBlocks)
    {
        _first = group.Add(new KlResnetBlock<T>(2 * channels, channels, 32));
        _dense = Enumerable.Range(0, denseBlocks).Select(_ => new MgldResidualDenseBlock<T>(group, channels)).ToArray();
        _last = group.Add(new KlResnetBlock<T>(channels, channels, 32));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> encoder, Tensor<T> decoder, double weight)
    {
        var h = _first.Forward(engine.TensorConcatenate(new[] { encoder, decoder }, 1));
        foreach (var block in _dense) h = block.Forward(engine, h);
        h = _last.Forward(h);
        return engine.TensorAdd(decoder, engine.TensorMultiplyScalar(h, MathHelper.GetNumericOperations<T>().FromDouble(weight)));
    }
}

/// <summary>
/// MGLD-VSR's video decoder (<c>VideoDecoder_Mix</c> behind <c>post_quant_conv</c>): SD's KL decoder (128 wide,
/// multipliers 1, 2, 4, 4, three residual blocks a level) with a temporal convolution after the middle block's first
/// residual block and after every up-path residual block, and a CFW module at the two middle resolutions fusing the
/// encoder's features of the LQ frames. SD's weights are frozen; the CFW modules and temporal convolutions train.
/// </summary>
internal sealed class MgldVideoDecoder<T>
{
    private readonly MgldConv2d<T> _postQuant, _convIn, _convOut;
    private readonly KlResnetBlock<T> _mid1, _mid2;
    private readonly KlAttention<T> _midAttn;
    private readonly MgldTemporalConv<T> _midTemporal;
    private readonly KlGroupNorm<T> _normOut;
    private readonly List<(List<(KlResnetBlock<T> Res, MgldTemporalConv<T> Temporal)> Blocks, MgldFuseBlock<T>? Fuse, MgldConv2d<T>? Up)> _levels = new();

    public MgldVideoDecoder(MgldParts<T> parts, int latentChannels, int channels, IReadOnlyList<int> multipliers, int resBlocks, int fuseBlocks)
    {
        var frozen = parts.Frozen;
        int levels = multipliers.Count, ch = channels * multipliers[levels - 1];
        _postQuant = frozen.Add(new MgldConv2d<T>(latentChannels, latentChannels, 1));
        _convIn = frozen.Add(new MgldConv2d<T>(latentChannels, ch, 3, padding: 1));
        _mid1 = frozen.Add(new KlResnetBlock<T>(ch, ch, 32));
        _midTemporal = new MgldTemporalConv<T>(parts, ch);
        _midAttn = frozen.Add(new KlAttention<T>(ch, 32));
        _mid2 = frozen.Add(new KlResnetBlock<T>(ch, ch, 32));
        for (int level = levels - 1; level >= 0; level--)
        {
            int width = channels * multipliers[level];
            var blocks = new List<(KlResnetBlock<T>, MgldTemporalConv<T>)>();
            for (int i = 0; i <= resBlocks; i++)
            {
                blocks.Add((frozen.Add(new KlResnetBlock<T>(ch, width, 32)), new MgldTemporalConv<T>(parts, width)));
                ch = width;
            }
            var fuse = level != levels - 1 && level != 0 ? new MgldFuseBlock<T>(parts.Trained, width, fuseBlocks) : null;
            var up = level != 0 ? frozen.Add(new MgldConv2d<T>(ch, ch, 3, padding: 1)) : null;
            _levels.Add((blocks, fuse, up));
        }
        _normOut = frozen.Add(new KlGroupNorm<T>(32, ch));
        _convOut = frozen.Add(new MgldConv2d<T>(ch, 3, 3, padding: 1));
    }

    /// <summary>The last convolution's kernel: the layer the adaptive GAN weight compares gradients at.</summary>
    public Tensor<T> LastLayer => _convOut.Kernel;

    /// <summary>
    /// Frames [B·T, 3, H, W] in [−1, 1] from unscaled latents [B·T, 4, H/8, W/8] and the encoder's features of the LQ
    /// frames at levels 1 and 2 (H/2 and H/4), fused at weight w.
    /// </summary>
    public Tensor<T> Forward(IEngine engine, Tensor<T> z, IReadOnlyList<Tensor<T>> encoderFeatures, double fusionWeight, int frames)
    {
        var h = _convIn.Forward(_postQuant.Forward(z));
        h = _midTemporal.Forward(engine, _mid1.Forward(h), frames);
        h = _mid2.Forward(_midAttn.Forward(h));
        int level = _levels.Count - 1;
        foreach (var (blocks, fuse, up) in _levels)
        {
            foreach (var (res, temporal) in blocks) h = temporal.Forward(engine, res.Forward(h), frames);
            if (fuse is not null) h = fuse.Forward(engine, encoderFeatures[level], h, fusionWeight);
            if (up is not null) h = up.Forward(engine.Upsample(h, 2, 2));
            level--;
        }
        return _convOut.Forward(engine.Swish(_normOut.Forward(h)));
    }
}

/// <summary>
/// taming's <c>NLayerDiscriminator</c> (PatchGAN, Isola et al. 2017) with three layers: 4x4 convolutions 3-64-128-256-512-1
/// (stride 2 for the first three, bias-free and batch-normed between, LeakyReLU 0.2), N(0, 0.02) weights and N(1, 0.02)
/// batch-norm scales. It always runs in training mode, normalizing with the batch's statistics.
/// </summary>
internal sealed class MgldDiscriminator<T> : LazyWeightModule<T>
{
    private readonly MgldConv2d<T>[] _convs;
    private readonly LazyWeight<T>[] _gamma, _beta;

    public MgldDiscriminator(int inChannels = 3, int width = 64, int layers = 3)
    {
        var convs = new List<MgldConv2d<T>> { new(inChannels, width, 4, stride: 2, padding: 1, init: MgldInit.Normal002) };
        var gamma = new List<LazyWeight<T>>();
        var beta = new List<LazyWeight<T>>();
        int previous = width;
        for (int n = 1; n <= layers; n++)
        {
            int next = width * Math.Min(1 << n, 8);
            convs.Add(new MgldConv2d<T>(previous, next, 4, stride: n < layers ? 2 : 1, padding: 1, init: MgldInit.Normal002, bias: false));
            var ops = MathHelper.GetNumericOperations<T>();
            gamma.Add(new LazyWeight<T>(new[] { next }, (tensor, random) =>
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                    tensor[i] = ops.FromDouble(1.0 + 0.02 * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                }
            }));
            beta.Add(LazyWeight<T>.Constant(new[] { next }, 0.0));
            previous = next;
        }
        convs.Add(new MgldConv2d<T>(previous, 1, 4, stride: 1, padding: 1, init: MgldInit.Normal002));
        _convs = convs.ToArray();
        _gamma = gamma.ToArray();
        _beta = beta.ToArray();
    }

    // BatchNorm2d in training mode: the batch's mean and biased variance over N, H, W (epsilon 1e-5).
    private Tensor<T> BatchNorm(Tensor<T> x, int index)
    {
        var shape = x.Shape.ToArray();
        var axes = new[] { 0, 2, 3 };
        var mean = Engine.TensorBroadcastTo(Engine.ReduceMean(x, axes, true), shape);
        var centred = Engine.TensorSubtract(x, mean);
        var variance = Engine.TensorBroadcastTo(Engine.ReduceMean(Engine.TensorSquare(centred), axes, true), shape);
        var normalized = Engine.TensorDivide(centred, Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-5))));
        var scaled = Engine.TensorMultiply(normalized, Engine.TensorBroadcastTo(Engine.Reshape(_gamma[index].Value, new[] { 1, shape[1], 1, 1 }), shape));
        return MgldOps.ChannelBias(Engine, scaled, _beta[index].Value);
    }

    /// <summary>Patch logits of frames [N, 3, H, W].</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        var slope = NumOps.FromDouble(0.2);
        var h = Engine.LeakyReLU(_convs[0].Forward(x), slope);
        for (int i = 1; i < _convs.Length - 1; i++) h = Engine.LeakyReLU(BatchNorm(_convs[i].Forward(h), i - 1), slope);
        return _convs[_convs.Length - 1].Forward(h);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        for (int i = 0; i < _gamma.Length; i++)
        {
            yield return _gamma[i];
            yield return _beta[i];
        }
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}

/// <summary>Which normalization a RAFT encoder uses.</summary>
internal enum RaftNorm
{
    /// <summary>InstanceNorm2d without affine parameters (the feature encoder).</summary>
    Instance,

    /// <summary>BatchNorm2d in evaluation mode, with running statistics (the context encoder).</summary>
    Batch,
}

/// <summary>A RAFT normalization: per-sample instance statistics, or a frozen batch norm's running ones.</summary>
internal sealed class RaftNormalization<T> : LazyWeightModule<T>
{
    private readonly RaftNorm _kind;
    private readonly LazyWeight<T>? _gamma, _beta, _mean, _variance;

    public RaftNormalization(RaftNorm kind, int channels)
    {
        _kind = kind;
        if (kind != RaftNorm.Batch) return;
        _gamma = LazyWeight<T>.Constant(new[] { channels }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { channels }, 0.0);
        _mean = LazyWeight<T>.Constant(new[] { channels }, 0.0);
        _variance = LazyWeight<T>.Constant(new[] { channels }, 1.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        var epsilon = NumOps.FromDouble(1e-5);
        if (_gamma is null || _beta is null || _mean is null || _variance is null)
        {
            var axes = new[] { 2, 3 };
            var centred = Engine.TensorSubtract(x, Engine.TensorBroadcastTo(Engine.ReduceMean(x, axes, true), shape));
            var variance = Engine.TensorBroadcastTo(Engine.ReduceMean(Engine.TensorSquare(centred), axes, true), shape);
            return Engine.TensorDivide(centred, Engine.TensorSqrt(Engine.TensorAddScalar(variance, epsilon)));
        }
        Tensor<T> Channel(Tensor<T> v) => Engine.TensorBroadcastTo(Engine.Reshape(v, new[] { 1, shape[1], 1, 1 }), shape);
        var normalized = Engine.TensorDivide(Engine.TensorSubtract(x, Channel(_mean.Value)),
            Engine.TensorSqrt(Engine.TensorAddScalar(Channel(_variance.Value), epsilon)));
        return Engine.TensorAdd(Engine.TensorMultiply(normalized, Channel(_gamma.Value)), Channel(_beta.Value));
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        if (_gamma is null || _beta is null || _mean is null || _variance is null) yield break;
        yield return _gamma;
        yield return _beta;
        yield return _mean;
        yield return _variance;
    }
}

/// <summary>RAFT's <c>ResidualBlock</c>: conv-norm-ReLU twice, a strided 1x1 conv and norm on the skip when downsampling, then ReLU.</summary>
internal sealed class RaftResidualBlock<T>
{
    private readonly MgldConv2d<T> _conv1, _conv2;
    private readonly RaftNormalization<T> _norm1, _norm2;
    private readonly MgldConv2d<T>? _down;
    private readonly RaftNormalization<T>? _norm3;

    public RaftResidualBlock(LazyModuleGroup<T> group, int inPlanes, int planes, RaftNorm norm, int stride)
    {
        _conv1 = group.Add(new MgldConv2d<T>(inPlanes, planes, 3, stride: stride, padding: 1));
        _conv2 = group.Add(new MgldConv2d<T>(planes, planes, 3, padding: 1));
        _norm1 = group.Add(new RaftNormalization<T>(norm, planes));
        _norm2 = group.Add(new RaftNormalization<T>(norm, planes));
        if (stride == 1) return;
        _down = group.Add(new MgldConv2d<T>(inPlanes, planes, 1, stride: stride));
        _norm3 = group.Add(new RaftNormalization<T>(norm, planes));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x)
    {
        var y = engine.ReLU(_norm1.Forward(_conv1.Forward(x)));
        y = engine.ReLU(_norm2.Forward(_conv2.Forward(y)));
        var skip = _down is null || _norm3 is null ? x : _norm3.Forward(_down.Forward(x));
        return engine.ReLU(engine.TensorAdd(skip, y));
    }
}

/// <summary>RAFT's <c>BasicEncoder</c>: a 7x7 stride-2 conv to 64, residual stages of 64, 96 (stride 2) and 128 (stride 2), and a 1x1 conv out (1/8 resolution).</summary>
internal sealed class RaftEncoder<T>
{
    private readonly MgldConv2d<T> _conv1, _conv2;
    private readonly RaftNormalization<T> _norm1;
    private readonly RaftResidualBlock<T>[] _blocks;

    public RaftEncoder(LazyModuleGroup<T> group, int outDim, RaftNorm norm)
    {
        _conv1 = group.Add(new MgldConv2d<T>(3, 64, 7, stride: 2, padding: 3));
        _norm1 = group.Add(new RaftNormalization<T>(norm, 64));
        _blocks = new[]
        {
            new RaftResidualBlock<T>(group, 64, 64, norm, 1), new RaftResidualBlock<T>(group, 64, 64, norm, 1),
            new RaftResidualBlock<T>(group, 64, 96, norm, 2), new RaftResidualBlock<T>(group, 96, 96, norm, 1),
            new RaftResidualBlock<T>(group, 96, 128, norm, 2), new RaftResidualBlock<T>(group, 128, 128, norm, 1),
        };
        _conv2 = group.Add(new MgldConv2d<T>(128, outDim, 1));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x)
    {
        var h = engine.ReLU(_norm1.Forward(_conv1.Forward(x)));
        foreach (var block in _blocks) h = block.Forward(engine, h);
        return _conv2.Forward(h);
    }
}

/// <summary>
/// RAFT (Teed &amp; Deng 2020) as MGLD-VSR's <c>RAFT_SR</c> runs it ("normal": a 256-wide instance-normed feature
/// encoder, a batch-normed context encoder split into a 128-wide hidden state and context, a four-level correlation
/// pyramid looked up at radius 4, the separable ConvGRU update block, and convex 8x upsampling), on frames in [0, 1]
/// without further normalization, ten iterations. Frozen: MGLD-VSR loads RAFT-things.
/// </summary>
internal sealed class MgldRaft<T>
{
    private const int Hidden = 128, Context = 128, Levels = 4, Radius = 4;
    private readonly RaftEncoder<T> _features, _context;
    private readonly MgldConv2d<T> _convc1, _convc2, _convf1, _convf2, _conv, _flow1, _flow2, _mask1, _mask2;
    private readonly MgldConv2d<T>[] _gru;

    public MgldRaft(LazyModuleGroup<T> group)
    {
        _features = new RaftEncoder<T>(group, 256, RaftNorm.Instance);
        _context = new RaftEncoder<T>(group, Hidden + Context, RaftNorm.Batch);
        int corrPlanes = Levels * (2 * Radius + 1) * (2 * Radius + 1);
        _convc1 = group.Add(new MgldConv2d<T>(corrPlanes, 256, 1));
        _convc2 = group.Add(new MgldConv2d<T>(256, 192, 3, padding: 1));
        _convf1 = group.Add(new MgldConv2d<T>(2, 128, 7, padding: 3));
        _convf2 = group.Add(new MgldConv2d<T>(128, 64, 3, padding: 1));
        _conv = group.Add(new MgldConv2d<T>(64 + 192, 128 - 2, 3, padding: 1));
        int gruIn = Hidden + Context + Hidden;
        _gru = new[]
        {
            group.Add(new MgldConv2d<T>(gruIn, Hidden, 1, 5, 0, 2)), group.Add(new MgldConv2d<T>(gruIn, Hidden, 1, 5, 0, 2)),
            group.Add(new MgldConv2d<T>(gruIn, Hidden, 1, 5, 0, 2)), group.Add(new MgldConv2d<T>(gruIn, Hidden, 5, 1, 2, 0)),
            group.Add(new MgldConv2d<T>(gruIn, Hidden, 5, 1, 2, 0)), group.Add(new MgldConv2d<T>(gruIn, Hidden, 5, 1, 2, 0)),
        };
        _flow1 = group.Add(new MgldConv2d<T>(Hidden, 256, 3, padding: 1));
        _flow2 = group.Add(new MgldConv2d<T>(256, 2, 3, padding: 1));
        _mask1 = group.Add(new MgldConv2d<T>(128, 256, 3, padding: 1));
        _mask2 = group.Add(new MgldConv2d<T>(256, 64 * 9, 1));
    }

    // One ConvGRU half: z, r and q from three convolutions of [h, x].
    private Tensor<T> GruStep(IEngine engine, Tensor<T> h, Tensor<T> x, int first)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var hx = engine.TensorConcatenate(new[] { h, x }, 1);
        var z = engine.Sigmoid(_gru[first].Forward(hx));
        var r = engine.Sigmoid(_gru[first + 1].Forward(hx));
        var q = engine.Tanh(_gru[first + 2].Forward(engine.TensorConcatenate(new[] { engine.TensorMultiply(r, h), x }, 1)));
        var keep = engine.TensorAddScalar(engine.TensorNegate(z), ops.One);
        return engine.TensorAdd(engine.TensorMultiply(keep, h), engine.TensorMultiply(z, q));
    }

    // CorrBlock lookup: each pyramid level sampled on a (2r + 1)² grid around coords / 2^i (bilinear, zeros, align_corners).
    private Tensor<T> Lookup(IEngine engine, IReadOnlyList<Tensor<T>> pyramid, Tensor<T> coords)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = coords.Shape[0], h = coords.Shape[2], w = coords.Shape[3], side = 2 * Radius + 1;
        var output = new List<Tensor<T>>();
        for (int level = 0; level < pyramid.Count; level++)
        {
            var corr = pyramid[level];
            int ch = corr.Shape[2], cw = corr.Shape[3];
            var grid = new Tensor<T>(new[] { n * h * w, side, side, 2 });
            for (int b = 0; b < n; b++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int row = (b * h + y) * w + x;
                        double cx = ops.ToDouble(coords[b, 0, y, x]) / (1 << level), cy = ops.ToDouble(coords[b, 1, y, x]) / (1 << level);
                        // torch.meshgrid(dy, dx) stacked: the last axis holds (dy, dx) and is read as (x, y), so the first
                        // grid axis offsets x and the second offsets y, as the release does.
                        for (int i = 0; i < side; i++)
                            for (int j = 0; j < side; j++)
                            {
                                double sx = cx + (i - Radius), sy = cy + (j - Radius);
                                grid[row, i, j, 0] = ops.FromDouble(cw > 1 ? 2 * sx / (cw - 1) - 1 : 0);
                                grid[row, i, j, 1] = ops.FromDouble(ch > 1 ? 2 * sy / (ch - 1) - 1 : 0);
                            }
                    }
            var sampled = engine.GridSample(corr, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, true);
            output.Add(engine.TensorPermute(engine.Reshape(sampled, new[] { n, h, w, side * side }), new[] { 0, 3, 1, 2 }));
        }
        return engine.TensorConcatenate(output.ToArray(), 1);
    }

    // Convex upsampling: each fine pixel a softmax-weighted mix of the 3x3 coarse neighbourhood of 8·flow.
    private static Tensor<T> Upsample(IEngine engine, Tensor<T> flow, Tensor<T> mask)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = flow.Shape[0], h = flow.Shape[2], w = flow.Shape[3];
        var weights = engine.TensorSoftmax(engine.Reshape(mask, new[] { n, 1, 9, 8, 8, h, w }), 2);
        var padded = engine.PadNd(engine.TensorMultiplyScalar(flow, ops.FromDouble(8.0)), new[] { 1, 1, 1, 1 }, PadMode.Constant, ops.Zero);
        var shifts = new List<Tensor<T>>();
        for (int dy = 0; dy < 3; dy++)
            for (int dx = 0; dx < 3; dx++)
                shifts.Add(engine.Reshape(engine.TensorNarrow(engine.TensorNarrow(padded, 2, dy, h), 3, dx, w), new[] { n, 2, 1, 1, 1, h, w }));
        var neighbours = engine.TensorBroadcastTo(engine.TensorConcatenate(shifts.ToArray(), 2), new[] { n, 2, 9, 8, 8, h, w });
        var mixed = engine.ReduceSum(engine.TensorMultiply(engine.TensorBroadcastTo(weights, new[] { n, 2, 9, 8, 8, h, w }), neighbours), new[] { 2 }, keepDims: false);
        return engine.Reshape(engine.TensorPermute(mixed, new[] { 0, 1, 4, 2, 5, 3 }), new[] { n, 2, 8 * h, 8 * w });
    }

    /// <summary>
    /// Flow [N, 2, H, W] from <paramref name="reference"/> to <paramref name="support"/> ([N, 3, H, W] in [0, 1]). The
    /// frames are replicate-padded to a multiple of 8 (InputPadder), and to at least 64 so the four-level pyramid exists.
    /// </summary>
    public Tensor<T> Flow(IEngine engine, Tensor<T> reference, Tensor<T> support, int iterations = 10)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int h = reference.Shape[2], w = reference.Shape[3];
        int ph = Math.Max(64, (h + 7) / 8 * 8) - h, pw = Math.Max(64, (w + 7) / 8 * 8) - w;
        var pads = new[] { pw / 2, pw - pw / 2, ph / 2, ph - ph / 2 };
        Tensor<T> Pad(Tensor<T> x) => ph + pw == 0 ? x : engine.PadNd(x, pads, PadMode.Replicate, ops.Zero);
        reference = Pad(reference);
        support = Pad(support);
        int n = reference.Shape[0], H = reference.Shape[2], W = reference.Shape[3], h8 = H / 8, w8 = W / 8;
        var f1 = _features.Forward(engine, reference);
        var f2 = _features.Forward(engine, support);
        int dim = f1.Shape[1];
        var corr = engine.TensorMultiplyScalar(engine.TensorBatchMatMul(
            engine.TensorPermute(engine.Reshape(f1, new[] { n, dim, h8 * w8 }), new[] { 0, 2, 1 }), engine.Reshape(f2, new[] { n, dim, h8 * w8 })),
            ops.FromDouble(1.0 / Math.Sqrt(dim)));
        var pyramid = new List<Tensor<T>> { engine.Reshape(corr, new[] { n * h8 * w8, 1, h8, w8 }) };
        for (int i = 1; i < Levels; i++) pyramid.Add(engine.AvgPool2D(pyramid[i - 1], 2, 2, padding: 0));
        var context = _context.Forward(engine, reference);
        var net = engine.Tanh(engine.TensorNarrow(context, 1, 0, Hidden));
        var inp = engine.ReLU(engine.TensorNarrow(context, 1, Hidden, Context));
        var coords0 = new Tensor<T>(new[] { n, 2, h8, w8 });
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h8; y++)
                for (int x = 0; x < w8; x++)
                {
                    coords0[b, 0, y, x] = ops.FromDouble(x);
                    coords0[b, 1, y, x] = ops.FromDouble(y);
                }
        var coords1 = coords0;
        Tensor<T> flowUp = new Tensor<T>(new[] { n, 2, H, W });
        for (int it = 0; it < iterations; it++)
        {
            var lookup = Lookup(engine, pyramid, coords1);
            var flow = engine.TensorSubtract(coords1, coords0);
            var cor = engine.ReLU(_convc2.Forward(engine.ReLU(_convc1.Forward(lookup))));
            var flo = engine.ReLU(_convf2.Forward(engine.ReLU(_convf1.Forward(flow))));
            var motion = engine.TensorConcatenate(new[] { engine.ReLU(_conv.Forward(engine.TensorConcatenate(new[] { cor, flo }, 1))), flow }, 1);
            var x = engine.TensorConcatenate(new[] { inp, motion }, 1);
            net = GruStep(engine, GruStep(engine, net, x, 0), x, 3);
            var delta = _flow2.Forward(engine.ReLU(_flow1.Forward(net)));
            var mask = engine.TensorMultiplyScalar(_mask2.Forward(engine.ReLU(_mask1.Forward(net))), ops.FromDouble(0.25));
            coords1 = engine.TensorAdd(coords1, delta);
            flowUp = Upsample(engine, engine.TensorSubtract(coords1, coords0), mask);
        }
        return engine.TensorNarrow(engine.TensorNarrow(flowUp, 2, pads[2], h), 3, pads[0], w);
    }
}

/// <summary>MGLD-VSR's flow utilities: occlusion masks, flow resizing, the latent motion guidance and the decoder's temporal losses.</summary>
internal static class MgldFlow
{
    private static Tensor<T> Norm<T>(IEngine engine, Tensor<T> flow)
        => engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(flow), new[] { 1 }, keepDims: true));

    /// <summary>
    /// <c>forward_backward_consistency_check</c> (UnFlow's α 0.01, β 0.5): a pixel is occluded where the flow and the
    /// warped reverse flow fail to cancel by more than α·(|f| + |b|) + β. Returns 0/1 masks [N, 1, H, W].
    /// </summary>
    internal static (Tensor<T> Forward, Tensor<T> Backward) Occlusion<T>(IEngine engine, Tensor<T> forward, Tensor<T> backward)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var magnitude = engine.TensorAdd(Norm(engine, forward), Norm(engine, backward));
        var threshold = engine.TensorAddScalar(engine.TensorMultiplyScalar(magnitude, ops.FromDouble(0.01)), ops.FromDouble(0.5));
        var diffForward = Norm(engine, engine.TensorAdd(forward, BasicVsrBlocks.FlowWarp(engine, backward, forward, GridSamplePadding.Zeros)));
        var diffBackward = Norm(engine, engine.TensorAdd(backward, BasicVsrBlocks.FlowWarp(engine, forward, backward, GridSamplePadding.Zeros)));
        Tensor<T> Above(Tensor<T> diff)
        {
            var mask = new Tensor<T>(diff.Shape.ToArray());
            for (int i = 0; i < mask.Length; i++) mask[i] = ops.ToDouble(diff[i]) > ops.ToDouble(threshold[i]) ? ops.One : ops.Zero;
            return mask;
        }
        return (Above(diffForward), Above(diffBackward));
    }

    /// <summary>BasicSR's <c>resize_flow</c> by shape: bilinear resizing with the vectors scaled by the size ratio.</summary>
    internal static Tensor<T> Resize<T>(IEngine engine, Tensor<T> flow, int height, int width)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int h = flow.Shape[2], w = flow.Shape[3];
        var resized = engine.Interpolate(flow.IsContiguous ? flow : flow.Contiguous(), new[] { height, width }, InterpolateMode.Bilinear, false);
        var dx = engine.TensorMultiplyScalar(engine.TensorNarrow(resized, 1, 0, 1), ops.FromDouble((double)width / w));
        var dy = engine.TensorMultiplyScalar(engine.TensorNarrow(resized, 1, 1, 1), ops.FromDouble((double)height / h));
        return engine.TensorConcatenate(new[] { dx, dy }, 1);
    }

    /// <summary>The masked L1 of a warped neighbour against a frame, mean over every element (<c>F.l1_loss</c>).</summary>
    internal static Tensor<T> MaskedL1<T>(IEngine engine, Tensor<T> warped, Tensor<T> frame, Tensor<T> visible, Tensor<T>? weight = null)
    {
        var shape = frame.Shape.ToArray();
        var keep = engine.TensorBroadcastTo(visible, shape);
        if (weight is not null) keep = engine.TensorMultiply(keep, weight);
        var diff = engine.TensorAbs(engine.TensorSubtract(engine.TensorMultiply(keep, warped), engine.TensorMultiply(keep, frame)));
        return engine.ReduceMean(diff, Enumerable.Range(0, diff.Rank).ToArray(), false);
    }

    /// <summary>
    /// The paper's warping loss over a clip of T frames [T, C, H, W] (Eq. 4 and the release's
    /// <c>compute_temporal_condition_v4</c> / <c>swc_loss</c>): for each neighbour pair, frame i+1 warped to frame i by
    /// the flow i→i+1 against frame i outside its occlusion, and frame i warped to frame i+1 by the flow i+1→i against
    /// frame i+1, optionally weighted. The release's loops compare each frame with the neighbour warped one step
    /// earlier (and the first term with zeros); the pairing here is the one the equation states.
    /// </summary>
    internal static Tensor<T> WarpingLoss<T>(IEngine engine, Tensor<T> frames, IReadOnlyList<Tensor<T>> flowsToNext,
        IReadOnlyList<Tensor<T>> flowsToPrevious, IReadOnlyList<Tensor<T>> visibleNext, IReadOnlyList<Tensor<T>> visiblePrevious,
        Tensor<T>? weights = null)
    {
        int t = frames.Shape[0];
        Tensor<T> Frame(Tensor<T> x, int i) => engine.TensorNarrow(x, 0, i, 1);
        Tensor<T>? loss = null;
        for (int i = 0; i < t - 1; i++)
        {
            var current = Frame(frames, i);
            var next = Frame(frames, i + 1);
            var backward = MaskedL1(engine, BasicVsrBlocks.FlowWarp(engine, next, flowsToNext[i], GridSamplePadding.Zeros), current,
                visibleNext[i], weights is null ? null : Frame(weights, i));
            var forward = MaskedL1(engine, BasicVsrBlocks.FlowWarp(engine, current, flowsToPrevious[i], GridSamplePadding.Zeros), next,
                visiblePrevious[i], weights is null ? null : Frame(weights, i + 1));
            var pair = engine.TensorAdd(backward, forward);
            loss = loss is null ? pair : engine.TensorAdd(loss, pair);
        }
        return loss ?? new Tensor<T>(new[] { 1 });
    }

    /// <summary>
    /// The gradient of the unweighted <see cref="WarpingLoss{T}"/> with respect to the frames, in closed form: the
    /// sign of each masked residual over the frame's element count, scattered back through the bilinear warp's adjoint
    /// (align_corners, zero padding). The motion guidance needs it while sampling, where no tape records.
    /// </summary>
    internal static Tensor<T> WarpingLossGradient<T>(Tensor<T> frames, IReadOnlyList<Tensor<T>> flowsToNext,
        IReadOnlyList<Tensor<T>> flowsToPrevious, IReadOnlyList<Tensor<T>> visibleNext, IReadOnlyList<Tensor<T>> visiblePrevious)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int t = frames.Shape[0], c = frames.Shape[1], h = frames.Shape[2], w = frames.Shape[3], plane = h * w, size = c * plane;
        double[] Values(Tensor<T> x)
        {
            var values = new double[x.Length];
            for (int i = 0; i < values.Length; i++) values[i] = ops.ToDouble(x[i]);
            return values;
        }
        var x = Values(frames.IsContiguous ? frames : frames.Contiguous());
        var grad = new double[x.Length];

        // One term: mean(|m ⊙ (warp(source, flow) − frame)|); source and frame are frame offsets into x.
        void Term(int source, int frame, Tensor<T> flowTensor, Tensor<T> visibleTensor)
        {
            var flow = Values(flowTensor.IsContiguous ? flowTensor : flowTensor.Contiguous());
            var visible = Values(visibleTensor.IsContiguous ? visibleTensor : visibleTensor.Contiguous());
            for (int y = 0; y < h; y++)
                for (int px = 0; px < w; px++)
                {
                    int p = y * w + px;
                    double m = visible[p];
                    if (m == 0) continue;
                    double sx = px + flow[p], sy = y + flow[plane + p];
                    int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
                    double ax = sx - x0, ay = sy - y0;
                    var corners = new (int X, int Y, double Weight)[]
                    {
                        (x0, y0, (1 - ax) * (1 - ay)), (x0 + 1, y0, ax * (1 - ay)), (x0, y0 + 1, (1 - ax) * ay), (x0 + 1, y0 + 1, ax * ay),
                    };
                    for (int ch = 0; ch < c; ch++)
                    {
                        double warped = 0;
                        foreach (var (cx, cy, weight) in corners)
                            if (cx >= 0 && cx < w && cy >= 0 && cy < h) warped += weight * x[source * size + ch * plane + cy * w + cx];
                        double residual = m * (warped - x[frame * size + ch * plane + p]);
                        if (residual == 0) continue;
                        double s = Math.Sign(residual) * m / size;
                        grad[frame * size + ch * plane + p] -= s;
                        foreach (var (cx, cy, weight) in corners)
                            if (cx >= 0 && cx < w && cy >= 0 && cy < h) grad[source * size + ch * plane + cy * w + cx] += s * weight;
                    }
                }
        }

        for (int i = 0; i < t - 1; i++)
        {
            Term(i + 1, i, flowsToNext[i], visibleNext[i]);
            Term(i, i + 1, flowsToPrevious[i], visiblePrevious[i]);
        }
        var result = new Tensor<T>(frames.Shape.ToArray());
        for (int i = 0; i < grad.Length; i++) result[i] = ops.FromDouble(grad[i]);
        return result;
    }

    /// <summary>kornia's <c>sobel</c>: the normalized 3x3 Sobel gradients (replicate padding) as √(gx² + gy² + 1e-6) per channel.</summary>
    internal static Tensor<T> Sobel<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var flat = engine.PadNd(engine.Reshape(x, new[] { n * c, 1, h, w }), new[] { 1, 1, 1, 1 }, PadMode.Replicate, ops.Zero);
        double[] sx = { -1, 0, 1, -2, 0, 2, -1, 0, 1 }, sy = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
        var kernel = new Tensor<T>(new[] { 2, 1, 3, 3 });
        for (int i = 0; i < 9; i++)
        {
            kernel[i] = ops.FromDouble(sx[i] / 8.0);
            kernel[9 + i] = ops.FromDouble(sy[i] / 8.0);
        }
        var g = engine.Conv2D(flat, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var magnitude = engine.TensorSqrt(engine.TensorAddScalar(engine.ReduceSum(engine.TensorSquare(g), new[] { 1 }, keepDims: true), ops.FromDouble(1e-6)));
        return engine.Reshape(magnitude, new[] { n, c, h, w });
    }
}
