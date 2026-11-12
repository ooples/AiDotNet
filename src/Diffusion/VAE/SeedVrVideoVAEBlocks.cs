using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// <c>InflatedCausalConv3d</c> of SeedVR's <c>video_vae_v3</c>: the first frame repeated twice the time padding in front (so an
/// output frame reads only its own and earlier frames), the symmetric spatial zero padding, then an unpadded convolution.
/// </summary>
internal sealed class SeedVrCausalConv3d<T> : CausalVideoModule<T>
{
    private readonly int _timeStride, _spaceStride, _timePad, _spacePad;
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="timeKernel">The kernel's length in time.</param>
    /// <param name="spaceKernel">The kernel's height and width.</param>
    /// <param name="timeStride">The stride in time.</param>
    /// <param name="spaceStride">The stride in height and width.</param>
    /// <param name="timePad">The time padding; twice it is replicated in front.</param>
    /// <param name="spacePad">The spatial zero padding on each side.</param>
    public SeedVrCausalConv3d(int inChannels, int outChannels, int timeKernel, int spaceKernel, int timeStride = 1, int spaceStride = 1,
        int timePad = -1, int spacePad = -1)
    {
        _timeStride = timeStride;
        _spaceStride = spaceStride;
        _timePad = timePad < 0 ? timeKernel / 2 : timePad;
        _spacePad = spacePad < 0 ? spaceKernel / 2 : spacePad;
        int fanIn = inChannels * timeKernel * spaceKernel * spaceKernel;
        _weight = LazyWeight<T>.Uniform(new[] { outChannels, inChannels, timeKernel, spaceKernel, spaceKernel }, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int front = 2 * _timePad;
        if (front > 0)
        {
            var s = x.Shape;
            var head = Engine.TensorBroadcastTo(Frames(x, 0, 1), new[] { s[0], s[1], front, s[3], s[4] });
            x = Engine.TensorConcatenate(new[] { head, x }, 2);
        }
        if (_spacePad > 0)
            x = Engine.PadNd(x, new[] { _spacePad, _spacePad, _spacePad, _spacePad }, PadMode.Constant, NumOps.Zero);
        var y = Engine.Conv3D(x, _weight.Value, new[] { _timeStride, _spaceStride, _spaceStride }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 });
        return AddChannelBias(y, _bias.Value);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _weight.Value);
        yield return (Join(prefix, "bias"), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary><c>causal_norm_wrapper(nn.GroupNorm)</c>: group normalization of each frame on its own.</summary>
internal sealed class SeedVrFrameGroupNorm<T> : CausalVideoModule<T>
{
    private readonly int _groups;
    private readonly double _epsilon;
    private readonly LazyWeight<T> _gamma, _beta;

    public SeedVrFrameGroupNorm(int channels, int groups, double epsilon = 1e-6)
    {
        _groups = groups;
        _epsilon = epsilon;
        _gamma = LazyWeight<T>.Constant(new[] { channels }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { channels }, 0.0);
    }

    /// <summary>Normalizes [B, C, T, H, W] per (sample, frame) or [N, C, H, W] per sample.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        if (x.Rank == 4) return Engine.GroupNorm(x, _groups, _gamma.Value, _beta.Value, _epsilon, out _, out _);
        var s = x.Shape;
        int b = s[0], c = s[1], t = s[2], h = s[3], w = s[4];
        var frames = Engine.Reshape(Engine.TensorPermute(x, new[] { 0, 2, 1, 3, 4 }), new[] { b * t, c, h, w });
        var normed = Engine.GroupNorm(frames, _groups, _gamma.Value, _beta.Value, _epsilon, out _, out _);
        return Engine.TensorPermute(Engine.Reshape(normed, new[] { b, t, c, h, w }), new[] { 0, 2, 1, 3, 4 });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _gamma.Value);
        yield return (Join(prefix, "bias"), _beta.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _gamma;
        yield return _beta;
    }
}

/// <summary>
/// <c>ResnetBlock3D</c> with <c>time_receptive_field="full"</c>: (frame group norm, SiLU, causal 3x3x3 convolution) twice, plus
/// a causal 1x1x1 shortcut when the widths differ.
/// </summary>
internal sealed class SeedVrResnetBlock<T> : CausalVideoModule<T>
{
    private readonly SeedVrFrameGroupNorm<T> _norm1, _norm2;
    private readonly SeedVrCausalConv3d<T> _conv1, _conv2;
    private readonly SeedVrCausalConv3d<T>? _shortcut;

    public SeedVrResnetBlock(int inChannels, int outChannels, int groups)
    {
        _norm1 = new SeedVrFrameGroupNorm<T>(inChannels, groups);
        _conv1 = new SeedVrCausalConv3d<T>(inChannels, outChannels, 3, 3);
        _norm2 = new SeedVrFrameGroupNorm<T>(outChannels, groups);
        _conv2 = new SeedVrCausalConv3d<T>(outChannels, outChannels, 3, 3);
        if (inChannels != outChannels) _shortcut = new SeedVrCausalConv3d<T>(inChannels, outChannels, 1, 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var h = _conv1.Forward(Engine.Swish(_norm1.Forward(x)));
        h = _conv2.Forward(Engine.Swish(_norm2.Forward(h)));
        return Engine.TensorAdd(_shortcut is null ? x : _shortcut.Forward(x), h);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
        => _norm1.NamedTensors(Join(prefix, "norm1")).Concat(_conv1.NamedTensors(Join(prefix, "conv1")))
            .Concat(_norm2.NamedTensors(Join(prefix, "norm2"))).Concat(_conv2.NamedTensors(Join(prefix, "conv2")))
            .Concat(_shortcut?.NamedTensors(Join(prefix, "conv_shortcut")) ?? Enumerable.Empty<(string, Tensor<T>)>());

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm1;
        yield return _conv1;
        yield return _norm2;
        yield return _conv2;
        yield return _shortcut;
    }
}

/// <summary>
/// <c>Downsample3D</c> with <c>padding=0</c>: zero padding on the right and bottom, then a stride-2 causal 3x3 convolution, in
/// time as well (kernel 3, stride 2, the first frame repeated twice in front) when the block compresses time.
/// </summary>
internal sealed class SeedVrDownsample<T> : CausalVideoModule<T>
{
    private readonly SeedVrCausalConv3d<T> _conv;

    public SeedVrDownsample(int channels, bool temporal)
    {
        _conv = temporal
            ? new SeedVrCausalConv3d<T>(channels, channels, 3, 3, 2, 2, 1, 0)
            : new SeedVrCausalConv3d<T>(channels, channels, 1, 3, 1, 2, 0, 0);
    }

    public Tensor<T> Forward(Tensor<T> x)
        => _conv.Forward(Engine.PadNd(x, new[] { 0, 1, 0, 1 }, PadMode.Constant, NumOps.Zero));

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) => _conv.NamedTensors(Join(prefix, "conv"));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _conv;
    }
}

/// <summary>
/// <c>Upsample3D</c> (MAGVIT-v2 style): a 1x1x1 convolution to r times the channels (initialized to repeat its input), a
/// depth-to-space rearrangement <c>b (x y z c) f h w -> b c (f z) (h x) (w y)</c>, the duplicated frame after the first dropped
/// when time doubles, and a causal 3x3x3 convolution.
/// </summary>
internal sealed class SeedVrUpsample<T> : CausalVideoModule<T>
{
    private readonly int _channels, _timeRatio;
    private readonly LazyWeight<T> _upscaleWeight, _upscaleBias;
    private readonly SeedVrCausalConv3d<T> _conv;

    public SeedVrUpsample(int channels, bool temporal)
    {
        _channels = channels;
        _timeRatio = temporal ? 2 : 1;
        int ratio = 4 * _timeRatio;
        // torch.eye(C).repeat(r, 1): output channel k·C + c copies input channel c.
        _upscaleWeight = new LazyWeight<T>(new[] { ratio * channels, channels, 1, 1, 1 }, (tensor, _) =>
        {
            tensor.Fill(NumOps.Zero);
            for (int k = 0; k < ratio; k++)
                for (int c = 0; c < channels; c++) tensor[(k * channels + c) * channels + c] = NumOps.One;
        });
        _upscaleBias = LazyWeight<T>.Constant(new[] { ratio * channels }, 0.0);
        _conv = new SeedVrCausalConv3d<T>(channels, channels, 3, 3);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var s = x.Shape;
        int b = s[0], c = _channels, f = s[2], h = s[3], w = s[4], z = _timeRatio;
        var up = AddChannelBias(Engine.Conv3D(x, _upscaleWeight.Value, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 }), _upscaleBias.Value);
        up = Engine.Reshape(up, new[] { b, 2, 2, z, c, f, h, w });
        // (b, x, y, z, c, f, h, w) -> (b, c, f, z, h, x, w, y)
        up = Engine.TensorPermute(up, new[] { 0, 4, 5, 3, 6, 1, 7, 2 });
        up = Engine.Reshape(up, new[] { b, c, f * z, 2 * h, 2 * w });
        if (z > 1 && f * z > 2)
            up = Engine.TensorConcatenate(new[] { Frames(up, 0, 1), Frames(up, 2, f * z - 2) }, 2);
        else if (z > 1)
            up = Frames(up, 0, 1);
        return _conv.Forward(up);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "upscale_conv.weight"), _upscaleWeight.Value);
        yield return (Join(prefix, "upscale_conv.bias"), _upscaleBias.Value);
        foreach (var named in _conv.NamedTensors(Join(prefix, "conv"))) yield return named;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _upscaleWeight;
        yield return _upscaleBias;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _conv;
    }
}

/// <summary>
/// The mid block's diffusers <c>Attention</c> (one head as wide as the channels, group norm in front, residual): self-attention
/// over each frame's pixels.
/// </summary>
internal sealed class SeedVrFrameAttention<T> : CausalVideoModule<T>
{
    private readonly int _channels;
    private readonly SeedVrFrameGroupNorm<T> _norm;
    private readonly LazyWeight<T> _q, _qBias, _k, _kBias, _v, _vBias, _out, _outBias;

    public SeedVrFrameAttention(int channels, int groups)
    {
        _channels = channels;
        _norm = new SeedVrFrameGroupNorm<T>(channels, groups);
        (_q, _qBias) = Linear(channels);
        (_k, _kBias) = Linear(channels);
        (_v, _vBias) = Linear(channels);
        (_out, _outBias) = Linear(channels);
    }

    private static (LazyWeight<T>, LazyWeight<T>) Linear(int channels)
        => (LazyWeight<T>.Uniform(new[] { channels, channels }, channels), LazyWeight<T>.Uniform(new[] { channels }, channels));

    private Tensor<T> Project(Tensor<T> tokens, LazyWeight<T> weight, LazyWeight<T> bias)
    {
        var y = Engine.TensorMatMul(tokens, Engine.TensorTranspose(weight.Value));
        return Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(bias.Value, new[] { 1, 1, _channels }), y.Shape.ToArray()));
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var s = x.Shape;
        int b = s[0], c = _channels, t = s[2], h = s[3], w = s[4];
        var tokens = Engine.Reshape(Engine.TensorPermute(_norm.Forward(x), new[] { 0, 2, 3, 4, 1 }), new[] { b * t, h * w, c });
        var q = Project(tokens, _q, _qBias);
        var k = Project(tokens, _k, _kBias);
        var v = Project(tokens, _v, _vBias);
        var scores = Engine.TensorMultiplyScalar(Engine.TensorBatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(c)));
        var attended = Project(Engine.TensorBatchMatMul(Engine.TensorSoftmax(scores, 2), v), _out, _outBias);
        var back = Engine.TensorPermute(Engine.Reshape(attended, new[] { b, t, h, w, c }), new[] { 0, 4, 1, 2, 3 });
        return Engine.TensorAdd(back, x);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var named in _norm.NamedTensors(Join(prefix, "group_norm"))) yield return named;
        yield return (Join(prefix, "to_q.weight"), _q.Value);
        yield return (Join(prefix, "to_q.bias"), _qBias.Value);
        yield return (Join(prefix, "to_k.weight"), _k.Value);
        yield return (Join(prefix, "to_k.bias"), _kBias.Value);
        yield return (Join(prefix, "to_v.weight"), _v.Value);
        yield return (Join(prefix, "to_v.bias"), _vBias.Value);
        yield return (Join(prefix, "to_out.0.weight"), _out.Value);
        yield return (Join(prefix, "to_out.0.bias"), _outBias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _q;
        yield return _qBias;
        yield return _k;
        yield return _kBias;
        yield return _v;
        yield return _vBias;
        yield return _out;
        yield return _outBias;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm;
    }
}

/// <summary>
/// SeedVR's causal video VAE (<c>video_vae_v3</c>, the SD3 autoencoder inflated to 3D): 8x spatial and 4x temporal compression,
/// the first frame alone in the first latent, frame-wise group norms and attention, causal 3D convolutions throughout, and
/// learned depth-to-space upsampling. Latents are the posterior mean times the released scaling factor.
/// </summary>
internal sealed class SeedVrVideoVae<T> : CausalVideoModule<T>
{
    private readonly SeedVrCausalConv3d<T> _encoderIn, _encoderOut, _decoderIn, _decoderOut;
    private readonly List<(List<SeedVrResnetBlock<T>> Resnets, SeedVrDownsample<T>? Down)> _down = new();
    private readonly List<(List<SeedVrResnetBlock<T>> Resnets, SeedVrUpsample<T>? Up)> _up = new();
    private readonly SeedVrResnetBlock<T> _encoderMid1, _encoderMid2, _decoderMid1, _decoderMid2;
    private readonly SeedVrFrameAttention<T> _encoderAttention, _decoderAttention;
    private readonly SeedVrFrameGroupNorm<T> _encoderNorm, _decoderNorm;
    private readonly double _scaling;

    /// <param name="channels">The four stages' widths (released: 128, 256, 512, 512).</param>
    /// <param name="layersPerBlock">The encoder's residual blocks per stage; the decoder has one more (released: 2).</param>
    /// <param name="groups">The group norms' groups (released: 32).</param>
    /// <param name="latentChannels">The latent channels (released: 16).</param>
    /// <param name="scaling">The latent scaling factor (released: 0.9152).</param>
    public SeedVrVideoVae(IReadOnlyList<int> channels, int layersPerBlock, int groups, int latentChannels, double scaling)
    {
        if (channels.Count != 4) throw new ArgumentOutOfRangeException(nameof(channels), "SeedVR's VAE has four stages.");
        _scaling = scaling;
        LatentChannels = latentChannels;
        _encoderIn = new SeedVrCausalConv3d<T>(3, channels[0], 3, 3);
        int previous = channels[0];
        for (int i = 0; i < 4; i++)
        {
            var resnets = new List<SeedVrResnetBlock<T>>();
            for (int j = 0; j < layersPerBlock; j++) resnets.Add(new SeedVrResnetBlock<T>(j == 0 ? previous : channels[i], channels[i], groups));
            // temporal_down_num = 2 takes the last two downsampling stages.
            _down.Add((resnets, i < 3 ? new SeedVrDownsample<T>(channels[i], temporal: i >= 1) : null));
            previous = channels[i];
        }
        _encoderMid1 = new SeedVrResnetBlock<T>(channels[3], channels[3], groups);
        _encoderAttention = new SeedVrFrameAttention<T>(channels[3], groups);
        _encoderMid2 = new SeedVrResnetBlock<T>(channels[3], channels[3], groups);
        _encoderNorm = new SeedVrFrameGroupNorm<T>(channels[3], groups);
        _encoderOut = new SeedVrCausalConv3d<T>(channels[3], 2 * latentChannels, 3, 3);

        _decoderIn = new SeedVrCausalConv3d<T>(latentChannels, channels[3], 3, 3);
        _decoderMid1 = new SeedVrResnetBlock<T>(channels[3], channels[3], groups);
        _decoderAttention = new SeedVrFrameAttention<T>(channels[3], groups);
        _decoderMid2 = new SeedVrResnetBlock<T>(channels[3], channels[3], groups);
        previous = channels[3];
        for (int i = 0; i < 4; i++)
        {
            int width = channels[3 - i];
            var resnets = new List<SeedVrResnetBlock<T>>();
            for (int j = 0; j <= layersPerBlock; j++) resnets.Add(new SeedVrResnetBlock<T>(j == 0 ? previous : width, width, groups));
            // temporal_up_num = 2 takes the first two upsampling stages.
            _up.Add((resnets, i < 3 ? new SeedVrUpsample<T>(width, temporal: i < 2) : null));
            previous = width;
        }
        _decoderNorm = new SeedVrFrameGroupNorm<T>(channels[0], groups);
        _decoderOut = new SeedVrCausalConv3d<T>(channels[0], 3, 3, 3);
    }

    /// <summary>The latent channels.</summary>
    internal int LatentChannels { get; }

    /// <summary>The spatial compression.</summary>
    internal int SpatialFactor => 8;

    /// <summary>The scaled posterior mean of [B, 3, 1 + 4k, H, W] in [-1, 1]: [B, C, 1 + k, H / 8, W / 8].</summary>
    /// <remarks>
    /// The release samples the posterior; the mean is used here so a clip always encodes to the same latent (the trained
    /// posterior's spread is small next to the latents).
    /// </remarks>
    public Tensor<T> Encode(Tensor<T> video)
    {
        var h = _encoderIn.Forward(video);
        foreach (var (resnets, down) in _down)
        {
            foreach (var resnet in resnets) h = resnet.Forward(h);
            if (down is not null) h = down.Forward(h);
        }
        h = _encoderMid2.Forward(_encoderAttention.Forward(_encoderMid1.Forward(h)));
        h = _encoderOut.Forward(Engine.Swish(_encoderNorm.Forward(h)));
        return Engine.TensorMultiplyScalar(Engine.TensorNarrow(h, 1, 0, LatentChannels), NumOps.FromDouble(_scaling));
    }

    /// <summary>Scaled latents [B, C, n, h, w] to video [B, 3, 4n − 3, 8h, 8w].</summary>
    public Tensor<T> Decode(Tensor<T> latents)
    {
        var h = _decoderIn.Forward(Engine.TensorMultiplyScalar(latents, NumOps.FromDouble(1.0 / _scaling)));
        h = _decoderMid2.Forward(_decoderAttention.Forward(_decoderMid1.Forward(h)));
        foreach (var (resnets, up) in _up)
        {
            foreach (var resnet in resnets) h = resnet.Forward(h);
            if (up is not null) h = up.Forward(h);
        }
        return _decoderOut.Forward(Engine.Swish(_decoderNorm.Forward(h)));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        IEnumerable<(string, Tensor<T>)> all = _encoderIn.NamedTensors(Join(prefix, "encoder.conv_in"));
        for (int i = 0; i < _down.Count; i++)
        {
            var (resnets, down) = _down[i];
            for (int j = 0; j < resnets.Count; j++) all = all.Concat(resnets[j].NamedTensors(Join(prefix, $"encoder.down_blocks.{i}.resnets.{j}")));
            if (down is not null) all = all.Concat(down.NamedTensors(Join(prefix, $"encoder.down_blocks.{i}.downsamplers.0")));
        }
        all = all.Concat(_encoderMid1.NamedTensors(Join(prefix, "encoder.mid_block.resnets.0")))
            .Concat(_encoderAttention.NamedTensors(Join(prefix, "encoder.mid_block.attentions.0")))
            .Concat(_encoderMid2.NamedTensors(Join(prefix, "encoder.mid_block.resnets.1")))
            .Concat(_encoderNorm.NamedTensors(Join(prefix, "encoder.conv_norm_out")))
            .Concat(_encoderOut.NamedTensors(Join(prefix, "encoder.conv_out")))
            .Concat(_decoderIn.NamedTensors(Join(prefix, "decoder.conv_in")))
            .Concat(_decoderMid1.NamedTensors(Join(prefix, "decoder.mid_block.resnets.0")))
            .Concat(_decoderAttention.NamedTensors(Join(prefix, "decoder.mid_block.attentions.0")))
            .Concat(_decoderMid2.NamedTensors(Join(prefix, "decoder.mid_block.resnets.1")));
        for (int i = 0; i < _up.Count; i++)
        {
            var (resnets, up) = _up[i];
            for (int j = 0; j < resnets.Count; j++) all = all.Concat(resnets[j].NamedTensors(Join(prefix, $"decoder.up_blocks.{i}.resnets.{j}")));
            if (up is not null) all = all.Concat(up.NamedTensors(Join(prefix, $"decoder.up_blocks.{i}.upsamplers.0")));
        }
        return all.Concat(_decoderNorm.NamedTensors(Join(prefix, "decoder.conv_norm_out")))
            .Concat(_decoderOut.NamedTensors(Join(prefix, "decoder.conv_out")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _encoderIn;
        foreach (var (resnets, down) in _down)
        {
            foreach (var resnet in resnets) yield return resnet;
            yield return down;
        }
        yield return _encoderMid1;
        yield return _encoderAttention;
        yield return _encoderMid2;
        yield return _encoderNorm;
        yield return _encoderOut;
        yield return _decoderIn;
        yield return _decoderMid1;
        yield return _decoderAttention;
        yield return _decoderMid2;
        foreach (var (resnets, up) in _up)
        {
            foreach (var resnet in resnets) yield return resnet;
            yield return up;
        }
        yield return _decoderNorm;
        yield return _decoderOut;
    }
}
