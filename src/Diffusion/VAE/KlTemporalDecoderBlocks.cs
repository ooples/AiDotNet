namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// A convolution over time only, <c>nn.Conv3d</c> with kernel (k, 1, 1) and padding (k / 2, 0, 0), on [B, C, F, H, W].
/// </summary>
internal sealed class KlTimeConv<T> : KlModule<T>
{
    private readonly int _kernel;
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    public KlTimeConv(int inChannels, int outChannels, int kernel)
    {
        if (kernel <= 0 || kernel % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernel), "The kernel must be odd.");
        _kernel = kernel;
        int fanIn = inChannels * kernel;
        _weight = LazyWeight<T>.Uniform(new[] { outChannels, inChannels, kernel, 1, 1 }, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x) => AddChannelBias(
        Engine.Conv3D(x, _weight.Value, new[] { 1, 1, 1 }, new[] { _kernel / 2, 0, 0 }, new[] { 1, 1, 1 }), _bias.Value);

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

/// <summary>
/// diffusers' <c>TemporalResnetBlock</c> without a time embedding, on [B, C, F, H, W]: GroupNorm, SiLU and a (3, 1, 1)
/// convolution over time, twice, plus the input (through a 1x1x1 convolution when the width changes).
/// </summary>
internal sealed class KlTemporalResnetBlock<T> : KlModule<T>
{
    private readonly KlGroupNorm<T> _norm1;
    private readonly KlTimeConv<T> _conv1;
    private readonly KlGroupNorm<T> _norm2;
    private readonly KlTimeConv<T> _conv2;
    private readonly KlTimeConv<T>? _shortcut;

    public KlTemporalResnetBlock(int inChannels, int outChannels, int groups, double epsilon)
    {
        _norm1 = new KlGroupNorm<T>(groups, inChannels, epsilon);
        _conv1 = new KlTimeConv<T>(inChannels, outChannels, 3);
        _norm2 = new KlGroupNorm<T>(groups, outChannels, epsilon);
        _conv2 = new KlTimeConv<T>(outChannels, outChannels, 3);
        if (inChannels != outChannels) _shortcut = new KlTimeConv<T>(inChannels, outChannels, 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var h = _conv1.Forward(Engine.Swish(_norm1.Forward(x)));
        h = _conv2.Forward(Engine.Swish(_norm2.Forward(h)));
        return Engine.TensorAdd(_shortcut is null ? x : _shortcut.Forward(x), h);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _norm1.NamedTensors(Join(prefix, "norm1"))
            .Concat(_conv1.NamedTensors(Join(prefix, "conv1")))
            .Concat(_norm2.NamedTensors(Join(prefix, "norm2")))
            .Concat(_conv2.NamedTensors(Join(prefix, "conv2")));
        return _shortcut is null ? all : all.Concat(_shortcut.NamedTensors(Join(prefix, "conv_shortcut")));
    }

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
/// diffusers' <c>SpatioTemporalResBlock</c> as the temporal decoder builds it: a spatial residual block on every frame,
/// a temporal residual block across the frames, and a learned blend of the two (<c>AlphaBlender</c>, strategy
/// "learned", spatial and temporal switched): <c>(1 - s) * spatial + s * temporal</c> with <c>s = sigmoid(m)</c> and
/// <c>m</c> starting at 0.
/// </summary>
internal sealed class KlSpatioTemporalResBlock<T> : KlModule<T>
{
    private readonly KlResnetBlock<T> _spatial;
    private readonly KlTemporalResnetBlock<T> _temporal;
    private readonly LazyWeight<T> _mixFactor = LazyWeight<T>.Constant(new[] { 1 }, 0.0);

    public KlSpatioTemporalResBlock(int inChannels, int outChannels, int groups)
    {
        _spatial = new KlResnetBlock<T>(inChannels, outChannels, groups);
        _temporal = new KlTemporalResnetBlock<T>(outChannels, outChannels, groups, 1e-5);
    }

    /// <summary>Runs the block on [B * F, C, H, W], frames of one clip adjacent.</summary>
    public Tensor<T> Forward(Tensor<T> x, int frames)
    {
        var spatial = _spatial.Forward(x);
        var s = spatial.Shape;
        int batch = s[0] / frames;
        // [B * F, C, H, W] -> [B, C, F, H, W].
        var clip = Engine.TensorPermute(Engine.Reshape(spatial, new[] { batch, frames, s[1], s[2], s[3] }), new[] { 0, 2, 1, 3, 4 });
        var temporal = _temporal.Forward(clip);
        var weight = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.Sigmoid(_mixFactor.Value), new[] { 1, 1, 1, 1, 1 }), clip.Shape.ToArray());
        var mixed = Engine.TensorAdd(clip, Engine.TensorMultiply(weight, Engine.TensorSubtract(temporal, clip)));
        return Engine.Reshape(Engine.TensorPermute(mixed, new[] { 0, 2, 1, 3, 4 }), new[] { s[0], s[1], s[2], s[3] });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _spatial.NamedTensors(Join(prefix, "spatial_res_block"))
            .Concat(_temporal.NamedTensors(Join(prefix, "temporal_res_block")))
            .Concat(new[] { (Join(prefix, "time_mixer.mix_factor"), _mixFactor.Value) });

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _mixFactor;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _spatial;
        yield return _temporal;
    }
}

/// <summary>
/// diffusers' <c>TemporalDecoder</c>, the decoder of Stable Video Diffusion's VAE (<c>AutoencoderKLTemporalDecoder</c>):
/// a 3x3 conv in; a middle block of spatio-temporal residual blocks around one single-head self-attention; one stage per
/// width from the deepest, each of <c>layersPerBlock + 1</c> spatio-temporal residual blocks and, but the last, a
/// nearest 2x upsample and 3x3 conv; GroupNorm, SiLU and a 3x3 conv out; then a (3, 1, 1) convolution over time.
/// </summary>
/// <remarks>
/// The middle block builds <c>layersPerBlock</c> residual blocks and one attention but, as diffusers' forward does,
/// runs only the first two blocks around the attention.
/// </remarks>
internal sealed class KlTemporalDecoder<T> : KlModule<T>
{
    private readonly KlConv2d<T> _convIn;
    private readonly List<KlSpatioTemporalResBlock<T>> _midResnets = new();
    private readonly KlAttention<T> _midAttention;
    private readonly List<(List<KlSpatioTemporalResBlock<T>> Resnets, KlConv2d<T>? Upsample)> _stages = new();
    private readonly KlGroupNorm<T> _normOut;
    private readonly KlConv2d<T> _convOut;
    private readonly KlTimeConv<T> _timeConvOut;

    public KlTemporalDecoder(int latentChannels, int outChannels, IReadOnlyList<int> widths, int layersPerBlock, int groups)
    {
        var reversed = widths.Reverse().ToArray();
        int deepest = reversed[0];
        _convIn = new KlConv2d<T>(latentChannels, deepest, 3, padding: 1);
        for (int i = 0; i < layersPerBlock; i++) _midResnets.Add(new KlSpatioTemporalResBlock<T>(deepest, deepest, groups));
        _midAttention = new KlAttention<T>(deepest, groups);
        int previous = deepest;
        for (int i = 0; i < reversed.Length; i++)
        {
            var resnets = new List<KlSpatioTemporalResBlock<T>>();
            for (int j = 0; j < layersPerBlock + 1; j++)
                resnets.Add(new KlSpatioTemporalResBlock<T>(j == 0 ? previous : reversed[i], reversed[i], groups));
            var up = i < reversed.Length - 1 ? new KlConv2d<T>(reversed[i], reversed[i], 3, padding: 1) : null;
            _stages.Add((resnets, up));
            previous = reversed[i];
        }
        _normOut = new KlGroupNorm<T>(groups, previous);
        _convOut = new KlConv2d<T>(previous, outChannels, 3, padding: 1);
        _timeConvOut = new KlTimeConv<T>(outChannels, outChannels, 3);
    }

    /// <summary>Decodes [B * F, latent, h, w], frames of one clip adjacent, to [B * F, out, H, W].</summary>
    public Tensor<T> Forward(Tensor<T> z, int frames)
    {
        var x = _convIn.Forward(z);
        x = _midResnets[0].Forward(x, frames);
        if (_midResnets.Count > 1) x = _midResnets[1].Forward(_midAttention.Forward(x), frames);
        foreach (var (resnets, up) in _stages)
        {
            foreach (var resnet in resnets) x = resnet.Forward(x, frames);
            if (up is not null) x = up.Forward(Engine.Upsample(x, 2, 2));
        }
        x = _convOut.Forward(Engine.Swish(_normOut.Forward(x)));
        var s = x.Shape;
        int batch = s[0] / frames;
        var clip = Engine.TensorPermute(Engine.Reshape(x, new[] { batch, frames, s[1], s[2], s[3] }), new[] { 0, 2, 1, 3, 4 });
        clip = _timeConvOut.Forward(clip);
        return Engine.Reshape(Engine.TensorPermute(clip, new[] { 0, 2, 1, 3, 4 }), new[] { s[0], s[1], s[2], s[3] });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _convIn.NamedTensors(Join(prefix, "conv_in"))
            .Concat(_midAttention.NamedTensors(Join(prefix, "mid_block.attentions.0")));
        for (int i = 0; i < _midResnets.Count; i++)
            all = all.Concat(_midResnets[i].NamedTensors(Join(prefix, $"mid_block.resnets.{i}")));
        for (int i = 0; i < _stages.Count; i++)
        {
            for (int j = 0; j < _stages[i].Resnets.Count; j++)
                all = all.Concat(_stages[i].Resnets[j].NamedTensors(Join(prefix, $"up_blocks.{i}.resnets.{j}")));
            if (_stages[i].Upsample is { } up) all = all.Concat(up.NamedTensors(Join(prefix, $"up_blocks.{i}.upsamplers.0.conv")));
        }
        return all
            .Concat(_normOut.NamedTensors(Join(prefix, "conv_norm_out")))
            .Concat(_convOut.NamedTensors(Join(prefix, "conv_out")))
            .Concat(_timeConvOut.NamedTensors(Join(prefix, "time_conv_out")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _convIn;
        foreach (var resnet in _midResnets) yield return resnet;
        yield return _midAttention;
        foreach (var (resnets, up) in _stages)
        {
            foreach (var resnet in resnets) yield return resnet;
            yield return up;
        }
        yield return _normOut;
        yield return _convOut;
        yield return _timeConvOut;
    }
}