using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>How a FLAVR convolution is initialized.</summary>
internal enum FlavrInit
{
    /// <summary>torch's default: U(±1/√fan_in) weights and bias.</summary>
    TorchDefault,

    /// <summary>VideoResNet._initialize_weights: kaiming_normal_(mode='fan_out', nonlinearity='relu'), zero bias.</summary>
    FanOutNormal,
}

/// <summary>nn.Conv3d with an optional bias.</summary>
internal sealed class FlavrConv3d<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;
    private readonly int[] _stride, _padding;

    public FlavrConv3d(int inChannels, int outChannels, int[] kernel, int[] stride, int[] padding, bool bias, FlavrInit init)
    {
        _stride = stride;
        _padding = padding;
        var shape = new[] { outChannels, inChannels, kernel[0], kernel[1], kernel[2] };
        int receptive = kernel[0] * kernel[1] * kernel[2];
        if (init == FlavrInit.FanOutNormal)
        {
            _weight = EmaInit.Normal<T>(shape, Math.Sqrt(2.0 / (outChannels * (double)receptive)));
            if (bias) _bias = LazyWeight<T>.Constant(new[] { outChannels }, 0.0);
        }
        else
        {
            _weight = LazyWeight<T>.Uniform(shape, inChannels * receptive);
            if (bias) _bias = LazyWeight<T>.Uniform(new[] { outChannels }, inChannels * receptive);
        }
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Engine.Conv3D(x, _weight.Value, _stride, _padding, new[] { 1, 1, 1 });
        return _bias is null ? y : MgldOps.ChannelBias(Engine, y, _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>SEGating: the clip's global average through a biased 1x1x1 convolution and a sigmoid gates each channel.</summary>
internal sealed class FlavrGating<T> : LazyWeightModule<T>
{
    private readonly FlavrConv3d<T> _attention;

    public FlavrGating(int channels, FlavrInit init)
        => _attention = new FlavrConv3d<T>(channels, channels, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, true, init);

    public Tensor<T> Forward(Tensor<T> x)
    {
        var gate = Engine.Sigmoid(_attention.Forward(Engine.ReduceMean(x, new[] { 2, 3, 4 }, true)));
        return Engine.TensorMultiply(x, Engine.TensorBroadcastTo(gate, x.Shape.ToArray()));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _attention };
}

/// <summary>
/// resnet_3D.BasicBlock without batch norm: two 3x3x3 convolutions (the first strided in space), feature gating, the
/// shortcut (a strided 1x1x1 projection when the shape changes) and a ReLU.
/// </summary>
internal sealed class FlavrBasicBlock<T> : LazyWeightModule<T>
{
    private readonly FlavrConv3d<T> _conv1, _conv2;
    private readonly FlavrGating<T> _gating;
    private readonly FlavrConv3d<T>? _downsample;

    public FlavrBasicBlock(int inChannels, int channels, int stride, bool bias)
    {
        var strides = new[] { 1, stride, stride };
        _conv1 = new FlavrConv3d<T>(inChannels, channels, new[] { 3, 3, 3 }, strides, new[] { 1, 1, 1 }, bias, FlavrInit.FanOutNormal);
        _conv2 = new FlavrConv3d<T>(channels, channels, new[] { 3, 3, 3 }, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }, bias, FlavrInit.FanOutNormal);
        _gating = new FlavrGating<T>(channels, FlavrInit.FanOutNormal);
        if (stride != 1 || inChannels != channels)
            _downsample = new FlavrConv3d<T>(inChannels, channels, new[] { 1, 1, 1 }, strides, new[] { 0, 0, 0 }, false, FlavrInit.FanOutNormal);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _gating.Forward(_conv2.Forward(Engine.ReLU(_conv1.Forward(x))));
        return Engine.ReLU(Engine.TensorAdd(y, _downsample is null ? x : _downsample.Forward(x)));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>?[] { _conv1, _conv2, _gating, _downsample };
}

/// <summary>
/// unet_18 / unet_34 (VideoResNet): a (3, 7, 7) stem at half size and four stages of basic blocks, the second and third
/// halving the size; time is never strided. Returns the stem's and every stage's output.
/// </summary>
internal sealed class FlavrEncoder<T> : LazyWeightModule<T>
{
    private readonly FlavrConv3d<T> _stem;
    private readonly FlavrBasicBlock<T>[][] _stages;

    public FlavrEncoder(int width, IReadOnlyList<int> blocks, bool bias)
    {
        _stem = new FlavrConv3d<T>(3, width, new[] { 3, 7, 7 }, new[] { 1, 2, 2 }, new[] { 1, 3, 3 }, bias, FlavrInit.FanOutNormal);
        _stages = new FlavrBasicBlock<T>[4][];
        int inChannels = width;
        for (int s = 0; s < 4; s++)
        {
            int channels = width << s;
            int stride = s is 1 or 2 ? 2 : 1;
            _stages[s] = new FlavrBasicBlock<T>[blocks[s]];
            for (int b = 0; b < blocks[s]; b++)
            {
                _stages[s][b] = new FlavrBasicBlock<T>(inChannels, channels, b == 0 ? stride : 1, bias);
                inChannels = channels;
            }
        }
    }

    public Tensor<T>[] Forward(Tensor<T> x)
    {
        var outputs = new Tensor<T>[5];
        outputs[0] = x = Engine.ReLU(_stem.Forward(x));
        for (int s = 0; s < 4; s++)
        {
            foreach (var block in _stages[s]) x = block.Forward(x);
            outputs[s + 1] = x;
        }
        return outputs;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _stem }.Concat(_stages.SelectMany(stage => stage));
}

/// <summary>upConv3D(upmode='transpose'): ConvTranspose3d(kernel (3, 4, 4), stride (1, 2, 2), padding 1) and gating.</summary>
internal sealed class FlavrUpConv<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;
    private readonly FlavrGating<T> _gating;

    public FlavrUpConv(int inChannels, int outChannels)
    {
        // ConvTranspose3d's fan_in is weight.size(1) · kernel volume: the output channels.
        _weight = LazyWeight<T>.Uniform(new[] { inChannels, outChannels, 3, 4, 4 }, outChannels * 48);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, outChannels * 48);
        _gating = new FlavrGating<T>(outChannels, FlavrInit.TorchDefault);
    }

    public Tensor<T> Forward(Tensor<T> x)
        => _gating.Forward(MgldOps.ChannelBias(Engine,
            Engine.ConvTranspose3D(x, _weight.Value, new[] { 1, 2, 2 }, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }), _bias.Value));

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _gating };
}

/// <summary>Conv_3d: a biased 3x3x3 convolution and gating.</summary>
internal sealed class FlavrDecoderConv<T> : LazyWeightModule<T>
{
    private readonly FlavrConv3d<T> _conv;
    private readonly FlavrGating<T> _gating;

    public FlavrDecoderConv(int inChannels, int outChannels)
    {
        _conv = new FlavrConv3d<T>(inChannels, outChannels, new[] { 3, 3, 3 }, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }, true, FlavrInit.TorchDefault);
        _gating = new FlavrGating<T>(outChannels, FlavrInit.TorchDefault);
    }

    public Tensor<T> Forward(Tensor<T> x) => _gating.Forward(_conv.Forward(x));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv, _gating };
}

/// <summary>nn.Conv2d with torch's default initialization and an optional bias.</summary>
internal sealed class FlavrConv2d<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;

    public FlavrConv2d(int inChannels, int outChannels, int kernel, bool bias)
    {
        int fanIn = inChannels * kernel * kernel;
        _weight = LazyWeight<T>.Uniform(new[] { outChannels, inChannels, kernel, kernel }, fanIn);
        if (bias) _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Engine.Conv2D(x, _weight.Value, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        return _bias is null ? y : MgldOps.ChannelBias(Engine, y, _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>
/// UNet_3D_3D (joinType 'concat', upmode 'transpose', no batch norm): the encoder over the mean-subtracted clip, five
/// decoder stages each followed by LeakyReLU(0.2) and concatenated with the matching encoder output, the time steps
/// stacked into channels, a 1x1 fusion and a reflection-padded 7x7 convolution to every output frame.
/// </summary>
internal sealed class FlavrNetwork<T> : LazyWeightModule<T>
{
    /// <summary>The encoder reaches 1/8 size.</summary>
    internal const int Divisor = 8;

    private readonly FlavrEncoder<T> _encoder;
    private readonly FlavrDecoderConv<T> _decoder0, _decoder3;
    private readonly FlavrUpConv<T> _decoder1, _decoder2, _decoder4;
    private readonly FlavrConv2d<T> _fuse, _out;
    private readonly int _outputs;

    public FlavrNetwork(int width, IReadOnlyList<int> blocks, int inputs, int outputs)
    {
        _outputs = outputs;
        // resnet_3D.useBias is set when more than one frame is synthesized.
        _encoder = new FlavrEncoder<T>(width, blocks, outputs > 1);
        _decoder0 = new FlavrDecoderConv<T>(8 * width, 4 * width);
        _decoder1 = new FlavrUpConv<T>(8 * width, 2 * width);
        _decoder2 = new FlavrUpConv<T>(4 * width, width);
        _decoder3 = new FlavrDecoderConv<T>(2 * width, width);
        _decoder4 = new FlavrUpConv<T>(2 * width, width);
        _fuse = new FlavrConv2d<T>(width * inputs, width, 1, false);
        _out = new FlavrConv2d<T>(width, 3 * outputs, 7, true);
    }

    /// <summary>The right and bottom zero padding that brings a side to a multiple of 8.</summary>
    internal static (int Right, int Bottom) Padding(int height, int width)
        => ((Divisor - width % Divisor) % Divisor, (Divisor - height % Divisor) % Divisor);

    /// <summary>
    /// UNet_3D_3D.forward on frames [N, 3, H, W] in [0, 1]: every output frame [N, 3·outputs, H, W], earliest first, with
    /// the clip's per-channel mean restored and nothing clamped. The release takes sides divisible by 8; other sizes are
    /// zero-padded on the right and bottom after the mean is subtracted, and the output cropped.
    /// </summary>
    public Tensor<T> Forward(IReadOnlyList<Tensor<T>> frames, IDictionary<string, Tensor<T>>? activations = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = frames[0].Shape[0], h = frames[0].Shape[2], w = frames[0].Shape[3], t = frames.Count;
        var clip = Engine.TensorConcatenate(frames.Select(f => Engine.Reshape(f, new[] { n, 3, 1, h, w })).ToArray(), 2);
        var mean = Engine.ReduceMean(clip, new[] { 2, 3, 4 }, true);
        var centered = Engine.TensorSubtract(clip, Engine.TensorBroadcastTo(mean, clip.Shape.ToArray()));
        var (right, bottom) = Padding(h, w);
        if (right + bottom > 0) centered = Engine.PadNd(centered, new[] { 0, right, 0, bottom }, PadMode.Constant, ops.Zero);
        int hp = h + bottom, wp = w + right;
        var x = _encoder.Forward(centered);
        if (activations is not null)
            for (int k = 0; k < x.Length; k++) activations[k == 0 ? "Stem" : $"Stage{k}"] = x[k];
        var slope = ops.FromDouble(0.2);
        var d = Engine.TensorConcatenate(new[] { Engine.LeakyReLU(_decoder0.Forward(x[4]), slope), x[3] }, 1);
        d = Engine.TensorConcatenate(new[] { Engine.LeakyReLU(_decoder1.Forward(d), slope), x[2] }, 1);
        d = Engine.TensorConcatenate(new[] { Engine.LeakyReLU(_decoder2.Forward(d), slope), x[1] }, 1);
        d = Engine.TensorConcatenate(new[] { Engine.LeakyReLU(_decoder3.Forward(d), slope), x[0] }, 1);
        d = Engine.LeakyReLU(_decoder4.Forward(d), slope);
        if (activations is not null) activations["Decoder"] = d;
        // torch.cat(torch.unbind(dx_out, 2), 1): time-major channels.
        int c = d.Shape[1];
        var stacked = Engine.Reshape(Engine.TensorPermute(d, new[] { 0, 2, 1, 3, 4 }).Contiguous(), new[] { n, t * c, hp, wp });
        var fused = Engine.LeakyReLU(_fuse.Forward(stacked), slope);
        if (activations is not null) activations["Fusion"] = fused;
        var output = _out.Forward(Engine.PadNd(fused, new[] { 3, 3, 3, 3 }, PadMode.Reflect, ops.Zero));
        if (right + bottom > 0) output = Engine.TensorSlice(output, new[] { 0, 0, 0, 0 }, new[] { n, 3 * _outputs, h, w });
        var perFrame = Engine.Reshape(output, new[] { n, _outputs, 3, h, w });
        var frameMean = Engine.TensorBroadcastTo(Engine.Reshape(mean, new[] { n, 1, 3, 1, 1 }), perFrame.Shape.ToArray());
        return Engine.Reshape(Engine.TensorAdd(perFrame, frameMean), new[] { n, 3 * _outputs, h, w });
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _encoder, _decoder0, _decoder1, _decoder2, _decoder3, _decoder4, _fuse, _out };
}
