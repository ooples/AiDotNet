namespace AiDotNet.ComputerVision.Segmentation.Salient;

/// <summary>
/// The channel widths of a U²-Net: six encoder stages (RSU-7, 6, 5, 4, 4F, 4F) and five decoder stages (RSU-4F, 4,
/// 5, 6, 7), each with a middle and an output width.
/// </summary>
public sealed class U2NetConfiguration
{
    /// <summary>Input channels. Default: 3.</summary>
    public int InputChannels { get; set; } = 3;

    /// <summary>Output channels (side outputs and fused map). Default: 1.</summary>
    public int OutputChannels { get; set; } = 1;

    /// <summary>Middle widths of encoder stages 1-6. U²-Net: 32, 32, 64, 128, 256, 256.</summary>
    public int[] EncoderMiddle { get; set; } = { 32, 32, 64, 128, 256, 256 };

    /// <summary>Output widths of encoder stages 1-6. U²-Net: 64, 128, 256, 512, 512, 512.</summary>
    public int[] EncoderOutput { get; set; } = { 64, 128, 256, 512, 512, 512 };

    /// <summary>Middle widths of decoder stages 5d, 4d, 3d, 2d, 1d. U²-Net: 256, 128, 64, 32, 16.</summary>
    public int[] DecoderMiddle { get; set; } = { 256, 128, 64, 32, 16 };

    /// <summary>Output widths of decoder stages 5d, 4d, 3d, 2d, 1d. U²-Net: 512, 256, 128, 64, 64.</summary>
    public int[] DecoderOutput { get; set; } = { 512, 256, 128, 64, 64 };

    /// <summary>BatchNorm epsilon (PyTorch's default). Default: 1e-5.</summary>
    public double BatchNormEpsilon { get; set; } = 1e-5;

    /// <summary>The full U²-Net (176.3 MB, the <c>u2net</c> model rembg uses).</summary>
    public static U2NetConfiguration Full() => new U2NetConfiguration();

    /// <summary>The small U²-Net† (<c>u2netp</c>, 4.7 MB): every middle width 16, every output width 64.</summary>
    public static U2NetConfiguration Small() => new U2NetConfiguration
    {
        EncoderMiddle = new[] { 16, 16, 16, 16, 16, 16 },
        EncoderOutput = new[] { 64, 64, 64, 64, 64, 64 },
        DecoderMiddle = new[] { 16, 16, 16, 16, 16 },
        DecoderOutput = new[] { 64, 64, 64, 64, 64 },
    };

    internal void Validate()
    {
        if (InputChannels < 1 || OutputChannels < 1) throw new ArgumentException("Channel counts must be positive.");
        if (EncoderMiddle is null || EncoderMiddle.Length != 6 || EncoderOutput is null || EncoderOutput.Length != 6)
            throw new ArgumentException("U²-Net has six encoder stages.");
        if (DecoderMiddle is null || DecoderMiddle.Length != 5 || DecoderOutput is null || DecoderOutput.Length != 5)
            throw new ArgumentException("U²-Net has five decoder stages.");
    }
}

/// <summary>
/// One <c>REBNCONV</c> of U²-Net: 3 x 3 convolution (padding = dilation) with bias, BatchNorm (inference statistics),
/// ReLU.
/// </summary>
internal sealed class U2NetConv<T>
{
    public U2NetConv(int inputs, int outputs, int dilation, Random random)
    {
        Inputs = inputs;
        Outputs = outputs;
        Dilation = dilation;
        // PyTorch's Conv2d default: kaiming_uniform(a = sqrt 5) = U(-1/sqrt(fan_in), 1/sqrt(fan_in)), same bound for
        // the bias; BatchNorm starts at weight 1, bias 0, running mean 0, running variance 1.
        double bound = 1.0 / Math.Sqrt(inputs * 9.0);
        Weight = U2NetMath<T>.Uniform(new[] { outputs, inputs, 3, 3 }, bound, random);
        Bias = U2NetMath<T>.Uniform(new[] { outputs }, bound, random);
        Gamma = U2NetMath<T>.Constant(outputs, 1.0);
        Beta = U2NetMath<T>.Constant(outputs, 0.0);
        RunningMean = U2NetMath<T>.Constant(outputs, 0.0);
        RunningVariance = U2NetMath<T>.Constant(outputs, 1.0);
    }

    public int Inputs { get; }
    public int Outputs { get; }
    public int Dilation { get; }
    public Tensor<T> Weight { get; set; }
    public Tensor<T> Bias { get; set; }
    public Tensor<T> Gamma { get; set; }
    public Tensor<T> Beta { get; set; }
    public Tensor<T> RunningMean { get; set; }
    public Tensor<T> RunningVariance { get; set; }

    public long ParameterCount => ((long)Outputs * Inputs * 9) + (3L * Outputs);

    public Tensor<T> Forward(Tensor<T> x, double epsilon)
    {
        var engine = AiDotNetEngine.Current;
        var y = U2NetMath<T>.Convolve(x, Weight, Bias, Dilation);
        y = engine.BatchNormInference(y, Gamma, Beta, RunningMean, RunningVariance, epsilon);
        return engine.ReLU(y);
    }

    public IEnumerable<Tensor<T>> Parameters()
    {
        yield return Weight;
        yield return Bias;
        yield return Gamma;
        yield return Beta;
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        Weight = U2NetMath<T>.Read(read, prefix + "conv_s1.weight", new[] { Outputs, Inputs, 3, 3 });
        Bias = U2NetMath<T>.Read(read, prefix + "conv_s1.bias", new[] { Outputs });
        Gamma = U2NetMath<T>.Read(read, prefix + "bn_s1.weight", new[] { Outputs });
        Beta = U2NetMath<T>.Read(read, prefix + "bn_s1.bias", new[] { Outputs });
        RunningMean = U2NetMath<T>.Read(read, prefix + "bn_s1.running_mean", new[] { Outputs });
        RunningVariance = U2NetMath<T>.Read(read, prefix + "bn_s1.running_var", new[] { Outputs });
    }
}

/// <summary>
/// A residual U-block (<c>RSU-L</c>, or <c>RSU-4F</c> with dilations instead of pooling): an input convolution, a
/// height-L U-Net of <c>REBNCONV</c>s whose innermost layer is dilated, and the residual sum of both.
/// </summary>
internal sealed class ResidualUBlock<T>
{
    public ResidualUBlock(int height, bool dilated, int inputs, int middle, int outputs, Random random)
    {
        if (height < 2) throw new ArgumentOutOfRangeException(nameof(height));
        Height = height;
        Dilated = dilated;
        Outputs = outputs;
        Input = new U2NetConv<T>(inputs, outputs, 1, random);
        // Encoder: rebnconv1 .. rebnconv{L}; the last is dilated 2 (RSU-L) or the dilations double (RSU-4F).
        for (int level = 1; level <= height; level++)
        {
            int dilation = dilated ? 1 << (level - 1) : (level == height ? 2 : 1);
            Encoder.Add(new U2NetConv<T>(level == 1 ? outputs : middle, middle, dilation, random));
        }

        // Decoder: rebnconv{L-1}d .. rebnconv1d.
        for (int level = height - 1; level >= 1; level--)
        {
            int dilation = dilated ? 1 << (level - 1) : 1;
            Decoder.Add(new U2NetConv<T>(2 * middle, level == 1 ? outputs : middle, dilation, random));
        }
    }

    public int Height { get; }
    public bool Dilated { get; }
    public int Outputs { get; }
    public U2NetConv<T> Input { get; }
    public List<U2NetConv<T>> Encoder { get; } = new();
    public List<U2NetConv<T>> Decoder { get; } = new();

    public IEnumerable<U2NetConv<T>> Convolutions()
    {
        yield return Input;
        foreach (var c in Encoder) yield return c;
        foreach (var c in Decoder) yield return c;
    }

    public Tensor<T> Forward(Tensor<T> x, double epsilon)
    {
        var engine = AiDotNetEngine.Current;
        var hin = Input.Forward(x, epsilon);
        var skips = new List<Tensor<T>>();
        var h = hin;
        for (int level = 1; level <= Height; level++)
        {
            h = Encoder[level - 1].Forward(h, epsilon);
            if (level < Height)
            {
                skips.Add(h);
                if (!Dilated && level < Height - 1)
                    h = U2NetMath<T>.MaxPoolCeil(h);
            }
        }

        // h = hx{L}; skips[i] = hx{i + 1}.
        for (int i = 0; i < Decoder.Count; i++)
        {
            var skip = skips[skips.Count - 1 - i];
            if (!Dilated)
                h = U2NetMath<T>.UpsampleLike(h, skip);
            h = Decoder[i].Forward(engine.TensorConcatenate(new[] { h, skip }, 1), epsilon);
        }

        return engine.TensorAdd(h, hin);
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        Input.Load(read, prefix + "rebnconvin.");
        for (int level = 1; level <= Height; level++)
            Encoder[level - 1].Load(read, $"{prefix}rebnconv{level}.");
        for (int i = 0; i < Decoder.Count; i++)
            Decoder[i].Load(read, $"{prefix}rebnconv{Height - 1 - i}d.");
    }
}

/// <summary>
/// U²-Net (Qin et al., "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection", Pattern
/// Recognition 2020; official code xuebinqin/U-2-Net <c>model/u2net.py</c>): a two-level nested U-structure of
/// residual U-blocks with six side outputs and their fused map.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The layer names follow the official state dict (<c>stage1.rebnconvin.conv_s1.weight</c>, <c>side1.weight</c>,
/// <c>outconv.weight</c>, ...), so the released <c>u2net.pth</c> / <c>u2netp.pth</c> load with
/// <see cref="LoadTorchWeights"/>. Pooling is <c>MaxPool2d(2, 2, ceil_mode=True)</c> and upsampling bilinear
/// (<c>align_corners=False</c>), as in the reference. BatchNorm runs with its running statistics (evaluation mode).
/// </para>
/// <para><b>For Beginners:</b> U²-Net looks at an image and paints the most noticeable object white and everything
/// else black. rembg uses it to cut objects out of photos.</para>
/// </remarks>
public class U2Net<T> : IParameterSource<T>
{
    private readonly U2NetConfiguration _configuration;
    private readonly ResidualUBlock<T>[] _encoder;
    private readonly ResidualUBlock<T>[] _decoder;
    private readonly Tensor<T>[] _sideWeight;
    private readonly Tensor<T>[] _sideBias;
    private readonly int[] _sideInputs;
    private Tensor<T> _fuseWeight;
    private Tensor<T> _fuseBias;

    /// <summary>Creates a U²-Net with PyTorch's default initialisation.</summary>
    /// <param name="configuration">Channel widths; null is the full U²-Net.</param>
    /// <param name="seed">Initialisation seed; null uses a secure random source.</param>
    public U2Net(U2NetConfiguration? configuration = null, int? seed = null)
    {
        _configuration = configuration ?? U2NetConfiguration.Full();
        _configuration.Validate();
        var c = _configuration;
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        int[] heights = { 7, 6, 5, 4, 4, 4 };
        _encoder = new ResidualUBlock<T>[6];
        int inputs = c.InputChannels;
        for (int s = 0; s < 6; s++)
        {
            _encoder[s] = new ResidualUBlock<T>(heights[s], s >= 4, inputs, c.EncoderMiddle[s], c.EncoderOutput[s], random);
            inputs = c.EncoderOutput[s];
        }

        // stage5d (RSU-4F), stage4d (RSU-4), stage3d (RSU-5), stage2d (RSU-6), stage1d (RSU-7).
        _decoder = new ResidualUBlock<T>[5];
        int below = c.EncoderOutput[5];
        for (int d = 0; d < 5; d++)
        {
            int skip = c.EncoderOutput[4 - d];
            _decoder[d] = new ResidualUBlock<T>(DecoderHeight(d), d == 0, below + skip, c.DecoderMiddle[d], c.DecoderOutput[d], random);
            below = c.DecoderOutput[d];
        }

        // side1 .. side6 read stage1d, 2d, 3d, 4d, 5d and stage6.
        _sideInputs = new[] { c.DecoderOutput[4], c.DecoderOutput[3], c.DecoderOutput[2], c.DecoderOutput[1], c.DecoderOutput[0], c.EncoderOutput[5] };
        _sideWeight = new Tensor<T>[6];
        _sideBias = new Tensor<T>[6];
        for (int s = 0; s < 6; s++)
        {
            double bound = 1.0 / Math.Sqrt(_sideInputs[s] * 9.0);
            _sideWeight[s] = U2NetMath<T>.Uniform(new[] { c.OutputChannels, _sideInputs[s], 3, 3 }, bound, random);
            _sideBias[s] = U2NetMath<T>.Uniform(new[] { c.OutputChannels }, bound, random);
        }

        double fuseBound = 1.0 / Math.Sqrt(6.0 * c.OutputChannels);
        _fuseWeight = U2NetMath<T>.Uniform(new[] { c.OutputChannels, 6 * c.OutputChannels, 1, 1 }, fuseBound, random);
        _fuseBias = U2NetMath<T>.Uniform(new[] { c.OutputChannels }, fuseBound, random);
    }

    private static int DecoderHeight(int d) => d switch
    {
        0 => 4,
        1 => 4,
        2 => 5,
        3 => 6,
        _ => 7,
    };

    /// <summary>The channel widths.</summary>
    public U2NetConfiguration Configuration => _configuration;

    internal IReadOnlyList<ResidualUBlock<T>> EncoderStages => _encoder;

    internal IReadOnlyList<ResidualUBlock<T>> DecoderStages => _decoder;

    /// <inheritdoc />
    /// <remarks>The trainable parameters (PyTorch's <c>parameters()</c>); BatchNorm running statistics are buffers.</remarks>
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var block in _encoder.Concat(_decoder))
                total += block.Convolutions().Sum(conv => conv.ParameterCount);
            int outputs = _configuration.OutputChannels;
            for (int s = 0; s < 6; s++)
                total += ((long)outputs * _sideInputs[s] * 9) + outputs;
            return total + ((long)outputs * 6 * outputs) + outputs;
        }
    }

    /// <summary>
    /// The seven saliency maps <c>[d0, d1, ..., d6]</c> (d0 the fused output), each [B, out, H, W] after the sigmoid.
    /// </summary>
    /// <param name="image">The normalised input [B, in, H, W].</param>
    public IReadOnlyList<Tensor<T>> Forward(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Rank != 4 || image.Shape[1] != _configuration.InputChannels)
            throw new ArgumentException($"The input must be [B, {_configuration.InputChannels}, H, W].", nameof(image));
        var engine = AiDotNetEngine.Current;
        double eps = _configuration.BatchNormEpsilon;
        var stage = new Tensor<T>[6];
        var h = image;
        for (int s = 0; s < 6; s++)
        {
            stage[s] = _encoder[s].Forward(h, eps);
            if (s < 5) h = U2NetMath<T>.MaxPoolCeil(stage[s]);
        }

        var decoded = new Tensor<T>[5];
        var below = stage[5];
        for (int d = 0; d < 5; d++)
        {
            var skip = stage[4 - d];
            var up = U2NetMath<T>.UpsampleLike(below, skip);
            decoded[d] = _decoder[d].Forward(engine.TensorConcatenate(new[] { up, skip }, 1), eps);
            below = decoded[d];
        }

        var sources = new[] { decoded[4], decoded[3], decoded[2], decoded[1], decoded[0], stage[5] };
        var sides = new Tensor<T>[6];
        for (int s = 0; s < 6; s++)
        {
            var side = U2NetMath<T>.Convolve(sources[s], _sideWeight[s], _sideBias[s], 1);
            sides[s] = s == 0 ? side : U2NetMath<T>.UpsampleLike(side, sides[0]);
        }

        var fused = engine.Conv2D(engine.TensorConcatenate(sides, 1), _fuseWeight, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        fused = U2NetMath<T>.AddChannelBias(fused, _fuseBias);
        var outputs = new List<Tensor<T>> { engine.Sigmoid(fused) };
        foreach (var side in sides)
            outputs.Add(engine.Sigmoid(side));
        return outputs;
    }

    /// <summary>
    /// Loads an official state dict (<c>u2net.pth</c> for <see cref="U2NetConfiguration.Full"/>, <c>u2netp.pth</c>
    /// for <see cref="U2NetConfiguration.Small"/>).
    /// </summary>
    /// <param name="read">Reads a named tensor of the given shape as row-major values.</param>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        for (int s = 0; s < 6; s++)
            _encoder[s].Load(read, $"stage{s + 1}.");
        for (int d = 0; d < 5; d++)
            _decoder[d].Load(read, $"stage{5 - d}d.");
        int outputs = _configuration.OutputChannels;
        for (int s = 0; s < 6; s++)
        {
            _sideWeight[s] = U2NetMath<T>.Read(read, $"side{s + 1}.weight", new[] { outputs, _sideInputs[s], 3, 3 });
            _sideBias[s] = U2NetMath<T>.Read(read, $"side{s + 1}.bias", new[] { outputs });
        }

        _fuseWeight = U2NetMath<T>.Read(read, "outconv.weight", new[] { outputs, 6 * outputs, 1, 1 });
        _fuseBias = U2NetMath<T>.Read(read, "outconv.bias", new[] { outputs });
    }

    private IEnumerable<Tensor<T>> AllParameters()
    {
        foreach (var block in _encoder.Concat(_decoder))
        {
            foreach (var conv in block.Convolutions())
            {
                foreach (var p in conv.Parameters())
                    yield return p;
            }
        }

        for (int s = 0; s < 6; s++)
        {
            yield return _sideWeight[s];
            yield return _sideBias[s];
        }

        yield return _fuseWeight;
        yield return _fuseBias;
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var all = new List<T>();
        foreach (var p in AllParameters())
            all.AddRange(p.ToArray());
        return new Vector<T>(all.ToArray());
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));
        var values = parameters.ToArray();
        int offset = 0;
        var slices = new List<T[]>();
        foreach (var p in AllParameters())
        {
            var slice = new T[p.Length];
            Array.Copy(values, offset, slice, 0, p.Length);
            offset += p.Length;
            slices.Add(slice);
        }

        int i = 0;
        foreach (var block in _encoder.Concat(_decoder))
        {
            foreach (var conv in block.Convolutions())
            {
                conv.Weight = new Tensor<T>(conv.Weight._shape, new Vector<T>(slices[i++]));
                conv.Bias = new Tensor<T>(conv.Bias._shape, new Vector<T>(slices[i++]));
                conv.Gamma = new Tensor<T>(conv.Gamma._shape, new Vector<T>(slices[i++]));
                conv.Beta = new Tensor<T>(conv.Beta._shape, new Vector<T>(slices[i++]));
            }
        }

        for (int s = 0; s < 6; s++)
        {
            _sideWeight[s] = new Tensor<T>(_sideWeight[s]._shape, new Vector<T>(slices[i++]));
            _sideBias[s] = new Tensor<T>(_sideBias[s]._shape, new Vector<T>(slices[i++]));
        }

        _fuseWeight = new Tensor<T>(_fuseWeight._shape, new Vector<T>(slices[i++]));
        _fuseBias = new Tensor<T>(_fuseBias._shape, new Vector<T>(slices[i]));
    }
}

/// <summary>The tensor helpers U²-Net shares between its blocks.</summary>
internal static class U2NetMath<T>
{
    public static Tensor<T> Uniform(int[] shape, double bound, Random random)
    {
        int length = 1;
        foreach (int s in shape) length *= s;
        var values = new double[length];
        for (int i = 0; i < length; i++)
            values[i] = ((2.0 * random.NextDouble()) - 1.0) * bound;
        return new Tensor<double>(shape, new Vector<double>(values)).Cast<T>();
    }

    public static Tensor<T> Constant(int length, double value)
    {
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = value;
        return new Tensor<double>(new[] { length }, new Vector<double>(values)).Cast<T>();
    }

    public static Tensor<T> Read(Func<string, int[], double[]> read, string name, int[] shape)
    {
        var values = read(name, shape) ?? throw new InvalidOperationException($"The state dict has no '{name}'.");
        int length = 1;
        foreach (int s in shape) length *= s;
        if (values.Length != length)
            throw new InvalidOperationException($"'{name}' has {values.Length} values; [{string.Join(", ", shape)}] needs {length}.");
        return new Tensor<double>(shape, new Vector<double>(values)).Cast<T>();
    }

    public static Tensor<T> Convolve(Tensor<T> x, Tensor<T> weight, Tensor<T> bias, int dilation)
    {
        var y = AiDotNetEngine.Current.Conv2D(x, weight, new[] { 1, 1 }, new[] { dilation, dilation }, new[] { dilation, dilation });
        return AddChannelBias(y, bias);
    }

    public static Tensor<T> AddChannelBias(Tensor<T> y, Tensor<T> bias)
    {
        var engine = AiDotNetEngine.Current;
        var b = engine.Reshape(bias, new[] { 1, bias.Length, 1, 1 });
        return engine.TensorAdd(y, engine.TensorBroadcastTo(b, y._shape));
    }

    /// <summary><c>MaxPool2d(2, stride=2, ceil_mode=True)</c>: an odd edge is padded with -infinity first.</summary>
    public static Tensor<T> MaxPoolCeil(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        if (h % 2 == 1)
        {
            x = engine.TensorConcatenate(new[] { x, Floor(new[] { b, c, 1, w }) }, 2);
            h++;
        }

        if (w % 2 == 1)
            x = engine.TensorConcatenate(new[] { x, Floor(new[] { b, c, h, 1 }) }, 3);
        return engine.MaxPool2D(x, 2, 2, 0);
    }

    private static Tensor<T> Floor(int[] shape)
    {
        int length = shape[0] * shape[1] * shape[2] * shape[3];
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = -1e30;
        return new Tensor<double>(shape, new Vector<double>(values)).Cast<T>();
    }

    /// <summary>The reference's <c>_upsample_like</c>: bilinear resize to the target's size.</summary>
    public static Tensor<T> UpsampleLike(Tensor<T> source, Tensor<T> target)
    {
        if (source.Shape[2] == target.Shape[2] && source.Shape[3] == target.Shape[3])
            return source;
        return AiDotNetEngine.Current.Interpolate(
            source, new[] { target.Shape[2], target.Shape[3] }, InterpolateMode.Bilinear, false);
    }
}
