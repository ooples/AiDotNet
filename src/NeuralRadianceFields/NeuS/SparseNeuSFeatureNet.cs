namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// SparseNeuS's image feature pyramid (reference <c>models/featurenet.py</c> <c>FeatureNet</c>, the CasMVSNet FPN with
/// <c>InPlaceABN</c>), fused to one map per view as <c>obtain_pyramid_feature_maps</c> does.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Three stages of convolution + batch norm + leaky ReLU (8 channels at full resolution, 16 at 1/2, 32 at 1/4), a
/// top-down path with 1×1 lateral convolutions and ×2 bilinear (<c>align_corners=True</c>) upsampling, and 3×3
/// smoothing to 16 and 8 channels. The fused map concatenates the 1/4 (×4 upsampled), 1/2 (×2) and full-resolution
/// maps: 32 + 16 + 8 = 56 channels at the input resolution. Images must be <c>[V, 3, H, W]</c> with H and W divisible
/// by 4.
/// </para>
/// </remarks>
internal sealed class SparseNeuSFeatureNet<T>
{
    private readonly SparseNeuSConvBlock<T>[] _conv0;
    private readonly SparseNeuSConvBlock<T>[] _conv1;
    private readonly SparseNeuSConvBlock<T>[] _conv2;
    private readonly SparseNeuSConvBlock<T> _top;
    private readonly SparseNeuSConvBlock<T> _lat1;
    private readonly SparseNeuSConvBlock<T> _lat0;
    private readonly SparseNeuSConvBlock<T> _smooth1;
    private readonly SparseNeuSConvBlock<T> _smooth0;

    /// <summary>The fused output channel count (56).</summary>
    public const int FusedChannels = 56;

    /// <summary>Creates the network.</summary>
    /// <param name="random">The initialization generator.</param>
    public SparseNeuSFeatureNet(Random random)
    {
        SparseNeuSConvBlock<T> Abn(int i, int o, int k, int s, int p)
            => new(2, i, o, k, s, p, false, true, SparseNeuSActivation.LeakyReLU, random);
        SparseNeuSConvBlock<T> Plain(int i, int o, int k, int p)
            => new(2, i, o, k, 1, p, false, false, SparseNeuSActivation.None, random, convPath: "");
        _conv0 = new[] { Abn(3, 8, 3, 1, 1), Abn(8, 8, 3, 1, 1) };
        _conv1 = new[] { Abn(8, 16, 5, 2, 2), Abn(16, 16, 3, 1, 1), Abn(16, 16, 3, 1, 1) };
        _conv2 = new[] { Abn(16, 32, 5, 2, 2), Abn(32, 32, 3, 1, 1), Abn(32, 32, 3, 1, 1) };
        _top = Plain(32, 32, 1, 0);
        _lat1 = Plain(16, 32, 1, 0);
        _lat0 = Plain(8, 32, 1, 0);
        _smooth1 = Plain(32, 16, 3, 1);
        _smooth0 = Plain(32, 8, 3, 1);
    }

    private IEnumerable<KeyValuePair<string, SparseNeuSConvBlock<T>>> Blocks()
    {
        // ConvBnReLU's convolution is ".conv" and its InPlaceABN ".bn"; Sequential children are indexed.
        for (int i = 0; i < _conv0.Length; i++) yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("conv0." + i, _conv0[i]);
        for (int i = 0; i < _conv1.Length; i++) yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("conv1." + i, _conv1[i]);
        for (int i = 0; i < _conv2.Length; i++) yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("conv2." + i, _conv2[i]);
        yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("toplayer", _top);
        yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("lat1", _lat1);
        yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("lat0", _lat0);
        yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("smooth1", _smooth1);
        yield return new KeyValuePair<string, SparseNeuSConvBlock<T>>("smooth0", _smooth0);
    }

    /// <summary>The trainable tensors.</summary>
    public IEnumerable<Tensor<T>> Parameters => Blocks().SelectMany(b => b.Value.Parameters);

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => Blocks().Sum(b => b.Value.ParameterCount);

    /// <summary>The parameters under <paramref name="prefix"/>.</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters(string prefix)
        => Blocks().SelectMany(b => b.Value.NamedParameters(prefix + b.Key));

    /// <summary>Loads one named parameter; returns whether the name belongs to this network.</summary>
    public bool TryLoad(string prefix, string name, Tensor<T> value)
    {
        foreach (var block in Blocks())
            if (block.Value.TryLoad(prefix + block.Key, name, value)) return true;
        return false;
    }

    /// <summary>The fused 56-channel feature maps, <c>[V, 56, H, W]</c>, of <c>[V, 3, H, W]</c> images.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="images">The images, colors in [0, 1].</param>
    /// <param name="training">Whether batch normalization uses batch statistics.</param>
    public Tensor<T> Forward(IEngine engine, Tensor<T> images, bool training)
    {
        if (images.Shape.Length != 4 || images.Shape[1] != 3)
            throw new ArgumentException("Images must be [V, 3, H, W].", nameof(images));
        int h = images.Shape[2], w = images.Shape[3];
        if (h % 4 != 0 || w % 4 != 0) throw new ArgumentException("Image height and width must be divisible by 4.", nameof(images));

        var conv0 = Run(engine, _conv0, images, training);
        var conv1 = Run(engine, _conv1, conv0, training);
        var conv2 = Run(engine, _conv2, conv1, training);
        var feat2 = _top.Forward(engine, conv2, training);
        var feat1 = engine.TensorAdd(Upsample(engine, feat2, h / 2, w / 2), _lat1.Forward(engine, conv1, training));
        var feat0 = engine.TensorAdd(Upsample(engine, feat1, h, w), _lat0.Forward(engine, conv0, training));
        feat1 = _smooth1.Forward(engine, feat1, training);
        feat0 = _smooth0.Forward(engine, feat0, training);
        return engine.TensorConcatenate(new[] { Upsample(engine, feat2, h, w), Upsample(engine, feat1, h, w), feat0 }, 1);
    }

    private static Tensor<T> Run(IEngine engine, SparseNeuSConvBlock<T>[] blocks, Tensor<T> x, bool training)
    {
        foreach (var block in blocks) x = block.Forward(engine, x, training);
        return x;
    }

    private static Tensor<T> Upsample(IEngine engine, Tensor<T> x, int height, int width)
        => engine.Interpolate(x, new[] { height, width }, InterpolateMode.Bilinear, true);
}
