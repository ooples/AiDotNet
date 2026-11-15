namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// SparseNeuS's generalizable color network (reference <c>models/rendering_network.py</c>
/// <c>GeneralRenderingNetwork</c>, after IBRNet): the color of a 3-D point as a learned blend of the colors it
/// projects to in the source views.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Per point and source view the input is the sampled color and 56-channel image feature, shifted by an embedding of
/// the ray-direction difference. Views are pooled into a weighted mean and variance (anti-alias pooling: weights
/// <c>exp(|s|(cos − 1)) − min</c>, s = 0.2 learned), concatenated back with each view's feature and the point's
/// 16-channel geometry feature, and passed through the base, visibility and blending MLPs (ELU). The blending logits
/// of views that do not see the point are masked out before the softmax over views. All layers are plain
/// <c>nn.Linear</c>; the base, visibility and blending MLPs use Kaiming-normal weights and zero biases.
/// </para>
/// </remarks>
internal sealed class SparseNeuSRenderingNetwork<T>
{
    private readonly NeuSLinear<T>[] _rayDir;
    private readonly NeuSLinear<T>[] _base;
    private readonly NeuSLinear<T>[] _vis;
    private readonly NeuSLinear<T>[] _vis2;
    private readonly NeuSLinear<T>[] _rgb;
    private readonly int _geometryChannels;
    private readonly int _renderingChannels;

    /// <summary>Creates the network.</summary>
    /// <param name="geometryChannels">The geometry feature width (<c>in_geometry_feat_ch</c> = 16).</param>
    /// <param name="renderingChannels">The image feature width (<c>in_rendering_feat_ch</c> = 56).</param>
    /// <param name="antiAliasPooling">Whether to use anti-alias pooling (true).</param>
    /// <param name="random">The initialization generator.</param>
    public SparseNeuSRenderingNetwork(int geometryChannels, int renderingChannels, bool antiAliasPooling, Random random)
    {
        _geometryChannels = geometryChannels;
        _renderingChannels = renderingChannels;
        AntiAliasPooling = antiAliasPooling;
        int feat = renderingChannels + 3;
        _rayDir = new[] { Default(4, 16, random), Default(16, feat, random) };
        _base = new[] { Kaiming(feat * 3 + geometryChannels, 64, random), Kaiming(64, 32, random) };
        _vis = new[] { Kaiming(32, 32, random), Kaiming(32, 33, random) };
        _vis2 = new[] { Kaiming(32, 32, random), Kaiming(32, 1, random) };
        _rgb = new[] { Kaiming(32 + 1 + 4, 16, random), Kaiming(16, 8, random), Kaiming(8, 1, random) };
        PoolingSharpness = NeuSHost<T>.FromDoubles(new[] { 0.2 }, 1);
    }

    /// <summary>Whether anti-alias pooling is on.</summary>
    public bool AntiAliasPooling { get; }

    /// <summary>The anti-alias pooling sharpness s (<c>self.s</c>, one element).</summary>
    public Tensor<T> PoolingSharpness { get; private set; }

    private IEnumerable<KeyValuePair<string, NeuSLinear<T>>> Linears()
    {
        // nn.Sequential indices: linear layers sit at 0, 2, 4 with activations between.
        for (int i = 0; i < _rayDir.Length; i++) yield return new KeyValuePair<string, NeuSLinear<T>>("ray_dir_fc." + (2 * i), _rayDir[i]);
        for (int i = 0; i < _base.Length; i++) yield return new KeyValuePair<string, NeuSLinear<T>>("base_fc." + (2 * i), _base[i]);
        for (int i = 0; i < _vis.Length; i++) yield return new KeyValuePair<string, NeuSLinear<T>>("vis_fc." + (2 * i), _vis[i]);
        for (int i = 0; i < _vis2.Length; i++) yield return new KeyValuePair<string, NeuSLinear<T>>("vis_fc2." + (2 * i), _vis2[i]);
        for (int i = 0; i < _rgb.Length; i++) yield return new KeyValuePair<string, NeuSLinear<T>>("rgb_fc." + (2 * i), _rgb[i]);
    }

    /// <summary>The trainable tensors.</summary>
    public IEnumerable<Tensor<T>> Parameters
        => Linears().SelectMany(l => l.Value.Parameters).Concat(AntiAliasPooling ? new[] { PoolingSharpness } : Array.Empty<Tensor<T>>());

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => Linears().Sum(l => l.Value.ParameterCount) + (AntiAliasPooling ? 1 : 0);

    /// <summary>The parameters under <paramref name="prefix"/>.</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters(string prefix)
    {
        foreach (var l in Linears())
            foreach (var p in l.Value.NamedParameters(prefix + l.Key)) yield return p;
        if (AntiAliasPooling) yield return new KeyValuePair<string, Tensor<T>>(prefix + "s", PoolingSharpness);
    }

    /// <summary>Loads one named parameter; returns whether the name belongs to this network.</summary>
    public bool TryLoad(string prefix, string name, Tensor<T> value)
    {
        if (AntiAliasPooling && name == prefix + "s")
        {
            if (value is null || value.Length != 1) throw new ArgumentException("s is a single value.", nameof(value));
            PoolingSharpness = value.Reshape(1);
            return true;
        }

        foreach (var l in Linears())
            if (l.Value.TryLoad(prefix + l.Key, name, value)) return true;
        return false;
    }

    /// <summary>Blends the source views' colors at <c>P</c> points.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="geometry">Geometry features, <c>[P, G]</c>.</param>
    /// <param name="rgbFeatures">Per point and view, color then image feature, <c>[P·V, 3 + C]</c> (point-major).</param>
    /// <param name="rayDiff">Per point and view, the direction difference and its dot product, <c>[P·V, 4]</c> (host).</param>
    /// <param name="mask">Per point and view, whether the view sees the point, <c>[P·V]</c> (host 0/1).</param>
    /// <param name="views">V.</param>
    /// <returns>The blended colors, <c>[P, 3]</c>.</returns>
    public Tensor<T> Forward(IEngine engine, Tensor<T> geometry, Tensor<T> rgbFeatures, double[] rayDiff, double[] mask, int views)
    {
        var e = engine;
        int pv = mask.Length;
        int points = pv / views;
        int feat = _renderingChannels + 3;
        var maskT = NeuSHost<T>.FromDoubles(mask, points, views, 1);
        var rayDiffT = NeuSHost<T>.FromDoubles(rayDiff, pv, 4);

        var geometryExpanded = e.TensorBroadcastTo(e.Reshape(geometry, new[] { points, 1, _geometryChannels }),
            new[] { points, views, _geometryChannels });
        var directionFeat = Mlp(e, _rayDir, rayDiffT, eluLast: true);
        var rgbIn = e.Reshape(e.TensorSlice(rgbFeatures, new[] { 0, 0 }, new[] { pv, 3 }), new[] { points, views, 3 });
        var rgbFeat = e.Reshape(e.TensorAdd(rgbFeatures, directionFeat), new[] { points, views, feat });

        Tensor<T> weight;
        if (AntiAliasPooling)
        {
            var dot = NeuSHost<T>.FromDoubles(ColumnOf(rayDiff, 4, 3), points, views, 1);
            var expDot = e.TensorExp(e.TensorMultiply(e.TensorAddScalar(dot, NeuSHost<T>.Scalar(-1.0)),
                e.Reshape(e.TensorAbs(PoolingSharpness), new[] { 1, 1, 1 })));
            var minimum = e.TensorNegate(e.ReduceMax(e.TensorNegate(expDot), new[] { 1 }, true));
            weight = e.TensorMultiply(e.TensorSubtract(expDot, minimum), maskT);
        }
        else
        {
            weight = maskT;
        }

        weight = e.TensorDivide(weight, e.TensorAddScalar(e.ReduceSum(weight, new[] { 1 }, true), NeuSHost<T>.Epsilon(1e-8)));
        var mean = e.ReduceSum(e.TensorMultiply(rgbFeat, weight), new[] { 1 }, true);
        var centered = e.TensorSubtract(rgbFeat, mean);
        var variance = e.ReduceSum(e.TensorMultiply(weight, e.TensorMultiply(centered, centered)), new[] { 1 }, true);
        var global = e.TensorBroadcastTo(e.TensorConcatenate(new[] { mean, variance }, 2), new[] { points, views, 2 * feat });
        var x = e.TensorConcatenate(new[] { geometryExpanded, global, rgbFeat }, 2);
        x = Reshape3(e, Mlp(e, _base, Flat(e, x), eluLast: true), points, views);

        var xVis = Reshape3(e, Mlp(e, _vis, Flat(e, e.TensorMultiply(x, weight)), eluLast: true), points, views);
        var xRes = e.TensorSlice(xVis, new[] { 0, 0, 0 }, new[] { points, views, 32 });
        var vis = e.TensorMultiply(e.TensorSigmoid(e.TensorSlice(xVis, new[] { 0, 0, 32 }, new[] { points, views, 1 })), maskT);
        x = e.TensorAdd(x, xRes);
        vis = e.TensorMultiply(Reshape3(e, e.TensorSigmoid(Mlp(e, _vis2, Flat(e, e.TensorMultiply(x, vis)), eluLast: false)), points, views), maskT);

        var rgbInput = e.TensorConcatenate(new[] { x, vis, e.Reshape(rayDiffT, new[] { points, views, 4 }) }, 2);
        var logits = Reshape3(e, Mlp(e, _rgb, Flat(e, rgbInput), eluLast: false), points, views);
        // masked_fill(mask == 0, -1e9), as a blend so an unseen view never multiplies an infinity.
        var fill = NeuSHost<T>.Scalar(LargeNegative());
        var inverseMask = e.TensorAddScalar(e.TensorNegate(maskT), NeuSHost<T>.Scalar(1.0));
        logits = e.TensorAdd(e.TensorMultiply(logits, maskT), e.TensorMultiplyScalar(inverseMask, fill));
        var blend = e.TensorSoftmax(logits, 1);
        return e.ReduceSum(e.TensorMultiply(rgbIn, blend), new[] { 1 }, false);
    }

    private static double LargeNegative()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double value = ops.ToDouble(ops.FromDouble(-1e9));
        return double.IsInfinity(value) ? -6.0e4 : -1e9;
    }

    private static Tensor<T> Mlp(IEngine e, NeuSLinear<T>[] layers, Tensor<T> x, bool eluLast)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            x = layers[i].Forward(e, x);
            if (i < layers.Length - 1 || eluLast) x = e.ELU(x, 1.0);
        }

        return x;
    }

    private static Tensor<T> Flat(IEngine e, Tensor<T> x)
        => e.Reshape(x, new[] { x.Shape[0] * x.Shape[1], x.Shape[2] });

    private static Tensor<T> Reshape3(IEngine e, Tensor<T> x, int points, int views)
        => e.Reshape(x, new[] { points, views, x.Shape[1] });

    private static double[] ColumnOf(double[] values, int width, int column)
    {
        var result = new double[values.Length / width];
        for (int i = 0; i < result.Length; i++) result[i] = values[i * width + column];
        return result;
    }

    private static NeuSLinear<T> Default(int inDim, int outDim, Random random)
    {
        var (weight, bias) = NeuSLinear<T>.DefaultInit(inDim, outDim, random);
        return new NeuSLinear<T>(weight, bias, inDim, outDim, weightNorm: false);
    }

    private static NeuSLinear<T> Kaiming(int inDim, int outDim, Random random)
    {
        // nn.init.kaiming_normal_ (fan_in, gain √2) and a zero bias: the reference's weights_init.
        double std = Math.Sqrt(2.0 / inDim);
        var weight = new double[inDim * outDim];
        for (int i = 0; i < weight.Length; i++) weight[i] = std * NeuSHost<T>.NextGaussian(random);
        return new NeuSLinear<T>(weight, new double[outDim], inDim, outDim, weightNorm: false);
    }
}
