namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The implicit function a Shap-E latent parameterizes: an MLP whose weights are read from the latent.
/// </summary>
/// <remarks>
/// <para>
/// Shap-E (Jun &amp; Nichol, 2023) represents every shape as the parameters of an implicit MLP that maps a
/// (positionally encoded) 3D point to an STF signed distance, a NeRF density and an RGB colour. Here the latent
/// vector is read, in order, as the weights of a two-hidden-layer SiLU MLP:
/// <c>W1 [E, H], b1 [H], W2 [H, H], b2 [H], W3 [H, 5], b3 [5]</c>, where <c>E = 3 + 6 L</c> is the width of the
/// positional encoding with <see cref="EncodingFrequencies"/> octaves and <c>H</c> is the widest hidden size whose
/// parameters fit in the latent. Weights are scaled by <c>1 / sqrt(fan_in)</c> so a unit-variance latent gives
/// unit-scale activations.
/// </para>
/// <para>
/// Outputs: <c>sdf = o0</c> (negative inside), <c>density = softplus(o1)</c>, <c>rgb = sigmoid(o2..o4)</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class ShapEImplicitFunction<T>
{
    /// <summary>Octaves of the positional encoding of the query point.</summary>
    internal const int EncodingFrequencies = 2;

    /// <summary>Outputs per point: SDF, density, and three colour channels.</summary>
    internal const int OutputDim = 5;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Tensor<T> _w1;
    private readonly Tensor<T> _b1;
    private readonly Tensor<T> _w2;
    private readonly Tensor<T> _b2;
    private readonly Tensor<T> _w3;
    private readonly Tensor<T> _b3;

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>The width of the positional encoding.</summary>
    internal static int EncodedDim => 3 + (6 * EncodingFrequencies);

    /// <summary>The hidden width the MLP uses.</summary>
    internal int HiddenDim { get; }

    /// <summary>
    /// The widest hidden size H with <c>H^2 + (E + 7) H + 5 &lt;= latentLength</c> (the MLP's parameter count).
    /// </summary>
    internal static int HiddenWidth(int latentLength)
    {
        double linear = EncodedDim + 7;
        int hidden = (int)Math.Floor((-linear + Math.Sqrt((linear * linear) - (4.0 * (OutputDim - latentLength)))) / 2.0);
        while (hidden > 0 && ParameterCount(hidden) > latentLength)
            hidden--;
        return hidden;
    }

    /// <summary>The number of latent values an MLP of hidden width <paramref name="hidden"/> reads.</summary>
    internal static int ParameterCount(int hidden)
        => (EncodedDim * hidden) + hidden + (hidden * hidden) + hidden + (hidden * OutputDim) + OutputDim;

    /// <summary>
    /// Reads the MLP weights from <paramref name="latent"/> (any shape; read in flat order).
    /// </summary>
    internal ShapEImplicitFunction(Tensor<T> latent)
    {
        int length = latent.Length;
        HiddenDim = HiddenWidth(length);
        if (HiddenDim < 1)
            throw new ArgumentException(
                $"A Shap-E latent needs at least {ParameterCount(1)} values to parameterize its implicit MLP; got {length}.",
                nameof(latent));

        var flat = Engine.Reshape(latent, new[] { length });
        int offset = 0;
        Tensor<T> Take(int rows, int columns, int fanIn)
        {
            var block = Engine.Reshape(Engine.TensorNarrow(flat, 0, offset, rows * columns), new[] { rows, columns });
            offset += rows * columns;
            return Engine.TensorMultiplyScalar(block, NumOps.FromDouble(1.0 / Math.Sqrt(fanIn)));
        }

        int h = HiddenDim;
        _w1 = Take(EncodedDim, h, EncodedDim);
        _b1 = Take(1, h, EncodedDim);
        _w2 = Take(h, h, h);
        _b2 = Take(1, h, h);
        _w3 = Take(h, OutputDim, h);
        _b3 = Take(1, OutputDim, h);
    }

    /// <summary>
    /// The positional encoding [N, E] of <paramref name="points"/> [N, 3]: the coordinates followed, for every
    /// octave k, by (sin, cos)(2^k pi x) of each coordinate.
    /// </summary>
    internal static Tensor<T> Encode(Tensor<T> points)
        => NeRFPositionalEncoding.Encode(Engine, points, EncodingFrequencies);

    /// <summary>The raw MLP outputs [N, 5] at <paramref name="points"/> [N, 3].</summary>
    internal Tensor<T> Evaluate(Tensor<T> points)
    {
        var h1 = Engine.TensorSiLU(AddBias(Engine.TensorMatMul(Encode(points), _w1), _b1));
        var h2 = Engine.TensorSiLU(AddBias(Engine.TensorMatMul(h1, _w2), _b2));
        return AddBias(Engine.TensorMatMul(h2, _w3), _b3);
    }

    private static Tensor<T> AddBias(Tensor<T> rows, Tensor<T> bias)
    {
        return Engine.TensorAdd(rows, Engine.TensorBroadcastTo(bias, new[] { rows.Shape[0], bias.Shape[1] }));
    }

    /// <summary>The signed distance o0 [N] (negative inside) of the MLP outputs [N, 5].</summary>
    internal static Tensor<T> SignedDistance(Tensor<T> outputs)
        => Engine.Reshape(Engine.TensorNarrow(outputs, 1, 0, 1), new[] { outputs.Shape[0] });

    /// <summary>The density softplus(o1) [N, 1].</summary>
    internal static Tensor<T> Density(Tensor<T> outputs) => Engine.Softplus(Engine.TensorNarrow(outputs, 1, 1, 1));

    /// <summary>The colour sigmoid(o2..o4) [N, 3].</summary>
    internal static Tensor<T> Colour(Tensor<T> outputs) => Engine.TensorSigmoid(Engine.TensorNarrow(outputs, 1, 2, 3));

    /// <summary>
    /// The points of the regular grid (origin + i step) on each axis, i &lt; <paramref name="resolution"/>,
    /// as [resolution^3, 3] in x-major order.
    /// </summary>
    internal static Tensor<T> GridPoints(int resolution, double origin, double step)
    {
        var axis = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, resolution, 1), NumOps.FromDouble(step)),
            NumOps.FromDouble(origin));
        var shape = new[] { resolution, resolution, resolution, 1 };
        var xs = Engine.TensorBroadcastTo(Engine.Reshape(axis, new[] { resolution, 1, 1, 1 }), shape);
        var ys = Engine.TensorBroadcastTo(Engine.Reshape(axis, new[] { 1, resolution, 1, 1 }), shape);
        var zs = Engine.TensorBroadcastTo(Engine.Reshape(axis, new[] { 1, 1, resolution, 1 }), shape);
        return Engine.Reshape(
            Engine.TensorConcatenate(new[] { xs, ys, zs }, 3), new[] { resolution * resolution * resolution, 3 });
    }
}
