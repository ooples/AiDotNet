using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// IART's ImplicitWarpModule (Xu et al. 2024; kai422/IART, archs/implicit_alignment.py): instead of interpolating the
/// propagated features at the flow-moved point, each current pixel queries the 2x2 propagated pixels around it with a
/// single cross-attention, so the resampling weights are learned.
/// </summary>
/// <remarks>
/// <para>
/// The point the flow lands on is split into its integer corner and sub-pixel offset. The four corner neighbours are
/// gathered with duplicate (clamped) padding and get a sine code of their place in the 2x2 window; the current pixel gets
/// a sine code of the sub-pixel offset. Both codes use temperature pe_temp = 0.01 and are added to the features (pe_dim
/// equals the feature width, so the release's repeat is a no-op). q, k and v are biased linear maps, the heads attend over
/// the four neighbours, and the result is returned without an output projection (aux_loss_out is off).
/// </para>
/// <para>
/// Gradients follow the release: through the gathered features, the query, and the flow via the offset code; the floor
/// and the neighbour indices are piecewise constant. The module is outside SwinIRFM, so its linear layers keep PyTorch's
/// default initialization.
/// </para>
/// </remarks>
internal sealed class IartImplicitWarp<T> : IPsrtFrameAligner<T>
{
    /// <summary>The neighbourhood side (window_size = 2).</summary>
    internal const int Window = 2;

    private const int Neighbours = Window * Window;
    private readonly int _q, _k, _v, _dim, _heads;
    private readonly double[] _frequency;
    private readonly double[] _windowCode;

    internal IartImplicitWarp(List<ILayer<T>> layers, int dim, int heads, double temperature)
    {
        if (dim % 2 != 0 || dim % heads != 0)
            throw new ArgumentException("The implicit warp needs an even width divisible by its heads.", nameof(dim));
        _dim = dim;
        _heads = heads;
        _q = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _k = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _v = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));

        int features = dim / 2;
        _frequency = new double[features];
        for (int i = 0; i < features; i++) _frequency[i] = 1.0 / Math.Pow(temperature, 2.0 * (i / 2) / features);

        // get_sine_position_encoding over the 2x2 window, normalized so the last row/column sits at 2*pi.
        _windowCode = new double[Neighbours * dim];
        for (int k = 0; k < Neighbours; k++)
        {
            double y = k / Window / (Window - 1 + 1e-6) * 2 * Math.PI, x = k % Window / (Window - 1 + 1e-6) * 2 * Math.PI;
            for (int i = 0; i < features; i++)
            {
                _windowCode[k * dim + i] = Code(y, i);
                _windowCode[k * dim + features + i] = Code(x, i);
            }
        }
    }

    // Channel i of a sine code: sin for even i, cos for odd, at frequency 1 / dim_t[i].
    private double Code(double value, int i) => i % 2 == 0 ? Math.Sin(value * _frequency[i]) : Math.Cos(value * _frequency[i]);

    /// <summary>The sine code of the 2x2 window, [4, C] in (row, column) order.</summary>
    internal IReadOnlyList<double> WindowCode => _windowCode;

    public Tensor<T> Align(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> propagated, Tensor<T> current, Tensor<T> flow)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = current.Shape[0], c = current.Shape[1], h = current.Shape[2], w = current.Shape[3];
        if (c != _dim) throw new ArgumentException($"The implicit warp is {_dim} wide; the features have {c} channels.", nameof(current));
        int pixels = n * h * w;

        // Integer corners and the constant part of the offset, from the flow's values (floor has no gradient).
        var index = new Tensor<int>(new[] { pixels * Neighbours });
        var restY = new Tensor<T>(new[] { pixels, 1 });
        var restX = new Tensor<T>(new[] { pixels, 1 });
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int p = (b * h + y) * w + x;
                    double fx = x + ops.ToDouble(flow[((b * 2) * h + y) * w + x]);
                    double fy = y + ops.ToDouble(flow[((b * 2 + 1) * h + y) * w + x]);
                    int iy = (int)Math.Floor(fy), ix = (int)Math.Floor(fx);
                    restY[p] = ops.FromDouble(y - iy);
                    restX[p] = ops.FromDouble(x - ix);
                    for (int k = 0; k < Neighbours; k++)
                        index[p * Neighbours + k] = (b * h + Math.Min(Math.Max(iy + k / Window, 0), h - 1)) * w
                            + Math.Min(Math.Max(ix + k % Window, 0), w - 1);
                }

        var rows = engine.Reshape(engine.TensorPermute(propagated, new[] { 0, 2, 3, 1 }), new[] { pixels, c });
        var windowCode = new Tensor<T>(new[] { pixels, Neighbours, c });
        for (int p = 0; p < pixels; p++)
            for (int j = 0; j < Neighbours * c; j++) windowCode[p * Neighbours * c + j] = ops.FromDouble(_windowCode[j]);
        var neighbours = engine.TensorAdd(engine.Reshape(engine.TensorIndexSelectDiff(rows, index, 0), new[] { pixels, Neighbours, c }), windowCode);

        // Sub-pixel offset = (pixel + flow) - floor, differentiable in the flow.
        var offsetY = engine.TensorAdd(engine.Reshape(engine.TensorNarrow(flow, 1, 1, 1), new[] { pixels, 1 }), restY);
        var offsetX = engine.TensorAdd(engine.Reshape(engine.TensorNarrow(flow, 1, 0, 1), new[] { pixels, 1 }), restX);
        var query = engine.TensorAdd(engine.Reshape(engine.TensorPermute(current, new[] { 0, 2, 3, 1 }), new[] { pixels, c }),
            engine.TensorConcatenate(new[] { OffsetCode(engine, offsetY, pixels), OffsetCode(engine, offsetX, pixels) }, 1));

        int hd = _dim / _heads;
        var q = engine.Reshape(engine.TensorPermute(engine.Reshape(layers[_q].Forward(query), new[] { pixels, 1, _heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { pixels * _heads, 1, hd });
        var k2 = engine.Reshape(engine.TensorPermute(engine.Reshape(layers[_k].Forward(neighbours), new[] { pixels, Neighbours, _heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { pixels * _heads, Neighbours, hd });
        var v = engine.Reshape(engine.TensorPermute(engine.Reshape(layers[_v].Forward(neighbours), new[] { pixels, Neighbours, _heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { pixels * _heads, Neighbours, hd });
        var attn = engine.TensorBatchMatMul(engine.TensorMultiplyScalar(q, ops.FromDouble(Math.Pow(hd, -0.5))), engine.TensorPermute(k2, new[] { 0, 2, 1 }));
        attn = engine.TensorSoftmax(attn, 2);
        var output = engine.Reshape(engine.TensorBatchMatMul(attn, v), new[] { n, h, w, _dim });
        return engine.TensorPermute(output, new[] { 0, 3, 1, 2 });
    }

    // get_sine_position_encoding_points for one axis: the offset over (window + eps) times 2*pi, then the sine code.
    private Tensor<T> OffsetCode(IEngine engine, Tensor<T> offset, int pixels)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int features = _dim / 2;
        var rate = new Tensor<T>(new[] { 1, features });
        var even = new Tensor<T>(new[] { pixels, features });
        var odd = new Tensor<T>(new[] { pixels, features });
        for (int i = 0; i < features; i++) rate[i] = ops.FromDouble(2 * Math.PI / (Window + 1e-6) * _frequency[i]);
        for (int p = 0; p < pixels; p++)
            for (int i = 0; i < features; i++)
            {
                even[p * features + i] = i % 2 == 0 ? ops.One : ops.Zero;
                odd[p * features + i] = i % 2 == 0 ? ops.Zero : ops.One;
            }
        var angle = engine.TensorMatMul(offset, rate);
        return engine.TensorAdd(engine.TensorMultiply(engine.TensorSin(angle), even), engine.TensorMultiply(engine.TensorCos(angle), odd));
    }
}
