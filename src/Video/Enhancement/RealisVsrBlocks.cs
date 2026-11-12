using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// RealisVSR's Consistency-Preserved ControlNet (Zhao et al. 2025, Sec. 3.2): a branch of Wan DiT blocks that reads only the
/// embedded condition, x_CPC = T_cond(c), with the noisy latent left out of its input, and whose block outputs are added to the
/// main network's (Eq. 1): after main block i, X_i += γ·F_⌊i/r⌋ with r = ⌈L_main / L_CPC⌉ and γ a learned scale.
/// </summary>
/// <remarks>
/// The paper does not state the branch depth, its initialization or where in a block the feature is added. The branch shares the
/// main network's prompt, time modulation and rotary tables; its blocks are copies of the main network's first blocks (the
/// ControlNet convention the paper builds on, done by the model once its weights exist); T_cond is a new patch embedding of the
/// condition latent; and each feature is added to the main block's output.
/// </remarks>
internal sealed class RealisCpc<T>
{
    private readonly int _condition, _scale;
    private readonly WanDiTBlock<T>[] _blocks;

    /// <param name="layers">The model's layer list.</param>
    /// <param name="dim">The DiT width.</param>
    /// <param name="heads">The attention heads.</param>
    /// <param name="ffnDim">The feed-forward width.</param>
    /// <param name="depth">L_CPC, the branch's block count.</param>
    /// <param name="epsilon">The norms' ε.</param>
    /// <param name="conditionScale">γ's initial value (the paper's condition scale).</param>
    internal RealisCpc(List<ILayer<T>> layers, int dim, int heads, int ffnDim, int depth, double epsilon, double conditionScale)
    {
        FirstLayer = layers.Count;
        _condition = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _blocks = new WanDiTBlock<T>[depth];
        for (int i = 0; i < depth; i++) _blocks[i] = new WanDiTBlock<T>(layers, dim, heads, ffnDim, 0, epsilon);
        _scale = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(1, conditionScale));
        EndLayer = layers.Count;
    }

    /// <summary>The branch's blocks.</summary>
    internal IReadOnlyList<WanDiTBlock<T>> Blocks => _blocks;

    /// <summary>The index of the branch's first layer.</summary>
    internal int FirstLayer { get; }

    /// <summary>One past the index of the branch's last layer.</summary>
    internal int EndLayer { get; }

    /// <summary>The scaled features γ·F_j [B, L, D] of condition latents [B, C, F, H, W].</summary>
    internal IReadOnlyList<Tensor<T>> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> condition, WanConditioning<T> c)
    {
        var x = layers[_condition].Forward(WanDiT<T>.Patches(engine, condition));
        var gamma = RealViformerBlocks.Read(layers, _scale);
        var features = new Tensor<T>[_blocks.Length];
        for (int j = 0; j < _blocks.Length; j++)
        {
            x = _blocks[j].Forward(engine, layers, x, c.Context, c.TimeModulation, 0, null, c.Cos, c.Sin);
            features[j] = engine.TensorMultiply(x, engine.TensorBroadcastTo(engine.Reshape(gamma, new[] { 1, 1, 1 }), x.Shape.ToArray()));
        }
        return features;
    }
}

/// <summary>
/// RealisVSR's High-Frequency Rectified Diffusion Loss terms beyond the flow-matching MSE (Sec. 3.3): the Haar wavelet loss on
/// the velocity (Eq. 4) and the HOG loss on it (Eq. 5).
/// </summary>
internal static class RealisLoss
{
    /// <summary>
    /// Eq. 4: Σ_k w_k‖f_k(v) − f_k(v*)‖² over one level of the 2D Haar transform (L = [1, 1]/√2, H = [−1, 1]/√2) of each latent
    /// frame and channel, with weights {LL, LH, HL, HH}; each band's squared error is averaged over its elements.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="prediction">The predicted velocity [B, C, F, H, W], H and W even.</param>
    /// <param name="target">The target velocity, the same shape.</param>
    /// <param name="weights">The four band weights in LL, LH, HL, HH order.</param>
    internal static Tensor<T> Wavelet<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target, IReadOnlyList<double> weights)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // The transform is linear, so the band errors are the bands of the error.
        var d = engine.TensorSubtract(prediction, target);
        int b = d.Shape[0], c = d.Shape[1], f = d.Shape[2], h = d.Shape[3] / 2, w = d.Shape[4] / 2;
        var grid = engine.Reshape(d, new[] { b, c, f, h, 2, w, 2 });
        Tensor<T> At(int row, int column) => engine.TensorNarrow(engine.TensorNarrow(grid, 4, row, 1), 6, column, 1);
        var p00 = At(0, 0);
        var p01 = At(0, 1);
        var p10 = At(1, 0);
        var p11 = At(1, 1);
        Tensor<T> Band(int sign01, int sign10, int sign11)
        {
            var sum = p00;
            sum = sign01 > 0 ? engine.TensorAdd(sum, p01) : engine.TensorSubtract(sum, p01);
            sum = sign10 > 0 ? engine.TensorAdd(sum, p10) : engine.TensorSubtract(sum, p10);
            sum = sign11 > 0 ? engine.TensorAdd(sum, p11) : engine.TensorSubtract(sum, p11);
            return engine.TensorMultiplyScalar(sum, ops.FromDouble(0.5));
        }
        // Rows then columns: LL = (a + b + c + d)/2, LH high across the width, HL high down the height, HH both. The
        // high-pass kernel's sign only flips a band's sign, which its squared error does not see.
        var bands = new[] { Band(1, 1, 1), Band(-1, 1, -1), Band(1, -1, -1), Band(-1, -1, 1) };
        Tensor<T>? loss = null;
        for (int k = 0; k < bands.Length; k++)
        {
            var squared = engine.TensorSquare(bands[k]);
            var term = engine.TensorMultiplyScalar(engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), false),
                ops.FromDouble(weights[k]));
            loss = loss is null ? term : engine.TensorAdd(loss, term);
        }
        return loss ?? new Tensor<T>(new[] { 1 });
    }

    /// <summary>
    /// Eq. 5: the squared error between the HOG descriptors of the predicted and the target velocity, per latent channel and
    /// frame, as scikit-image's <c>hog</c> computes them: central-difference gradients (zero on the border rows and columns),
    /// unsigned orientations in <paramref name="bins"/> equal bins over [0°, 180°), each cell's histogram the bin-wise mean
    /// magnitude, and overlapping blocks of cells normalized by L2-Hys (ε 1e-5, clip 0.2).
    /// </summary>
    /// <remarks>
    /// The orientation bin of a pixel is a hard choice, so the descriptor's gradient flows through the magnitudes, as it does
    /// for any HOG. A plane too small for one block adds nothing.
    /// </remarks>
    internal static Tensor<T> Hog<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target, int bins, int cell, int block)
    {
        int b = prediction.Shape[0], c = prediction.Shape[1], f = prediction.Shape[2], h = prediction.Shape[3], w = prediction.Shape[4];
        int cellsY = h / cell, cellsX = w / cell, blocksY = cellsY - block + 1, blocksX = cellsX - block + 1;
        if (blocksY < 1 || blocksX < 1 || h < 3 || w < 3) return new Tensor<T>(new[] { 1 });
        Tensor<T> Planes(Tensor<T> x) => engine.Reshape(x, new[] { b * c * f, h, w });
        Tensor<T> expected;
        using (new NoGradScope<T>()) expected = Descriptor(engine, Planes(target), bins, cell, block, cellsY, cellsX, blocksY, blocksX);
        var actual = Descriptor(engine, Planes(prediction), bins, cell, block, cellsY, cellsX, blocksY, blocksX);
        var squared = engine.TensorSquare(engine.TensorSubtract(actual, expected));
        return engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), false);
    }

    // The HOG blocks [N, blocksY, blocksX, block²·bins] of planes [N, H, W].
    private static Tensor<T> Descriptor<T>(IEngine engine, Tensor<T> planes, int bins, int cell, int block,
        int cellsY, int cellsX, int blocksY, int blocksX)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = planes.Shape[0], h = planes.Shape[1], w = planes.Shape[2];
        // np.gradient-free central differences as skimage's _hog_channel_gradient: interior only, border rows/columns zero.
        Tensor<T> Pad(Tensor<T> inner, int axis)
        {
            var edge = inner.Shape.ToArray();
            edge[axis] = 1;
            var zero = new Tensor<T>(edge);
            return engine.TensorConcatenate(new[] { zero, inner, zero }, axis);
        }
        var gy = Pad(engine.TensorSubtract(engine.TensorNarrow(planes, 1, 2, h - 2), engine.TensorNarrow(planes, 1, 0, h - 2)), 1);
        var gx = Pad(engine.TensorSubtract(engine.TensorNarrow(planes, 2, 2, w - 2), engine.TensorNarrow(planes, 2, 0, w - 2)), 2);
        int hc = cellsY * cell, wc = cellsX * cell;
        gy = engine.TensorNarrow(engine.TensorNarrow(gy, 1, 0, hc), 2, 0, wc);
        gx = engine.TensorNarrow(engine.TensorNarrow(gx, 1, 0, hc), 2, 0, wc);
        // A tiny floor keeps the square root's derivative finite where both gradients vanish.
        var magnitude = engine.TensorSqrt(engine.TensorAddScalar(engine.TensorAdd(engine.TensorSquare(gx), engine.TensorSquare(gy)),
            ops.FromDouble(1e-12)));

        // The orientation bins, read on the host: atan2(gy, gx) in degrees modulo 180, cut into equal bins.
        var mask = new Tensor<T>(new[] { n, hc, wc, bins });
        using (new NoGradScope<T>())
        {
            var gyValues = gy.IsContiguous ? gy : gy.Contiguous();
            var gxValues = gx.IsContiguous ? gx : gx.Contiguous();
            for (int i = 0; i < n * hc * wc; i++)
            {
                double angle = Math.Atan2(ops.ToDouble(gyValues[i]), ops.ToDouble(gxValues[i])) * 180.0 / Math.PI;
                angle %= 180.0;
                if (angle < 0) angle += 180.0;
                int bin = Math.Min(bins - 1, (int)(angle / (180.0 / bins)));
                mask[i * bins + bin] = ops.One;
            }
        }
        var voted = engine.TensorMultiply(engine.TensorBroadcastTo(engine.Reshape(magnitude, new[] { n, hc, wc, 1 }), new[] { n, hc, wc, bins }), mask);
        var cells = engine.ReduceMean(engine.Reshape(voted, new[] { n, cellsY, cell, cellsX, cell, bins }), new[] { 2, 4 }, false);

        var parts = new List<Tensor<T>>();
        for (int dy = 0; dy < block; dy++)
            for (int dx = 0; dx < block; dx++)
                parts.Add(engine.TensorNarrow(engine.TensorNarrow(cells, 1, dy, blocksY), 2, dx, blocksX));
        var blocks = engine.TensorConcatenate(parts.ToArray(), 3);

        // L2-Hys: L2-normalize, clip at 0.2, L2-normalize again (ε = 1e-5 as skimage).
        Tensor<T> L2(Tensor<T> v)
        {
            var shape = v.Shape.ToArray();
            var norm = engine.TensorSqrt(engine.TensorAddScalar(engine.ReduceSum(engine.TensorSquare(v), new[] { 3 }, true), ops.FromDouble(1e-10)));
            return engine.TensorDivide(v, engine.TensorBroadcastTo(norm, shape));
        }
        return L2(engine.TensorClamp(L2(blocks), ops.Zero, ops.FromDouble(0.2)));
    }
}
