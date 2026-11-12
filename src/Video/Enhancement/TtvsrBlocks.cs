using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// TTVSRNet (Liu et al., CVPR 2022, "Learning Trajectory-Aware Transformer for Video Super-Resolution"; authors'
/// release researchmm/TTVSR, mmedit/models/backbones/sr_backbones/ttvsrnet.py): per-frame features from five residual
/// blocks, then backward and forward recurrent propagation in which every frame attends, along motion trajectories,
/// to tokens of earlier keyframes (every <c>frame_stride</c>-th frame) at three scales; the two directions are fused
/// with the frame's own features and reconstructed at x4.
/// </summary>
/// <remarks>
/// <para>
/// A trajectory is a location map on the 1/stride token grid: each keyframe starts one as the identity grid, and every
/// later step warps all of them by the 4x4-average-pooled flow (nearest sampling, border padding). Nearest sampling
/// passes no gradient to the map in PyTorch, so the maps are computed on the host and only the tokens they select carry
/// gradients. Tokens are stride x stride patches (scale 1), and 1.5x and 2x patches resampled to the same size (scales
/// 2 and 3); the attention picks, per token position, the keyframe whose trajectory token is most cosine-similar to
/// the current frame's, takes its three tokens, fuses them with a 3x3 convolution and scales by the similarity.
/// </para>
/// </remarks>
internal sealed class TtvsrBackbone<T>
{
    private readonly int _midChannels;
    private readonly int _stride;
    private readonly int _frameStride;
    private readonly int _ltamFusion;
    private readonly int _fusion;

    internal TtvsrBackbone(List<ILayer<T>> layers, int midChannels, int numBlocks, int stride, int frameStride)
    {
        _midChannels = midChannels;
        _stride = stride;
        _frameStride = frameStride;
        SpyNet = new SpyNetFlowBlock<T>(layers);
        Features = new ResidualTrunkBlock<T>(layers, midChannels, 5);
        _ltamFusion = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(midChannels, 3, null));
        Propagation = new ResidualTrunkBlock<T>(layers, midChannels, numBlocks);
        _fusion = layers.Count;
        layers.Add(BasicVsrBlocks.Conv(midChannels, 1, new LeakyReLUActivation<T>(0.1)));
        Reconstruction = new X4ReconstructionBlock<T>(layers, midChannels);
    }

    internal SpyNetFlowBlock<T> SpyNet { get; }

    internal ResidualTrunkBlock<T> Features { get; }

    internal ResidualTrunkBlock<T> Propagation { get; }

    internal X4ReconstructionBlock<T> Reconstruction { get; }

    /// <summary>The token scales' (kernel, padding) for unfolding at the stride: s, 1.5s and 2s patches.</summary>
    private (int Kernel, int Padding)[] Scales => new[] { (_stride, 0), (3 * _stride / 2, _stride / 4), (2 * _stride, _stride / 2) };

    /// <summary>
    /// The three-scale tokenization of a feature [N, C, H, W] into [N, C·s², H/s, W/s] each: scale 1 is F.unfold at
    /// stride s refolded on the token grid (a pixel unshuffle); scales 2 and 3 unfold k = 1.5s, 2s patches at stride s,
    /// tile them into a (k/s)H x (k/s)W image, adaptively average-pool it back to H x W and unshuffle.
    /// </summary>
    internal Tensor<T>[] Tokenize(IEngine engine, Tensor<T> feature)
    {
        int h = feature.Shape[2], w = feature.Shape[3];
        var tokens = new Tensor<T>[3];
        tokens[0] = engine.TensorPixelUnshuffle(feature, _stride);
        for (int s = 1; s < 3; s++)
        {
            var (k, pad) = Scales[s];
            var patches = engine.Unfold(feature, new[] { k, k }, new[] { _stride, _stride }, new[] { pad, pad });
            var tiled = engine.Fold(patches, new[] { k * h / _stride, k * w / _stride }, new[] { k, k }, new[] { k, k }, new[] { 0, 0 });
            tokens[s] = engine.TensorPixelUnshuffle(engine.AdaptiveAvgPool2D(tiled, h, w), _stride);
        }
        return tokens;
    }

    /// <summary>
    /// mmagic flow_warp of a host location map [2, h, w] (x then y) with nearest sampling and border padding:
    /// out(y, x) = map(round(y + fy), round(x + fx)) with the coordinates clamped into the grid.
    /// </summary>
    internal static double[] WarpLocation(double[] map, double[] flowX, double[] flowY, int h, int w)
    {
        var warped = new double[map.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int p = y * w + x;
                int sx = Math.Min(w - 1, Math.Max(0, (int)Math.Round(x + flowX[p], MidpointRounding.ToEven)));
                int sy = Math.Min(h - 1, Math.Max(0, (int)Math.Round(y + flowY[p], MidpointRounding.ToEven)));
                warped[p] = map[sy * w + sx];
                warped[h * w + p] = map[h * w + sy * w + sx];
            }
        return warped;
    }

    private static double[] IdentityLocation(int h, int w)
    {
        var map = new double[2 * h * w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                map[y * w + x] = x;
                map[h * w + y * w + x] = y;
            }
        return map;
    }

    // F.adaptive_avg_pool2d(flow, (h/s, w/s)) / s for one batch item, on the host.
    private (double[] X, double[] Y) TokenFlow(Tensor<T> flow, int item, int h, int w)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hs = h / _stride, ws = w / _stride;
        var fx = new double[hs * ws];
        var fy = new double[hs * ws];
        var data = flow.Data.Span;
        int plane = h * w, baseIndex = item * 2 * plane;
        double norm = 1.0 / (_stride * (double)_stride * _stride);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int t = (y / _stride) * ws + x / _stride;
                fx[t] += ops.ToDouble(data[baseIndex + y * w + x]) * norm;
                fy[t] += ops.ToDouble(data[baseIndex + plane + y * w + x]) * norm;
            }
        return (fx, fy);
    }

    private sealed class TrajectoryMemory
    {
        internal readonly List<Tensor<T>[]> Tokens = new();
        internal readonly List<Tensor<T>> IndexTokens = new();
        // Per keyframe, per batch item, the location map [2, h/s, w/s].
        internal readonly List<double[][]> Locations = new();
    }

    // [N, C', h, w] tokens as rows [N·h·w, C'].
    private static Tensor<T> AsRows(IEngine engine, Tensor<T> tokens)
    {
        int n = tokens.Shape[0], c = tokens.Shape[1], hw = tokens.Shape[2] * tokens.Shape[3];
        return engine.Reshape(engine.TensorPermute(engine.Reshape(tokens, new[] { n, c, hw }), new[] { 0, 2, 1 }), new[] { n * hw, c });
    }

    // Rows [N·h·w, C'] back to [N, C', h, w].
    private static Tensor<T> FromRows(IEngine engine, Tensor<T> rows, int n, int h, int w)
    {
        int c = rows.Shape[1];
        return engine.Reshape(engine.TensorPermute(engine.Reshape(rows, new[] { n, h * w, c }), new[] { 0, 2, 1 }), new[] { n, c, h, w });
    }

    // F.normalize(x, dim=1) for rows: x / max(‖x‖, 1e-12).
    private static Tensor<T> Normalized(IEngine engine, Tensor<T> rows)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(rows, rows), new[] { 1 }, true));
        return engine.TensorDivide(rows, engine.TensorMax(norm, ops.FromDouble(1e-12)));
    }

    /// <summary>
    /// LTAM: the long-range trajectory-aware attention of the current frame's features over the memory, added to the
    /// propagated feature <paramref name="anchor"/>.
    /// </summary>
    private Tensor<T> Attend(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> current, Tensor<T> anchor, TrajectoryMemory memory)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = anchor.Shape[0], h = anchor.Shape[2], w = anchor.Shape[3];
        int hs = h / _stride, ws = w / _stride, hw = hs * ws, rows = n * hw, frames = memory.Tokens.Count;

        // Row r = b·hw + p of keyframe j's trajectory token: its token at the location the trajectory reached.
        var select = new int[frames][];
        for (int j = 0; j < frames; j++)
        {
            select[j] = new int[rows];
            for (int b = 0; b < n; b++)
            {
                var map = memory.Locations[j][b];
                for (int p = 0; p < hw; p++)
                    select[j][b * hw + p] = b * hw + (int)map[hw + p] * ws + (int)map[p];
            }
        }

        var currentRows = Normalized(engine, AsRows(engine, engine.TensorPixelUnshuffle(current, _stride)));
        var similarities = new Tensor<T>[frames];
        for (int j = 0; j < frames; j++)
        {
            var candidate = Normalized(engine, engine.TensorIndexSelectDiff(AsRows(engine, memory.IndexTokens[j]), IndexTensor(select[j]), 0));
            similarities[j] = engine.ReduceSum(engine.TensorMultiply(candidate, currentRows), new[] { 1 }, false);
        }
        var stacked = engine.TensorConcatenate(similarities, 0);

        // torch.max over keyframes: the winner per row (host), and its similarity on the tape.
        var values = stacked.Data.Span;
        var best = new int[rows];
        var flat = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            int arg = 0;
            double top = ops.ToDouble(values[r]);
            for (int j = 1; j < frames; j++)
            {
                double v = ops.ToDouble(values[j * rows + r]);
                if (v > top)
                {
                    top = v;
                    arg = j;
                }
            }
            best[r] = arg;
            flat[r] = arg * rows + r;
        }
        // IndexSelect takes rows of a 2-D tensor, so the stacked similarities are a column.
        var soft = engine.TensorIndexSelectDiff(engine.Reshape(stacked, new[] { frames * rows, 1 }), IndexTensor(flat), 0);
        var softMap = engine.Interpolate(engine.Reshape(soft, new[] { n, 1, hs, ws }), new[] { h, w }, InterpolateMode.Nearest, false);

        var gathered = new Tensor<T>[3];
        for (int s = 0; s < 3; s++)
        {
            var all = engine.TensorConcatenate(Enumerable.Range(0, frames).Select(j => AsRows(engine, memory.Tokens[j][s])).ToArray(), 0);
            var picks = new int[rows];
            for (int r = 0; r < rows; r++) picks[r] = best[r] * rows + select[best[r]][r];
            var chosen = engine.TensorIndexSelectDiff(all, IndexTensor(picks), 0);
            gathered[s] = engine.PixelShuffle(FromRows(engine, chosen, n, hs, ws), _stride);
        }
        var fused = BasicVsrBlocks.Run(layers, _ltamFusion, engine.TensorConcatenate(gathered, 1));
        return engine.TensorAdd(engine.TensorMultiply(fused, softMap), anchor);
    }

    private static Tensor<int> IndexTensor(int[] indices)
    {
        var tensor = new Tensor<int>(new[] { indices.Length });
        for (int i = 0; i < indices.Length; i++) tensor[i] = indices[i];
        return tensor;
    }

    /// <summary>TTVSRNet.forward: clips [N, T, 3, H, W] in [0, 1] (H, W multiples of the stride) to [N, T, 3, 4H, 4W].</summary>
    internal Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips)
    {
        int n = clips.Shape[0], t = clips.Shape[1], c = clips.Shape[2], h = clips.Shape[3], w = clips.Shape[4];
        if (h % _stride != 0 || w % _stride != 0)
            throw new ArgumentException($"TTVSR needs frame sizes divisible by the token stride {_stride}; got {h}x{w}.", nameof(clips));
        var frames = new Tensor<T>[t];
        for (int i = 0; i < t; i++) frames[i] = engine.Reshape(engine.TensorNarrow(clips, 1, i, 1), new[] { n, c, h, w });

        var flowsBackward = new Tensor<T>[Math.Max(0, t - 1)];
        var flowsForward = new Tensor<T>[Math.Max(0, t - 1)];
        if (t > 1)
        {
            var first = engine.TensorConcatenate(frames.Take(t - 1).ToArray(), 0);
            var second = engine.TensorConcatenate(frames.Skip(1).ToArray(), 0);
            var backward = SpyNet.Flow(engine, layers, first, second);
            var forward = SpyNet.Flow(engine, layers, second, first);
            for (int i = 0; i < t - 1; i++)
            {
                flowsBackward[i] = engine.TensorNarrow(backward, 0, i * n, n);
                flowsForward[i] = engine.TensorNarrow(forward, 0, i * n, n);
            }
        }

        var all = Features.Forward(engine, layers, engine.TensorConcatenate(frames, 0));
        var features = new Tensor<T>[t];
        for (int i = 0; i < t; i++) features[i] = engine.TensorNarrow(all, 0, i * n, n);

        var backwardKeyframes = new HashSet<int>();
        for (int i = t - 1; i > 0; i -= _frameStride) backwardKeyframes.Add(i);
        var forwardKeyframes = new HashSet<int>();
        for (int i = 0; i < t; i += _frameStride) forwardKeyframes.Add(i);

        var backwardStates = Propagate(engine, layers, features, Enumerable.Range(0, t).Reverse().ToArray(),
            i => flowsBackward[i], backwardKeyframes, h, w, out _);

        var outputs = new Tensor<T>[t];
        var forwardStates = Propagate(engine, layers, features, Enumerable.Range(0, t).ToArray(),
            i => flowsForward[i - 1], forwardKeyframes, h, w, out _);
        for (int i = 0; i < t; i++)
        {
            var fused = BasicVsrBlocks.Run(layers, _fusion, engine.TensorConcatenate(new[] { backwardStates[i], features[i], forwardStates[i] }, 1));
            var output = Reconstruction.Forward(engine, layers, fused, frames[i]);
            outputs[i] = engine.Reshape(output, new[] { n, 1, c, 4 * h, 4 * w });
        }
        return engine.TensorConcatenate(outputs, 1);
    }

    // One propagation direction over `order`; flowFor(i) is the flow that aligns the previous state to step i.
    private Tensor<T>[] Propagate(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T>[] features, int[] order,
        Func<int, Tensor<T>> flowFor, HashSet<int> keyframes, int h, int w, out int keyframeCount)
    {
        int n = features[0].Shape[0], hs = h / _stride, ws = w / _stride;
        var states = new Tensor<T>[features.Length];
        var memory = new TrajectoryMemory();
        var identity = IdentityLocation(hs, ws);
        // The first step's own keyframe trajectory starts before it, as in the release.
        var pending = Enumerable.Range(0, n).Select(_ => (double[])identity.Clone()).ToArray();
        var state = new Tensor<T>(new[] { n, _midChannels, h, w });
        for (int step = 0; step < order.Length; step++)
        {
            int i = order[step];
            if (step > 0)
            {
                var flow = flowFor(i);
                state = BasicVsrBlocks.FlowWarp(engine, state, flow, GridSamplePadding.Border);
                for (int b = 0; b < n; b++)
                {
                    var (fx, fy) = TokenFlow(flow, b, h, w);
                    for (int j = 0; j < memory.Locations.Count; j++) memory.Locations[j][b] = WarpLocation(memory.Locations[j][b], fx, fy, hs, ws);
                }
                state = Attend(engine, layers, features[i], state, memory);
            }
            state = Propagation.Forward(engine, layers, engine.TensorConcatenate(new[] { features[i], state }, 1));
            states[i] = state;
            if (keyframes.Contains(i))
            {
                memory.Tokens.Add(Tokenize(engine, state));
                memory.IndexTokens.Add(engine.TensorPixelUnshuffle(features[i], _stride));
                memory.Locations.Add(step == 0 ? pending : Enumerable.Range(0, n).Select(_ => (double[])identity.Clone()).ToArray());
            }
        }
        keyframeCount = memory.Tokens.Count;
        return states;
    }

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(SpyNet.First).Take(SpyNet.Count));
}
