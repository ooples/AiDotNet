using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// RVRT's GuidedDeformAttnPack: offsets predicted from the query clip, the flow-warped propagated features and the
/// flows (residues of at most <c>max_residue_magnitude</c> on top of the flow), then the release's deform_attn op: per
/// query pixel and head, a scaled softmax over a deformable 3x3 window sampled from every frame of the clip.
/// </summary>
/// <remarks>
/// deform_attn (models/op/deform_attn_cuda_pt110.cpp): query frame i samples clip frame (n + i) mod clip with offset
/// block n, without a modulation mask; the deformable groups span the concatenated key and value channels; the key and
/// value columns are split per head (head-major channels) and the query is scaled by attn_dim^-0.5.
/// </remarks>
internal sealed class RvrtGuidedDeformableAttention<T>
{
    private readonly int[] _offsetConvs = new int[6];
    private readonly int _projQ, _projK, _projV, _proj, _fc1, _fc2;
    private readonly int _channels, _clip, _groups, _heads;
    private readonly double _maxResidue;

    internal RvrtGuidedDeformableAttention(List<ILayer<T>> layers, int channels, int clip, int groups, int heads, double maxResidue)
    {
        _channels = channels;
        _clip = clip;
        _groups = groups;
        _heads = heads;
        _maxResidue = maxResidue;
        int outputs = clip * groups * 9 * 2;
        for (int i = 0; i < 6; i++)
        {
            _offsetConvs[i] = layers.Count;
            if (i == 5)
                layers.Add(BasicVsrBlocks.Conv<T>(outputs, 1, null, new ZeroInitializationStrategy<T>()));
            else
                layers.Add(BasicVsrBlocks.Conv<T>(64, i == 0 ? 1 : 3, new LeakyReLUActivation<T>(0.1)));
        }
        _projQ = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(2 * channels));
        _projK = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(2 * channels));
        _projV = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(2 * channels));
        _proj = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(channels));
        _fc1 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(2 * channels, new GELUActivation<T>()));
        _fc2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(channels));
    }

    // nn.Linear over the channel axis of [n, d, c, h, w].
    private static Tensor<T> Linear(IEngine engine, IReadOnlyList<ILayer<T>> layers, int index, Tensor<T> x)
        => RvrtBlocks.ChannelsFirst(engine, layers[index].Forward(RvrtBlocks.ChannelsLast(engine, x)));

    // [n, d, 2, h, w] (dx, dy) to (dy, dx), mmcv's offset order.
    private static Tensor<T> Flip(IEngine engine, Tensor<T> flow)
        => engine.TensorConcatenate(new[] { engine.TensorNarrow(flow, 2, 1, 1), engine.TensorNarrow(flow, 2, 0, 1) }, 2);

    /// <summary>The attention, and with <paramref name="updateFlows"/> the offsets averaged back into two flows.</summary>
    /// <param name="q">The query clip [n, clip, C, h, w].</param>
    /// <param name="k">The previous clip's features [n, clip, C, h, w].</param>
    /// <param name="v">The propagated features [n, clip, C, h, w].</param>
    /// <param name="warped">The propagated features warped by each flow.</param>
    /// <param name="flows">The two flows [n, clip, 2, h, w].</param>
    internal (Tensor<T> Output, Tensor<T>? Flow1, Tensor<T>? Flow2) Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers,
        Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T>[] warped, Tensor<T>[] flows, bool updateFlows)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = q.Shape[0], h = q.Shape[3], w = q.Shape[4], block = _groups * 18;

        var offsets = engine.TensorConcatenate(new[] { q }.Concat(warped).Concat(flows).ToArray(), 2);
        foreach (int conv in _offsetConvs) offsets = RvrtBlocks.PerFrame(engine, layers, conv, offsets);
        offsets = engine.TensorMultiplyScalar(engine.Tanh(offsets), ops.FromDouble(_maxResidue));
        var offset1 = engine.TensorAdd(engine.TensorNarrow(offsets, 2, 0, block), engine.TensorTile(Flip(engine, flows[0]), new[] { 1, 1, _groups * 9, 1, 1 }));
        var offset2 = engine.TensorAdd(engine.TensorNarrow(offsets, 2, block, block), engine.TensorTile(Flip(engine, flows[1]), new[] { 1, 1, _groups * 9, 1, 1 }));

        var query = Linear(engine, layers, _projQ, q);
        var kv = engine.TensorConcatenate(new[] { Linear(engine, layers, _projK, k), Linear(engine, layers, _projV, v) }, 2);
        var attended = DeformAttention(engine, query, kv, new[] { offset1, offset2 }, _groups, _heads);

        var result = Linear(engine, layers, _proj, attended);
        result = engine.TensorAdd(result, Linear(engine, layers, _fc2, Linear(engine, layers, _fc1, result)));
        if (!updateFlows) return (result, null, null);

        Tensor<T> Average(Tensor<T> offset)
            => Flip(engine, engine.ReduceMean(engine.Reshape(offset, new[] { n, _clip, _groups * 9, 2, h, w }), new[] { 2 }, false));
        return (result, Average(offset1), Average(offset2));
    }

    /// <summary>
    /// The release's deform_attn op. Query frame i of [n, clip, P, h, w] attends over a deformable 3x3 window in every
    /// frame of kv [n, clip, 2P, h, w] (keys then values): clip frame (b + i) mod clip, sampled with
    /// <paramref name="offsetBlocks"/>[b] at frame i ([n, clip, groups·9·2, h, w], (dy, dx) pairs per group and tap,
    /// groups spanning all 2P channels), with a per-head softmax scaled by (P / heads)^-0.5.
    /// </summary>
    internal static Tensor<T> DeformAttention(IEngine engine, Tensor<T> query, Tensor<T> kv, Tensor<T>[] offsetBlocks, int groups, int heads)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = query.Shape[0], clip = query.Shape[1], projected = query.Shape[2], h = query.Shape[3], w = query.Shape[4];
        int kvChannels = 2 * projected, attnDim = projected / heads, samples = clip * 9, area = h * w, block = groups * 18;

        // im2col of the deformable 3x3 window: a depthwise one-hot kernel puts tap t of channel c in output c·9 + t.
        var kernel = new Tensor<T>(new[] { kvChannels * 9, 1, 3, 3 });
        for (int c = 0; c < kvChannels; c++)
            for (int t = 0; t < 9; t++) kernel[(c * 9 + t) * 9 + t] = ops.One;
        var one = new[] { 1, 1 };

        var outputs = new Tensor<T>[clip];
        for (int i = 0; i < clip; i++)
        {
            var columns = new Tensor<T>[clip];
            for (int b = 0; b < clip; b++)
            {
                var frame = engine.Reshape(engine.TensorNarrow(kv, 1, (b + i) % clip, 1), new[] { n, kvChannels, h, w });
                // mmcv interleaves (dy, dx) per tap; the engine reads each group's dy for all taps, then dx.
                var offset = engine.Reshape(engine.TensorPermute(
                    engine.Reshape(engine.TensorNarrow(offsetBlocks[b], 1, i, 1), new[] { n, groups, 9, 2, h, w }),
                    new[] { 0, 1, 3, 2, 4, 5 }), new[] { n, block, h, w });
                var sampled = engine.DeformableConv2DGrouped(frame, kernel, offset, null, one, one, one, kvChannels, groups);
                columns[b] = engine.Reshape(sampled, new[] { n, 2, heads, attnDim, 1, 9, area });
            }
            var all = engine.Reshape(engine.TensorConcatenate(columns, 4), new[] { n, 2, heads, attnDim, samples, area });
            var keys = engine.Reshape(engine.TensorNarrow(all, 1, 0, 1), new[] { n, heads, attnDim, samples, area });
            var values = engine.Reshape(engine.TensorNarrow(all, 1, 1, 1), new[] { n, heads, attnDim, samples, area });
            var qi = engine.TensorMultiplyScalar(
                engine.Reshape(engine.TensorNarrow(query, 1, i, 1), new[] { n, heads, attnDim, 1, area }), ops.FromDouble(Math.Pow(attnDim, -0.5)));
            var attn = engine.TensorSoftmax(engine.ReduceSum(engine.TensorMultiply(qi, keys), new[] { 2 }, false), 2);
            var o = engine.ReduceSum(engine.TensorMultiply(engine.Reshape(attn, new[] { n, heads, 1, samples, area }), values), new[] { 3 }, false);
            outputs[i] = engine.Reshape(o, new[] { n, 1, projected, h, w });
        }
        return engine.TensorConcatenate(outputs, 1);
    }
}

/// <summary>
/// RVRT for x4 video super-resolution (network_rvrt.py, RVRT with upscale=4): SPyNet flow, an RSTB feature extractor,
/// four recurrent propagation passes (backward, forward, backward, forward) over two-frame clips, each aligning the
/// propagated clip with guided deformable attention and refining it with an RSTB group over every earlier pass, and an
/// RSTB reconstruction with x4 pixel-shuffle upsampling plus the trilinearly upsampled input.
/// </summary>
internal sealed class RvrtBackbone<T>
{
    internal const int ClipSize = 2;
    private static readonly string[] Passes = { "backward_1", "forward_1", "backward_2", "forward_2" };

    private readonly int _channels;
    private readonly RvrtStage<T> _extract, _reconstruction;
    private readonly RvrtStage<T>[] _refine = new RvrtStage<T>[4];
    private readonly RvrtGuidedDeformableAttention<T>[] _align = new RvrtGuidedDeformableAttention<T>[4];
    private readonly int _beforeUpsample, _up1, _up2, _upLast, _last;

    internal RvrtBackbone(List<ILayer<T>> layers, int channels, int heads, int window, int[] numBlocks, int depth,
        double mlpRatio, int deformableGroups, int attentionHeads, double maxResidue)
    {
        _channels = channels;
        SpyNet = new SpyNetFlowBlock<T>(layers);
        _extract = new RvrtStage<T>(layers, channels, numBlocks[0], depth, heads, new[] { 1, window, window }, mlpRatio);
        for (int i = 0; i < 4; i++)
        {
            _align[i] = new RvrtGuidedDeformableAttention<T>(layers, channels, ClipSize, deformableGroups, attentionHeads, maxResidue);
            _refine[i] = new RvrtStage<T>(layers, channels, numBlocks[1], depth, heads, new[] { ClipSize, window, window }, mlpRatio);
        }
        _reconstruction = new RvrtStage<T>(layers, channels, numBlocks[2], depth, heads, new[] { 1, window, window }, mlpRatio);
        _beforeUpsample = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(64, 1, new LeakyReLUActivation<T>(0.1)));
        // LeakyReLU is elementwise and pixel shuffle a permutation, so the activation sits on the convolution.
        _up1 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(256, 3, new LeakyReLUActivation<T>(0.1)));
        _up2 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(256, 3, new LeakyReLUActivation<T>(0.1)));
        _upLast = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(64, 3, null));
        _last = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(3, 3, null));
    }

    internal SpyNetFlowBlock<T> SpyNet { get; }

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(SpyNet.First).Take(SpyNet.Count));

    /// <summary>RVRT.forward: clips [N, T, 3, H, W] in [0, 1], T a multiple of the clip size, to [N, T, 3, 4H, 4W].</summary>
    internal Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> lqs)
    {
        int n = lqs.Shape[0], t = lqs.Shape[1], h = lqs.Shape[3], w = lqs.Shape[4];
        if (t % ClipSize != 0)
            throw new ArgumentException($"RVRT processes {ClipSize}-frame clips; the sequence has {t} frames.", nameof(lqs));
        int clips = t / ClipSize;

        var shallow = _extract.Forward(engine, layers, lqs);
        var feats = new List<(string Name, Tensor<T>[] Clips)>
        {
            ("shallow", Enumerable.Range(0, clips).Select(c => engine.TensorNarrow(shallow, 1, c * ClipSize, ClipSize)).ToArray()),
        };

        // compute_flow: backward flows run frame i to i + 1, forward flows i + 1 to i.
        var flowsBackward = new Tensor<T>[t - 1];
        var flowsForward = new Tensor<T>[t - 1];
        if (t > 1)
        {
            var first = engine.Reshape(engine.TensorNarrow(lqs, 1, 0, t - 1), new[] { n * (t - 1), 3, h, w });
            var second = engine.Reshape(engine.TensorNarrow(lqs, 1, 1, t - 1), new[] { n * (t - 1), 3, h, w });
            var backward = engine.Reshape(SpyNet.Flow(engine, layers, first, second), new[] { n, t - 1, 2, h, w });
            var forward = engine.Reshape(SpyNet.Flow(engine, layers, second, first), new[] { n, t - 1, 2, h, w });
            for (int i = 0; i < t - 1; i++)
            {
                flowsBackward[i] = engine.Reshape(engine.TensorNarrow(backward, 1, i, 1), new[] { n, 2, h, w });
                flowsForward[i] = engine.Reshape(engine.TensorNarrow(forward, 1, i, 1), new[] { n, 2, h, w });
            }
        }

        var updated = new Dictionary<string, List<(Tensor<T> N1, Tensor<T> N2)>>();
        for (int p = 0; p < 4; p++)
            feats.Add((Passes[p], Propagate(engine, layers, feats, p, p % 2 == 0 ? flowsBackward : flowsForward, updated, n, h, w)));

        // upsample: every pass's features, concatenated along channels in pass order, reconstructed frame by frame.
        var stacked = engine.TensorConcatenate(feats.Select(f => engine.TensorConcatenate(f.Clips, 1)).ToArray(), 2);
        var hr = _reconstruction.Forward(engine, layers, stacked);
        hr = RvrtBlocks.PerFrame(engine, layers, _beforeUpsample, hr);
        hr = Upsample(engine, layers, _up1, hr);
        hr = Upsample(engine, layers, _up2, hr);
        hr = RvrtBlocks.PerFrame(engine, layers, _upLast, hr);
        hr = RvrtBlocks.PerFrame(engine, layers, _last, hr);
        // F.interpolate trilinear to (T, 4H, 4W) keeps T, so it is bilinear per frame (align_corners=False).
        var lowUp = engine.Interpolate(engine.Reshape(lqs, new[] { n * t, 3, h, w }), new[] { 4 * h, 4 * w }, InterpolateMode.Bilinear, false);
        return engine.TensorAdd(hr, engine.Reshape(lowUp, new[] { n, t, 3, 4 * h, 4 * w }));
    }

    // conv (1, 3, 3) to 4C with LeakyReLU 0.1, then pixel shuffle x2.
    private static Tensor<T> Upsample(IEngine engine, IReadOnlyList<ILayer<T>> layers, int conv, Tensor<T> x)
    {
        var y = RvrtBlocks.PerFrame(engine, layers, conv, x);
        int n = y.Shape[0], d = y.Shape[1];
        var shuffled = engine.PixelShuffle(engine.Reshape(y, new[] { n * d, y.Shape[2], y.Shape[3], y.Shape[4] }), 2);
        return engine.Reshape(shuffled, new[] { n, d, shuffled.Shape[1], shuffled.Shape[2], shuffled.Shape[3] });
    }

    private static Tensor<T> Stack(IEngine engine, Tensor<T> a, Tensor<T> b, int n, int h, int w)
        => engine.TensorConcatenate(new[] { engine.Reshape(a, new[] { n, 1, 2, h, w }), engine.Reshape(b, new[] { n, 1, 2, h, w }) }, 1);

    // flow_warp of every frame of a clip [n, clip, C, h, w] by its own flow [n, clip, 2, h, w].
    private static Tensor<T> WarpClip(IEngine engine, Tensor<T> clip, Tensor<T> flow, int n, int h, int w)
    {
        int c = clip.Shape[2];
        var warped = BasicVsrBlocks.FlowWarp(engine, engine.Reshape(clip, new[] { n * ClipSize, c, h, w }),
            engine.Reshape(flow, new[] { n * ClipSize, 2, h, w }), GridSamplePadding.Zeros);
        return engine.Reshape(warped, new[] { n, ClipSize, c, h, w });
    }

    /// <summary>RVRT.propagate for pass <paramref name="pass"/> (clip size 2).</summary>
    private Tensor<T>[] Propagate(IEngine engine, IReadOnlyList<ILayer<T>> layers, List<(string Name, Tensor<T>[] Clips)> feats,
        int pass, Tensor<T>[] flows, Dictionary<string, List<(Tensor<T> N1, Tensor<T> N2)>> updated, int n, int h, int w)
    {
        string name = Passes[pass];
        bool backward = name.StartsWith("backward", StringComparison.Ordinal), first = name.EndsWith("_1", StringComparison.Ordinal);
        int t = flows.Length + 1, clips = t / ClipSize;
        var flowIndex = backward ? Enumerable.Range(0, t).Reverse().ToArray() : Enumerable.Range(-1, t).ToArray();
        var clipIndex = backward ? Enumerable.Range(0, clips).Reverse().ToArray() : Enumerable.Range(0, clips).ToArray();
        if (first) updated[name] = new List<(Tensor<T>, Tensor<T>)>();

        Tensor<T> Oriented(Tensor<T> clip) => backward ? engine.TensorFlip(clip, new[] { 1 }) : clip;
        var last = feats[feats.Count - 1].Clips;
        var propagated = new Tensor<T>(new[] { n, ClipSize, _channels, h, w });
        var results = new Tensor<T>[clips];
        for (int i = 0; i < clips; i++)
        {
            int idx = clipIndex[i];
            if (i > 0)
            {
                Tensor<T> flowN1, flowN2;
                if (first)
                {
                    var n01 = flows[flowIndex[ClipSize * i - 1]];
                    var n12 = flows[flowIndex[ClipSize * i]];
                    var n23 = flows[flowIndex[ClipSize * i + 1]];
                    var n02 = engine.TensorAdd(n12, BasicVsrBlocks.FlowWarp(engine, n01, n12, GridSamplePadding.Zeros));
                    var n13 = engine.TensorAdd(n23, BasicVsrBlocks.FlowWarp(engine, n12, n23, GridSamplePadding.Zeros));
                    var n03 = engine.TensorAdd(n23, BasicVsrBlocks.FlowWarp(engine, n02, n23, GridSamplePadding.Zeros));
                    flowN1 = Stack(engine, n02, n13, n, h, w);
                    flowN2 = Stack(engine, n12, n03, n, h, w);
                }
                else
                {
                    (flowN1, flowN2) = updated[name.Replace("_2", "_1")][i - 1];
                }

                var query = Oriented(last[idx]);
                var key = Oriented(last[clipIndex[i - 1]]);
                var warped1 = WarpClip(engine, propagated, flowN1, n, h, w);
                var warped2 = WarpClip(engine, engine.TensorFlip(propagated, new[] { 1 }), flowN2, n, h, w);
                var (aligned, f1, f2) = _align[pass].Forward(engine, layers, query, key, propagated,
                    new[] { warped1, warped2 }, new[] { flowN1, flowN2 }, first);
                propagated = aligned;
                if (first && f1 is not null && f2 is not null) updated[name].Add((f1, f2));
            }

            var inputs = feats.Select(f => Oriented(f.Clips[idx])).Append(propagated).ToArray();
            propagated = engine.TensorAdd(propagated, _refine[pass].Forward(engine, layers, engine.TensorConcatenate(inputs, 2)));
            results[i] = propagated;
        }

        if (backward)
        {
            Array.Reverse(results);
            for (int i = 0; i < clips; i++) results[i] = engine.TensorFlip(results[i], new[] { 1 });
        }
        return results;
    }
}
