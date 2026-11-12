using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// PSRT's SwinIRFM as the recurrent network uses it (psrt_sliding_arch.py): a three-frame stack is LayerNorm-embedded,
/// passes through RSTBs (alternately shifted window-attention layers, a 3x3 convolution and a residual) and a final
/// LayerNorm, and its centre frame, after conv_after_body, is added to the centre input.
/// </summary>
/// <remarks>
/// The release's conv_first, conv_first_feat and conv_before_upsample inside SwinIRFM never run on PSRT-recurrent's
/// embed-width input (forward only applies them to 3- or 64-channel stacks, and to a reference that is never passed),
/// so they are not built. Linear layers and the relative-position table take trunc_normal_(std=0.02), whose +-2 bounds
/// lie a hundred deviations out, so a plain normal is the same distribution; stochastic depth rises linearly from 0 to
/// drop_path_rate over every layer of the module.
/// </remarks>
internal sealed class PsrtAlignment<T>
{
    private readonly int _patchNorm, _norm, _afterBody;
    private readonly List<(RvrtSwinLayer<T>[] Layers, int Conv)> _groups = new();

    internal PsrtAlignment(List<ILayer<T>> layers, int dim, int[] depths, int heads, int[] window, double mlpRatio,
        double dropPathRate, Random random)
    {
        _patchNorm = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        var shift = new[] { window[0], window[1] / 2, window[2] / 2 };
        int total = depths.Sum(), k = 0;
        foreach (int depth in depths)
        {
            var group = new RvrtSwinLayer<T>[depth];
            for (int i = 0; i < depth; i++, k++)
                group[i] = new RvrtSwinLayer<T>(layers, dim, heads, window, i % 2 == 0 ? new int[3] : shift, mlpRatio, true,
                    clampWindow: false, linearInit: new NormalInitializationStrategy<T>(0.0, 0.02),
                    tableInit: new NormalInitializationStrategy<T>(0.0, 0.02),
                    dropPath: total > 1 ? dropPathRate * k / (total - 1) : 0.0, random: random);
            _groups.Add((group, layers.Count));
            layers.Add(BasicVsrBlocks.Conv<T>(dim, 3, null));
        }
        _norm = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _afterBody = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(dim, 3, null));
    }

    /// <summary>SwinIRFM.forward: [N, 3, C, H, W] to the aligned centre frame [N, C, H, W].</summary>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> stack, bool training)
    {
        int n = stack.Shape[0], t = stack.Shape[1], c = stack.Shape[2], h = stack.Shape[3], w = stack.Shape[4];
        var centre = engine.Reshape(engine.TensorNarrow(stack, 1, t / 2, 1), new[] { n, c, h, w });
        var x = layers[_patchNorm].Forward(RvrtBlocks.ChannelsLast(engine, stack));
        foreach (var (group, conv) in _groups)
        {
            var y = x;
            foreach (var layer in group) y = layer.Forward(engine, layers, y, training);
            y = RvrtBlocks.ChannelsLast(engine, RvrtBlocks.PerFrame(engine, layers, conv, RvrtBlocks.ChannelsFirst(engine, y)));
            x = engine.TensorAdd(x, y);
        }
        x = RvrtBlocks.ChannelsFirst(engine, layers[_norm].Forward(x));
        var mid = engine.Reshape(engine.TensorNarrow(x, 1, t / 2, 1), new[] { n, c, h, w });
        return engine.TensorAdd(layers[_afterBody].Forward(mid), centre);
    }
}

/// <summary>Aligns a propagated feature map to the current frame along a flow (PSRT's patch warp, IART's implicit warp).</summary>
internal interface IPsrtFrameAligner<T>
{
    /// <param name="propagated">[N, C, H, W], the output being carried to the current frame.</param>
    /// <param name="current">[N, C, H, W], the current frame's features.</param>
    /// <param name="flow">[N, 2, H, W], (dx, dy) from the current frame into the propagated one.</param>
    Tensor<T> Align(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> propagated, Tensor<T> current, Tensor<T> flow);
}

/// <summary>PSRT's flow_warp_avg_patch as an aligner; it reads only the propagated features.</summary>
internal sealed class PsrtPatchAligner<T> : IPsrtFrameAligner<T>
{
    public Tensor<T> Align(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> propagated, Tensor<T> current, Tensor<T> flow)
        => PsrtBackbone<T>.PatchWarp(engine, propagated, flow);
}

/// <summary>
/// PSRT-recurrent's BasicRecurrentSwin (Shi et al. 2022; XPixelGroup/RethinkVSRAlignment, psrt_recurrent_arch.py):
/// SPyNet flow, a 3x3 embedding, four recurrent passes (backward, forward, backward, forward) that each align a frame
/// with the patch-warped previous and second-previous outputs through a SwinIRFM, and x4 pixel-shuffle reconstruction
/// of the last pass plus the bilinearly upsampled input.
/// </summary>
internal sealed class PsrtBackbone<T>
{
    /// <summary>flow_warp_avg_patch's patch size (fixed at 8 in the release).</summary>
    internal const int PatchSize = 8;

    private readonly int _beforeUpsample, _convFirst, _up1, _up2, _hr, _last;
    private readonly PsrtAlignment<T>[] _align = new PsrtAlignment<T>[4];
    private readonly IPsrtFrameAligner<T> _aligner;

    /// <param name="aligner">Builds the frame aligner after every other layer (IART's implicit_warp is registered
    /// last); null is PSRT's patch alignment.</param>
    internal PsrtBackbone(List<ILayer<T>> layers, int midChannels, int embedDim, int[] depths, int heads, int[] window,
        double mlpRatio, double dropPathRate, Random random, Func<List<ILayer<T>>, IPsrtFrameAligner<T>>? aligner = null)
    {
        _beforeUpsample = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(midChannels, 3, null));
        SpyNet = new SpyNetFlowBlock<T>(layers);
        _convFirst = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(embedDim, 3, null));
        for (int i = 0; i < 4; i++)
            _align[i] = new PsrtAlignment<T>(layers, embedDim, depths, heads, window, mlpRatio, dropPathRate, random);
        // LeakyReLU is elementwise and pixel shuffle a permutation, so the activation sits on the convolution.
        _up1 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(midChannels * 4, 3, new LeakyReLUActivation<T>(0.1)));
        _up2 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(64 * 4, 3, new LeakyReLUActivation<T>(0.1)));
        _hr = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(64, 3, new LeakyReLUActivation<T>(0.1)));
        _last = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(3, 3, null));
        _aligner = aligner?.Invoke(layers) ?? new PsrtPatchAligner<T>();
    }

    internal SpyNetFlowBlock<T> SpyNet { get; }

    /// <summary>The frame aligner every pass uses.</summary>
    internal IPsrtFrameAligner<T> Aligner => _aligner;

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(SpyNet.First).Take(SpyNet.Count));

    /// <summary>BasicRecurrentSwin.forward: [N, T, 3, H, W] in [0, 1] to [N, T, 3, 4H, 4W].</summary>
    internal Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> lqs, bool training)
    {
        int n = lqs.Shape[0], t = lqs.Shape[1], h = lqs.Shape[3], w = lqs.Shape[4];
        var spatial = RvrtBlocks.PerFrame(engine, layers, _convFirst, lqs);
        int c = spatial.Shape[2];
        var feats = Enumerable.Range(0, t)
            .Select(i => engine.Reshape(engine.TensorNarrow(spatial, 1, i, 1), new[] { n, c, h, w })).ToArray();

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

        // Each pass reads only the previous pass; upsample reads only the last (forward_2).
        for (int p = 0; p < 4; p++)
            feats = Propagate(engine, layers, feats, p % 2 == 0 ? flowsBackward : flowsForward, p % 2 == 0, _align[p], _aligner, training);

        var hr = engine.TensorConcatenate(feats.Select(f => engine.Reshape(f, new[] { n, 1, c, h, w })).ToArray(), 1);
        hr = RvrtBlocks.PerFrame(engine, layers, _beforeUpsample, hr);
        hr = Upsample(engine, layers, _up1, hr);
        hr = Upsample(engine, layers, _up2, hr);
        hr = RvrtBlocks.PerFrame(engine, layers, _hr, hr);
        hr = RvrtBlocks.PerFrame(engine, layers, _last, hr);
        var lowUp = engine.Interpolate(engine.Reshape(lqs, new[] { n * t, 3, h, w }), new[] { 4 * h, 4 * w }, InterpolateMode.Bilinear, false);
        return engine.TensorAdd(hr, engine.Reshape(lowUp, new[] { n, t, 3, 4 * h, 4 * w }));
    }

    // conv to 4C with LeakyReLU 0.1, then pixel shuffle x2.
    private static Tensor<T> Upsample(IEngine engine, IReadOnlyList<ILayer<T>> layers, int conv, Tensor<T> x)
    {
        var y = RvrtBlocks.PerFrame(engine, layers, conv, x);
        int n = y.Shape[0], d = y.Shape[1];
        var shuffled = engine.PixelShuffle(engine.Reshape(y, new[] { n * d, y.Shape[2], y.Shape[3], y.Shape[4] }), 2);
        return engine.Reshape(shuffled, new[] { n, d, shuffled.Shape[1], shuffled.Shape[2], shuffled.Shape[3] });
    }

    /// <summary>
    /// BasicRecurrentSwin.propagate: the first frame of a pass aligns three copies of itself; every later frame aligns
    /// [aligned previous output, current, aligned second-previous output (zeros for the second frame)].
    /// </summary>
    private static Tensor<T>[] Propagate(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T>[] feats, Tensor<T>[] flows,
        bool backward, PsrtAlignment<T> align, IPsrtFrameAligner<T> aligner, bool training)
    {
        int t = feats.Length;
        var order = backward ? Enumerable.Range(0, t).Reverse().ToArray() : Enumerable.Range(0, t).ToArray();
        var outputs = new List<Tensor<T>>();
        for (int i = 0; i < t; i++)
        {
            int idx = order[i];
            var current = feats[idx];
            Tensor<T> stack;
            if (i == 0)
            {
                stack = Stack(engine, current, current, current);
            }
            else
            {
                var flowN1 = backward ? flows[idx] : flows[idx - 1];
                var condN1 = aligner.Align(engine, layers, outputs[i - 1], current, flowN1);
                var condN2 = new Tensor<T>(condN1.Shape.ToArray());
                if (i > 1)
                {
                    var flowN2 = backward ? flows[idx + 1] : flows[idx - 2];
                    flowN2 = engine.TensorAdd(flowN1, BasicVsrBlocks.FlowWarp(engine, flowN2, flowN1, GridSamplePadding.Zeros));
                    condN2 = aligner.Align(engine, layers, outputs[i - 2], current, flowN2);
                }
                stack = Stack(engine, condN1, current, condN2);
            }
            outputs.Add(align.Forward(engine, layers, stack, training));
        }
        if (backward) outputs.Reverse();
        return outputs.ToArray();
    }

    private static Tensor<T> Stack(IEngine engine, Tensor<T> a, Tensor<T> b, Tensor<T> c)
    {
        int n = a.Shape[0], ch = a.Shape[1], h = a.Shape[2], w = a.Shape[3];
        return engine.TensorConcatenate(new[] { a, b, c }.Select(f => engine.Reshape(f, new[] { n, 1, ch, h, w })).ToArray(), 1);
    }

    /// <summary>
    /// flow_warp_avg_patch: the flow, reflect-padded to whole 8x8 patches, is averaged per patch and spread back over
    /// it, so every pixel of a patch moves by the same offset; features are then sampled nearest-neighbour with zero
    /// padding and align_corners=True.
    /// </summary>
    internal static Tensor<T> PatchWarp(IEngine engine, Tensor<T> x, Tensor<T> flow)
    {
        int n = flow.Shape[0], h = flow.Shape[2], w = flow.Shape[3];
        int hp = (h + PatchSize - 1) / PatchSize * PatchSize, wp = (w + PatchSize - 1) / PatchSize * PatchSize;
        var padded = ReflectPadEnd(engine, ReflectPadEnd(engine, flow, 2, hp - h), 3, wp - w);
        var pooled = engine.ReduceMean(engine.Reshape(padded, new[] { n, 2, hp / PatchSize, PatchSize, wp / PatchSize, PatchSize }),
            new[] { 3, 5 }, true);
        var spread = engine.Reshape(engine.TensorTile(pooled, new[] { 1, 1, 1, PatchSize, 1, PatchSize }), new[] { n, 2, hp, wp });
        if (hp != h || wp != w) spread = engine.TensorNarrow(engine.TensorNarrow(spread, 2, 0, h), 3, 0, w);
        return BasicVsrBlocks.FlowWarp(engine, x, spread, GridSamplePadding.Zeros, GridSampleMode.Nearest);
    }

    // F.pad(mode='reflect') at the end of an axis: x[s-2], x[s-3], ..., x[s-1-extra].
    private static Tensor<T> ReflectPadEnd(IEngine engine, Tensor<T> x, int axis, int extra)
    {
        if (extra == 0) return x;
        int size = x.Shape[axis];
        if (extra >= size)
            throw new ArgumentException($"Reflect padding of {extra} needs more than {size} positions.", nameof(extra));
        var mirrored = engine.TensorFlip(engine.TensorNarrow(x, axis, size - 1 - extra, extra), new[] { axis });
        return engine.TensorConcatenate(new[] { x, mirrored }, axis);
    }
}
