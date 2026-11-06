using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// mmcv's <c>ConvModule</c> as EDVR uses it: a biased convolution with an optional LeakyReLU(0.1), initialized by
/// <c>kaiming_init(mode='fan_out', distribution='normal')</c> with the activation's gain and zero bias.
/// </summary>
internal static class EdvrConv
{
    internal static ConvolutionalLayer<T> Module<T>(int outChannels, int kernel, bool activation, int stride = 1)
    {
        double slope = activation ? 0.1 : 0.0;
        double std = Math.Sqrt(2.0 / (1 + slope * slope)) / Math.Sqrt((double)outChannels * kernel * kernel);
        return new ConvolutionalLayer<T>(outChannels, kernel, stride, kernel / 2,
            activation ? new LeakyReLUActivation<T>(0.1) : null, new NormalInitializationStrategy<T>(0.0, std), null, 1,
            Enums.BiasMode.Always);
    }
}

/// <summary>
/// EDVR's <c>ModulatedDCNPack</c>: a zero-initialized 3x3 convolution of a guiding feature predicts the offsets
/// (first two thirds of its 27·G channels) and, through a sigmoid, the modulation masks (last third) of a 3x3 modulated
/// deformable convolution (DCNv2, G deformable groups) of the input. The deformable kernel takes mmcv's uniform
/// ±1/√(C·9) initialization with zero bias.
/// </summary>
internal sealed class ModulatedDcnPack<T>
{
    private readonly int _offset;
    private readonly int _main;
    private readonly int _deformGroups;

    internal ModulatedDcnPack(List<ILayer<T>> layers, int channels, int deformGroups)
    {
        _deformGroups = deformGroups;
        _offset = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(3 * deformGroups * 9, 3, null, new ZeroInitializationStrategy<T>()));
        _main = layers.Count;
        var main = new ConvolutionalLayer<T>(channels, 3, 1, 1, null,
            new UniformInitializationStrategy<T>(1.0 / Math.Sqrt(channels * 9.0)), null, 1, Enums.BiasMode.Always);
        layers.Add(main);
        // Materialize the kernel: it is read directly below, never through the holder's own forward.
        using (new NoGradScope<T>()) main.Forward(new Tensor<T>(new[] { 1, channels, 4, 4 }));
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> guide)
    {
        var raw = BasicVsrBlocks.Run(layers, _offset, guide);
        int k = _deformGroups * 9;
        var offset = engine.TensorNarrow(raw, 1, 0, 2 * k);
        var mask = engine.Sigmoid(engine.TensorNarrow(raw, 1, 2 * k, k));
        var holder = (ConvolutionalLayer<T>)layers[_main];
        var y = engine.DeformableConv2DGrouped(x, holder.KernelsForFolding, offset, mask, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, 1, _deformGroups);
        var bias = holder.BiasesForFolding;
        return bias is null ? y : engine.TensorAdd(y, engine.Reshape(bias, new[] { 1, bias.Length, 1, 1 }));
    }
}

/// <summary>
/// EDVR's <c>PCDAlignment</c>: pyramid (three levels, coarse to fine) and cascading deformable alignment of a
/// neighbor's features to the reference frame's.
/// </summary>
internal sealed class PcdAlignmentBlock<T>
{
    // Index 0 is level 1 (full size), 2 is level 3 (quarter size).
    private readonly int[] _offsetConv1 = new int[3];
    private readonly int[] _offsetConv2 = new int[3];
    private readonly int[] _offsetConv3 = new int[3];
    private readonly ModulatedDcnPack<T>[] _dcn = new ModulatedDcnPack<T>[3];
    private readonly int[] _featConv = new int[3];
    private readonly int _casOffsetConv1;
    private readonly int _casOffsetConv2;
    private readonly ModulatedDcnPack<T> _casDcn;

    internal PcdAlignmentBlock(List<ILayer<T>> layers, int channels, int deformGroups)
    {
        for (int level = 2; level >= 0; level--)
        {
            _offsetConv1[level] = Add(layers, EdvrConv.Module<T>(channels, 3, true));
            _offsetConv2[level] = Add(layers, EdvrConv.Module<T>(channels, 3, true));
            if (level < 2) _offsetConv3[level] = Add(layers, EdvrConv.Module<T>(channels, 3, true));
            _dcn[level] = new ModulatedDcnPack<T>(layers, channels, deformGroups);
            // feat_conv: with LeakyReLU at level 2, without at level 1.
            if (level < 2) _featConv[level] = Add(layers, EdvrConv.Module<T>(channels, 3, level == 1));
        }
        _casOffsetConv1 = Add(layers, EdvrConv.Module<T>(channels, 3, true));
        _casOffsetConv2 = Add(layers, EdvrConv.Module<T>(channels, 3, true));
        _casDcn = new ModulatedDcnPack<T>(layers, channels, deformGroups);
    }

    private static int Add(List<ILayer<T>> layers, ILayer<T> layer)
    {
        layers.Add(layer);
        return layers.Count - 1;
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T>[] neighbor, Tensor<T>[] reference)
    {
        var slope = MathHelper.GetNumericOperations<T>().FromDouble(0.1);
        Tensor<T>? upOffset = null, upFeat = null, feat = null;
        for (int level = 2; level >= 0; level--)
        {
            var offset = BasicVsrBlocks.Run(layers, _offsetConv1[level], engine.TensorConcatenate(new[] { neighbor[level], reference[level] }, 1));
            if (upOffset is null || upFeat is null)
            {
                offset = BasicVsrBlocks.Run(layers, _offsetConv2[level], offset);
                feat = engine.LeakyReLU(_dcn[level].Forward(engine, layers, neighbor[level], offset), slope);
            }
            else
            {
                offset = BasicVsrBlocks.Run(layers, _offsetConv2[level], engine.TensorConcatenate(new[] { offset, upOffset }, 1));
                offset = BasicVsrBlocks.Run(layers, _offsetConv3[level], offset);
                var aligned = _dcn[level].Forward(engine, layers, neighbor[level], offset);
                feat = BasicVsrBlocks.Run(layers, _featConv[level], engine.TensorConcatenate(new[] { aligned, upFeat }, 1));
            }
            if (level > 0)
            {
                upOffset = engine.TensorMultiplyScalar(Up(engine, offset), MathHelper.GetNumericOperations<T>().FromDouble(2.0));
                upFeat = Up(engine, feat);
            }
        }
        var current = feat ?? throw new InvalidOperationException("PCD alignment produced no features.");
        var casOffset = BasicVsrBlocks.Run(layers, _casOffsetConv2,
            BasicVsrBlocks.Run(layers, _casOffsetConv1, engine.TensorConcatenate(new[] { current, reference[0] }, 1)));
        return engine.LeakyReLU(_casDcn.Forward(engine, layers, current, casOffset), slope);
    }

    // nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).
    internal static Tensor<T> Up(IEngine engine, Tensor<T> x)
        => engine.Interpolate(x, new[] { x.Shape[2] * 2, x.Shape[3] * 2 }, InterpolateMode.Bilinear, false);
}

/// <summary>
/// EDVR's <c>TSAFusion</c>: temporal attention (sigmoid of each frame's embedding correlation with the reference)
/// weights the aligned features, a 1x1 convolution fuses them, and a two-level spatial attention pyramid (max and
/// average pooling) produces a multiplicative (sigmoid · 2) and an additive modulation.
/// </summary>
internal sealed class TsaFusionBlock<T>
{
    private readonly int _center;
    private readonly int _temporal1;
    private readonly int _temporal2;
    private readonly int _fusion;
    private readonly int _spatial1, _spatial2, _spatial3, _spatial4, _spatial5;
    private readonly int _spatialL1, _spatialL2, _spatialL3;
    private readonly int _spatialAdd1, _spatialAdd2;

    internal TsaFusionBlock(List<ILayer<T>> layers, int channels, int centerFrame)
    {
        _center = centerFrame;
        int Add(ILayer<T> layer)
        {
            layers.Add(layer);
            return layers.Count - 1;
        }
        _temporal1 = Add(BasicVsrBlocks.Conv<T>(channels, 3, null));
        _temporal2 = Add(BasicVsrBlocks.Conv<T>(channels, 3, null));
        _fusion = Add(EdvrConv.Module<T>(channels, 1, true));
        _spatial1 = Add(EdvrConv.Module<T>(channels, 1, true));
        _spatial2 = Add(EdvrConv.Module<T>(channels, 1, true));
        _spatial3 = Add(EdvrConv.Module<T>(channels, 3, true));
        _spatial4 = Add(EdvrConv.Module<T>(channels, 1, true));
        _spatial5 = Add(BasicVsrBlocks.Conv<T>(channels, 3, null));
        _spatialL1 = Add(EdvrConv.Module<T>(channels, 1, true));
        _spatialL2 = Add(EdvrConv.Module<T>(channels, 3, true));
        _spatialL3 = Add(EdvrConv.Module<T>(channels, 3, true));
        _spatialAdd1 = Add(EdvrConv.Module<T>(channels, 1, true));
        _spatialAdd2 = Add(BasicVsrBlocks.Conv<T>(channels, 1, null));
    }

    /// <summary>Fuses aligned features (one [N, C, H, W] per frame) into [N, C, H, W].</summary>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T>[] aligned)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = aligned[0].Shape[0], c = aligned[0].Shape[1], h = aligned[0].Shape[2], w = aligned[0].Shape[3], t = aligned.Length;
        var reference = BasicVsrBlocks.Run(layers, _temporal1, aligned[_center]);
        var embeddings = BasicVsrBlocks.Run(layers, _temporal2, engine.TensorConcatenate(aligned, 0));
        var weighted = new Tensor<T>[t];
        for (int i = 0; i < t; i++)
        {
            var corr = engine.ReduceSum(engine.TensorMultiply(engine.TensorNarrow(embeddings, 0, i * n, n), reference), new[] { 1 }, true);
            weighted[i] = engine.TensorMultiply(aligned[i], engine.Sigmoid(corr));
        }
        var stacked = engine.TensorConcatenate(weighted, 1);
        var feat = BasicVsrBlocks.Run(layers, _fusion, stacked);

        Tensor<T> Pool(Tensor<T> x) => engine.TensorConcatenate(new[] { engine.MaxPool2D(x, 3, 2, 1), engine.AvgPool2D(x, 3, 2, 1, true) }, 1);
        var attn = BasicVsrBlocks.Run(layers, _spatial2, Pool(BasicVsrBlocks.Run(layers, _spatial1, stacked)));
        var level = BasicVsrBlocks.Run(layers, _spatialL1, attn);
        level = BasicVsrBlocks.Run(layers, _spatialL3, BasicVsrBlocks.Run(layers, _spatialL2, Pool(level)));
        level = PcdAlignmentBlock<T>.Up(engine, level);
        attn = engine.TensorAdd(BasicVsrBlocks.Run(layers, _spatial3, attn), level);
        attn = BasicVsrBlocks.Run(layers, _spatial4, attn);
        attn = BasicVsrBlocks.Run(layers, _spatial5, PcdAlignmentBlock<T>.Up(engine, attn));
        var add = BasicVsrBlocks.Run(layers, _spatialAdd2, BasicVsrBlocks.Run(layers, _spatialAdd1, attn));
        var gate = engine.TensorMultiplyScalar(engine.Sigmoid(attn), ops.FromDouble(2.0));
        return engine.TensorAdd(engine.TensorMultiply(feat, gate), add);
    }
}

/// <summary>
/// IconVSR's <c>EDVRFeatureExtractor</c> (EDVR-M without reconstruction): a 3x3 convolution with LeakyReLU and
/// residual blocks per frame, a three-level feature pyramid (stride-2 ConvModules), PCD alignment of every frame to
/// the center frame, and TSA fusion. Frames [N, T, 3, H, W] (H, W multiples of 4) to features [N, C, H, W].
/// </summary>
internal sealed class EdvrFeatureExtractorBlock<T>
{
    private readonly ResidualTrunkBlock<T> _extraction;
    private readonly int _l2Conv1, _l2Conv2, _l3Conv1, _l3Conv2;
    private readonly PcdAlignmentBlock<T> _pcd;
    private readonly TsaFusionBlock<T> _tsa;
    private readonly int _center;

    internal EdvrFeatureExtractorBlock(List<ILayer<T>> layers, int channels, int numFrames, int deformGroups, int extractionBlocks)
    {
        First = layers.Count;
        _center = numFrames / 2;
        _extraction = new ResidualTrunkBlock<T>(layers, channels, extractionBlocks);
        _l2Conv1 = layers.Count;
        layers.Add(EdvrConv.Module<T>(channels, 3, true, stride: 2));
        _l2Conv2 = layers.Count;
        layers.Add(EdvrConv.Module<T>(channels, 3, true));
        _l3Conv1 = layers.Count;
        layers.Add(EdvrConv.Module<T>(channels, 3, true, stride: 2));
        _l3Conv2 = layers.Count;
        layers.Add(EdvrConv.Module<T>(channels, 3, true));
        _pcd = new PcdAlignmentBlock<T>(layers, channels, deformGroups);
        _tsa = new TsaFusionBlock<T>(layers, channels, _center);
        Count = layers.Count - First;
    }

    internal int First { get; }

    internal int Count { get; }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T>[] frames)
    {
        int n = frames[0].Shape[0], t = frames.Length;
        var l1 = _extraction.Forward(engine, layers, engine.TensorConcatenate(frames, 0));
        var l2 = BasicVsrBlocks.Run(layers, _l2Conv2, BasicVsrBlocks.Run(layers, _l2Conv1, l1));
        var l3 = BasicVsrBlocks.Run(layers, _l3Conv2, BasicVsrBlocks.Run(layers, _l3Conv1, l2));
        Tensor<T>[] Levels(int i) => new[] { engine.TensorNarrow(l1, 0, i * n, n), engine.TensorNarrow(l2, 0, i * n, n), engine.TensorNarrow(l3, 0, i * n, n) };
        var reference = Levels(_center);
        var aligned = new Tensor<T>[t];
        for (int i = 0; i < t; i++) aligned[i] = _pcd.Forward(engine, layers, Levels(i), reference);
        return _tsa.Forward(engine, layers, aligned);
    }
}

/// <summary>
/// IconVSRNet (Chan et al., CVPR 2021; open-mmlab/mmagic, models/editors/iconvsr): BasicVSR's bidirectional
/// propagation with information refill (at keyframes, every <c>keyframe_stride</c> frames and the last, the propagated
/// feature is fused by a 3x3 convolution with EDVR-M features of the surrounding 2p+1 frames) and coupled propagation
/// (the forward branch also takes the backward branch's feature).
/// </summary>
internal sealed class IconVsrBackbone<T>
{
    private readonly int _midChannels;
    private readonly int _padding;
    private readonly int _keyframeStride;
    private readonly int _backwardFusion;
    private readonly int _forwardFusion;

    internal IconVsrBackbone(List<ILayer<T>> layers, int midChannels, int numBlocks, int keyframeStride, int padding,
        int edvrChannels, int deformGroups, int edvrExtractionBlocks)
    {
        _midChannels = midChannels;
        _padding = padding;
        _keyframeStride = keyframeStride;
        SpyNet = new SpyNetFlowBlock<T>(layers);
        Edvr = new EdvrFeatureExtractorBlock<T>(layers, edvrChannels, 2 * padding + 1, deformGroups, edvrExtractionBlocks);
        _backwardFusion = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(midChannels, 3, null));
        _forwardFusion = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(midChannels, 3, null));
        Backward = new ResidualTrunkBlock<T>(layers, midChannels, numBlocks);
        Forward = new ResidualTrunkBlock<T>(layers, midChannels, numBlocks);
        Reconstruction = new X4ReconstructionBlock<T>(layers, midChannels);
    }

    internal SpyNetFlowBlock<T> SpyNet { get; }

    internal EdvrFeatureExtractorBlock<T> Edvr { get; }

    internal ResidualTrunkBlock<T> Backward { get; }

    internal ResidualTrunkBlock<T> Forward { get; }

    internal X4ReconstructionBlock<T> Reconstruction { get; }

    /// <summary>The keyframes of a t-frame clip: every stride-th frame, and the last.</summary>
    internal static List<int> Keyframes(int t, int stride)
    {
        var keyframes = new List<int>();
        for (int i = 0; i < t; i += stride) keyframes.Add(i);
        if (keyframes[keyframes.Count - 1] != t - 1) keyframes.Add(t - 1);
        return keyframes;
    }

    /// <summary>
    /// compute_refill_features' temporal padding: frames 2p, ..., p+1 before the clip and t-p-2, ..., t-2p-1 after it
    /// (for p = 2: [4, 3] and [-4, -5]). The release needs t ≥ 2p+1; shorter clips clamp the indices into the clip.
    /// </summary>
    internal static int[] RefillIndices(int t, int padding)
    {
        var indices = new List<int>();
        for (int j = 0; j < padding; j++) indices.Add(2 * padding - j);
        for (int i = 0; i < t; i++) indices.Add(i);
        for (int j = 0; j < padding; j++) indices.Add(t - padding - 2 - j);
        return indices.Select(i => Math.Min(t - 1, Math.Max(0, i))).ToArray();
    }

    /// <summary>
    /// spatial_padding: reflect-pad H and W at the bottom/right up to multiples of 4 (PCD's pyramid halves twice).
    /// Input frames carry no gradient, so the padding runs on the host.
    /// </summary>
    internal static Tensor<T> ReflectPadTo4(Tensor<T> frames)
    {
        int h = frames.Shape[2], w = frames.Shape[3];
        int ph = (4 - h % 4) % 4, pw = (4 - w % 4) % 4;
        if (ph == 0 && pw == 0) return frames;
        int planes = frames.Shape[0] * frames.Shape[1], hp = h + ph, wp = w + pw;
        var padded = new Tensor<T>(new[] { frames.Shape[0], frames.Shape[1], hp, wp });
        var source = frames.Data.Span;
        var target = padded.Data.Span;
        for (int p = 0; p < planes; p++)
            for (int y = 0; y < hp; y++)
                for (int x = 0; x < wp; x++)
                    target[(p * hp + y) * wp + x] = source[(p * h + RealBasicVsrCore<T>.Reflect101(y, h)) * w + RealBasicVsrCore<T>.Reflect101(x, w)];
        return padded;
    }

    /// <summary>IconVSRNet.forward: clips [N, T, 3, H, W] in [0, 1] to [N, T, 3, 4H, 4W].</summary>
    internal Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips)
    {
        int n = clips.Shape[0], t = clips.Shape[1], c = clips.Shape[2], hIn = clips.Shape[3], wIn = clips.Shape[4];
        var frames = new Tensor<T>[t];
        for (int i = 0; i < t; i++)
            frames[i] = ReflectPadTo4(engine.Reshape(engine.TensorNarrow(clips, 1, i, 1), new[] { n, c, hIn, wIn }));
        int h = frames[0].Shape[2], w = frames[0].Shape[3];

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

        var keyframes = Keyframes(t, _keyframeStride);
        var padded = RefillIndices(t, _padding).Select(i => frames[i]).ToArray();
        var refill = new Dictionary<int, Tensor<T>>();
        foreach (int k in keyframes) refill[k] = Edvr.Forward(engine, layers, padded.Skip(k).Take(2 * _padding + 1).ToArray());

        var backwardStates = new Tensor<T>[t];
        var state = new Tensor<T>(new[] { n, _midChannels, h, w });
        for (int i = t - 1; i >= 0; i--)
        {
            if (i < t - 1) state = BasicVsrBlocks.FlowWarp(engine, state, flowsBackward[i], GridSamplePadding.Zeros);
            if (refill.TryGetValue(i, out var feature))
                state = BasicVsrBlocks.Run(layers, _backwardFusion, engine.TensorConcatenate(new[] { state, feature }, 1));
            state = Backward.Forward(engine, layers, engine.TensorConcatenate(new[] { frames[i], state }, 1));
            backwardStates[i] = state;
        }

        var outputs = new Tensor<T>[t];
        state = new Tensor<T>(new[] { n, _midChannels, h, w });
        for (int i = 0; i < t; i++)
        {
            if (i > 0) state = BasicVsrBlocks.FlowWarp(engine, state, flowsForward[i - 1], GridSamplePadding.Zeros);
            if (refill.TryGetValue(i, out var feature))
                state = BasicVsrBlocks.Run(layers, _forwardFusion, engine.TensorConcatenate(new[] { state, feature }, 1));
            state = Forward.Forward(engine, layers, engine.TensorConcatenate(new[] { frames[i], backwardStates[i], state }, 1));
            var output = Reconstruction.Forward(engine, layers, state, frames[i]);
            if (h != hIn || w != wIn)
                output = engine.TensorNarrow(engine.TensorNarrow(output, 2, 0, 4 * hIn), 3, 0, 4 * wIn);
            outputs[i] = engine.Reshape(output, new[] { n, 1, c, 4 * hIn, 4 * wIn });
        }
        return engine.TensorConcatenate(outputs, 1);
    }

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(SpyNet.First).Take(SpyNet.Count));

    /// <summary>The EDVR-M feature extractor's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> EdvrParameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(Edvr.First).Take(Edvr.Count));
}
