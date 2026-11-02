using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// Building blocks shared by the BasicVSR family (BasicVSR, IconVSR, RealBasicVSR), following the authors' release
/// (open-mmlab/mmagic, models/editors/basicvsr/basicvsr_net.py and archs/sr_backbone.py).
/// </summary>
/// <remarks>
/// Every block adds its layers to the owning model's <c>Layers</c> list and keeps their indices, so a model whose
/// layers are replaced by deserialization stays wired to the restored layers.
/// </remarks>
internal static class BasicVsrBlocks
{
    /// <summary>mmagic's <c>flow_warp</c>: bilinear <c>grid_sample</c> with <c>align_corners=True</c>.</summary>
    /// <param name="engine">The engine, so the warp is recorded on the caller's tape.</param>
    /// <param name="image">[N, C, H, W].</param>
    /// <param name="flow">[N, 2, H, W], (dx, dy) in pixels.</param>
    /// <param name="padding">Zeros for features, border for SPyNet's image warp.</param>
    internal static Tensor<T> FlowWarp<T>(IEngine engine, Tensor<T> image, Tensor<T> flow, GridSamplePadding padding)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = image.Shape[0], h = image.Shape[2], w = image.Shape[3];
        var baseX = new Tensor<T>(new[] { n, 1, h, w });
        var baseY = new Tensor<T>(new[] { n, 1, h, w });
        var bx = baseX.Data.Span;
        var by = baseY.Data.Span;
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    bx[(b * h + y) * w + x] = ops.FromDouble(x);
                    by[(b * h + y) * w + x] = ops.FromDouble(y);
                }

        // align_corners=True: pixel p sits at 2p / (extent - 1) - 1.
        Tensor<T> Normalize(Tensor<T> pixels, int extent) => extent > 1
            ? engine.TensorAddScalar(engine.TensorMultiplyScalar(pixels, ops.FromDouble(2.0 / (extent - 1))), ops.FromDouble(-1.0))
            : engine.TensorMultiplyScalar(pixels, ops.Zero);
        var gx = Normalize(engine.TensorAdd(engine.TensorNarrow(flow, 1, 0, 1), baseX), w);
        var gy = Normalize(engine.TensorAdd(engine.TensorNarrow(flow, 1, 1, 1), baseY), h);
        if (padding == GridSamplePadding.Border)
        {
            // With align_corners=True, border padding samples the clamped coordinate, and its grid gradient is zero
            // where the clamp is active, exactly as PyTorch's; the engine differentiates only zero padding.
            gx = engine.TensorClamp(gx, ops.FromDouble(-1.0), ops.One);
            gy = engine.TensorClamp(gy, ops.FromDouble(-1.0), ops.One);
            padding = GridSamplePadding.Zeros;
        }
        var grid = engine.TensorConcatenate(new[]
        {
            engine.Reshape(gx, new[] { n, h, w, 1 }),
            engine.Reshape(gy, new[] { n, h, w, 1 }),
        }, 3);
        return engine.GridSample(image, grid, GridSampleMode.Bilinear, padding, true);
    }

    /// <summary>mmagic's <c>default_init_weights(scale)</c>: Kaiming-normal (fan-in, ReLU gain) times scale, zero bias.</summary>
    internal static IInitializationStrategy<T> ScaledKaiming<T>(int fanIn, double scale)
        => new NormalInitializationStrategy<T>(0.0, scale * Math.Sqrt(2.0 / fanIn));

    internal static ConvolutionalLayer<T> Conv<T>(int outChannels, int kernel, IActivationFunction<T>? activation,
        IInitializationStrategy<T>? init = null, bool bias = true)
        => new(outChannels, kernel, 1, kernel / 2, activation, init ?? new TorchDefaultInitializationStrategy<T>(), null, 1,
            bias ? Enums.BiasMode.Always : Enums.BiasMode.Never);

    internal static Tensor<T> Run<T>(IReadOnlyList<ILayer<T>> layers, int index, Tensor<T> x) => layers[index].Forward(x);
}

/// <summary>
/// SPyNet as BasicVSR uses it (Ranjan &amp; Black 2017; mmagic <c>SPyNet</c>): six pyramid levels, each a basic module of
/// five 7x7 convolutions (8-32-64-32-16-2, ReLU between), over ImageNet-normalized frames resized to a multiple of 32.
/// </summary>
internal sealed class SpyNetFlowBlock<T>
{
    internal const int Levels = 6;
    private static readonly int[] Widths = { 32, 64, 32, 16, 2 };
    private static readonly double[] Mean = { 0.485, 0.456, 0.406 };
    private static readonly double[] Std = { 0.229, 0.224, 0.225 };

    private readonly int[,] _convs = new int[Levels, 5];

    internal SpyNetFlowBlock(List<ILayer<T>> layers)
    {
        First = layers.Count;
        for (int level = 0; level < Levels; level++)
            for (int k = 0; k < 5; k++)
            {
                _convs[level, k] = layers.Count;
                layers.Add(BasicVsrBlocks.Conv(Widths[k], 7, k < 4 ? new ReLUActivation<T>() : null));
            }
        Count = layers.Count - First;
    }

    /// <summary>Index of the first SPyNet layer in the owner's list.</summary>
    internal int First { get; }

    /// <summary>Number of SPyNet layers.</summary>
    internal int Count { get; }

    /// <summary>Flow from <paramref name="reference"/> to <paramref name="support"/>, both [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> Flow(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> reference, Tensor<T> support)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int h = reference.Shape[2], w = reference.Shape[3];
        int hUp = (h + 31) / 32 * 32, wUp = (w + 31) / 32 * 32;
        if (hUp != h || wUp != w)
        {
            reference = engine.Interpolate(reference, new[] { hUp, wUp }, InterpolateMode.Bilinear, false);
            support = engine.Interpolate(support, new[] { hUp, wUp }, InterpolateMode.Bilinear, false);
        }

        var refs = new Tensor<T>[Levels];
        var supps = new Tensor<T>[Levels];
        refs[0] = Normalize(engine, ops, reference);
        supps[0] = Normalize(engine, ops, support);
        for (int level = 1; level < Levels; level++)
        {
            refs[level] = engine.AvgPool2D(refs[level - 1], 2, 2, 0, false);
            supps[level] = engine.AvgPool2D(supps[level - 1], 2, 2, 0, false);
        }

        int n = reference.Shape[0];
        var flow = new Tensor<T>(new[] { n, 2, hUp / 32, wUp / 32 });
        for (int level = 0; level < Levels; level++)
        {
            var r = refs[Levels - 1 - level];
            var s = supps[Levels - 1 - level];
            var flowUp = level == 0
                ? flow
                : engine.TensorMultiplyScalar(
                    engine.Interpolate(flow, new[] { r.Shape[2], r.Shape[3] }, InterpolateMode.Bilinear, true), ops.FromDouble(2.0));
            var x = engine.TensorConcatenate(new[] { r, BasicVsrBlocks.FlowWarp(engine, s, flowUp, GridSamplePadding.Border), flowUp }, 1);
            for (int k = 0; k < 5; k++) x = BasicVsrBlocks.Run(layers, _convs[level, k], x);
            flow = engine.TensorAdd(flowUp, x);
        }

        if (hUp == h && wUp == w) return flow;
        flow = engine.Interpolate(flow, new[] { h, w }, InterpolateMode.Bilinear, false);
        return engine.TensorConcatenate(new[]
        {
            engine.TensorMultiplyScalar(engine.TensorNarrow(flow, 1, 0, 1), ops.FromDouble((double)w / wUp)),
            engine.TensorMultiplyScalar(engine.TensorNarrow(flow, 1, 1, 1), ops.FromDouble((double)h / hUp)),
        }, 1);
    }

    private static Tensor<T> Normalize(IEngine engine, INumericOperations<T> ops, Tensor<T> image)
    {
        var channels = new Tensor<T>[3];
        for (int c = 0; c < 3; c++)
            channels[c] = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.TensorNarrow(image, 1, c, 1), ops.FromDouble(1.0 / Std[c])),
                ops.FromDouble(-Mean[c] / Std[c]));
        return engine.TensorConcatenate(channels, 1);
    }
}

/// <summary>
/// mmagic's <c>ResidualBlocksWithInputConv</c>: a 3x3 convolution with LeakyReLU(0.1), then residual blocks without
/// batch norm (x + conv(ReLU(conv(x))), both convolutions Kaiming-initialized at scale 0.1).
/// </summary>
internal sealed class ResidualTrunkBlock<T>
{
    private readonly int _input;
    private readonly int[] _conv1;
    private readonly int[] _conv2;

    internal ResidualTrunkBlock(List<ILayer<T>> layers, int outChannels, int numBlocks)
    {
        _input = layers.Count;
        layers.Add(BasicVsrBlocks.Conv(outChannels, 3, new LeakyReLUActivation<T>(0.1)));
        _conv1 = new int[numBlocks];
        _conv2 = new int[numBlocks];
        int fanIn = outChannels * 9;
        for (int i = 0; i < numBlocks; i++)
        {
            _conv1[i] = layers.Count;
            layers.Add(BasicVsrBlocks.Conv(outChannels, 3, new ReLUActivation<T>(), BasicVsrBlocks.ScaledKaiming<T>(fanIn, 0.1)));
            _conv2[i] = layers.Count;
            layers.Add(BasicVsrBlocks.Conv<T>(outChannels, 3, null, BasicVsrBlocks.ScaledKaiming<T>(fanIn, 0.1)));
        }
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x)
    {
        x = BasicVsrBlocks.Run(layers, _input, x);
        for (int i = 0; i < _conv1.Length; i++)
            x = engine.TensorAdd(x, BasicVsrBlocks.Run(layers, _conv2[i], BasicVsrBlocks.Run(layers, _conv1[i], x)));
        return x;
    }
}

/// <summary>
/// BasicVSR's reconstruction: two <c>PixelShufflePack</c> x2 stages (3x3 conv to 4C, pixel shuffle, LeakyReLU 0.1),
/// a 3x3 conv with LeakyReLU, a 3x3 conv to RGB, plus the bilinearly x4-upsampled low-resolution frame.
/// </summary>
internal sealed class X4ReconstructionBlock<T>
{
    private readonly int _up1;
    private readonly int _up2;
    private readonly int _convHr;
    private readonly int _convLast;

    internal X4ReconstructionBlock(List<ILayer<T>> layers, int midChannels)
    {
        // PixelShufflePack.init_weights: default_init_weights(self, 1).
        _up1 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(midChannels * 4, 3, null, BasicVsrBlocks.ScaledKaiming<T>(midChannels * 9, 1.0)));
        _up2 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(64 * 4, 3, null, BasicVsrBlocks.ScaledKaiming<T>(midChannels * 9, 1.0)));
        _convHr = layers.Count;
        layers.Add(BasicVsrBlocks.Conv(64, 3, new LeakyReLUActivation<T>(0.1)));
        _convLast = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(3, 3, null));
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> features, Tensor<T> lowRes)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> LeakyRelu(Tensor<T> x) => engine.LeakyReLU(x, ops.FromDouble(0.1));
        var x = LeakyRelu(engine.PixelShuffle(BasicVsrBlocks.Run(layers, _up1, features), 2));
        x = LeakyRelu(engine.PixelShuffle(BasicVsrBlocks.Run(layers, _up2, x), 2));
        x = BasicVsrBlocks.Run(layers, _convLast, BasicVsrBlocks.Run(layers, _convHr, x));
        var baseImage = engine.Interpolate(lowRes, new[] { lowRes.Shape[2] * 4, lowRes.Shape[3] * 4 }, InterpolateMode.Bilinear, false);
        return engine.TensorAdd(x, baseImage);
    }
}

/// <summary>
/// BasicVSRNet (mmagic): SPyNet, a backward and a forward propagation branch of <see cref="ResidualTrunkBlock{T}"/>,
/// a 1x1 fusion and the x4 reconstruction. RealBasicVSR runs it after its cleaning module.
/// </summary>
internal sealed class BasicVsrBackbone<T>
{
    private readonly int _midChannels;
    private readonly int _fusion;

    internal BasicVsrBackbone(List<ILayer<T>> layers, int midChannels, int numBlocks)
    {
        _midChannels = midChannels;
        SpyNet = new SpyNetFlowBlock<T>(layers);
        Backward = new ResidualTrunkBlock<T>(layers, midChannels, numBlocks);
        Forward = new ResidualTrunkBlock<T>(layers, midChannels, numBlocks);
        _fusion = layers.Count;
        layers.Add(BasicVsrBlocks.Conv(midChannels, 1, new LeakyReLUActivation<T>(0.1)));
        Reconstruction = new X4ReconstructionBlock<T>(layers, midChannels);
    }

    internal SpyNetFlowBlock<T> SpyNet { get; }

    internal ResidualTrunkBlock<T> Backward { get; }

    internal ResidualTrunkBlock<T> Forward { get; }

    internal X4ReconstructionBlock<T> Reconstruction { get; }

    /// <summary>BasicVSRNet.forward: clips [N, T, 3, H, W] in [0, 1] to [N, T, 3, 4H, 4W].</summary>
    internal Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips)
    {
        int n = clips.Shape[0], t = clips.Shape[1], c = clips.Shape[2], h = clips.Shape[3], w = clips.Shape[4];
        var frames = new Tensor<T>[t];
        for (int i = 0; i < t; i++) frames[i] = engine.Reshape(engine.TensorNarrow(clips, 1, i, 1), new[] { n, c, h, w });

        // flows_backward[i]: frame i to frame i+1; flows_forward[i]: frame i+1 to frame i.
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

        var backwardStates = new Tensor<T>[t];
        var state = new Tensor<T>(new[] { n, _midChannels, h, w });
        for (int i = t - 1; i >= 0; i--)
        {
            if (i < t - 1) state = BasicVsrBlocks.FlowWarp(engine, state, flowsBackward[i], GridSamplePadding.Zeros);
            state = Backward.Forward(engine, layers, engine.TensorConcatenate(new[] { frames[i], state }, 1));
            backwardStates[i] = state;
        }

        var outputs = new Tensor<T>[t];
        state = new Tensor<T>(new[] { n, _midChannels, h, w });
        for (int i = 0; i < t; i++)
        {
            if (i > 0) state = BasicVsrBlocks.FlowWarp(engine, state, flowsForward[i - 1], GridSamplePadding.Zeros);
            state = Forward.Forward(engine, layers, engine.TensorConcatenate(new[] { frames[i], state }, 1));
            var fused = layers[_fusion].Forward(engine.TensorConcatenate(new[] { backwardStates[i], state }, 1));
            var output = Reconstruction.Forward(engine, layers, fused, frames[i]);
            outputs[i] = engine.Reshape(output, new[] { n, 1, c, 4 * h, 4 * w });
        }
        return engine.TensorConcatenate(outputs, 1);
    }

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(SpyNet.First).Take(SpyNet.Count));
}
