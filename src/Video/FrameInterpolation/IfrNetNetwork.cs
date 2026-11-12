using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>IFRNet's Encoder: four levels of a stride-2 convrelu and a convrelu, 1/2 to 1/16 size.</summary>
internal sealed class IfrEncoder<T> : LazyWeightModule<T>
{
    private readonly AmtConvPRelu<T>[] _convs;

    public IfrEncoder(IReadOnlyList<int> channels, int firstKernel)
    {
        _convs = new AmtConvPRelu<T>[8];
        int inChannels = 3;
        for (int i = 0; i < 4; i++)
        {
            _convs[2 * i] = new AmtConvPRelu<T>(inChannels, channels[i], i == 0 ? firstKernel : 3, 2);
            _convs[2 * i + 1] = new AmtConvPRelu<T>(channels[i], channels[i]);
            inChannels = channels[i];
        }
    }

    public Tensor<T>[] Forward(Tensor<T> x)
    {
        var levels = new Tensor<T>[4];
        for (int i = 0; i < 4; i++) levels[i] = x = _convs[2 * i + 1].Forward(_convs[2 * i].Forward(x));
        return levels;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}

/// <summary>One forward pass: the refined intermediate features at 1/8, 1/4 and 1/2 size (finest first), the flows of
/// every decoder, and the synthesized frame.</summary>
internal sealed class IfrNetOutput<T>
{
    public Tensor<T>[] IntermediateFeatures { get; set; } = Array.Empty<Tensor<T>>();
    public Tensor<T> Prediction { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> Flow0 { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> Flow1 { get; set; } = new Tensor<T>(new[] { 0 });
}

/// <summary>
/// IFRNet's Model: a shared encoder and four decoders from 1/16 size, each refining both flows from time t and the
/// intermediate feature; the last predicts the blending mask and a residual.
/// </summary>
internal sealed class IfrNetNetwork<T> : LazyWeightModule<T>
{
    private readonly IfrEncoder<T> _encoder;
    private readonly AmtDecoderBlock<T>[] _decoders;

    public IfrNetNetwork(IReadOnlyList<int> e, int side, int firstKernel)
    {
        _encoder = new IfrEncoder<T>(e, firstKernel);
        _decoders = new[]
        {
            new AmtDecoderBlock<T>(2 * e[3] + 1, 2 * e[3], side, 4 + e[2]),
            new AmtDecoderBlock<T>(3 * e[2] + 4, 3 * e[2], side, 4 + e[1]),
            new AmtDecoderBlock<T>(3 * e[1] + 4, 3 * e[1], side, 4 + e[0]),
            new AmtDecoderBlock<T>(3 * e[0] + 4, 3 * e[0], side, 8),
        };
    }

    /// <summary>The encoder's pyramid for a mean-subtracted frame (the target's features supervise the decoders').</summary>
    public Tensor<T>[] Encode(Tensor<T> image) => _encoder.Forward(image);

    /// <summary>
    /// Model.forward / inference on frames [N, 3, H, W] in [0, 1] with sides divisible by 16: the mean of both frames is
    /// subtracted, the decoders run coarse-to-fine, and the blend plus residual is clamped to [0, 1].
    /// </summary>
    public IfrNetOutput<T> Forward(Tensor<T> image0, Tensor<T> image1, double t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = image0.Shape.ToArray();
        var meanFull = FrameMean(image0, image1, shape);
        var img0 = Engine.TensorSubtract(image0, meanFull);
        var img1 = Engine.TensorSubtract(image1, meanFull);
        var f0 = _encoder.Forward(img0);
        var f1 = _encoder.Forward(img1);

        int n = shape[0], h4 = f0[3].Shape[2], w4 = f0[3].Shape[3];
        var embt = new Tensor<T>(new[] { n, 1, h4, w4 });
        embt.Fill(ops.FromDouble(t));
        var output = _decoders[0].Forward(Engine.TensorConcatenate(new[] { f0[3], f1[3], embt }, 1));
        var flow0 = Engine.TensorNarrow(output, 1, 0, 2);
        var flow1 = Engine.TensorNarrow(output, 1, 2, 2);
        var features = new List<Tensor<T>>();
        Tensor<T> Up(Tensor<T> flow)
            => Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, flow.Shape[2] * 2, flow.Shape[3] * 2), ops.FromDouble(2.0));
        for (int d = 1; d < 4; d++)
        {
            var ft = Engine.TensorNarrow(output, 1, 4, output.Shape[1] - 4);
            features.Add(ft);
            int level = 3 - d;
            output = _decoders[d].Forward(Engine.TensorConcatenate(new[]
            {
                ft, AmtOps.Warp(Engine, f0[level], flow0), AmtOps.Warp(Engine, f1[level], flow1), flow0, flow1,
            }, 1));
            flow0 = Engine.TensorAdd(Engine.TensorNarrow(output, 1, 0, 2), Up(flow0));
            flow1 = Engine.TensorAdd(Engine.TensorNarrow(output, 1, 2, 2), Up(flow1));
        }
        var mask = Engine.Sigmoid(Engine.TensorNarrow(output, 1, 4, 1));
        var residual = Engine.TensorNarrow(output, 1, 5, 3);
        var maskFull = Engine.TensorBroadcastTo(mask, shape);
        var merged = Engine.TensorAdd(Engine.TensorAdd(Engine.TensorMultiply(maskFull, AmtOps.Warp(Engine, img0, flow0)),
            Engine.TensorMultiply(Engine.ScalarMinusTensor(ops.One, maskFull), AmtOps.Warp(Engine, img1, flow1))), meanFull);
        features.Reverse();
        return new IfrNetOutput<T>
        {
            IntermediateFeatures = features.ToArray(),
            Prediction = Engine.TensorClamp(Engine.TensorAdd(merged, residual), ops.Zero, ops.One),
            Flow0 = flow0,
            Flow1 = flow1,
        };
    }

    /// <summary>The mean of both frames, broadcast to <paramref name="shape"/>, that Forward subtracts.</summary>
    public Tensor<T> FrameMean(Tensor<T> image0, Tensor<T> image1, int[] shape)
        => Engine.TensorBroadcastTo(Engine.TensorMultiplyScalar(Engine.TensorAdd(Engine.ReduceMean(image0, new[] { 1, 2, 3 }, true),
            Engine.ReduceMean(image1, new[] { 1, 2, 3 }, true)), MathHelper.GetNumericOperations<T>().FromDouble(0.5)), shape);

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _encoder }.Concat(_decoders);
}

/// <summary>IFRNet's loss.py terms beyond AMT's shared Charbonnier and ternary.</summary>
internal static class IfrOps
{
    /// <summary>
    /// Geometry(3): each channel's 3x3 census d / √(0.81 + d²), the mean over channels·taps of δ²/(0.1 + δ²) between the
    /// two features (neither detached), zeroed on the one-pixel border and averaged over every pixel.
    /// </summary>
    internal static Tensor<T> Geometry<T>(IEngine engine, Tensor<T> x, Tensor<T> y)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var identity = new Tensor<T>(new[] { 9, 1, 3, 3 });
        for (int k = 0; k < 9; k++) identity[k * 9 + k] = ops.One;
        Tensor<T> Census(Tensor<T> f)
        {
            var flat = engine.Reshape(f, new[] { b * c, 1, h, w });
            var difference = engine.TensorSubtract(engine.Conv2D(flat, identity, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }),
                engine.TensorBroadcastTo(flat, new[] { b * c, 9, h, w }));
            var normalized = engine.TensorDivide(difference, engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(difference), ops.FromDouble(0.81))));
            return engine.Reshape(normalized, new[] { b, c * 9, h, w });
        }
        var squared = engine.TensorSquare(engine.TensorSubtract(Census(x), Census(y)));
        var dist = engine.ReduceMean(engine.TensorDivide(squared, engine.TensorAddScalar(squared, ops.FromDouble(0.1))), new[] { 1 }, true);
        var mask = new Tensor<T>(dist._shape);
        for (int n = 0; n < b; n++)
            for (int yy = 1; yy < h - 1; yy++)
                for (int xx = 1; xx < w - 1; xx++)
                    mask[(n * h + yy) * w + xx] = ops.One;
        return engine.ReduceMean(engine.TensorMultiply(dist, mask), new[] { 0, 1, 2, 3 }, false);
    }
}
