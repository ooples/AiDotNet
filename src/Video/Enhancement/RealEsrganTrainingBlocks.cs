using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// mmagic's <c>UNetDiscriminatorWithSpectralNorm</c> (Real-ESRGAN's U-Net discriminator): a 3x3 convolution, three
/// spectrally normalized 4x4 stride-2 convolutions, three bilinear x2 upsampling stages with spectrally normalized 3x3
/// convolutions and skip additions, two more normalized 3x3 convolutions and a 3x3 convolution to one logit per pixel,
/// LeakyReLU(0.2) throughout.
/// </summary>
internal sealed class SpectralNormUNetDiscriminator<T>
{
    private const double Slope = 0.2;
    private readonly int _conv0;
    private readonly int[] _normalized = new int[8];
    private readonly int _conv9;
    private readonly double[][] _u = new double[8][];
    private readonly double[][] _v = new double[8][];
    private readonly Random _random;

    internal SpectralNormUNetDiscriminator(List<ILayer<T>> layers, int midChannels, Random random)
    {
        _random = random;
        First = layers.Count;
        int m = midChannels;
        _conv0 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(m, 3, null));
        // (out, kernel, stride) of conv_1 .. conv_8, all bias-free.
        var specs = new[] { (2 * m, 4, 2), (4 * m, 4, 2), (8 * m, 4, 2), (4 * m, 3, 1), (2 * m, 3, 1), (m, 3, 1), (m, 3, 1), (m, 3, 1) };
        var inputs = new[] { m, 2 * m, 4 * m, 8 * m, 4 * m, 2 * m, m, m };
        Specs = specs;
        for (int i = 0; i < 8; i++)
        {
            _normalized[i] = layers.Count;
            var holder = new ConvolutionalLayer<T>(specs[i].Item1, specs[i].Item2, specs[i].Item3, 1,
                null, new AiDotNet.Initialization.TorchDefaultInitializationStrategy<T>(), null, 1, Enums.BiasMode.Never);
            layers.Add(holder);
            // Materialize the kernel: it is read directly below, never through the holder's own forward.
            using (new NoGradScope<T>()) holder.Forward(new Tensor<T>(new[] { 1, inputs[i], 8, 8 }));
            // torch.nn.utils.spectral_norm starts u and v as normalized standard normal draws.
            _u[i] = UnitVector(specs[i].Item1);
            _v[i] = UnitVector(inputs[i] * specs[i].Item2 * specs[i].Item2);
        }
        _conv9 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(1, 3, null));
        Count = layers.Count - First;
    }

    private (int Out, int Kernel, int Stride)[] Specs { get; }

    internal int First { get; }

    internal int Count { get; }

    private double[] UnitVector(int length)
    {
        var v = new double[length];
        double norm = 0;
        for (int i = 0; i < length; i++)
        {
            v[i] = Math.Sqrt(-2.0 * Math.Log(1.0 - _random.NextDouble())) * Math.Cos(2.0 * Math.PI * _random.NextDouble());
            norm += v[i] * v[i];
        }
        norm = Math.Sqrt(norm) + 1e-12;
        for (int i = 0; i < length; i++) v[i] /= norm;
        return v;
    }

    /// <summary>
    /// torch.nn.utils.spectral_norm: in training one power iteration on the host (v = Wᵀu/‖·‖, u = Wv/‖·‖, both
    /// detached and kept), in evaluation the kept u and v; then W / (uᵀWv) on the tape, so the gradient flows through σ
    /// as PyTorch's does.
    /// </summary>
    private Tensor<T> NormalizedKernel(IEngine engine, IReadOnlyList<ILayer<T>> layers, int i, bool training)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var kernel = ((ConvolutionalLayer<T>)layers[_normalized[i]]).KernelsForFolding;
        int rows = kernel.Shape[0], cols = kernel.Length / rows;
        var w = kernel.Data.Span;
        var u = _u[i];
        var v = _v[i];
        if (training)
        {
            for (int c = 0; c < cols; c++)
            {
                double s = 0;
                for (int r = 0; r < rows; r++) s += ops.ToDouble(w[r * cols + c]) * u[r];
                v[c] = s;
            }
            Normalize(v);
            for (int r = 0; r < rows; r++)
            {
                double s = 0;
                for (int c = 0; c < cols; c++) s += ops.ToDouble(w[r * cols + c]) * v[c];
                u[r] = s;
            }
            Normalize(u);
        }
        var uTensor = new Tensor<T>(new[] { rows });
        var vTensor = new Tensor<T>(new[] { 1, cols });
        for (int r = 0; r < rows; r++) uTensor[r] = ops.FromDouble(u[r]);
        for (int c = 0; c < cols; c++) vTensor[c] = ops.FromDouble(v[c]);
        var matrix = engine.Reshape(kernel, new[] { rows, cols });
        var wv = engine.ReduceSum(engine.TensorMultiply(matrix, vTensor), new[] { 1 }, false);
        var sigma = engine.ReduceSum(engine.TensorMultiply(wv, uTensor), new[] { 0 }, false);
        return engine.TensorMultiply(kernel, engine.Reshape(engine.TensorReciprocal(sigma), new[] { 1, 1, 1, 1 }));
    }

    private static void Normalize(double[] x)
    {
        double norm = 0;
        for (int i = 0; i < x.Length; i++) norm += x[i] * x[i];
        norm = Math.Sqrt(norm) + 1e-12;
        for (int i = 0; i < x.Length; i++) x[i] /= norm;
    }

    private Tensor<T> Normalized(IEngine engine, IReadOnlyList<ILayer<T>> layers, int i, Tensor<T> x, bool training)
    {
        var spec = Specs[i];
        var y = engine.Conv2D(x, NormalizedKernel(engine, layers, i, training), spec.Stride, 1, 1);
        return engine.LeakyReLU(y, MathHelper.GetNumericOperations<T>().FromDouble(Slope));
    }

    /// <summary>Per-pixel logits [N, 1, H, W] for images [N, 3, H, W] in [0, 1]; H and W must be multiples of 8.</summary>
    internal Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> images, bool training)
    {
        var slope = MathHelper.GetNumericOperations<T>().FromDouble(Slope);
        Tensor<T> Up(Tensor<T> x) => engine.Interpolate(x, new[] { x.Shape[2] * 2, x.Shape[3] * 2 }, InterpolateMode.Bilinear, false);
        var x0 = engine.LeakyReLU(BasicVsrBlocks.Run(layers, _conv0, images), slope);
        var x1 = Normalized(engine, layers, 0, x0, training);
        var x2 = Normalized(engine, layers, 1, x1, training);
        var x3 = Normalized(engine, layers, 2, x2, training);
        var x4 = engine.TensorAdd(Normalized(engine, layers, 3, Up(x3), training), x2);
        var x5 = engine.TensorAdd(Normalized(engine, layers, 4, Up(x4), training), x1);
        var x6 = engine.TensorAdd(Normalized(engine, layers, 5, Up(x5), training), x0);
        var y = Normalized(engine, layers, 7, Normalized(engine, layers, 6, x6, training), training);
        return BasicVsrBlocks.Run(layers, _conv9, y);
    }

    /// <summary>The discriminator's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> Parameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(First).Take(Count));
}

/// <summary>
/// mmagic's <c>PerceptualLoss</c> as Real-ESRGAN and RealBasicVSR configure it: VGG19 features (ImageNet-normalized input)
/// taken before the ReLU of conv1_2, conv2_2, conv3_4, conv4_4 and conv5_4 (torchvision layers 2, 7, 16, 25, 34),
/// weighted 0.1, 0.1, 1, 1, 1, with an L1 criterion. The VGG is frozen.
/// </summary>
internal sealed class Vgg19PerceptualBlock<T>
{
    private static readonly int[] Widths = { 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512 };
    // Index of the conv (0-based among the 16) after which a 2x2 max-pool follows, and the tapped convs with weights.
    private static readonly int[] PoolAfter = { 1, 3, 7, 11 };
    private static readonly (int Conv, double Weight)[] Taps = { (1, 0.1), (3, 0.1), (7, 1.0), (11, 1.0), (15, 1.0) };
    private static readonly double[] Mean = { 0.485, 0.456, 0.406 };
    private static readonly double[] Std = { 0.229, 0.224, 0.225 };
    private readonly int[] _convs = new int[16];

    internal Vgg19PerceptualBlock(List<ILayer<T>> layers)
    {
        First = layers.Count;
        for (int i = 0; i < 16; i++)
        {
            _convs[i] = layers.Count;
            layers.Add(BasicVsrBlocks.Conv<T>(Widths[i], 3, null));
        }
        Count = layers.Count - First;
    }

    internal int First { get; }

    internal int Count { get; }

    private List<Tensor<T>> Features(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> images)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var channels = new Tensor<T>[3];
        for (int c = 0; c < 3; c++)
            channels[c] = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.TensorNarrow(images, 1, c, 1), ops.FromDouble(1.0 / Std[c])),
                ops.FromDouble(-Mean[c] / Std[c]));
        var x = engine.TensorConcatenate(channels, 1);
        var features = new List<Tensor<T>>();
        int tap = 0;
        for (int i = 0; i < 16 && tap < Taps.Length; i++)
        {
            x = BasicVsrBlocks.Run(layers, _convs[i], x);
            if (Taps[tap].Conv == i)
            {
                features.Add(x);
                tap++;
            }
            x = engine.ReLU(x);
            if (Array.IndexOf(PoolAfter, i) >= 0) x = engine.MaxPool2D(x, 2, 2, 0);
        }
        return features;
    }

    /// <summary>Σ wₖ · mean |φₖ(output) - φₖ(target)| for images [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> output, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        List<Tensor<T>> targets;
        using (new NoGradScope<T>()) targets = Features(engine, layers, target);
        var outputs = Features(engine, layers, output);
        Tensor<T>? total = null;
        for (int k = 0; k < outputs.Count; k++)
        {
            var diff = engine.TensorAbs(engine.TensorSubtract(outputs[k], targets[k]));
            var term = engine.TensorMultiplyScalar(
                engine.ReduceMean(diff, Enumerable.Range(0, diff.Rank).ToArray(), false), ops.FromDouble(Taps[k].Weight));
            total = total is null ? term : engine.TensorAdd(total, term);
        }
        return total ?? throw new InvalidOperationException("The perceptual block produced no features.");
    }

    /// <summary>The VGG parameters (frozen).</summary>
    internal IReadOnlyList<Tensor<T>> Parameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(First).Take(Count));
}

/// <summary>
/// BasicSR's <c>RRDBNet</c> (Wang et al. 2018, ESRGAN; Real-ESRGAN's generator) at x4: a 3x3 convolution, residual-in-
/// residual dense blocks (three five-convolution dense blocks with growth 32 and residual scaling 0.2, the whole block
/// again scaled by 0.2), a 3x3 trunk convolution added back, two nearest-neighbour x2 upsamplings each followed by a 3x3
/// convolution and LeakyReLU(0.2), then a 3x3 convolution with LeakyReLU and a 3x3 convolution to RGB. The dense
/// blocks' convolutions take Kaiming-normal initialization scaled by 0.1; the rest keep PyTorch's default.
/// </summary>
internal sealed class RrdbNetBlock<T>
{
    private const double Slope = 0.2;
    private const double ResidualScale = 0.2;
    private readonly int _convFirst;
    private readonly int[][][] _dense;
    private readonly int _convBody;
    private readonly int _convUp1;
    private readonly int _convUp2;
    private readonly int _convHr;
    private readonly int _convLast;

    internal RrdbNetBlock(List<ILayer<T>> layers, int features, int blocks, int growth)
    {
        First = layers.Count;
        AiDotNet.ActivationFunctions.LeakyReLUActivation<T> Lrelu() => new(Slope);
        int Add(ILayer<T> layer)
        {
            layers.Add(layer);
            return layers.Count - 1;
        }
        _convFirst = Add(BasicVsrBlocks.Conv<T>(features, 3, null));
        _dense = new int[blocks][][];
        for (int b = 0; b < blocks; b++)
        {
            _dense[b] = new int[3][];
            for (int d = 0; d < 3; d++)
            {
                _dense[b][d] = new int[5];
                for (int c = 0; c < 5; c++)
                {
                    int fanIn = (features + c * growth) * 9;
                    _dense[b][d][c] = Add(BasicVsrBlocks.Conv(c < 4 ? growth : features, 3, c < 4 ? Lrelu() : null,
                        BasicVsrBlocks.ScaledKaiming<T>(fanIn, 0.1)));
                }
            }
        }
        _convBody = Add(BasicVsrBlocks.Conv<T>(features, 3, null));
        _convUp1 = Add(BasicVsrBlocks.Conv(features, 3, Lrelu()));
        _convUp2 = Add(BasicVsrBlocks.Conv(features, 3, Lrelu()));
        _convHr = Add(BasicVsrBlocks.Conv(features, 3, Lrelu()));
        _convLast = Add(BasicVsrBlocks.Conv<T>(3, 3, null));
        Count = layers.Count - First;
    }

    internal int First { get; }

    internal int Count { get; }

    private Tensor<T> DenseBlock(IEngine engine, IReadOnlyList<ILayer<T>> layers, int[] convs, Tensor<T> x)
    {
        var inputs = new List<Tensor<T>> { x };
        Tensor<T> output = x;
        for (int c = 0; c < 5; c++)
        {
            output = BasicVsrBlocks.Run(layers, convs[c], inputs.Count == 1 ? x : engine.TensorConcatenate(inputs.ToArray(), 1));
            inputs.Add(output);
        }
        return engine.TensorAdd(engine.TensorMultiplyScalar(output, MathHelper.GetNumericOperations<T>().FromDouble(ResidualScale)), x);
    }

    /// <summary>Images [N, 3, H, W] in [0, 1] to [N, 3, 4H, 4W].</summary>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> images)
    {
        var scale = MathHelper.GetNumericOperations<T>().FromDouble(ResidualScale);
        var feat = BasicVsrBlocks.Run(layers, _convFirst, images);
        var body = feat;
        foreach (var block in _dense)
        {
            var y = body;
            foreach (var dense in block) y = DenseBlock(engine, layers, dense, y);
            body = engine.TensorAdd(engine.TensorMultiplyScalar(y, scale), body);
        }
        feat = engine.TensorAdd(feat, BasicVsrBlocks.Run(layers, _convBody, body));
        Tensor<T> Nearest2(Tensor<T> x) => engine.Interpolate(x, new[] { x.Shape[2] * 2, x.Shape[3] * 2 }, InterpolateMode.Nearest, false);
        feat = BasicVsrBlocks.Run(layers, _convUp1, Nearest2(feat));
        feat = BasicVsrBlocks.Run(layers, _convUp2, Nearest2(feat));
        return BasicVsrBlocks.Run(layers, _convLast, BasicVsrBlocks.Run(layers, _convHr, feat));
    }

    /// <summary>The generator's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> Parameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(First).Take(Count));
}
