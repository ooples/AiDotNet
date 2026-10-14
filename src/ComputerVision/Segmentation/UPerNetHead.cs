using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation;

/// <summary>
/// Exact spatial resampling of NCHW maps as constant matrices on each axis, so it is differentiable through ordinary
/// matrix products: PyTorch's bilinear interpolation with <c>align_corners=False</c>, and adaptive average pooling.
/// </summary>
internal static class SpatialResampling
{
    /// <summary>The <c>[output, input]</c> weights of linear interpolation along one axis (<c>align_corners=False</c>):
    /// output o reads source (o + ½)·in/out − ½, clamped at 0.</summary>
    public static double[,] Bilinear(int input, int output)
    {
        var m = new double[output, input];
        double scale = (double)input / output;
        for (int o = 0; o < output; o++)
        {
            double source = Math.Max(0.0, (o + 0.5) * scale - 0.5);
            int low = Math.Min((int)Math.Floor(source), input - 1);
            int high = Math.Min(low + 1, input - 1);
            double lambda = source - low;
            m[o, low] += 1.0 - lambda;
            m[o, high] += lambda;
        }
        return m;
    }

    /// <summary>
    /// The <c>[output, input]</c> weights of PyTorch's antialiased bilinear resize along one axis
    /// (<c>F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)</c>): a triangle filter whose support
    /// widens with the downscaling factor, normalized per output.
    /// </summary>
    public static double[,] BilinearAntialias(int input, int output)
    {
        var m = new double[output, input];
        double scale = (double)input / output;
        double support = scale >= 1.0 ? scale : 1.0;
        double inverse = scale >= 1.0 ? 1.0 / scale : 1.0;
        for (int o = 0; o < output; o++)
        {
            double center = scale * (o + 0.5);
            int start = Math.Max((int)(center - support + 0.5), 0);
            int end = Math.Min((int)(center + support + 0.5), input);
            double total = 0;
            for (int j = start; j < end; j++)
            {
                double w = Math.Max(0.0, 1.0 - Math.Abs((j - center + 0.5) * inverse));
                m[o, j] = w;
                total += w;
            }
            if (total > 0)
                for (int j = start; j < end; j++) m[o, j] /= total;
        }
        return m;
    }

    /// <summary><c>F.interpolate(x, size, mode="bilinear", align_corners=False, antialias=True)</c>.</summary>
    public static Tensor<T> ResizeAntialias<T>(IEngine engine, Tensor<T> x, int height, int width)
        => Apply(engine, x, BilinearAntialias(x.Shape[2], height), BilinearAntialias(x.Shape[3], width));

    /// <summary>The <c>[output, input]</c> weights of adaptive average pooling along one axis: output i averages
    /// inputs ⌊i·in/out⌋ to ⌈(i+1)·in/out⌉ − 1.</summary>
    public static double[,] AdaptiveAverage(int input, int output)
    {
        var m = new double[output, input];
        for (int i = 0; i < output; i++)
        {
            int start = i * input / output;
            int end = ((i + 1) * input + output - 1) / output;
            for (int j = start; j < end; j++) m[i, j] = 1.0 / (end - start);
        }
        return m;
    }

    /// <summary>Applies <paramref name="rows"/> <c>[H′, H]</c> and <paramref name="columns"/> <c>[W′, W]</c> to an NCHW map.</summary>
    public static Tensor<T> Apply<T>(IEngine engine, Tensor<T> x, double[,] rows, double[,] columns)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        int h2 = rows.GetLength(0), w2 = columns.GetLength(0);
        Tensor<T> Transposed(double[,] m)
        {
            int outputs = m.GetLength(0), inputs = m.GetLength(1);
            var t = new Tensor<T>(new[] { inputs, outputs });
            for (int i = 0; i < inputs; i++)
                for (int o = 0; o < outputs; o++) t[i, o] = ops.FromDouble(m[o, i]);
            return t;
        }
        var y = engine.TensorMatMul(engine.Reshape(x, new[] { b * c * h, w }), Transposed(columns));            // [B·C·H, W′]
        y = engine.TensorPermute(engine.Reshape(y, new[] { b, c, h, w2 }), new[] { 0, 1, 3, 2 });
        y = engine.TensorMatMul(engine.Reshape(y, new[] { b * c * w2, h }), Transposed(rows));                  // [B·C·W′, H′]
        return engine.TensorPermute(engine.Reshape(y, new[] { b, c, w2, h2 }), new[] { 0, 1, 3, 2 });
    }

    /// <summary><c>F.interpolate(x, size, mode="bilinear", align_corners=False)</c>.</summary>
    public static Tensor<T> Resize<T>(IEngine engine, Tensor<T> x, int height, int width)
        => Apply(engine, x, Bilinear(x.Shape[2], height), Bilinear(x.Shape[3], width));

    /// <summary><c>nn.AdaptiveAvgPool2d(size)</c>.</summary>
    public static Tensor<T> AdaptiveAveragePool<T>(IEngine engine, Tensor<T> x, int size)
        => Apply(engine, x, AdaptiveAverage(x.Shape[2], size), AdaptiveAverage(x.Shape[3], size));
}

/// <summary>An mmcv <c>ConvModule</c>: a bias-free convolution, BatchNorm and ReLU.</summary>
internal sealed record ConvModule<T>(ConvolutionalLayer<T> Conv, BatchNormalizationLayer<T> Norm)
{
    public static ConvModule<T> Create(Func<LayerBase<T>, LayerBase<T>> own, int inputs, int outputs, int kernel)
        => new((ConvolutionalLayer<T>)own(ConvolutionalLayer<T>.WithInputDepth(inputs, outputs, kernel, 1, padding: kernel / 2,
                initializationStrategy: new Initialization.ZeroInitializationStrategy<T>(), biasMode: Enums.BiasMode.Never)),
            (BatchNormalizationLayer<T>)own(new BatchNormalizationLayer<T>(outputs, 1e-5, 0.9)));

    public Tensor<T> Forward(IEngine engine, Tensor<T> x) => engine.ReLU(Norm.Forward(Conv.Forward(x)));

    /// <summary>mmcv's initialization: Kaiming-normal (fan-out, ReLU) convolution, BatchNorm scale 1 and shift 0.</summary>
    public void Initialize(Random random)
    {
        int k = Conv.KernelSize, outputs = Conv.OutputDepth;
        Conv.SetParameters(TorchParameters.Gaussian<T>(random, outputs * Conv.InputDepth * k * k, Math.Sqrt(2.0 / (outputs * k * k))));
        TorchParameters.BatchNorm(Norm, Enumerable.Repeat(1.0, outputs).ToArray(), new double[outputs], new double[outputs],
            Enumerable.Repeat(1.0, outputs).ToArray());
    }

    public void Load(Func<string, int[], double[]> read, string name)
    {
        int k = Conv.KernelSize, outputs = Conv.OutputDepth;
        var w = read($"{name}.conv.weight", new[] { outputs, Conv.InputDepth, k, k });
        Conv.SetParameters(new Vector<T>(Array.ConvertAll(w, MathHelper.GetNumericOperations<T>().FromDouble)));
        var shape = new[] { outputs };
        TorchParameters.BatchNorm(Norm, read($"{name}.bn.weight", shape), read($"{name}.bn.bias", shape),
            read($"{name}.bn.running_mean", shape), read($"{name}.bn.running_var", shape));
    }
}

/// <summary>A 1×1 classifier convolution with a bias (mmseg's <c>conv_seg</c>), after dropout.</summary>
internal sealed class SegmentationClassifier<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public SegmentationClassifier(IEngine engine, Func<LayerBase<T>, LayerBase<T>> own, int inputs, int classes, double dropout)
    {
        _engine = engine;
        Dropout = dropout;
        Conv = (ConvolutionalLayer<T>)own(ConvolutionalLayer<T>.WithInputDepth(inputs, classes, 1, 1, padding: 0,
            initializationStrategy: new Initialization.ZeroInitializationStrategy<T>()));
    }

    public ConvolutionalLayer<T> Conv { get; }
    public double Dropout { get; }

    public Tensor<T> Forward(Tensor<T> x, bool training, Random random)
    {
        if (training && Dropout > 0)
        {
            // nn.Dropout2d: whole channels.
            var mask = new Tensor<T>(new[] { x.Shape[0], x.Shape[1], 1, 1 });
            var keep = NumOps.FromDouble(1.0 / (1.0 - Dropout));
            for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < Dropout ? NumOps.Zero : keep;
            x = _engine.TensorMultiply(x, _engine.TensorBroadcastTo(mask, x.Shape.ToArray()));
        }
        return Conv.Forward(x);
    }

    /// <summary>mmseg's initialization: N(0, 0.01²) weight, zero bias.</summary>
    public void Initialize(Random random)
    {
        int classes = Conv.OutputDepth, inputs = Conv.InputDepth;
        var v = TorchParameters.Gaussian<T>(random, classes * inputs, 0.01);
        var full = new Vector<T>(classes * inputs + classes);
        for (int i = 0; i < v.Length; i++) full[i] = v[i];
        Conv.SetParameters(full);
    }

    public void Load(Func<string, int[], double[]> read, string name)
    {
        int classes = Conv.OutputDepth, inputs = Conv.InputDepth;
        var w = read($"{name}.weight", new[] { classes, inputs, 1, 1 });
        var b = read($"{name}.bias", new[] { classes });
        Conv.SetParameters(new Vector<T>(w.Concat(b).Select(NumOps.FromDouble).ToArray()));
    }
}

/// <summary>
/// UPerNet's decode head (Xiao et al. 2018) as mmsegmentation's <c>UPerHead</c> builds it: a pyramid pooling module
/// on the coarsest input, lateral 1×1 convolutions on the others, a top-down path adding each coarser map, upsampled,
/// to the next finer one, 3×3 convolutions on the finer three, every level upsampled to the finest and concatenated,
/// a 3×3 fusion convolution, dropout and a 1×1 classifier.
/// </summary>
internal sealed class UPerNetHead<T>
{
    private readonly IEngine _engine;

    public UPerNetHead(IEngine engine, List<LayerBase<T>> layers, int[] inputChannels, int channels, int classes,
        int[] poolScales, double dropout = 0.1)
    {
        _engine = engine;
        PoolScales = poolScales;
        LayerBase<T> Own(LayerBase<T> layer)
        {
            layers.Add(layer);
            return layer;
        }
        int last = inputChannels[^1];
        foreach (var _ in poolScales) PoolConvs.Add(ConvModule<T>.Create(Own, last, channels, 1));
        Bottleneck = ConvModule<T>.Create(Own, last + poolScales.Length * channels, channels, 3);
        for (int i = 0; i < inputChannels.Length - 1; i++) Laterals.Add(ConvModule<T>.Create(Own, inputChannels[i], channels, 1));
        for (int i = 0; i < inputChannels.Length - 1; i++) FpnConvs.Add(ConvModule<T>.Create(Own, channels, channels, 3));
        FpnBottleneck = ConvModule<T>.Create(Own, inputChannels.Length * channels, channels, 3);
        Classifier = new SegmentationClassifier<T>(engine, Own, channels, classes, dropout);
    }

    public int[] PoolScales { get; }
    public List<ConvModule<T>> PoolConvs { get; } = new();
    public ConvModule<T> Bottleneck { get; }
    public List<ConvModule<T>> Laterals { get; } = new();
    public List<ConvModule<T>> FpnConvs { get; } = new();
    public ConvModule<T> FpnBottleneck { get; }
    public SegmentationClassifier<T> Classifier { get; }

    /// <summary>The class logits at the finest input's resolution.</summary>
    public Tensor<T> Forward(IReadOnlyList<Tensor<T>> inputs, bool training, Random random)
    {
        var laterals = new List<Tensor<T>>();
        for (int i = 0; i < Laterals.Count; i++) laterals.Add(Laterals[i].Forward(_engine, inputs[i]));
        laterals.Add(PyramidPooling(inputs[^1]));
        for (int i = laterals.Count - 1; i > 0; i--)
            laterals[i - 1] = _engine.TensorAdd(laterals[i - 1],
                SpatialResampling.Resize(_engine, laterals[i], laterals[i - 1].Shape[2], laterals[i - 1].Shape[3]));
        var outputs = new List<Tensor<T>>();
        for (int i = 0; i < FpnConvs.Count; i++) outputs.Add(FpnConvs[i].Forward(_engine, laterals[i]));
        outputs.Add(laterals[^1]);
        int h = outputs[0].Shape[2], w = outputs[0].Shape[3];
        for (int i = 1; i < outputs.Count; i++) outputs[i] = SpatialResampling.Resize(_engine, outputs[i], h, w);
        var fused = FpnBottleneck.Forward(_engine, _engine.TensorConcatenate(outputs.ToArray(), 1));
        return Classifier.Forward(fused, training, random);
    }

    private Tensor<T> PyramidPooling(Tensor<T> x)
    {
        int h = x.Shape[2], w = x.Shape[3];
        var parts = new List<Tensor<T>> { x };
        for (int i = 0; i < PoolScales.Length; i++)
            parts.Add(SpatialResampling.Resize(_engine,
                PoolConvs[i].Forward(_engine, SpatialResampling.AdaptiveAveragePool(_engine, x, PoolScales[i])), h, w));
        return Bottleneck.Forward(_engine, _engine.TensorConcatenate(parts.ToArray(), 1));
    }

    public void Initialize(Random random)
    {
        foreach (var m in PoolConvs.Append(Bottleneck).Concat(Laterals).Concat(FpnConvs).Append(FpnBottleneck)) m.Initialize(random);
        Classifier.Initialize(random);
    }

    /// <summary>Loads an mmseg <c>UPerHead</c> state dict under <paramref name="prefix"/>.</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        for (int i = 0; i < PoolConvs.Count; i++) PoolConvs[i].Load(read, $"{prefix}.psp_modules.{i}.1");
        Bottleneck.Load(read, $"{prefix}.bottleneck");
        for (int i = 0; i < Laterals.Count; i++) Laterals[i].Load(read, $"{prefix}.lateral_convs.{i}");
        for (int i = 0; i < FpnConvs.Count; i++) FpnConvs[i].Load(read, $"{prefix}.fpn_convs.{i}");
        FpnBottleneck.Load(read, $"{prefix}.fpn_bottleneck");
        Classifier.Load(read, $"{prefix}.conv_seg");
    }
}

/// <summary>mmsegmentation's <c>FCNHead</c> with one 3×3 ConvModule and no input concatenation: the auxiliary head.</summary>
internal sealed class FcnHead<T>
{
    private readonly IEngine _engine;

    public FcnHead(IEngine engine, List<LayerBase<T>> layers, int inputChannels, int channels, int classes, double dropout = 0.1)
    {
        _engine = engine;
        LayerBase<T> Own(LayerBase<T> layer)
        {
            layers.Add(layer);
            return layer;
        }
        Conv = ConvModule<T>.Create(Own, inputChannels, channels, 3);
        Classifier = new SegmentationClassifier<T>(engine, Own, channels, classes, dropout);
    }

    public ConvModule<T> Conv { get; }
    public SegmentationClassifier<T> Classifier { get; }

    public Tensor<T> Forward(Tensor<T> x, bool training, Random random) => Classifier.Forward(Conv.Forward(_engine, x), training, random);

    public void Initialize(Random random)
    {
        Conv.Initialize(random);
        Classifier.Initialize(random);
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        Conv.Load(read, $"{prefix}.convs.0");
        Classifier.Load(read, $"{prefix}.conv_seg");
    }
}
