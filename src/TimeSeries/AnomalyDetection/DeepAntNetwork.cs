using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TimeSeries.AnomalyDetection;

/// <summary>The sizes of DeepAnT's time-series predictor.</summary>
/// <param name="Window">The history window w: the values the predictor reads.</param>
/// <param name="Horizon">The prediction window p_w: the values it predicts.</param>
/// <param name="Filters">The filters of each convolution (the paper's 32).</param>
/// <param name="KernelSize">The width of each convolution ('valid', stride 1).</param>
/// <param name="PoolSize">The width and stride of each max-pooling.</param>
/// <param name="HiddenUnits">The width of the fully connected layer before the output.</param>
/// <param name="Dropout">The dropout rate after the fully connected layer.</param>
public sealed record DeepAntConfiguration(int Window, int Horizon, int Filters, int KernelSize, int PoolSize, int HiddenUnits,
    double Dropout)
{
    /// <summary>The time steps after the first convolution and pooling.</summary>
    public int FirstLength => (Window - KernelSize + 1) / PoolSize;

    /// <summary>The time steps after the second convolution and pooling.</summary>
    public int SecondLength => (FirstLength - KernelSize + 1) / PoolSize;
}

/// <summary>
/// DeepAnT's time-series predictor (Munir et al. 2019, §IV-A and Fig. 1): two blocks of a convolution with ReLU and a
/// max-pooling, a fully connected layer with ReLU, and a linear output of p_w values. The paper fixes the structure,
/// 32 filters and ReLU; the kernel, pooling and hidden widths and the dropout are not stated there and follow the
/// PyTorch implementation at github.com/datacubeR/DeepAnt (no author code was released).
/// </summary>
/// <remarks>Values run time-major, [batch, time, channels]. Each convolution is a dense map of the KernelSize shifted
/// slices concatenated on the channel axis (feature k·C + c is channel c at offset k).</remarks>
internal sealed class DeepAntNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public DeepAntConfiguration Configuration { get; }
    public DenseLayer<T> FirstConvolution { get; }
    public DenseLayer<T> SecondConvolution { get; }
    public DenseLayer<T> Hidden { get; }
    public DenseLayer<T> Output { get; }

    public DeepAntNetwork(IEngine engine, List<LayerBase<T>> layers, DeepAntConfiguration configuration)
    {
        _engine = engine;
        var c = configuration;
        if (c.Window <= 0 || c.Horizon <= 0 || c.Filters <= 0 || c.KernelSize <= 0 || c.PoolSize <= 0 || c.HiddenUnits <= 0)
            throw new ArgumentException("The DeepAnT sizes must be positive.", nameof(configuration));
        if (c.Dropout < 0 || c.Dropout >= 1)
            throw new ArgumentException("The dropout rate must be in [0, 1).", nameof(configuration));
        if (c.SecondLength <= 0)
            throw new ArgumentException(
                $"A window of {c.Window} leaves no time step after two convolutions of width {c.KernelSize} and poolings of {c.PoolSize}.",
                nameof(configuration));
        Configuration = c;
        FirstConvolution = new DenseLayer<T>(c.Filters, (IActivationFunction<T>)new ReLUActivation<T>());
        SecondConvolution = new DenseLayer<T>(c.Filters, (IActivationFunction<T>)new ReLUActivation<T>());
        Hidden = new DenseLayer<T>(c.HiddenUnits, (IActivationFunction<T>)new ReLUActivation<T>());
        Output = new DenseLayer<T>(c.Horizon, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.AddRange(new LayerBase<T>[] { FirstConvolution, SecondConvolution, Hidden, Output });
    }

    /// <summary>The prediction [batch, Horizon] for windows [batch, Window, 1].</summary>
    public Tensor<T> Forward(Tensor<T> windows, bool training, Random random)
    {
        var c = Configuration;
        int batch = windows.Shape[0];
        var x = Pool(Convolve(FirstConvolution, windows));
        x = Pool(Convolve(SecondConvolution, x));
        x = _engine.Reshape(x, new[] { batch, c.SecondLength * c.Filters });
        x = Drop(Hidden.Forward(x), training, random);
        return Output.Forward(x);
    }

    // A 'valid' stride-1 convolution of [B, L, C] to [B, L − K + 1, Filters], ReLU applied by the layer.
    private Tensor<T> Convolve(DenseLayer<T> layer, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], channels = x.Shape[2], k = Configuration.KernelSize;
        int outLength = length - k + 1;
        var shifted = new Tensor<T>[k];
        for (int j = 0; j < k; j++)
            shifted[j] = _engine.TensorSlice(x, new[] { 0, j, 0 }, new[] { batch, outLength, channels });
        var unfolded = k == 1 ? shifted[0] : _engine.TensorConcatenate(shifted, 2);
        var y = layer.Forward(_engine.Reshape(unfolded, new[] { batch * outLength, k * channels }));
        return _engine.Reshape(y, new[] { batch, outLength, y.Shape[1] });
    }

    // nn.MaxPool1d(PoolSize): windows of PoolSize steps at stride PoolSize; a trailing partial window is dropped.
    private Tensor<T> Pool(Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], channels = x.Shape[2], p = Configuration.PoolSize;
        int pooled = length / p;
        if (pooled * p != length) x = _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, pooled * p, channels });
        var grouped = _engine.Reshape(x, new[] { batch, pooled, p, channels });
        return _engine.ReduceMax(grouped, new[] { 2 }, keepDims: false, out _);
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random)
    {
        double rate = Configuration.Dropout;
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    /// <summary>
    /// PyTorch's default initialization of nn.Conv1d and nn.Linear (kaiming_uniform with a = √5, which is U(±1/√fan_in)
    /// for the weight, and U(±1/√fan_in) for the bias), drawn in the order of the reference module's parameters.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        (double[] w, double[] b) Draw(int outputs, int fanIn)
        {
            double bound = 1.0 / Math.Sqrt(fanIn);
            var w = new double[outputs * fanIn];
            var b = new double[outputs];
            for (int i = 0; i < w.Length; i++) w[i] = (2 * random.NextDouble() - 1) * bound;
            for (int i = 0; i < b.Length; i++) b[i] = (2 * random.NextDouble() - 1) * bound;
            return (w, b);
        }
        var (w1, b1) = Draw(c.Filters, c.KernelSize);
        var (w2, b2) = Draw(c.Filters, c.Filters * c.KernelSize);
        var (w3, b3) = Draw(c.HiddenUnits, c.Filters * c.SecondLength);
        var (w4, b4) = Draw(c.Horizon, c.HiddenUnits);
        Load(w1, b1, w2, b2, w3, b3, w4, b4);
    }

    /// <summary>Loads the reference module's state dict (convblock1.0, convblock2.0, denseblock.0 and out).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int flat = c.Filters * c.SecondLength;
        Load(read("convblock1.0.weight", new[] { c.Filters, 1, c.KernelSize }), read("convblock1.0.bias", new[] { c.Filters }),
            read("convblock2.0.weight", new[] { c.Filters, c.Filters, c.KernelSize }), read("convblock2.0.bias", new[] { c.Filters }),
            read("denseblock.0.weight", new[] { c.HiddenUnits, flat }), read("denseblock.0.bias", new[] { c.HiddenUnits }),
            read("out.weight", new[] { c.Horizon, c.HiddenUnits }), read("out.bias", new[] { c.Horizon }));
    }

    // Torch layouts: a convolution weight [out, in, K]; the hidden weight [hidden, Filters · SecondLength] over the
    // channel-major flatten of [B, C, L] (column c·L + t).
    private void Load(double[] w1, double[] b1, double[] w2, double[] b2, double[] w3, double[] b3, double[] w4, double[] b4)
    {
        var c = Configuration;
        TorchParameters.Linear(_engine, FirstConvolution, c.KernelSize, c.Filters, Unfold(w1, c.Filters, 1, c.KernelSize), b1);
        TorchParameters.Linear(_engine, SecondConvolution, c.Filters * c.KernelSize, c.Filters,
            Unfold(w2, c.Filters, c.Filters, c.KernelSize), b2);
        int length = c.SecondLength, flat = c.Filters * length;
        var hidden = new double[c.HiddenUnits * flat];
        for (int o = 0; o < c.HiddenUnits; o++)
            for (int t = 0; t < length; t++)
                for (int ch = 0; ch < c.Filters; ch++)
                    hidden[o * flat + t * c.Filters + ch] = w3[o * flat + ch * length + t];
        TorchParameters.Linear(_engine, Hidden, flat, c.HiddenUnits, hidden, b3);
        TorchParameters.Linear(_engine, Output, c.HiddenUnits, c.Horizon, w4, b4);
    }

    // [out, in, K] to the dense weight [out, K · in] over the unfolded features (column k·in + c).
    private static double[] Unfold(double[] weight, int outputs, int inputs, int kernel)
    {
        var result = new double[weight.Length];
        for (int o = 0; o < outputs; o++)
            for (int ch = 0; ch < inputs; ch++)
                for (int k = 0; k < kernel; k++)
                    result[o * inputs * kernel + k * inputs + ch] = weight[(o * inputs + ch) * kernel + k];
        return result;
    }
}
