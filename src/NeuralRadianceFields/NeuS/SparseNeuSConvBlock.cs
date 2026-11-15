namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>The activation after a SparseNeuS convolution block.</summary>
internal enum SparseNeuSActivation
{
    /// <summary>No activation.</summary>
    None,

    /// <summary>ReLU (torchsparse's <c>spnn.ReLU</c>).</summary>
    ReLU,

    /// <summary>Leaky ReLU with slope 0.01 (<c>InPlaceABN</c>'s default activation).</summary>
    LeakyReLU,
}

/// <summary>
/// A 2-D or 3-D convolution, optionally transposed, optionally followed by batch normalization and an activation: the
/// <c>ConvBnReLU</c> (with <c>InPlaceABN</c>), <c>BasicSparseConvolutionBlock</c> and
/// <c>BasicSparseDeconvolutionBlock</c> of SparseNeuS, and its bias-carrying 1×1 and 3×3 FPN convolutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kernels use PyTorch's layouts (<c>[out, in, k…]</c>, transposed <c>[in, out, k…]</c>) and default initialization
/// (Kaiming-uniform, <c>a = √5</c>); batch normalization starts at γ = 1, β = 0 with running mean 0 and variance 1,
/// momentum 0.1 and ε = 10⁻⁵. In training mode the batch statistics are taken over the active voxels only when a mask
/// is given, which is how torchsparse's <c>spnn.BatchNorm</c> normalizes a sparse tensor's features.
/// </para>
/// </remarks>
internal sealed class SparseNeuSConvBlock<T>
{
    private const double Momentum = 0.1;
    private const double BatchNormEpsilon = 1e-5;

    private readonly int _spatialRank;
    private readonly string _convPath;
    private readonly string _normPath;
    private double[] _runningMean;
    private double[] _runningVariance;

    /// <summary>Creates a block.</summary>
    /// <param name="spatialRank">2 or 3.</param>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernel">Kernel size.</param>
    /// <param name="stride">Stride.</param>
    /// <param name="padding">Padding.</param>
    /// <param name="transposed">Whether the convolution is transposed (output padding <c>stride − 1</c>).</param>
    /// <param name="batchNorm">Whether batch normalization follows (then the convolution has no bias).</param>
    /// <param name="activation">The activation.</param>
    /// <param name="random">The initialization generator.</param>
    /// <param name="convPath">The convolution's state-dict path inside the block ("" for a bare convolution).</param>
    /// <param name="normPath">The batch normalization's state-dict path inside the block.</param>
    public SparseNeuSConvBlock(int spatialRank, int inChannels, int outChannels, int kernel, int stride, int padding,
        bool transposed, bool batchNorm, SparseNeuSActivation activation, Random random, string convPath = "conv",
        string normPath = "bn")
    {
        _convPath = convPath.Length == 0 ? string.Empty : "." + convPath;
        _normPath = "." + normPath;
        if (spatialRank != 2 && spatialRank != 3) throw new ArgumentOutOfRangeException(nameof(spatialRank));
        if (inChannels <= 0 || outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        _spatialRank = spatialRank;
        InChannels = inChannels;
        OutChannels = outChannels;
        KernelSize = kernel;
        Stride = stride;
        Padding = padding;
        Transposed = transposed;
        Activation = activation;
        int taps = spatialRank == 2 ? kernel * kernel : kernel * kernel * kernel;
        // PyTorch's fan-in for a transposed convolution's [in, out, k...] weight is out·k^d.
        int fanIn = (transposed ? outChannels : inChannels) * taps;
        double bound = 1.0 / Math.Sqrt(fanIn);
        var weight = new double[inChannels * outChannels * taps];
        for (int i = 0; i < weight.Length; i++) weight[i] = (random.NextDouble() * 2.0 - 1.0) * bound;
        var kernelShape = new int[spatialRank + 2];
        kernelShape[0] = transposed ? inChannels : outChannels;
        kernelShape[1] = transposed ? outChannels : inChannels;
        for (int d = 0; d < spatialRank; d++) kernelShape[2 + d] = kernel;
        Kernel = NeuSHost<T>.FromDoubles(weight, kernelShape);
        if (batchNorm)
        {
            Gamma = NeuSHost<T>.Filled(1.0, outChannels);
            Beta = NeuSHost<T>.Filled(0.0, outChannels);
        }
        else
        {
            var bias = new double[outChannels];
            for (int i = 0; i < bias.Length; i++) bias[i] = (random.NextDouble() * 2.0 - 1.0) * bound;
            Bias = NeuSHost<T>.FromDoubles(bias, outChannels);
        }

        _runningMean = new double[outChannels];
        _runningVariance = new double[outChannels];
        for (int c = 0; c < outChannels; c++) _runningVariance[c] = 1.0;
    }

    /// <summary>Input channels.</summary>
    public int InChannels { get; }

    /// <summary>Output channels.</summary>
    public int OutChannels { get; }

    /// <summary>Kernel size.</summary>
    public int KernelSize { get; }

    /// <summary>Stride.</summary>
    public int Stride { get; }

    /// <summary>Padding.</summary>
    public int Padding { get; }

    /// <summary>Whether the convolution is transposed.</summary>
    public bool Transposed { get; }

    /// <summary>The activation.</summary>
    public SparseNeuSActivation Activation { get; }

    /// <summary>The kernel.</summary>
    public Tensor<T> Kernel { get; private set; }

    /// <summary>The convolution bias (blocks without batch normalization), or null.</summary>
    public Tensor<T>? Bias { get; private set; }

    /// <summary>The batch-normalization scale γ, or null.</summary>
    public Tensor<T>? Gamma { get; private set; }

    /// <summary>The batch-normalization shift β, or null.</summary>
    public Tensor<T>? Beta { get; private set; }

    /// <summary>The trainable tensors.</summary>
    public IEnumerable<Tensor<T>> Parameters
    {
        get
        {
            yield return Kernel;
            if (Bias is not null) yield return Bias;
            if (Gamma is not null) yield return Gamma;
            if (Beta is not null) yield return Beta;
        }
    }

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => Parameters.Sum(p => p.Length);

    /// <summary>The parameters (and running statistics) under <paramref name="prefix"/>.</summary>
    /// <param name="prefix">The module path.</param>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters(string prefix)
    {
        yield return new KeyValuePair<string, Tensor<T>>(prefix + _convPath + ".weight", Kernel);
        if (Bias is not null) yield return new KeyValuePair<string, Tensor<T>>(prefix + _convPath + ".bias", Bias);
        if (Gamma is not null && Beta is not null)
        {
            yield return new KeyValuePair<string, Tensor<T>>(prefix + _normPath + ".weight", Gamma);
            yield return new KeyValuePair<string, Tensor<T>>(prefix + _normPath + ".bias", Beta);
            yield return new KeyValuePair<string, Tensor<T>>(prefix + _normPath + ".running_mean", NeuSHost<T>.FromDoubles(_runningMean, OutChannels));
            yield return new KeyValuePair<string, Tensor<T>>(prefix + _normPath + ".running_var", NeuSHost<T>.FromDoubles(_runningVariance, OutChannels));
        }
    }

    /// <summary>Loads one named parameter; returns whether the name belongs to this block.</summary>
    public bool TryLoad(string prefix, string name, Tensor<T> value)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (name == prefix + _convPath + ".weight") { Kernel = Checked(value, Kernel); return true; }
        if (Bias is not null && name == prefix + _convPath + ".bias") { Bias = Checked(value, Bias); return true; }
        if (Gamma is not null && name == prefix + _normPath + ".weight") { Gamma = Checked(value, Gamma); return true; }
        if (Beta is not null && name == prefix + _normPath + ".bias") { Beta = Checked(value, Beta); return true; }
        if (Gamma is not null && name == prefix + _normPath + ".running_mean") { _runningMean = CheckedHost(value); return true; }
        if (Gamma is not null && name == prefix + _normPath + ".running_var") { _runningVariance = CheckedHost(value); return true; }
        return false;
    }

    /// <summary>Applies the block to <c>[N, C, …]</c> input.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="input">The input.</param>
    /// <param name="training">Whether batch normalization uses (and updates) batch statistics.</param>
    /// <param name="activeMask">For 3-D blocks, the 0/1 active-voxel mask at the output resolution, <c>[1, 1, X, Y, Z]</c>
    /// on the host (null when every voxel is active): the output is zeroed off it and the statistics exclude it.</param>
    public Tensor<T> Forward(IEngine engine, Tensor<T> input, bool training, double[]? activeMask = null)
    {
        Tensor<T> x;
        if (_spatialRank == 2)
        {
            x = engine.Conv2D(input, Kernel, Stride, Padding, 1);
        }
        else if (!Transposed)
        {
            x = engine.Conv3D(input, Kernel, Stride, Padding, 1);
        }
        else
        {
            int outputPadding = Stride - 1;
            x = engine.ConvTranspose3D(input, Kernel, new[] { Stride, Stride, Stride }, new[] { Padding, Padding, Padding },
                new[] { outputPadding, outputPadding, outputPadding });
        }

        var channelShape = ChannelShape(x.Shape.Length);
        if (Bias is not null) x = engine.TensorAdd(x, engine.Reshape(Bias, channelShape));
        Tensor<T>? mask = activeMask is null ? null : NeuSHost<T>.FromDoubles(activeMask, MaskShape(x));
        if (Gamma is not null && Beta is not null) x = Normalize(engine, x, training, mask, activeMask, channelShape);
        x = Activation switch
        {
            SparseNeuSActivation.ReLU => engine.TensorReLU(x),
            SparseNeuSActivation.LeakyReLU => engine.TensorLeakyReLU(x, NeuSHost<T>.Scalar(0.01)),
            _ => x,
        };
        return mask is null ? x : engine.TensorMultiply(x, mask);
    }

    private Tensor<T> Normalize(IEngine engine, Tensor<T> x, bool training, Tensor<T>? mask, double[]? activeMask, int[] channelShape)
    {
        var gamma = Gamma ?? throw new InvalidOperationException("No batch normalization.");
        var beta = Beta ?? throw new InvalidOperationException("No batch normalization.");
        var axes = new int[x.Shape.Length - 1];
        axes[0] = 0;
        for (int a = 2; a < x.Shape.Length; a++) axes[a - 1] = a;
        Tensor<T> mean, variance;
        if (training)
        {
            int perChannel = x.Length / OutChannels;
            double count = perChannel;
            if (activeMask is not null)
            {
                count = 0;
                foreach (double m in activeMask) count += m;
                count *= x.Shape[0];
            }

            count = Math.Max(count, 1.0);
            var scale = NeuSHost<T>.Scalar(1.0 / count);
            var weighted = mask is null ? x : engine.TensorMultiply(x, mask);
            mean = engine.TensorMultiplyScalar(engine.ReduceSum(weighted, axes, true), scale);
            var centered = engine.TensorSubtract(x, mean);
            var squares = engine.TensorMultiply(centered, centered);
            if (mask is not null) squares = engine.TensorMultiply(squares, mask);
            variance = engine.TensorMultiplyScalar(engine.ReduceSum(squares, axes, true), scale);
            UpdateRunningStatistics(mean, variance, count);
        }
        else
        {
            mean = engine.Reshape(NeuSHost<T>.FromDoubles(_runningMean, OutChannels), channelShape);
            variance = engine.Reshape(NeuSHost<T>.FromDoubles(_runningVariance, OutChannels), channelShape);
        }

        var inverseStd = engine.TensorDivide(NeuSHost<T>.Filled(1.0, channelShape),
            engine.TensorSqrt(engine.TensorAddScalar(variance, NeuSHost<T>.Epsilon(BatchNormEpsilon))));
        var normalized = engine.TensorMultiply(engine.TensorSubtract(x, mean), inverseStd);
        return engine.TensorAdd(engine.TensorMultiply(normalized, engine.Reshape(gamma, channelShape)), engine.Reshape(beta, channelShape));
    }

    private void UpdateRunningStatistics(Tensor<T> mean, Tensor<T> variance, double count)
    {
        double[] batchMean, batchVariance;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            batchMean = NeuSHost<T>.ToDoubles(mean);
            batchVariance = NeuSHost<T>.ToDoubles(variance);
        }

        // PyTorch tracks the unbiased variance.
        double correction = count > 1 ? count / (count - 1) : 1.0;
        for (int c = 0; c < OutChannels; c++)
        {
            _runningMean[c] = (1 - Momentum) * _runningMean[c] + Momentum * batchMean[c];
            _runningVariance[c] = (1 - Momentum) * _runningVariance[c] + Momentum * batchVariance[c] * correction;
        }
    }

    private int[] ChannelShape(int rank)
    {
        var shape = new int[rank];
        for (int i = 0; i < rank; i++) shape[i] = 1;
        shape[1] = OutChannels;
        return shape;
    }

    private static int[] MaskShape(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        shape[0] = 1;
        shape[1] = 1;
        return shape;
    }

    private static Tensor<T> Checked(Tensor<T> value, Tensor<T> current)
    {
        if (value.Length != current.Length)
            throw new ArgumentException($"Expected {current.Length} values, got {value.Length}.", nameof(value));
        return value.Reshape(current.Shape.ToArray());
    }

    private double[] CheckedHost(Tensor<T> value)
    {
        if (value.Length != OutChannels) throw new ArgumentException($"Expected {OutChannels} values, got {value.Length}.", nameof(value));
        return NeuSHost<T>.ToDoubles(value);
    }
}
