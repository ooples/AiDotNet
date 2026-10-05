using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>How a <see cref="NormedConv1DLayer{T}"/> reparameterizes its kernel.</summary>
public enum ConvolutionNormalization
{
    /// <summary>Weight normalization (Salimans and Kingma 2016): W = g · V / ‖V‖ per output channel.</summary>
    Weight = 0,

    /// <summary>Spectral normalization (Miyato et al. 2018): W = V / σ(V), σ estimated by one power iteration per
    /// training step.</summary>
    Spectral = 1,

    /// <summary>No reparameterization: an ordinary convolution (W = V).</summary>
    None = 2,
}

/// <summary>
/// A 1-D convolution (or transposed convolution) with weight or spectral normalization, stride, dilation and groups —
/// PyTorch's <c>weight_norm(Conv1d(...))</c> / <c>spectral_norm(Conv1d(...))</c> as GAN vocoders use them.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The direction V is initialized as PyTorch's default for the layer (U(−1/√fan_in, 1/√fan_in), with fan_in the input
/// channels per group times the kernel; for a transposed convolution the output channels per group times the kernel),
/// as is the bias. With weight normalization the per-output-channel norm g starts at ‖V‖ so W starts equal to V; with
/// spectral normalization the left singular-vector estimate u (a persistent buffer, as in PyTorch) is refined by one power
/// iteration on every training forward and W = V / (uᵀ V v). Input and output are <c>[batch, channels, time]</c>.
/// </para>
/// <para><b>For Beginners:</b> An ordinary convolution whose weights are kept at a controlled scale, which makes GAN
/// training much more stable.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4, 8",
    TestConstructorArgs = "4, 4, 3, 1, 1, 1, 2, false, AiDotNet.NeuralNetworks.Layers.ConvolutionNormalization.Weight")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class NormedConv1DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _in;
    private readonly int _out;
    private readonly int _kernel;
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _dilation;
    private readonly int _groups;
    private readonly bool _transposed;
    private readonly ConvolutionNormalization _normalization;
    private readonly bool _useBias;
    private readonly int _outputPadding;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _direction;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _gain;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;
    [Buffer]
    private Tensor<T> _u;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernelSize">Kernel length.</param>
    /// <param name="stride">Stride (the upsampling factor of a transposed convolution).</param>
    /// <param name="dilation">Dilation (1 for a transposed convolution).</param>
    /// <param name="groups">Channel groups.</param>
    /// <param name="padding">Zero padding (removed from each end of a transposed convolution's output).</param>
    /// <param name="transposed">A transposed convolution.</param>
    /// <param name="normalization">Weight, spectral or no normalization.</param>
    /// <param name="useBias">Whether the layer has a bias (true).</param>
    /// <param name="outputPadding">Extra samples at the end of a transposed convolution's output (PyTorch
    /// <c>output_padding</c>, below the stride; 0).</param>
    public NormedConv1DLayer([LayerState] int inChannels, [LayerState] int outChannels, [LayerState] int kernelSize,
        [LayerState] int stride, [LayerState] int dilation, [LayerState] int groups, [LayerState] int padding,
        [LayerState] bool transposed, [LayerState] ConvolutionNormalization normalization, [LayerState] bool useBias = true,
        [LayerState] int outputPadding = 0)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        _useBias = useBias;
        if (outputPadding < 0 || (outputPadding > 0 && (!transposed || outputPadding >= stride)))
            throw new ArgumentOutOfRangeException(nameof(outputPadding), "Output padding applies to a transposed convolution and must be below its stride.");
        _outputPadding = outputPadding;
        if (groups <= 0 || inChannels % groups != 0 || outChannels % groups != 0)
            throw new ArgumentException($"Groups ({groups}) must divide the input ({inChannels}) and output ({outChannels}) channels.", nameof(groups));
        if (kernelSize <= 0 || stride <= 0 || dilation <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (transposed && (dilation != 1 || groups != 1))
            throw new ArgumentException("A transposed convolution here has dilation 1 and one group.", nameof(transposed));
        _in = inChannels;
        _out = outChannels;
        _kernel = kernelSize;
        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _groups = groups;
        _transposed = transposed;
        _normalization = normalization;

        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        int fanIn = (transposed ? outChannels : inChannels / groups) * kernelSize;
        double bound = 1.0 / Math.Sqrt(fanIn);
        _direction = transposed
            ? new Tensor<T>(new[] { inChannels, outChannels, 1, kernelSize })
            : new Tensor<T>(new[] { outChannels, inChannels / groups, 1, kernelSize });
        for (int i = 0; i < _direction.Length; i++) _direction[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        _bias = new Tensor<T>(new[] { outChannels });
        if (useBias)
            for (int i = 0; i < outChannels; i++) _bias[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);

        // weight_norm(dim=0) normalizes over everything but dim 0: output channels for Conv1d, input channels for
        // ConvTranspose1d (whose weight is [in, out, k]).
        int rows = _direction.Shape[0], per = _direction.Length / rows;
        _gain = new Tensor<T>(new[] { rows });
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int i = 0; i < per; i++)
            {
                double v = NumOps.ToDouble(_direction[r * per + i]);
                sum += v * v;
            }
            _gain[r] = NumOps.FromDouble(Math.Sqrt(sum));
        }
        // Spectral norm's u ~ N(0, 1), normalized (torch.nn.utils.spectral_norm).
        _u = new Tensor<T>(new[] { rows });
        double norm = 0;
        for (int r = 0; r < rows; r++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            double g = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            _u[r] = NumOps.FromDouble(g);
            norm += g * g;
        }
        for (int r = 0; r < rows; r++) _u[r] = NumOps.FromDouble(NumOps.ToDouble(_u[r]) / Math.Max(Math.Sqrt(norm), 1e-12));

        RegisterTrainableParameter(_direction, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gain, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>
    /// Re-initializes the direction V from <paramref name="sample"/> (one call per element) and the bias to
    /// <paramref name="bias"/>, then resets the weight-norm gain to ‖V‖ so W = V — for models whose reference initializes
    /// a weight-normalized convolution differently from PyTorch's default (Parallel WaveGAN's Kaiming-normal kernels and
    /// constant upsampling kernels).
    /// </summary>
    internal void Reinitialize(Func<double> sample, double bias = 0.0, bool keepBias = false)
    {
        for (int i = 0; i < _direction.Length; i++) _direction[i] = NumOps.FromDouble(sample());
        if (!keepBias)
            for (int i = 0; i < _bias.Length; i++) _bias[i] = NumOps.FromDouble(_useBias ? bias : 0.0);
        int rows = _direction.Shape[0], per = _direction.Length / rows;
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int i = 0; i < per; i++)
            {
                double v = NumOps.ToDouble(_direction[r * per + i]);
                sum += v * v;
            }
            _gain[r] = NumOps.FromDouble(Math.Sqrt(sum));
        }
        Engine.InvalidatePersistentTensor(_direction);
        Engine.InvalidatePersistentTensor(_gain);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <summary>The fan-in of a direction element: input channels per group times the kernel (output channels per
    /// group for a transposed convolution).</summary>
    internal int FanIn => (_transposed ? _out : _in / _groups) * _kernel;

    /// <summary>The effective kernel W.</summary>
    internal Tensor<T> Kernel()
    {
        int rows = _direction.Shape[0], per = _direction.Length / rows;
        if (_normalization == ConvolutionNormalization.None) return _direction;
        var v = Engine.Reshape(_direction, new[] { rows, per });
        if (_normalization == ConvolutionNormalization.Weight)
        {
            var norms = Engine.TensorPow(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(v, v), new[] { 1 }, keepDims: true),
                NumOps.FromDouble(1e-24)), NumOps.FromDouble(-0.5));                                          // [rows, 1]
            var scale = Engine.TensorTile(Engine.TensorMultiply(Engine.Reshape(_gain, new[] { rows, 1 }), norms), new[] { 1, per });
            return Engine.Reshape(Engine.TensorMultiply(v, scale), _direction._shape);
        }

        // One power iteration (training only, as torch does), then σ = uᵀ V v with u and v held constant.
        var vHost = v.ToVector();
        double[] uVec = new double[rows], vVec = new double[per];
        for (int r = 0; r < rows; r++) uVec[r] = NumOps.ToDouble(_u[r]);
        void Normalize(double[] a)
        {
            double n = Math.Sqrt(a.Sum(x => x * x));
            for (int i = 0; i < a.Length; i++) a[i] /= Math.Max(n, 1e-12);
        }
        for (int c = 0; c < per; c++)
        {
            double s = 0;
            for (int r = 0; r < rows; r++) s += NumOps.ToDouble(vHost[r * per + c]) * uVec[r];
            vVec[c] = s;
        }
        Normalize(vVec);
        if (IsTrainingMode)
        {
            for (int r = 0; r < rows; r++)
            {
                double s = 0;
                for (int c = 0; c < per; c++) s += NumOps.ToDouble(vHost[r * per + c]) * vVec[c];
                uVec[r] = s;
            }
            Normalize(uVec);
            for (int r = 0; r < rows; r++) _u[r] = NumOps.FromDouble(uVec[r]);
            Engine.InvalidatePersistentTensor(_u);
            for (int c = 0; c < per; c++)
            {
                double s = 0;
                for (int r = 0; r < rows; r++) s += NumOps.ToDouble(vHost[r * per + c]) * uVec[r];
                vVec[c] = s;
            }
            Normalize(vVec);
        }
        var uT = new Tensor<T>(new[] { 1, rows });
        var vT = new Tensor<T>(new[] { per, 1 });
        for (int r = 0; r < rows; r++) uT[0, r] = NumOps.FromDouble(uVec[r]);
        for (int c = 0; c < per; c++) vT[c, 0] = NumOps.FromDouble(vVec[c]);
        var sigma = Engine.TensorMatMul(Engine.TensorMatMul(uT, v), vT);                                    // [1, 1]
        var inverse = Engine.TensorTile(Engine.TensorPow(sigma, NumOps.FromDouble(-1)), new[] { rows, per });
        return Engine.Reshape(Engine.TensorMultiply(v, inverse), _direction._shape);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != _in)
            throw new ArgumentException($"Expected [batch, {_in}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2];
        var kernel = Kernel();
        var x4 = Engine.Reshape(input, new[] { batch, _in, 1, time });
        Tensor<T> y;
        if (_transposed)
        {
            y = Engine.ConvTranspose2D(x4, kernel, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 0, _outputPadding });
        }
        else if (_groups == 1)
        {
            y = Engine.Conv2D(x4, kernel, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 1, _dilation });
        }
        else
        {
            int inPer = _in / _groups, outPer = _out / _groups;
            var parts = new Tensor<T>[_groups];
            for (int g = 0; g < _groups; g++)
                parts[g] = Engine.Conv2D(Engine.TensorSlice(x4, new[] { 0, g * inPer, 0, 0 }, new[] { batch, inPer, 1, time }),
                    Engine.TensorSlice(kernel, new[] { g * outPer, 0, 0, 0 }, new[] { outPer, inPer, 1, _kernel }),
                    new[] { 1, _stride }, new[] { 0, _padding }, new[] { 1, _dilation });
            y = Engine.TensorConcatenate(parts, 1);
        }
        int outTime = y.Shape[3];
        if (!_useBias) return Engine.Reshape(y, new[] { batch, _out, outTime });
        var bias = Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _out, 1, 1 }), new[] { batch, 1, 1, outTime });
        return Engine.Reshape(Engine.TensorAdd(y, bias), new[] { batch, _out, outTime });
    }

    /// <summary>One output step <c>[1, out, 1]</c> of the convolution from the inputs its kernel positions see:
    /// <paramref name="taps"/>[j] <c>[1, in, 1]</c> is the input under kernel position j (for a causal dilated
    /// convolution at time t, <c>x[t − (k − 1 − j) · dilation]</c>). Autoregressive models generate one sample at a time
    /// with it instead of re-running the whole receptive field.</summary>
    internal Tensor<T> ForwardTaps(IReadOnlyList<Tensor<T>> taps)
    {
        if (_transposed || _groups != 1 || _stride != 1)
            throw new InvalidOperationException("Single-step evaluation needs an ungrouped, unstrided, non-transposed convolution.");
        if (taps is null || taps.Count != _kernel)
            throw new ArgumentException($"Expected {_kernel} taps.", nameof(taps));
        var x = Engine.Reshape(Engine.TensorConcatenate(taps.ToArray(), 2), new[] { 1, _in, 1, _kernel });
        var y = Engine.Conv2D(x, Kernel(), new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var output = Engine.Reshape(y, new[] { 1, _out, 1 });
        return _useBias ? Engine.TensorAdd(output, Engine.Reshape(_bias, new[] { 1, _out, 1 })) : output;
    }

    /// <summary>Multiplies the kernel's direction elementwise by <paramref name="mask"/> (same shape; 0 removes a weight),
    /// as magnitude pruning does after each update. Only for unnormalized kernels, whose direction is the kernel.</summary>
    internal void MaskKernel(Tensor<T> mask)
    {
        if (_normalization != ConvolutionNormalization.None)
            throw new InvalidOperationException("Only an unnormalized kernel can be masked directly.");
        if (mask is null || mask.Length != _direction.Length)
            throw new ArgumentException("The mask must match the kernel.", nameof(mask));
        for (int i = 0; i < _direction.Length; i++)
            if (NumOps.ToDouble(mask[i]) == 0) _direction[i] = NumOps.Zero;
        Engine.InvalidatePersistentTensor(_direction);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _in.ToString(inv);
        metadata["OutChannels"] = _out.ToString(inv);
        metadata["KernelSize"] = _kernel.ToString(inv);
        metadata["Stride"] = _stride.ToString(inv);
        metadata["Dilation"] = _dilation.ToString(inv);
        metadata["Groups"] = _groups.ToString(inv);
        metadata["Padding"] = _padding.ToString(inv);
        metadata["Transposed"] = _transposed.ToString(inv);
        metadata["Normalization"] = _normalization.ToString();
        metadata["UseBias"] = _useBias.ToString(inv);
        metadata["OutputPadding"] = _outputPadding.ToString(inv);
        return metadata;
    }
}
