using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Training;

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

    // The public constructor delegates initialization; these names let generated save/clone
    // metadata read its unchanged construction signature from the canonical geometry fields.
    private int InChannels => _in;
    private int OutChannels => _out;
    private int KernelSize => _kernel;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _direction;
    [TrainableParameter(Role = PersistentTensorRole.Weights, Condition = nameof(UsesWeightNormalization))]
    private Tensor<T> _gain;
    [TrainableParameter(Role = PersistentTensorRole.Biases, Condition = nameof(HasBias))]
    private Tensor<T> _bias;
    [Buffer]
    private Tensor<T> _u;
    [Buffer]
    private Tensor<T> _v;

    private readonly DerivedTensorCache<T> _inferenceWeightKernel = new(disposeReplacedValues:
        Environment.GetEnvironmentVariable("AIDOTNET_DERIVED_WEIGHT_LIFETIMES") == "1");

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
        : this(inChannels, outChannels, kernelSize, stride, dilation, groups, padding, transposed, normalization,
            outputPadding, useBias, initialization: null)
    {
    }

    // Module constructors can supply an explicit stream before any default weight, bias or spectral-vector draw.
    // The public constructor and its serialized shape signature keep their existing behavior and compatibility.
    [AiDotNet.Attributes.ScalarKernel("Cold initialization preserves the existing sequential uniform/Gaussian random draws and scalar gain-reduction order so an explicit seed determines every parameter and buffer without changing its distribution.")]
    internal NormedConv1DLayer(int inChannels, int outChannels, int kernelSize,
        int stride, int dilation, int groups, int padding, bool transposed, ConvolutionNormalization normalization,
        int outputPadding, bool useBias, Random? initialization = null)
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
        if (transposed && (dilation != 1 || (groups != 1 && groups != outChannels)))
            throw new ArgumentException("A transposed convolution here has dilation 1 and either one group or one output channel per group (depthwise when the input has as many channels).", nameof(transposed));
        _in = inChannels;
        _out = outChannels;
        _kernel = kernelSize;
        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _groups = groups;
        _transposed = transposed;
        _normalization = normalization;

        var random = initialization ?? (RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom());
        int fanIn = (transposed ? outChannels / groups : inChannels / groups) * kernelSize;
        double bound = 1.0 / Math.Sqrt(fanIn);
        _direction = transposed
            ? new Tensor<T>(new[] { inChannels, outChannels / groups, 1, kernelSize })
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
        // Spectral norm's u ~ N(0, 1) and v ~ N(0, 1), each normalized (torch.nn.utils.spectral_norm).
        _u = NormalizedGaussian(rows, random);
        _v = NormalizedGaussian(per, random);

        RegisterTrainableParameter(_direction, PersistentTensorRole.Weights);
        if (UsesWeightNormalization) RegisterTrainableParameter(_gain, PersistentTensorRole.Weights);
        if (HasBias) RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private Tensor<T> NormalizedGaussian(int length, Random random)
    {
        var result = new Tensor<T>(new[] { length });
        var values = new double[length];
        double norm = 0;
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            norm += values[i] * values[i];
        }
        for (int i = 0; i < length; i++) result[i] = NumOps.FromDouble(values[i] / Math.Max(Math.Sqrt(norm), 1e-12));
        return result;
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The output channels.</summary>
    internal int InputChannels => _in;
    internal int OutputChannels => _out;

    /// <summary>Whether the layer has a bias.</summary>
    internal bool HasBias => _useBias;

    /// <summary>The weight reparameterization; without one, the gain tensor is unused.</summary>
    internal ConvolutionNormalization Normalization => _normalization;
    private bool UsesWeightNormalization => _normalization == ConvolutionNormalization.Weight;

    /// <summary>Sets the bias to zero (references that initialize a convolution's bias with <c>nn.init.constant_(0)</c>).</summary>
    internal void ZeroBias()
    {
        for (int i = 0; i < _bias.Length; i++) _bias[i] = NumOps.Zero;
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <summary>The kernel's shape in PyTorch's layout: <c>[out, in / groups, k]</c> for a convolution, <c>[in, out, k]</c>
    /// for a transposed one.</summary>
    internal int[] TorchWeightShape => new[] { _direction.Shape[0], _direction.Shape[1], _kernel };

    /// <summary>
    /// Loads a torch convolution straight from <paramref name="source"/>: the kernel <paramref name="weightName"/> of torch
    /// shape <paramref name="weightShape"/> (the direction <c>weight_v</c> of a weight-normalized one, with its gain
    /// <paramref name="gainName"/>, else each row's norm is its gain) and the bias <paramref name="biasName"/> (zero without
    /// one).
    /// </summary>
    internal void LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string weightName, int[] weightShape, string? gainName,
        string? biasName)
    {
        if (gainName is not null && _normalization != ConvolutionNormalization.Weight)
            throw new InvalidOperationException("A weight-norm gain loads only into a weight-normalized convolution.");
        if (_normalization == ConvolutionNormalization.Spectral)
            throw new InvalidOperationException("Loading PyTorch weights into a spectrally normalized convolution is not supported.");
        source.CopyTo(weightName, weightShape, _direction);
        int rows = _direction.Shape[0], per = _direction.Length / rows;
        if (gainName is not null)
        {
            // weight_norm(dim=0)'s gain keeps the weight's rank: [rows, 1, ...].
            source.CopyTo(gainName, new[] { weightShape[0] }.Concat(Enumerable.Repeat(1, weightShape.Length - 1)).ToArray(), _gain);
        }
        else
        {
            var direction = _direction.AsSpan();
            for (int r = 0; r < rows; r++)
            {
                double sum = 0;
                for (int i = 0; i < per; i++)
                {
                    double v = NumOps.ToDouble(direction[r * per + i]);
                    sum += v * v;
                }
                _gain[r] = NumOps.FromDouble(Math.Sqrt(sum));
            }
        }
        if (biasName is null)
        {
            _bias.AsWritableSpan().Clear();
        }
        else
        {
            source.CopyTo(biasName, new[] { _bias.Length }, _bias);
            if (!_useBias)
                foreach (var b in _bias.AsSpan())
                    if (NumOps.ToDouble(b) != 0) throw new InvalidOperationException("The checkpoint has a bias but this convolution has none.");
        }
        Engine.InvalidatePersistentTensor(_direction);
        Engine.InvalidatePersistentTensor(_gain);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <summary>
    /// Loads a PyTorch convolution's parameters: the weight in <see cref="TorchWeightShape"/> order (weight_norm's
    /// <c>weight_v</c> when <paramref name="gain"/> is given), the weight-norm gain <c>weight_g</c> (one per row), and the
    /// bias. Without a gain a weight-normalized layer takes the weight as its direction with gain ‖V‖, so W = V.
    /// </summary>
    internal void LoadTorchWeights(double[] weight, double[]? gain, double[] bias)
    {
        if (weight.Length != _direction.Length)
            throw new ArgumentException($"Expected {_direction.Length} kernel values, got {weight.Length}.", nameof(weight));
        if (bias.Length != _bias.Length)
            throw new ArgumentException($"Expected {_bias.Length} bias values, got {bias.Length}.", nameof(bias));
        if (gain is not null && _normalization != ConvolutionNormalization.Weight)
            throw new InvalidOperationException("A weight-norm gain loads only into a weight-normalized convolution.");
        if (_normalization == ConvolutionNormalization.Spectral)
            throw new InvalidOperationException("Loading PyTorch weights into a spectrally normalized convolution is not supported.");
        for (int i = 0; i < weight.Length; i++) _direction[i] = NumOps.FromDouble(weight[i]);
        int rows = _direction.Shape[0], per = _direction.Length / rows;
        if (gain is not null && gain.Length != rows)
            throw new ArgumentException($"Expected {rows} gain values, got {gain.Length}.", nameof(gain));
        for (int r = 0; r < rows; r++)
        {
            double g;
            if (gain is not null)
            {
                g = gain[r];
            }
            else
            {
                double sum = 0;
                for (int i = 0; i < per; i++) sum += weight[r * per + i] * weight[r * per + i];
                g = Math.Sqrt(sum);
            }
            _gain[r] = NumOps.FromDouble(g);
        }
        for (int i = 0; i < bias.Length; i++)
        {
            if (!_useBias && bias[i] != 0)
                throw new InvalidOperationException("The checkpoint has a bias but this convolution has none.");
            _bias[i] = NumOps.FromDouble(bias[i]);
        }
        Engine.InvalidatePersistentTensor(_direction);
        Engine.InvalidatePersistentTensor(_gain);
        Engine.InvalidatePersistentTensor(_bias);
    }

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
        if (_normalization == ConvolutionNormalization.None) return _direction;
        if (_normalization == ConvolutionNormalization.Weight)
        {
            // Eval reuses the same effective weights until their sources change. Keeping their owned storage also
            // lets convolution backends retain a packed kernel instead of packing fresh normalization scratch.
            if (!IsTrainingMode && !GraphMode.IsActive
                && (GradientTape<T>.Current is null || NoGradScope<T>.IsSuppressed))
                return _inferenceWeightKernel.GetOrBuild(Engine,
                    static (engine, sources) => WeightNormalizedKernel(engine, sources[0], sources[1]), _direction, _gain);
            return WeightNormalizedKernel(Engine, _direction, _gain);
        }
        int rows = _direction.Shape[0], per = _direction.Length / rows;
        var v = Engine.Reshape(_direction, new[] { rows, per });

        // Match torch.nn.utils.parametrizations.spectral_norm: update u from the stored v FIRST, then v from the
        // new u. The older spectral_norm hook used the opposite order; those orders disagree after a weight update.
        // Both vectors are kept; σ = uᵀ V v with u and v held constant. Evaluation uses the stored state unchanged.
        Tensor<T> uT, vT;
        // Power iteration treats u/v as state, not differentiable functions of the weights. Keep the matrix and
        // state updates on the engine: downloading V and walking its elements dominated GPU vocoder training.
        using (new NoGradScope<T>())
        {
            uT = Engine.Reshape(_u, new[] { 1, rows });
            vT = Engine.Reshape(_v, new[] { per, 1 });
            if (IsTrainingMode)
            {
                Tensor<T> Normalize(Tensor<T> x)
                {
                    var norm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(x, x), null));
                    return Engine.TensorMultiply(x, Engine.TensorReciprocal(Engine.TensorMax(norm, NumOps.FromDouble(1e-12))));
                }
                var nextU = Normalize(Engine.TensorMatMul(v, vT));
                var nextV = Normalize(Engine.TensorMatMul(Engine.Reshape(nextU, new[] { 1, rows }), v));
                if (CompiledStepRandom<T>.InNonStepForward)
                {
                    // Shape-resolution forwards must not advance the accepted state.
                    uT = Engine.Reshape(nextU, new[] { 1, rows });
                    vT = Engine.Reshape(nextV, new[] { per, 1 });
                }
                else
                {
                    // A closure probe must record the same copies and snapshots as training. Its lazy scope
                    // abandons these writes without executing them, preserving both state and node positions.

                    Engine.TensorCopyInPlace(_u, Engine.Reshape(nextU, _u._shape));
                    Engine.TensorCopyInPlace(_v, Engine.Reshape(nextV, _v._shape));
                    // Snapshot this forward's state so another forward cannot overwrite the vectors its backward uses.
                    uT = Engine.Reshape(Engine.TensorAddScalar(_u, NumOps.Zero), new[] { 1, rows });
                    vT = Engine.Reshape(Engine.TensorAddScalar(_v, NumOps.Zero), new[] { per, 1 });
                }
            }
        }
        var sigma = Engine.TensorMatMul(Engine.TensorMatMul(uT, v), vT);                                    // [1, 1]
        var inverse = Engine.TensorPow(sigma, NumOps.FromDouble(-1));
        return InvalidateComputedKernelCaches(Engine.Reshape(Engine.TensorMultiply(v, inverse), _direction._shape));
    }

    private static Tensor<T> WeightNormalizedKernel(IEngine engine, Tensor<T> direction, Tensor<T> gain)
    {
        // A single semantic operation preserves the per-first-dimension gain (also for transposed weights) without
        // retaining a full squared-weight tensor and two full broadcast-gradient products in a compiled training plan.
        var kernel = engine.TensorWeightNorm(direction, gain, epsilon: 1e-24);
        // A rebuilt kernel may reuse scratch whose array still keys an older packed/transpose cache.
        engine.InvalidatePersistentTensor(kernel);
        return kernel;
    }

    private Tensor<T> InvalidateComputedKernelCaches(Tensor<T> kernel)
    {
        // A recomputed kernel can reuse scratch storage whose array still keys an older
        // convolution transpose or packed GEMM panel. Consumers must use the current bytes.
        Engine.InvalidatePersistentTensor(kernel);
        return kernel;
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => ForwardCore(input, false, NumOps.Zero);

    /// <summary>Explicit composition used only when a private inference convolution feeds LeakyReLU immediately.</summary>
    internal Tensor<T> ForwardLeakyInference(Tensor<T> input, T alpha, bool detachedTraining = false)
    {
        var ports = InputPorts;
        if (Environment.GetEnvironmentVariable("AIDOTNET_GPU_INFERENCE_CHANNEL_BIAS_LEAKY") == "0"
            || GetType() != typeof(NormedConv1DLayer<T>) || typeof(T) != typeof(float) || IsInferringShapes
            || (IsTrainingMode && !CanFuseDetachedForward(detachedTraining))
            || Engine.Unwrap().GetType() != typeof(DirectGpuTensorEngine) || !ReferenceEquals(Engine, Engine.Unwrap())
            || GradientTape<T>.Current is not null || GraphMode.IsActive
            || AiDotNet.NeuralNetworks.GraphCaptureCompatibility.IsActive
            || AiDotNet.NeuralNetworks.Graph.LayerForwardObserver<T>.Current is not null)
            return Engine.LeakyReLU(Forward(input), alpha);
        if (ports.Count != 1 || ports[0].Source != TensorPortSource.External
            || ports[0].ValueDomain.Kind != LayerInputDomainKind.Continuous || ports[0].ShapeConstraint.IsConstrained)
            BindInputContract(input.Shape.ToArray()).Validate(input);
        return ForwardCore(input, true, alpha);
    }

    /// <summary>Inference convolution followed by bias/residual addition and an optional separately owned activated carry.</summary>
    internal Tensor<T> ForwardResidualInference(Tensor<T> input, Tensor<T> residual, T alpha,
        bool writeActivation, out Tensor<T>? activated, bool detachedTraining = false)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_GPU_INFERENCE_CHANNEL_BIAS_RESIDUAL") == "0"
            || GetType() != typeof(NormedConv1DLayer<T>) || typeof(T) != typeof(float) || IsInferringShapes
            || (IsTrainingMode && !CanFuseDetachedForward(detachedTraining))
            || Engine.Unwrap().GetType() != typeof(DirectGpuTensorEngine) || !ReferenceEquals(Engine, Engine.Unwrap())
            || GradientTape<T>.Current is not null || GraphMode.IsActive
            || AiDotNet.NeuralNetworks.GraphCaptureCompatibility.IsActive
            || AiDotNet.NeuralNetworks.Graph.LayerForwardObserver<T>.Current is not null)
        {
            var ordinary = Engine.TensorAdd(Forward(input), residual);
            activated = writeActivation ? Engine.LeakyReLU(ordinary, alpha) : null;
            return ordinary;
        }
        var ports = InputPorts;
        if (ports.Count != 1 || ports[0].Source != TensorPortSource.External
            || ports[0].ValueDomain.Kind != LayerInputDomainKind.Continuous || ports[0].ShapeConstraint.IsConstrained)
            BindInputContract(input.Shape.ToArray()).Validate(input);
        Tensor<T>? computedKernel = null;
        if (Environment.GetEnvironmentVariable("AIDOTNET_GPU_INFERENCE_CONV_BIAS_RESIDUAL") == "1"
            && !_transposed && _groups == 1 && _useBias && input.Rank == 3 && input.Shape[1] == _in
            && residual.Rank == 3 && residual.Shape[0] == input.Shape[0] && residual.Shape[1] == _out)
        {
            computedKernel = Kernel();
            var input4 = Engine.Reshape(input, new[] { input.Shape[0], _in, 1, input.Shape[2] });
            var residual4 = Engine.Reshape(residual, new[] { residual.Shape[0], _out, 1, residual.Shape[2] });
            if (((DirectGpuTensorEngine)Engine).TryConvBiasResidualInference(input4, computedKernel, _bias, residual4,
                1, _stride, 0, _padding, 1, _dilation, alpha, writeActivation, out var fusedSum, out var fusedActivated))
            {
                activated = fusedActivated is not null
                    ? Engine.Reshape(fusedActivated, new[] { input.Shape[0], _out, residual.Shape[2] }) : null;
                return Engine.Reshape(fusedSum, new[] { input.Shape[0], _out, residual.Shape[2] });
            }
        }
        var y = ForwardConvolution(input, computedKernel);
        int batch = y.Shape[0], outTime = y.Shape[3];
        if (_useBias && residual.Rank == 3 && residual.Shape[0] == batch
            && residual.Shape[1] == _out && residual.Shape[2] == outTime)
        {
            var residual4 = Engine.Reshape(residual, new[] { batch, _out, 1, outTime });
            if (((DirectGpuTensorEngine)Engine).TryChannelBiasResidualInference(y, _bias, residual4, alpha,
                writeActivation, out var sum4, out var activated4))
            {
                activated = activated4 is not null ? Engine.Reshape(activated4, new[] { batch, _out, outTime }) : null;
                return Engine.Reshape(sum4, new[] { batch, _out, outTime });
            }
        }
        var biased = _useBias ? Engine.TensorChannelBiasAdd(y, _bias) : y;
        var sum = Engine.TensorAdd(Engine.Reshape(biased, new[] { batch, _out, outTime }), residual);
        activated = writeActivation ? Engine.LeakyReLU(sum, alpha) : null;
        return sum;
    }

    // The discriminator consumes only the generated waveform. Its detached forward can combine
    // epilogues while Kernel() still computes the current training weights for every convolution.
    private bool CanFuseDetachedForward(bool detachedTraining) =>
        detachedTraining && NoGradScope<T>.IsSuppressed && _normalization != ConvolutionNormalization.Spectral;

    private Tensor<T> ForwardConvolution(Tensor<T> input, Tensor<T>? computedKernel = null)
    {
        if (input.Rank != 3 || input.Shape[1] != _in)
            throw new ArgumentException($"Expected [batch, {_in}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2];
        var kernel = computedKernel ?? Kernel();
        var x4 = Engine.Reshape(input, new[] { batch, _in, 1, time });
        Tensor<T> y;
        if (_transposed && _groups > 1)
        {
            // One output channel per group: each input channel's own transposed convolution, summed over its group.
            var depthwise = DepthwiseTransposed(input, kernel);
            int length = depthwise.Shape[2], inPer = _in / _groups;
            if (inPer > 1)
                depthwise = Engine.ReduceSum(Engine.Reshape(depthwise, new[] { batch, _out, inPer, length }), new[] { 2 }, keepDims: false);
            y = Engine.Reshape(depthwise, new[] { batch, _out, 1, length });
        }
        else if (_transposed)
        {
            y = Engine.ConvTranspose2D(x4, kernel, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 0, _outputPadding });
        }
        else if (_groups == 1)
        {
            y = Engine.Conv2D(x4, kernel, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 1, _dilation });
        }
        else
        {
            y = GroupedConvolution.Forward(Engine, x4, kernel,
                new[] { 1, _stride }, new[] { 0, _padding }, new[] { 1, _dilation }, _groups);
        }
        return y;
    }

    private Tensor<T> ForwardCore(Tensor<T> input, bool leakyAfterBias, T alpha)
    {
        var y = ForwardConvolution(input);
        int batch = y.Shape[0], outTime = y.Shape[3];
        if (leakyAfterBias && _useBias && ((DirectGpuTensorEngine)Engine).TryChannelBiasLeakyInference(y, _bias, alpha, out var activated))
            return Engine.Reshape(activated, new[] { batch, _out, outTime });
        // Keep the channel semantic explicit: no activation-sized bias tile, and backward reduces batch/time.
        var biased = _useBias ? Engine.TensorChannelBiasAdd(y, _bias) : y;
        var output = Engine.Reshape(biased, new[] { batch, _out, outTime });
        return leakyAfterBias ? Engine.LeakyReLU(output, alpha) : output;
    }

    /// <summary>
    /// Each input channel's transposed convolution with its own kernel (PyTorch's <c>ConvTranspose1d(C, C, k, stride,
    /// padding, output_padding, groups=C)</c>, <c>[batch, in, time']</c>): the input zero-stuffed by the stride, padded by <c>k − 1 − padding</c> on the left and that plus the
    /// output padding on the right, convolved per channel with the flipped kernel.
    /// </summary>
    private Tensor<T> DepthwiseTransposed(Tensor<T> input, Tensor<T> kernel)
    {
        int batch = input.Shape[0], time = input.Shape[2];
        var x = input;
        if (_stride > 1)
        {
            var columns = Engine.Reshape(input, new[] { batch, _in, time, 1 });
            var zeros = new Tensor<T>(new[] { batch, _in, time, _stride - 1 });
            x = Engine.Reshape(Engine.TensorConcatenate(new[] { columns, zeros }, 3), new[] { batch, _in, time * _stride });
            x = Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, _in, (time - 1) * _stride + 1 });
        }
        int left = _kernel - 1 - _padding, right = left + _outputPadding;
        if (left < 0) throw new InvalidOperationException("The padding exceeds the kernel.");
        var parts = new List<Tensor<T>>();
        if (left > 0) parts.Add(new Tensor<T>(new[] { batch, _in, left }));
        parts.Add(x);
        if (right > 0) parts.Add(new Tensor<T>(new[] { batch, _in, right }));
        if (parts.Count > 1) x = Engine.TensorConcatenate(parts.ToArray(), 2);
        // [C, 1, 1, k] → [C, 1, k], flipped along the kernel.
        var flat = Engine.Reshape(kernel, new[] { _in, 1, _kernel });
        var flipped = Engine.TensorConcatenate(Enumerable.Range(0, _kernel).Reverse()
            .Select(j => Engine.TensorSlice(flat, new[] { 0, 0, j }, new[] { _in, 1, 1 })).ToArray(), 2);
        return Engine.DepthwiseConv1D(x, flipped, 1, 0);
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
