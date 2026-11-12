using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements a Diagonal State Space (S4D) layer from Gu et al., 2022.
/// </summary>
/// <remarks>
/// <para>
/// S4D simplifies the original S4 model by using a diagonal state matrix A, which greatly reduces
/// computational complexity while maintaining competitive performance. The diagonal structure means
/// each state dimension evolves independently, enabling efficient parallelization.
/// </para>
/// <para>
/// The layer implements the continuous-time state space model:
/// <code>
///   h'(t) = A * h(t) + B * x(t)
///   y(t)  = Re(C * h(t)) + D * x(t)
/// </code>
/// where A is diagonal (and typically complex-valued, stored as real/imaginary pairs for generic compatibility).
/// Discretization uses the Zero-Order Hold (ZOH) method:
/// <code>
///   A_bar = exp(delta * A)
///   B_bar = (A_bar - I) * A^{-1} * B
/// </code>
/// </para>
/// <para>
/// During training, the layer supports a global convolution mode that convolves the entire sequence
/// at once using the closed-form convolution kernel. During inference, it uses an efficient recurrent
/// mode that processes one step at a time with O(1) per-step cost.
/// </para>
/// <para>
/// The A matrix is initialized using HiPPO-LegS (Legendre polynomials) from Gu et al., 2020,
/// which provides a mathematically principled initialization that captures long-range dependencies.
/// S4D-Lin uses A_n = -1/2 + ni for state dimension n, giving logarithmically-spaced frequencies.
/// </para>
/// <para><b>For Beginners:</b> Think of S4D as a set of independent oscillators, each tuned to a
/// different frequency. When you feed in a sequence, each oscillator responds to the parts of the signal
/// at its frequency, building up a rich representation of the input over time.
///
/// The key insight is that diagonal state matrices are much simpler than full matrices:
/// - Full S4: A is N x N -> complex eigenvalue decomposition needed
/// - S4D: A is diagonal -> each state dimension is just one number
///
/// This makes S4D much faster while being nearly as expressive. It's the foundation that led to
/// more advanced models like Mamba.
/// </para>
/// <para>
/// <b>Reference:</b> Gu et al., "On the Parameterization and Initialization of Diagonal State Space Models", 2022.
/// https://arxiv.org/abs/2206.11893
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving. Relations discovered by probing; roles read from the forward - this folder's
// convention is seqLen = Shape[rank-2], modelDim = Shape[rank-1], so rank 2 is [Time, Features].
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class S4DLayer<T> : LayerBase<T>, IShapeContract
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _stateDimension;
    private readonly int _innerDimension;

    // A parameter stored as real/imaginary pairs: [innerDim, stateDim, 2]
    // A = a_real + i * a_imag, initialized with S4D-Lin: A_n = -1/2 + n*i
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _aReal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _aImag;

    // B projection: [modelDim, innerDim * stateDim] (complex, stored as real/imag pairs)
    // In S4D-Lin, B is typically initialized to ones.
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _bReal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _bImag;

    // C projection: [innerDim * stateDim, modelDim] (complex, learned)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _cReal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _cImag;

    // D: [innerDim] (skip connection, real-valued)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _dParam;

    // Input projection: [modelDim, innerDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputProjectionBias;

    // Output projection: [innerDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Delta (discretization step size): [innerDim] (learned, stored as log for positivity)
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _logDelta;

    // Cached values for backward pass
    private int[]? _originalInputShape;

    // Gradients
    [Scratch]
    private Tensor<T>? _aRealGradient;
    [Scratch]
    private Tensor<T>? _aImagGradient;
    [Scratch]
    private Tensor<T>? _bRealGradient;
    [Scratch]
    private Tensor<T>? _bImagGradient;
    [Scratch]
    private Tensor<T>? _cRealGradient;
    [Scratch]
    private Tensor<T>? _cImagGradient;
    [Scratch]
    private Tensor<T>? _dParamGradient;
    [Scratch]
    private Tensor<T>? _inputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _inputProjectionBiasGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionBiasGradient;
    [Scratch]
    private Tensor<T>? _logDeltaGradient;

    /// <inheritdoc />
    /// <summary>
    /// Training uses the fused complex diagonal SSM scan's analytic BPTT node. The ZOH
    /// discretization remains expressed through differentiable engine operations.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension (d_model) of this S4D layer.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the SSM state dimension (N) controlling the number of independent oscillators.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each state dimension corresponds to a different "frequency" that
    /// the model can detect in the input sequence. More state dimensions means the model can capture
    /// a wider range of temporal patterns. Typical values are 64 (S4D default) or 16 for efficiency.
    /// </para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the inner dimension used for the SSM computation.
    /// </summary>
    public int InnerDimension => _innerDimension;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>Construction state: the 'expandFactor' the layer was built with.</summary>
    private readonly int _expandFactor;

    /// <summary>
    /// Creates a new S4D (Diagonal State Space) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of the representation at each sequence position.</para>
    /// </param>
    /// <param name="stateDimension">
    /// SSM state dimension (N). Default: 64.
    /// <para><b>For Beginners:</b> The number of independent oscillators tracking different frequencies.
    /// The original S4D paper uses N=64. This is typically larger than Mamba's N=16 because S4D relies
    /// more heavily on the state for expressivity (no input-dependent selection mechanism).</para>
    /// </param>
    /// <param name="expandFactor">
    /// Expansion factor for inner dimension. Default: 1.
    /// <para><b>For Beginners:</b> Controls the ratio of inner computation width to model dimension.
    /// S4D typically uses 1 (no expansion), unlike Mamba which uses 2.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when modelDimension or stateDimension is not positive.</exception>
    public S4DLayer(
        int sequenceLength,
        int modelDimension = 256,
        int stateDimension = 64,
        int expandFactor = 1,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        _expandFactor = expandFactor;
        _sequenceLength = sequenceLength;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (modelDimension <= 0)
        {
            throw new ArgumentException(
                $"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        }

        if (stateDimension <= 0)
        {
            throw new ArgumentException(
                $"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));
        }

        if (expandFactor <= 0)
        {
            throw new ArgumentException(
                $"Expand factor ({expandFactor}) must be positive.", nameof(expandFactor));
        }

        _modelDimension = modelDimension;
        _stateDimension = stateDimension;
        _innerDimension = modelDimension * expandFactor;

        // A stored as separate real/imaginary: each [innerDim, stateDim]
        _aReal = new Tensor<T>([_innerDimension, stateDimension]);
        _aImag = new Tensor<T>([_innerDimension, stateDimension]);

        // B: [innerDim, stateDim] real/imag
        _bReal = new Tensor<T>([_innerDimension, stateDimension]);
        _bImag = new Tensor<T>([_innerDimension, stateDimension]);

        // C: [innerDim, stateDimension] real/imag
        _cReal = new Tensor<T>([_innerDimension, stateDimension]);
        _cImag = new Tensor<T>([_innerDimension, stateDimension]);

        // D: [innerDim] skip connection
        _dParam = new Tensor<T>([_innerDimension]);

        // Input projection: [modelDim, innerDim]
        _inputProjectionWeights = new Tensor<T>([modelDimension, _innerDimension]);
        _inputProjectionBias = new Tensor<T>([_innerDimension]);

        // Output projection: [innerDim, modelDim]
        _outputProjectionWeights = new Tensor<T>([_innerDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        // Log delta (discretization step): [innerDim]
        _logDelta = new Tensor<T>([_innerDimension]);

        InitializeParameters();
    }

    /// <summary>Compute B_bar = (exp(dt*A) - I) / A * B for given complex A and B.</summary>
    private static (double r, double i) ComputeBBar(double dt, double ar, double ai, double br, double bi)
    {
        double expR = Math.Exp(dt * ar);
        double cosA = Math.Cos(dt * ai);
        double sinA = Math.Sin(dt * ai);
        double diffR = expR * cosA - 1.0;
        double diffI = expR * sinA;
        double aMagSq = ar * ar + ai * ai;
        double quotR, quotI;
        if (aMagSq < 1e-12)
        {
            quotR = dt * br;
            quotI = dt * bi;
        }
        else
        {
            // (diffR + i*diffI) / (ar + i*ai)
            quotR = (diffR * ar + diffI * ai) / aMagSq;
            quotI = (diffI * ar - diffR * ai) / aMagSq;
        }
        // * B = (quotR + i*quotI) * (br + i*bi)
        return (quotR * br - quotI * bi, quotR * bi + quotI * br);
    }

    private void InitializeParameters()
    {
        // S4D-Lin initialization: A_n = -1/2 + n*i
        // Real part: -0.5 for all, Imaginary part: n for each state dim
        // S4D-Lin: A_n = -1/2 + i * pi * (n + 1), the same row for every channel.
        _aReal.Fill(NumOps.FromDouble(-0.5));
        Engine.TensorCopy(
            Engine.TensorBroadcastTo(
                Engine.Reshape(
                    Engine.TensorMultiplyScalar(Engine.TensorArange<T>(1, _stateDimension + 1, 1), NumOps.FromDouble(Math.PI)),
                    new[] { 1, _stateDimension }),
                new[] { _innerDimension, _stateDimension }),
            _aImag);

        // B initialized to ones (real part), zeros (imaginary part) per S4D paper
        _bReal.Fill(NumOps.One);
        _bImag.Fill(NumOps.Zero);

        // C initialized with Xavier (random) for both real and imaginary
        InitializeTensor(_cReal);
        InitializeTensor(_cImag);

        // D initialized to ones (skip connection)
        _dParam.Fill(NumOps.One);

        // Input/output projections: Xavier initialization
        InitializeTensor(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);
        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

        // Log delta: initialize so delta ~ 0.001 to 0.1 (uniform in log space)
        // log(0.001) ~ -6.9, log(0.1) ~ -2.3
        for (int i = 0; i < _innerDimension; i++)
        {
            // Use a moderate fixed delta for stable gradients
            // (random deltas can cause numerical instability in gradient check)
            double logVal = -6.9 + Random.NextDouble() * 4.6; // delta ~ [0.001, 0.1] per S4D paper
            _logDelta[i] = NumOps.FromDouble(logVal);
        }
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, modelDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, modelDim });

        // Step 1: Input projection [batch*seq, modelDim] -> [batch*seq, innerDim]
        var input2D = Engine.Reshape(input3D, new[] { batchSize * seqLen, modelDim });
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = Engine.Reshape(_inputProjectionBias, new[] { 1, _innerDimension });
        var projectedWithBias = Engine.TensorAdd(projected, bias2D);
        var projected3D = Engine.Reshape(projectedWithBias, new[] { batchSize, seqLen, _innerDimension });

        // Step 2: ZOH discretization followed by one fused differentiable grouped scan.
        var scanOutput = FusedScanForward(projected3D, batchSize, seqLen);

        // Step 3: Output projection [batch*seq, innerDim] -> [batch*seq, modelDim]
        var scanFlat = Engine.Reshape(scanOutput, new[] { batchSize * seqLen, _innerDimension });
        var outputFlat = Engine.TensorMatMul(scanFlat, _outputProjectionWeights);
        var outBias2D = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        var outputWithBias = Engine.TensorAdd(outputFlat, outBias2D);
        var output3D = Engine.Reshape(outputWithBias, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }

    /// <summary>
    /// Performs the S4D recurrent scan with complex-valued state transitions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each state dimension n has complex parameters (a_n, b_n, c_n).
    /// The recurrence is:
    ///   h_n[t] = A_bar_n * h_n[t-1] + B_bar_n * x[t]
    ///   y[t] = sum_n Re(C_n * h_n[t]) + D * x[t]
    /// where A_bar_n = exp(delta * a_n) (complex exponential).
    /// </para>
    /// <para>
    /// Complex multiplication is done using real arithmetic:
    ///   (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    /// </para>
    /// </remarks>
    /// <summary>
    /// Discretizes the continuous SSM parameters and evaluates the recurrence through
    /// the engine's fused, tape-aware complex diagonal scan.
    /// </summary>
    private Tensor<T> FusedScanForward(Tensor<T> x, int batchSize, int seqLen)
    {
        var delta = Engine.TensorExp(_logDelta);
        _cachedKernelDelta = delta;
        var delta2D = Engine.Reshape(delta, new[] { _innerDimension, 1 });
        var scaledReal = Engine.TensorMultiply(_aReal, delta2D);
        var scaledImag = Engine.TensorMultiply(_aImag, delta2D);
        var expReal = Engine.TensorExp(scaledReal);
        var aBarReal = Engine.TensorMultiply(expReal, Engine.TensorCos(scaledImag));
        var aBarImag = Engine.TensorMultiply(expReal, Engine.TensorSin(scaledImag));
        var diffReal = Engine.TensorSubtractScalar(aBarReal, NumOps.One);
        var denominator = Engine.TensorAddScalar(
            Engine.TensorAdd(
                Engine.TensorMultiply(_aReal, _aReal),
                Engine.TensorMultiply(_aImag, _aImag)),
            NumOps.FromDouble(1e-12));
        var quotientReal = Engine.TensorDivide(
            Engine.TensorAdd(
                Engine.TensorMultiply(diffReal, _aReal),
                Engine.TensorMultiply(aBarImag, _aImag)), denominator);
        var quotientImag = Engine.TensorDivide(
            Engine.TensorSubtract(
                Engine.TensorMultiply(aBarImag, _aReal),
                Engine.TensorMultiply(diffReal, _aImag)), denominator);
        var bBarReal = Engine.TensorSubtract(
            Engine.TensorMultiply(quotientReal, _bReal),
            Engine.TensorMultiply(quotientImag, _bImag));
        var bBarImag = Engine.TensorAdd(
            Engine.TensorMultiply(quotientReal, _bImag),
            Engine.TensorMultiply(quotientImag, _bReal));
        var groupedInput = Engine.Reshape(x, new[] { batchSize, seqLen, _innerDimension, 1 });
        var inputMapReal = Engine.Reshape(bBarReal, new[] { _innerDimension, _stateDimension, 1 });
        var inputMapImag = Engine.Reshape(bBarImag, new[] { _innerDimension, _stateDimension, 1 });
        var outputMapReal = Engine.Reshape(_cReal, new[] { _innerDimension, 1, _stateDimension });
        var outputMapImag = Engine.Reshape(_cImag, new[] { _innerDimension, 1, _stateDimension });
        var skip = Engine.Reshape(_dParam, new[] { _innerDimension, 1 });
        var output = Engine.ComplexDiagonalSsmScanForward(
            groupedInput, aBarReal, aBarImag, inputMapReal, inputMapImag,
            outputMapReal, outputMapImag, skip);
        return Engine.Reshape(output, new[] { batchSize, seqLen, _innerDimension });
    }

    [Scratch]
    private Tensor<T>? _cachedKernelDelta;

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_aRealGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        Engine.TensorAddInPlace(_aReal, Engine.TensorMultiplyScalar(_aRealGradient, negLR));
        Engine.TensorAddInPlace(_aImag, Engine.TensorMultiplyScalar((_aImagGradient ?? throw new InvalidOperationException("_aImagGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_bReal, Engine.TensorMultiplyScalar((_bRealGradient ?? throw new InvalidOperationException("_bRealGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_bImag, Engine.TensorMultiplyScalar((_bImagGradient ?? throw new InvalidOperationException("_bImagGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_cReal, Engine.TensorMultiplyScalar((_cRealGradient ?? throw new InvalidOperationException("_cRealGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_cImag, Engine.TensorMultiplyScalar((_cImagGradient ?? throw new InvalidOperationException("_cImagGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_dParam, Engine.TensorMultiplyScalar((_dParamGradient ?? throw new InvalidOperationException("_dParamGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_logDelta, Engine.TensorMultiplyScalar((_logDeltaGradient ?? throw new InvalidOperationException("_logDeltaGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_inputProjectionWeights, Engine.TensorMultiplyScalar((_inputProjectionWeightsGradient ?? throw new InvalidOperationException("_inputProjectionWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_inputProjectionBias, Engine.TensorMultiplyScalar((_inputProjectionBiasGradient ?? throw new InvalidOperationException("_inputProjectionBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionWeights, Engine.TensorMultiplyScalar((_outputProjectionWeightsGradient ?? throw new InvalidOperationException("_outputProjectionWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionBias, Engine.TensorMultiplyScalar((_outputProjectionBiasGradient ?? throw new InvalidOperationException("_outputProjectionBiasGradient is null where it is required.")), negLR));

        // Register trainable parameters for tape-based autodiff. The state-space core belongs here
        // with the projections: UpdateParameters above applies gradients to A, B, C, D and Delta, so
        // they are trained on the legacy path, but registering only the projections made them
        // invisible to the tape path -- which trains exclusively from registered parameters -- and to
        // GetParameters/SetParameters, and therefore to serialization.
        RegisterTrainableParameter(_aReal, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_aImag, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bReal, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bImag, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_cReal, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_cImag, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_dParam, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_logDelta, PersistentTensorRole.ScaleParameters);
        RegisterTrainableParameter(_inputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    /// <inheritdoc />
    public override void ClearGradients()
    {
        base.ClearGradients();
        _aRealGradient = null; _aImagGradient = null; _bRealGradient = null; _bImagGradient = null; _cRealGradient = null; _cImagGradient = null; _dParamGradient = null; _logDeltaGradient = null;
        _inputProjectionWeightsGradient = null; _inputProjectionBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _originalInputShape = null;
        _aRealGradient = null;
        _aImagGradient = null;
        _bRealGradient = null;
        _bImagGradient = null;
        _cRealGradient = null;
        _cImagGradient = null;
        _dParamGradient = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
        _logDeltaGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["StateDimension"] = _stateDimension.ToString();
        metadata["InnerDimension"] = _innerDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the A parameter (real part) for external inspection.
    /// </summary>
    public Tensor<T> GetAReal() => _aReal;

    /// <summary>
    /// Gets the A parameter (imaginary part) for external inspection.
    /// </summary>
    public Tensor<T> GetAImag() => _aImag;

    /// <summary>
    /// Gets the D skip connection parameter for external inspection.
    /// </summary>
    public Tensor<T> GetDParameter() => _dParam;

    /// <summary>
    /// Gets the log-delta (discretization step) parameter for external inspection.
    /// </summary>
    public Tensor<T> GetLogDelta() => _logDelta;
}
