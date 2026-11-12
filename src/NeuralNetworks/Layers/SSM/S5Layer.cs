using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Simplified State Space (S5) layer from Smith et al., 2023.
/// </summary>
/// <remarks>
/// <para>
/// S5 is a multi-input multi-output (MIMO) state space model that uses a diagonalized state
/// matrix with parallel scan for efficient sequence processing. Unlike S4D, which applies
/// independent single-input single-output (SISO) SSMs per feature, S5 uses a single MIMO SSM
/// that couples all input dimensions through shared state dynamics.
/// </para>
/// <para>
/// The continuous-time MIMO SSM is:
/// <code>
///   x'(t) = A * x(t) + B * u(t)       (state equation, x in R^N, u in R^H)
///   y(t)  = Re(C * x(t)) + D * u(t)   (output equation, y in R^H)
/// </code>
/// where A is diagonalized via eigendecomposition: A = V * Lambda * V^{-1}, with Lambda containing
/// complex eigenvalues. This diagonalization decouples the state dimensions while retaining the
/// MIMO structure through B and C matrices that mix input/output dimensions with state dimensions.
/// </para>
/// <para>
/// Discretization uses the Zero-Order Hold (ZOH) method:
/// <code>
///   A_bar = exp(Delta * Lambda)                    (diagonal, element-wise complex exp)
///   B_bar = (A_bar - I) * Lambda^{-1} * B_tilde   (where B_tilde = V^{-1} * B)
/// </code>
/// The discrete recurrence x_t = A_bar * x_{t-1} + B_bar * u_t is then computed efficiently
/// using a parallel associative scan in O(L log L) time, where L is the sequence length.
/// </para>
/// <para>
/// The A matrix is initialized using the HiPPO framework (High-order Polynomial Projection Operator)
/// which provides optimal polynomial approximations for continuous signal history. The S5 paper
/// uses the HiPPO-LegS (Legendre-Scaled) initialization, which gives the diagonal eigenvalues
/// Lambda_n = -1/2 + n*pi*i, placing them along a vertical line in the left half-plane with
/// logarithmically spaced frequencies. This ensures stable dynamics with a rich frequency spectrum
/// for capturing long-range dependencies.
/// </para>
/// <para><b>For Beginners:</b> S5 is an efficient sequence model that competes with Transformers
/// on long-range tasks while being much more computationally efficient.
///
/// Think of S5 as a bank of coupled oscillators processing a signal:
/// - Each oscillator has a complex frequency (how fast it vibrates) and a decay rate
/// - Unlike S4D where each feature has its own independent oscillator bank, S5 shares
///   one oscillator bank across ALL features -- this is the "MIMO" (multi-input multi-output) part
/// - The B matrix controls how each input feature excites the oscillators
/// - The C matrix controls how oscillator states are combined to produce each output feature
/// - The D matrix provides a direct skip connection from input to output
///
/// The key advantages of S5 over S4D:
/// 1. MIMO formulation: Features interact through shared state, enabling richer representations
/// 2. Parallel scan: The entire sequence is processed in O(L log L) instead of O(L) sequential steps
/// 3. Simpler architecture: Fewer parameters than S4D with comparable or better performance
///
/// The "simplified" in S5 refers to replacing S4's complex NPLR (Normal Plus Low-Rank) structure
/// with a straightforward diagonalization, making the model much easier to implement and reason about
/// while maintaining strong empirical performance.
/// </para>
/// <para>
/// <b>Reference:</b> Smith et al., "Simplified State Space Layers for Sequence Modeling", ICLR 2023.
/// https://arxiv.org/abs/2208.04933
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
public partial class S5Layer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _stateDimension;

    // Diagonal A stored as real/imaginary pairs: [stateDim]
    // Lambda = a_real + i * a_imag, initialized with HiPPO-LegS: Lambda_n = -1/2 + n*pi*i
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _aReal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _aImag;

    // B input matrix (complex, after V^{-1} transformation): [stateDim, modelDim] real/imag
    // Maps H-dimensional input to N-dimensional complex state
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _bReal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _bImag;

    // C output matrix (complex): [modelDim, stateDim] real/imag
    // Maps N-dimensional complex state to H-dimensional real output via Re(C * x)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _cReal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _cImag;

    // D skip connection: [modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _dParam;

    // Discretization step size: [stateDim] (stored as log for positivity)
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _logDelta;

    // Input projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputProjectionBias;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

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
    private Tensor<T>? _logDeltaGradient;
    [Scratch]
    private Tensor<T>? _inputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _inputProjectionBiasGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension (H), the width of input and output at each sequence position.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the SSM state dimension (N), the number of diagonal complex eigenvalues.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each state dimension is an independent complex oscillator with
    /// a learned frequency and decay rate. The HiPPO initialization spaces these frequencies so that
    /// the model can capture patterns at many different time scales simultaneously. Typical values
    /// are 64 (as in the S5 paper) for a good balance of expressivity and efficiency.</para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// Creates a new S5 (Simplified State Space) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (H). Default: 256.
    /// <para><b>For Beginners:</b> The width of the representation at each sequence position.
    /// This determines both the input and output size of the layer.</para>
    /// </param>
    /// <param name="stateDimension">
    /// SSM state dimension (N). Default: 64.
    /// <para><b>For Beginners:</b> The number of independent complex oscillators in the diagonalized
    /// state matrix. Unlike S4D where each feature has N states (total H*N), S5 shares N states
    /// across all H features via the MIMO B and C matrices. The S5 paper uses N=64 as default.
    /// Larger N captures more temporal patterns but increases computation.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public S5Layer(
        int sequenceLength,
        int modelDimension = 256,
        int stateDimension = 64,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        _sequenceLength = sequenceLength;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (sequenceLength <= 0)
            throw new ArgumentException(
                $"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException(
                $"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (stateDimension <= 0)
            throw new ArgumentException(
                $"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));

        _modelDimension = modelDimension;
        _stateDimension = stateDimension;

        // Diagonal A: [stateDim] real and imaginary (complex eigenvalues)
        _aReal = new Tensor<T>([stateDimension]);
        _aImag = new Tensor<T>([stateDimension]);

        // B (complex): [stateDim, modelDim] -- maps input to state
        _bReal = new Tensor<T>([stateDimension, modelDimension]);
        _bImag = new Tensor<T>([stateDimension, modelDimension]);

        // C (complex): [modelDim, stateDim] -- maps state to output
        _cReal = new Tensor<T>([modelDimension, stateDimension]);
        _cImag = new Tensor<T>([modelDimension, stateDimension]);

        // D: [modelDim] skip connection
        _dParam = new Tensor<T>([modelDimension]);

        // Log delta (discretization step): [stateDim]
        _logDelta = new Tensor<T>([stateDimension]);

        // Input projection: [modelDim, modelDim]
        _inputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _inputProjectionBias = new Tensor<T>([modelDimension]);

        // Output projection: [modelDim, modelDim]
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // HiPPO-LegS diagonal initialization: Lambda_n = -1/2 + n*pi*i
        // Real part: -0.5 (controls decay rate, placing eigenvalues in the stable left half-plane)
        // Imaginary part: n*pi (logarithmically spaced frequencies for multi-scale pattern capture)
        // S4D-Lin initialization: A_n = -1/2 + i * pi * (n + 1).
        _aReal.Fill(NumOps.FromDouble(-0.5));
        Engine.TensorCopy(
            Engine.TensorMultiplyScalar(Engine.TensorArange<T>(1, _stateDimension + 1, 1), NumOps.FromDouble(Math.PI)),
            _aImag);

        // B initialized with Xavier/Glorot for both real and imaginary parts
        // This ensures proper signal scaling through the MIMO input mapping
        InitializeTensor2D(_bReal);
        InitializeTensor2D(_bImag);

        // C initialized with Xavier/Glorot for both real and imaginary parts
        InitializeTensor2D(_cReal);
        InitializeTensor2D(_cImag);

        // D initialized to ones (identity skip connection, following S4/S5 convention)
        _dParam.Fill(NumOps.One);

        // Input/output projections: Xavier initialization
        InitializeTensor2D(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

        // Log delta: initialize so delta spans [0.001, 0.1] uniformly in log space
        // log(0.001) ~ -6.9, log(0.1) ~ -2.3
        for (int n = 0; n < _stateDimension; n++)
        {
            double logVal = -6.9 + Random.NextDouble() * 4.6;
            _logDelta[n] = NumOps.FromDouble(logVal);
        }
    }

    private void InitializeTensor2D(Tensor<T> tensor)
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

        // Step 1: Input projection [batch*seq, modelDim] -> [batch*seq, modelDim]
        var input2D = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = Engine.Reshape(_inputProjectionBias, new[] { 1, _modelDimension });
        var projectedWithBias = Engine.TensorAdd(projected, bias2D);
        var projected3D = Engine.Reshape(projectedWithBias, new[] { batchSize, seqLen, _modelDimension });

        // Step 2: ZOH discretization followed by one fused differentiable MIMO scan.
        var scanOutput = FusedMimoScanForward(projected3D, batchSize, seqLen);

        // Step 3: Output projection [batch*seq, modelDim] -> [batch*seq, modelDim]
        var scanFlat = Engine.Reshape(scanOutput, new[] { batchSize * seqLen, _modelDimension });
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
    /// Performs the S5 MIMO recurrence with diagonalized complex state transitions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The MIMO recurrence for diagonal A with N state dimensions and H input/output dimensions:
    /// <code>
    ///   x_n[t] = A_bar_n * x_n[t-1] + sum_h(B_bar[n,h] * u_h[t])   for n = 1..N
    ///   y_h[t] = sum_n Re(C[h,n] * x_n[t]) + D[h] * u_h[t]          for h = 1..H
    /// </code>
    /// where A_bar_n = exp(delta_n * A_n) is the discretized diagonal state transition,
    /// and B_bar[n,h] = (A_bar_n - 1) / A_n * B[n,h] is the discretized input matrix.
    /// </para>
    /// <para>
    /// The parallel scan exploits the associative property of the linear recurrence:
    /// each (A_bar, B_bar * u) pair can be combined using the binary operator
    /// (a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2), enabling O(log L) parallel computation.
    /// Here we implement a sequential scan for clarity; parallelism is at the batch level.
    /// </para>
    /// </remarks>
    private static (double r, double i) ComputeBBarScalar(double dt, double ar, double ai, double br, double bi)
    {
        double expR = Math.Exp(dt * ar);
        double cosA = Math.Cos(dt * ai);
        double sinA = Math.Sin(dt * ai);
        double diffR = expR * cosA - 1.0;
        double diffI = expR * sinA;
        double aMagSq = ar * ar + ai * ai;
        double quotR, quotI;
        if (aMagSq < 1e-12) { quotR = dt * br; quotI = dt * bi; }
        else { quotR = (diffR * ar + diffI * ai) / aMagSq; quotI = (diffI * ar - diffR * ai) / aMagSq; }
        return (quotR * br - quotI * bi, quotR * bi + quotI * br);
    }

    /// <summary>
    /// Discretizes the continuous MIMO SSM parameters and evaluates the recurrence through
    /// the engine's fused, tape-aware complex diagonal scan.
    /// </summary>
    private Tensor<T> FusedMimoScanForward(Tensor<T> u, int batchSize, int seqLen)
    {
        var delta = Engine.TensorExp(_logDelta);
        var scaledReal = Engine.TensorMultiply(delta, _aReal);
        var scaledImag = Engine.TensorMultiply(delta, _aImag);
        var expReal = Engine.TensorExp(scaledReal);
        var aBarReal = Engine.TensorMultiply(expReal, Engine.TensorCos(scaledImag));
        var aBarImag = Engine.TensorMultiply(expReal, Engine.TensorSin(scaledImag));
        var diffReal = Engine.TensorSubtractScalar(aBarReal, NumOps.One);
        var denominator = Engine.TensorAddScalar(
            Engine.TensorAdd(Engine.TensorMultiply(_aReal, _aReal), Engine.TensorMultiply(_aImag, _aImag)),
            NumOps.FromDouble(1e-12));
        var quotientReal = Engine.TensorDivide(
            Engine.TensorAdd(Engine.TensorMultiply(diffReal, _aReal), Engine.TensorMultiply(aBarImag, _aImag)),
            denominator);
        var quotientImag = Engine.TensorDivide(
            Engine.TensorSubtract(Engine.TensorMultiply(aBarImag, _aReal), Engine.TensorMultiply(diffReal, _aImag)),
            denominator);
        var quotientReal2D = Engine.Reshape(quotientReal, new[] { _stateDimension, 1 });
        var quotientImag2D = Engine.Reshape(quotientImag, new[] { _stateDimension, 1 });
        var bBarReal = Engine.TensorSubtract(
            Engine.TensorMultiply(_bReal, quotientReal2D),
            Engine.TensorMultiply(_bImag, quotientImag2D));
        var bBarImag = Engine.TensorAdd(
            Engine.TensorMultiply(_bImag, quotientReal2D),
            Engine.TensorMultiply(_bReal, quotientImag2D));

        var groupedInput = Engine.Reshape(u, new[] { batchSize, seqLen, 1, _modelDimension });
        var output = Engine.ComplexDiagonalSsmScanForward(
            groupedInput,
            Engine.Reshape(aBarReal, new[] { 1, _stateDimension }),
            Engine.Reshape(aBarImag, new[] { 1, _stateDimension }),
            Engine.Reshape(bBarReal, new[] { 1, _stateDimension, _modelDimension }),
            Engine.Reshape(bBarImag, new[] { 1, _stateDimension, _modelDimension }),
            Engine.Reshape(_cReal, new[] { 1, _modelDimension, _stateDimension }),
            Engine.Reshape(_cImag, new[] { 1, _modelDimension, _stateDimension }),
            Engine.Reshape(_dParam, new[] { 1, _modelDimension }));
        return Engine.Reshape(output, new[] { batchSize, seqLen, _modelDimension });
    }


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
        Engine.TensorAddInPlace(_inputProjectionWeights,
            Engine.TensorMultiplyScalar((_inputProjectionWeightsGradient ?? throw new InvalidOperationException("_inputProjectionWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_inputProjectionBias,
            Engine.TensorMultiplyScalar((_inputProjectionBiasGradient ?? throw new InvalidOperationException("_inputProjectionBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionWeights,
            Engine.TensorMultiplyScalar((_outputProjectionWeightsGradient ?? throw new InvalidOperationException("_outputProjectionWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionBias,
            Engine.TensorMultiplyScalar((_outputProjectionBiasGradient ?? throw new InvalidOperationException("_outputProjectionBiasGradient is null where it is required.")), negLR));

        // Register trainable parameters for tape-based autodiff
        // Register trainable parameters for tape-based autodiff. The state-space core belongs here
        // with the projections: UpdateParameters above applies gradients to every one of these, so
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

    private Tensor<T>[] GetAllTensors() =>
    [
        _aReal, _aImag,
        _bReal, _bImag,
        _cReal, _cImag,
        _dParam, _logDelta,
        _inputProjectionWeights, _inputProjectionBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

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
        _logDeltaGradient = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["StateDimension"] = _stateDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the A parameter (real part of diagonal eigenvalues) for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The real part of the diagonal eigenvalues controls the decay rate of each state dimension.
    /// Initialized to -0.5 per the HiPPO-LegS scheme.
    /// </para>
    /// </remarks>
    public Tensor<T> GetAReal() => _aReal;

    /// <summary>
    /// Gets the A parameter (imaginary part of diagonal eigenvalues) for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The imaginary part of the diagonal eigenvalues controls the oscillation frequency.
    /// Initialized to n*pi per the HiPPO-LegS scheme, giving logarithmically spaced frequencies.
    /// </para>
    /// </remarks>
    public Tensor<T> GetAImag() => _aImag;

    /// <summary>
    /// Gets the D skip connection parameter for external inspection.
    /// </summary>
    public Tensor<T> GetDParameter() => _dParam;

    /// <summary>
    /// Gets the log-delta (discretization step) parameter for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Delta controls the discretization step size per state dimension. Stored as log(delta)
    /// to ensure positivity. Smaller delta values give finer temporal resolution.
    /// </para>
    /// </remarks>
    public Tensor<T> GetLogDelta() => _logDelta;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;
}
