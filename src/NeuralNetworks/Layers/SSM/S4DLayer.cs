using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
public class S4DLayer<T> : LayerBase<T>
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _stateDimension;
    private readonly int _innerDimension;

    // A parameter stored as real/imaginary pairs: [innerDim, stateDim, 2]
    // A = a_real + i * a_imag, initialized with S4D-Lin: A_n = -1/2 + n*i
    private Tensor<T> _aReal;
    private Tensor<T> _aImag;

    // B projection: [modelDim, innerDim * stateDim] (complex, stored as real/imag pairs)
    // In S4D-Lin, B is typically initialized to ones.
    private Tensor<T> _bReal;
    private Tensor<T> _bImag;

    // C projection: [innerDim * stateDim, modelDim] (complex, learned)
    private Tensor<T> _cReal;
    private Tensor<T> _cImag;

    // D: [innerDim] (skip connection, real-valued)
    private Tensor<T> _dParam;

    // Input projection: [modelDim, innerDim]
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Output projection: [innerDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Delta (discretization step size): [innerDim] (learned, stored as log for positivity)
    private Tensor<T> _logDelta;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedInput;
    private Tensor<T>? _lastHiddenStatesReal;
    private Tensor<T>? _lastHiddenStatesImag;
    private Tensor<T>? _lastScanOutputReal;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _aRealGradient;
    private Tensor<T>? _aImagGradient;
    private Tensor<T>? _bRealGradient;
    private Tensor<T>? _bImagGradient;
    private Tensor<T>? _cRealGradient;
    private Tensor<T>? _cImagGradient;
    private Tensor<T>? _dParamGradient;
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;
    private Tensor<T>? _logDeltaGradient;

    /// <inheritdoc />
    /// <summary>
    /// Training is not yet supported. The backward pass uses simplified gradient paths and skips
    /// the chain rule through exp(delta*A) and delta*B discretization. Full backpropagation through
    /// the S4D recurrence is required before enabling training.
    /// </summary>
    public override bool SupportsTraining => false;

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

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _aReal.Length + _aImag.Length +
        _bReal.Length + _bImag.Length +
        _cReal.Length + _cImag.Length +
        _dParam.Length +
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length +
        _logDelta.Length;

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

    /// <summary>
    /// Runs the recurrence for a single (d, n) state dimension with given A, B, C parameters.
    /// Returns the contribution of state n to the output at dimension d: Re(C_n * h_n[t]).
    /// </summary>
    private double[,] RunSingleStateRecurrence(
        Tensor<T> x, int batchSize, int seqLen,
        int d, int n, double dt,
        double ar, double ai, double br, double bi, double cr, double ci)
    {
        // Discretize
        double expR = Math.Exp(dt * ar);
        double abrV = expR * Math.Cos(dt * ai);
        double abiV = expR * Math.Sin(dt * ai);
        var bbar = ComputeBBar(dt, ar, ai, br, bi);

        var output = new double[batchSize, seqLen];

        for (int b = 0; b < batchSize; b++)
        {
            double hR = 0, hI = 0;
            for (int t = 0; t < seqLen; t++)
            {
                double xt = NumOps.ToDouble(x[b, t, d]);
                // h[t+1] = A_bar * h[t] + B_bar * x[t]
                double newHR = abrV * hR - abiV * hI + bbar.r * xt;
                double newHI = abrV * hI + abiV * hR + bbar.i * xt;
                hR = newHR;
                hI = newHI;
                // y_contribution = Re(C * h) = cr*hR - ci*hI
                output[b, t] = cr * hR - ci * hI;
            }
        }

        return output;
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
        for (int d = 0; d < _innerDimension; d++)
        {
            for (int n = 0; n < _stateDimension; n++)
            {
                _aReal[new[] { d, n }] = NumOps.FromDouble(-0.5);
                _aImag[new[] { d, n }] = NumOps.FromDouble(Math.PI * (n + 1));
            }
        }

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
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape.ToArray();

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? input.Reshape(1, seqLen, modelDim)
            : input.Reshape(batchSize, seqLen, modelDim);

        _lastInput = input3D;

        // Step 1: Input projection [batch*seq, modelDim] -> [batch*seq, innerDim]
        var input2D = input3D.Reshape(batchSize * seqLen, modelDim);
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = _inputProjectionBias.Reshape(1, _innerDimension);
        var projectedWithBias = Engine.TensorBroadcastAdd(projected, bias2D);
        var projected3D = projectedWithBias.Reshape(batchSize, seqLen, _innerDimension);
        _lastProjectedInput = projected3D;

        // Step 2: Compute discretized SSM via recurrent scan (complex arithmetic)
        var scanOutput = ComplexRecurrentScan(projected3D, batchSize, seqLen);
        _lastScanOutputReal = scanOutput;

        // Step 3: Output projection [batch*seq, innerDim] -> [batch*seq, modelDim]
        var scanFlat = scanOutput.Reshape(batchSize * seqLen, _innerDimension);
        var outputFlat = Engine.TensorMatMul(scanFlat, _outputProjectionWeights);
        var outBias2D = _outputProjectionBias.Reshape(1, _modelDimension);
        var outputWithBias = Engine.TensorBroadcastAdd(outputFlat, outBias2D);
        var output3D = outputWithBias.Reshape(batchSize, seqLen, _modelDimension);

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return result.Reshape(seqLen, _modelDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return result.Reshape(outputShape);
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
    private Tensor<T> ComplexRecurrentScan(Tensor<T> x, int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _innerDimension });

        // Compute delta = exp(logDelta): [innerDim]
        var delta = Engine.TensorExp(_logDelta);

        // Pre-compute discretized A_bar = exp(delta * A) for complex A
        // exp((delta*a_real) + i*(delta*a_imag)) = exp(delta*a_real) * (cos(delta*a_imag) + i*sin(delta*a_imag))
        // A_bar_real[d,n] = exp(delta[d] * a_real[d,n]) * cos(delta[d] * a_imag[d,n])
        // A_bar_imag[d,n] = exp(delta[d] * a_real[d,n]) * sin(delta[d] * a_imag[d,n])
        var aBarReal = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        var aBarImag = new Tensor<T>(new[] { _innerDimension, _stateDimension });

        // B_bar = (A_bar - I) * A^{-1} * B, simplified for diagonal A:
        // B_bar_n = (exp(delta*a_n) - 1) / a_n * b_n
        // For numerical stability with small a_n, use: B_bar_n ~ delta * b_n (first-order approx)
        var bBarReal = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        var bBarImag = new Tensor<T>(new[] { _innerDimension, _stateDimension });

        for (int d = 0; d < _innerDimension; d++)
        {
            double dt = NumOps.ToDouble(delta[d]);
            for (int n = 0; n < _stateDimension; n++)
            {
                double ar = NumOps.ToDouble(_aReal[new[] { d, n }]);
                double ai = NumOps.ToDouble(_aImag[new[] { d, n }]);
                double br = NumOps.ToDouble(_bReal[new[] { d, n }]);
                double bi = NumOps.ToDouble(_bImag[new[] { d, n }]);

                // Complex exp: exp(dt*ar) * (cos(dt*ai) + i*sin(dt*ai))
                double expDtAr = Math.Exp(dt * ar);
                double cosVal = Math.Cos(dt * ai);
                double sinVal = Math.Sin(dt * ai);

                double abrVal = expDtAr * cosVal;
                double abiVal = expDtAr * sinVal;

                aBarReal[new[] { d, n }] = NumOps.FromDouble(abrVal);
                aBarImag[new[] { d, n }] = NumOps.FromDouble(abiVal);

                // B_bar = (A_bar - I) / A * B (complex division and multiplication)
                // (A_bar - I) = (abrVal - 1) + i * abiVal
                double diffReal = abrVal - 1.0;
                double diffImag = abiVal;

                // Complex division by A = ar + i*ai: (diffR + i*diffI) / (ar + i*ai)
                double aMagSq = ar * ar + ai * ai;
                double quotReal, quotImag;
                if (aMagSq < 1e-12)
                {
                    // When A is very small, use first-order approximation: B_bar ~ delta * B
                    quotReal = dt * br;
                    quotImag = dt * bi;
                }
                else
                {
                    double divReal = (diffReal * ar + diffImag * ai) / aMagSq;
                    double divImag = (diffImag * ar - diffReal * ai) / aMagSq;

                    // Multiply by B: (divR + i*divI) * (br + i*bi)
                    quotReal = divReal * br - divImag * bi;
                    quotImag = divReal * bi + divImag * br;
                }

                bBarReal[new[] { d, n }] = NumOps.FromDouble(quotReal);
                bBarImag[new[] { d, n }] = NumOps.FromDouble(quotImag);
            }
        }

        // Hidden state: [batch, innerDim, stateDim] real and imaginary
        var hReal = new Tensor<T>(new[] { batchSize, _innerDimension, _stateDimension });
        var hImag = new Tensor<T>(new[] { batchSize, _innerDimension, _stateDimension });

        // Store hidden states for backward pass
        var allHiddenReal = new Tensor<T>(new[] { batchSize, seqLen + 1, _innerDimension, _stateDimension });
        var allHiddenImag = new Tensor<T>(new[] { batchSize, seqLen + 1, _innerDimension, _stateDimension });

        // D for skip connection: [1, innerDim]
        var D2D = _dParam.Reshape(1, _innerDimension);

        // Expand A_bar and B_bar for batch broadcasting: [1, innerDim, stateDim]
        var aBarReal3D = Engine.TensorExpandDims(aBarReal, 0);
        var aBarImag3D = Engine.TensorExpandDims(aBarImag, 0);
        var bBarReal3D = Engine.TensorExpandDims(bBarReal, 0);
        var bBarImag3D = Engine.TensorExpandDims(bBarImag, 0);

        // Pre-compute C for output: [1, innerDim, stateDim]
        var cReal3D = Engine.TensorExpandDims(_cReal, 0);
        var cImag3D = Engine.TensorExpandDims(_cImag, 0);

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);  // [batch, innerDim]
            var x_t_3D = Engine.TensorExpandDims(x_t, 2);  // [batch, innerDim, 1]

            // State update: h = A_bar * h_prev + B_bar * x
            // Complex multiply A_bar * h:
            //   real = A_bar_r * h_r - A_bar_i * h_i
            //   imag = A_bar_r * h_i + A_bar_i * h_r
            var ahReal = Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(aBarReal3D, hReal),
                Engine.TensorNegate(Engine.TensorBroadcastMultiply(aBarImag3D, hImag)));
            var ahImag = Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(aBarReal3D, hImag),
                Engine.TensorBroadcastMultiply(aBarImag3D, hReal));

            // Complex B_bar * x (x is real, so imag part of x is 0):
            //   real = B_bar_r * x
            //   imag = B_bar_i * x
            var bxReal = Engine.TensorBroadcastMultiply(bBarReal3D, x_t_3D);
            var bxImag = Engine.TensorBroadcastMultiply(bBarImag3D, x_t_3D);

            hReal = Engine.TensorAdd(ahReal, bxReal);
            hImag = Engine.TensorAdd(ahImag, bxImag);

            // Output: y_t = sum_n Re(C * h) + D * x
            // Re(C * h) = C_r * h_r - C_i * h_i
            var chReal = Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(cReal3D, hReal),
                Engine.TensorNegate(Engine.TensorBroadcastMultiply(cImag3D, hImag)));
            var y_t = Engine.ReduceSum(chReal, new int[] { 2 });  // [batch, innerDim]

            // Skip connection
            var Dx = Engine.TensorBroadcastMultiply(D2D, x_t);
            y_t = Engine.TensorAdd(y_t, Dx);

            allHiddenReal.SetSlice(1, t + 1, hReal);
            allHiddenImag.SetSlice(1, t + 1, hImag);
            output.SetSlice(1, t, y_t);
        }

        _lastHiddenStatesReal = allHiddenReal;
        _lastHiddenStatesImag = allHiddenImag;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastProjectedInput == null || _lastScanOutputReal == null ||
            _lastHiddenStatesReal == null || _lastHiddenStatesImag == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int rank = outputGradient.Shape.Length;
        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var scanFlat = _lastScanOutputReal.Reshape(batchSize * seqLen, _innerDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(
            scanFlat.Transpose([1, 0]), gradFlat);

        var dScan = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _innerDimension);

        // Recurrent scan backward (complex)
        var dProjected = ComplexRecurrentScanBackward(
            dScan, _lastProjectedInput, batchSize, seqLen);

        // Input projection backward
        var dProjFlat = dProjected.Reshape(batchSize * seqLen, _innerDimension);
        _inputProjectionBiasGradient = Engine.ReduceSum(dProjected, new int[] { 0, 1 });

        var input2D = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        _inputProjectionWeightsGradient = Engine.TensorMatMul(
            input2D.Transpose([1, 0]), dProjFlat);

        var inputGradFlat = Engine.TensorMatMul(
            dProjFlat, _inputProjectionWeights.Transpose([1, 0]));
        var inputGrad3D = inputGradFlat.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return inputGrad3D.Reshape(_originalInputShape);

        return inputGrad3D;
    }

    /// <summary>
    /// Backward pass through the complex recurrent scan.
    /// </summary>
    private Tensor<T> ComplexRecurrentScanBackward(
        Tensor<T> dOutput, Tensor<T> x, int batchSize, int seqLen)
    {
        var dX = new Tensor<T>(new[] { batchSize, seqLen, _innerDimension });

        // Initialize gradient accumulators
        _aRealGradient = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        _aImagGradient = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        _bRealGradient = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        _bImagGradient = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        _cRealGradient = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        _cImagGradient = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        _dParamGradient = new Tensor<T>(new[] { _innerDimension });
        _logDeltaGradient = new Tensor<T>(new[] { _innerDimension });

        // Recompute discretized parameters
        var delta = Engine.TensorExp(_logDelta);

        var aBarReal = new Tensor<T>(new[] { _innerDimension, _stateDimension });
        var aBarImag = new Tensor<T>(new[] { _innerDimension, _stateDimension });

        for (int d = 0; d < _innerDimension; d++)
        {
            double dt = NumOps.ToDouble(delta[d]);
            for (int n = 0; n < _stateDimension; n++)
            {
                double ar = NumOps.ToDouble(_aReal[new[] { d, n }]);
                double ai = NumOps.ToDouble(_aImag[new[] { d, n }]);
                double expDtAr = Math.Exp(dt * ar);
                aBarReal[new[] { d, n }] = NumOps.FromDouble(expDtAr * Math.Cos(dt * ai));
                aBarImag[new[] { d, n }] = NumOps.FromDouble(expDtAr * Math.Sin(dt * ai));
            }
        }

        var aBarReal3D = Engine.TensorExpandDims(aBarReal, 0);
        var aBarImag3D = Engine.TensorExpandDims(aBarImag, 0);
        var cReal3D = Engine.TensorExpandDims(_cReal, 0);
        var cImag3D = Engine.TensorExpandDims(_cImag, 0);
        var D2D = _dParam.Reshape(1, _innerDimension);

        // Running gradient of hidden state (complex)
        var dhReal = new Tensor<T>(new[] { batchSize, _innerDimension, _stateDimension });
        var dhImag = new Tensor<T>(new[] { batchSize, _innerDimension, _stateDimension });

        // Hoist invariant null guards out of the loop
        var lastHiddenReal = _lastHiddenStatesReal ?? throw new InvalidOperationException("_lastHiddenStatesReal has not been initialized.");
        var lastHiddenImag = _lastHiddenStatesImag ?? throw new InvalidOperationException("_lastHiddenStatesImag has not been initialized.");

        for (int t = seqLen - 1; t >= 0; t--)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);
            var hReal_t = lastHiddenReal.GetSliceAlongDimension(t + 1, 1);
            var hImag_t = lastHiddenImag.GetSliceAlongDimension(t + 1, 1);
            var hReal_prev = lastHiddenReal.GetSliceAlongDimension(t, 1);
            var hImag_prev = lastHiddenImag.GetSliceAlongDimension(t, 1);

            var dOut_t_3D = Engine.TensorExpandDims(dOut_t, 2);

            // D skip gradient
            var dD_t = Engine.ReduceSum(Engine.TensorMultiply(x_t, dOut_t), new int[] { 0 });
            _dParamGradient = Engine.TensorAdd(_dParamGradient, dD_t);

            var dX_t = Engine.TensorBroadcastMultiply(D2D, dOut_t);

            // Gradient from output: y = sum_n Re(C * h)
            // dh_real += C_r * dOut, dh_imag -= C_i * dOut (from Re(C*h) = C_r*h_r - C_i*h_i)
            dhReal = Engine.TensorAdd(dhReal,
                Engine.TensorBroadcastMultiply(cReal3D, dOut_t_3D));
            dhImag = Engine.TensorAdd(dhImag,
                Engine.TensorNegate(Engine.TensorBroadcastMultiply(cImag3D, dOut_t_3D)));

            // dC_real = sum_batch sum_d(h_real * dOut), dC_imag = -sum_batch sum_d(h_imag * dOut)
            var hR_dOut = Engine.TensorBroadcastMultiply(hReal_t, dOut_t_3D);
            var hI_dOut = Engine.TensorBroadcastMultiply(hImag_t, dOut_t_3D);
            var dCr_t = Engine.ReduceSum(hR_dOut, new int[] { 0 });
            var dCi_t = Engine.TensorNegate(Engine.ReduceSum(hI_dOut, new int[] { 0 }));
            _cRealGradient = Engine.TensorAdd(_cRealGradient, dCr_t);
            _cImagGradient = Engine.TensorAdd(_cImagGradient, dCi_t);

            // Backprop through complex state update: h = A_bar * h_prev + B_bar * x
            // dh -> d_A_bar, d_h_prev, d_B_bar, d_x
            // Complex A_bar * h_prev backward:
            // d_A_bar_real = dh_r * h_prev_r + dh_i * h_prev_i
            // d_A_bar_imag = dh_i * h_prev_r - dh_r * h_prev_i
            // d_h_prev_real = A_bar_r * dh_r + A_bar_i * dh_i
            // d_h_prev_imag = A_bar_r * dh_i - A_bar_i * dh_r

            // d_x from B_bar * x (x is real):
            // dx += sum_n(B_bar_r * dh_r + B_bar_i * dh_i)
            var bBarRDh = Engine.TensorBroadcastMultiply(
                Engine.TensorExpandDims(
                    new Tensor<T>(new[] { _innerDimension, _stateDimension }), 0),
                dhReal);

            // Compute d_x contribution from B_bar
            // Since B_bar * x: real = B_bar_r * x, imag = B_bar_i * x
            // dB_bar_r contribution to dx: sum_n dh_r[b,d,n] * B_bar_r[d,n]
            // dB_bar_i contribution to dx: sum_n dh_i[b,d,n] * B_bar_i[d,n]
            // Actually: dx += sum_n(B_bar_r * dh_r + B_bar_i * dh_i) via chain rule

            // Recompute B_bar for this gradient step (simplified: use delta * B approximation for gradient)
            var x_t_3D = Engine.TensorExpandDims(x_t, 2);


            // dB: dh * x (for each complex component)
            var dBr_t = Engine.ReduceSum(Engine.TensorBroadcastMultiply(dhReal, x_t_3D), new int[] { 0 });
            var dBi_t = Engine.ReduceSum(Engine.TensorBroadcastMultiply(dhImag, x_t_3D), new int[] { 0 });
            _bRealGradient = Engine.TensorAdd(_bRealGradient, dBr_t);
            _bImagGradient = Engine.TensorAdd(_bImagGradient, dBi_t);

            // d_x from state: sum_n(B_bar_r * dh_r + B_bar_i * dh_i) using simplified B_bar ~ delta*B
            var bBarReal3D = Engine.TensorExpandDims(
                new Tensor<T>(new[] { _innerDimension, _stateDimension }), 0);
            var bBarImag3D = Engine.TensorExpandDims(
                new Tensor<T>(new[] { _innerDimension, _stateDimension }), 0);

            // Approximate: dx += delta * sum_n(B_r * dh_r + B_i * dh_i)
            var bR3D = Engine.TensorExpandDims(_bReal, 0);
            var bI3D = Engine.TensorExpandDims(_bImag, 0);
            var dXFromState = Engine.TensorAdd(
                Engine.ReduceSum(Engine.TensorBroadcastMultiply(bR3D, dhReal), new int[] { 2 }),
                Engine.ReduceSum(Engine.TensorBroadcastMultiply(bI3D, dhImag), new int[] { 2 }));

            var deltaExpanded = Engine.TensorExpandDims(delta, 0);
            dX_t = Engine.TensorAdd(dX_t, Engine.TensorBroadcastMultiply(deltaExpanded, dXFromState));

            // Accumulate A gradient BEFORE BPTT (dhReal is dL/dh[t+1] at this point).
            // dL/dA_bar = (dhReal * hReal_prev + dhImag * hImag_prev,
            //              dhImag * hReal_prev - dhReal * hImag_prev)
            var dAbarR = Engine.ReduceSum(Engine.TensorAdd(
                Engine.TensorMultiply(dhReal, hReal_prev),
                Engine.TensorMultiply(dhImag, hImag_prev)), new int[] { 0 });
            var dAbarI = Engine.ReduceSum(Engine.TensorAdd(
                Engine.TensorMultiply(dhImag, hReal_prev),
                Engine.TensorNegate(Engine.TensorMultiply(dhReal, hImag_prev))), new int[] { 0 });

            // Propagate gradient to previous hidden state (BPTT)
            // dh_prev = conj(A_bar) * dh (for complex: A_bar^* * dh)
            var newDhReal = Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(aBarReal3D, dhReal),
                Engine.TensorBroadcastMultiply(aBarImag3D, dhImag));
            var newDhImag = Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(aBarReal3D, dhImag),
                Engine.TensorNegate(Engine.TensorBroadcastMultiply(aBarImag3D, dhReal)));
            dhReal = newDhReal;
            dhImag = newDhImag;

            // A gradient: chain through discretization A_bar = exp(dt * A)
            // dL/dA = dL/dA_bar * dt * A_bar  (PyTorch reference)
            // For complex: dL/dA_real = Re(dt * A_bar * dL/dA_bar_complex)
            for (int d = 0; d < _innerDimension; d++)
            {
                double dt = NumOps.ToDouble(delta[d]);
                for (int n = 0; n < _stateDimension; n++)
                {
                    double ar = NumOps.ToDouble(_aReal[d, n]);
                    double ai = NumOps.ToDouble(_aImag[d, n]);
                    double expR = Math.Exp(dt * ar);
                    double cosA = Math.Cos(dt * ai);
                    double sinA = Math.Sin(dt * ai);

                    double dAbR = NumOps.ToDouble(dAbarR[d, n]);
                    double dAbI = NumOps.ToDouble(dAbarI[d, n]);

                    // dL/dA_real = Re(dt * A_bar * dL/dA_bar_complex)
                    //            = dt * (dAbR * abar_r - dAbI * abar_i)
                    double abar_r = expR * cosA;
                    double abar_i = expR * sinA;
                    double gradAr = dt * (dAbR * abar_r - dAbI * abar_i);
                    double gradAi = dt * (dAbR * abar_i + dAbI * abar_r);

                    // B_bar contribution: dL/dA += dL/dB_bar * dB_bar/dA
                    // B_bar = f(A) * B where f(A) = (exp(Δ*A) - 1) / A
                    // df/dA = [Δ*A_bar*A - (A_bar - 1)] / A²
                    // dB_bar/dA = df/dA * B
                    // dL/dA += Re(conj(dL/dB_bar) * dB_bar/dA)... but since we track real/imag separately:
                    // dL/dA_real = Re(dL_dBbar_complex * dBbar/dA_complex)
                    double dBbR = NumOps.ToDouble(dBr_t[d, n]);
                    double dBbI = NumOps.ToDouble(dBi_t[d, n]);
                    double brV = NumOps.ToDouble(_bReal[d, n]);
                    double biV = NumOps.ToDouble(_bImag[d, n]);

                    double aMagSq = ar * ar + ai * ai;
                    if (aMagSq > 1e-12)
                    {
                        // num = Δ * A_bar * A - (A_bar - 1)  (complex)
                        // A_bar * A: (abar_r + i*abar_i) * (ar + i*ai)
                        double abarA_r = abar_r * ar - abar_i * ai;
                        double abarA_i = abar_r * ai + abar_i * ar;
                        double num_r = dt * abarA_r - (abar_r - 1);
                        double num_i = dt * abarA_i - abar_i;

                        // A² = (ar² - ai²) + 2*ar*ai*i
                        double aSq_r = ar * ar - ai * ai;
                        double aSq_i = 2 * ar * ai;
                        double aSqMagSq = aSq_r * aSq_r + aSq_i * aSq_i;

                        // df/dA = num / A² (complex division)
                        double dfda_r = (num_r * aSq_r + num_i * aSq_i) / aSqMagSq;
                        double dfda_i = (num_i * aSq_r - num_r * aSq_i) / aSqMagSq;

                        // dBbar/dA = df/dA * B (complex multiply)
                        double dBbardA_r = dfda_r * brV - dfda_i * biV;
                        double dBbardA_i = dfda_r * biV + dfda_i * brV;

                        // dL/dA_real += dBbR * dBbardA_r + dBbI * dBbardA_i
                        gradAr += dBbR * dBbardA_r + dBbI * dBbardA_i;
                        // dL/dA_imag += dBbR * dBbardA_i_wrt_ai + dBbI * ...
                        // For A_imag: df/dA_imag = i * df/dA (since d(A)/d(ai) = i)
                        // So dBbar/dA_imag = i * dBbar/dA = (-dBbardA_i, dBbardA_r)
                        gradAi += dBbR * (-dBbardA_i) + dBbI * dBbardA_r;
                    }


                    _aRealGradient[d, n] = NumOps.Add(_aRealGradient[d, n], NumOps.FromDouble(gradAr));
                    _aImagGradient[d, n] = NumOps.Add(_aImagGradient[d, n], NumOps.FromDouble(gradAi));
                }
            }

            dX.SetSlice(1, t, dX_t);
        }

        return dX;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_aRealGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _aReal = Engine.TensorAdd(_aReal, Engine.TensorMultiplyScalar(_aRealGradient, negLR));
        _aImag = Engine.TensorAdd(_aImag, Engine.TensorMultiplyScalar(_aImagGradient!, negLR));
        _bReal = Engine.TensorAdd(_bReal, Engine.TensorMultiplyScalar(_bRealGradient!, negLR));
        _bImag = Engine.TensorAdd(_bImag, Engine.TensorMultiplyScalar(_bImagGradient!, negLR));
        _cReal = Engine.TensorAdd(_cReal, Engine.TensorMultiplyScalar(_cRealGradient!, negLR));
        _cImag = Engine.TensorAdd(_cImag, Engine.TensorMultiplyScalar(_cImagGradient!, negLR));
        _dParam = Engine.TensorAdd(_dParam, Engine.TensorMultiplyScalar(_dParamGradient!, negLR));
        _logDelta = Engine.TensorAdd(_logDelta, Engine.TensorMultiplyScalar(_logDeltaGradient!, negLR));
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights, Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient!, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias, Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var tensor in new[]
        {
            _aReal, _aImag, _bReal, _bImag, _cReal, _cImag, _dParam,
            _inputProjectionWeights, _inputProjectionBias,
            _outputProjectionWeights, _outputProjectionBias,
            _logDelta
        })
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = ParameterCount;
        if (parameters.Length != expectedParams)
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");

        int index = 0;
        foreach (var tensor in new[]
        {
            _aReal, _aImag, _bReal, _bImag, _cReal, _cImag, _dParam,
            _inputProjectionWeights, _inputProjectionBias,
            _outputProjectionWeights, _outputProjectionBias,
            _logDelta
        })
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
    }

    public override Vector<T> GetParameterGradients()
    {
        if (_aRealGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_aRealGradient!.ToArray()),
            new Vector<T>(_aImagGradient!.ToArray()),
            new Vector<T>(_bRealGradient!.ToArray()),
            new Vector<T>(_bImagGradient!.ToArray()),
            new Vector<T>(_cRealGradient!.ToArray()),
            new Vector<T>(_cImagGradient!.ToArray()),
            new Vector<T>(_dParamGradient!.ToArray()),
            _inputProjectionWeightsGradient != null ? new Vector<T>(_inputProjectionWeightsGradient.ToArray()) : new Vector<T>(_inputProjectionWeights.Length),
            _inputProjectionBiasGradient != null ? new Vector<T>(_inputProjectionBiasGradient.ToArray()) : new Vector<T>(_inputProjectionBias.Length),
            _outputProjectionWeightsGradient != null ? new Vector<T>(_outputProjectionWeightsGradient.ToArray()) : new Vector<T>(_outputProjectionWeights.Length),
            _outputProjectionBiasGradient != null ? new Vector<T>(_outputProjectionBiasGradient.ToArray()) : new Vector<T>(_outputProjectionBias.Length),
            new Vector<T>(_logDeltaGradient!.ToArray()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _aRealGradient = null; _aImagGradient = null; _bRealGradient = null; _bImagGradient = null; _cRealGradient = null; _cImagGradient = null; _dParamGradient = null; _logDeltaGradient = null;
        _inputProjectionWeightsGradient = null; _inputProjectionBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedInput = null;
        _lastHiddenStatesReal = null;
        _lastHiddenStatesImag = null;
        _lastScanOutputReal = null;
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _innerDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "x_t");

        var hPrevRealPlaceholder = new Tensor<T>(new int[] { 1, _innerDimension * _stateDimension });
        var hPrevRealNode = TensorOperations<T>.Variable(hPrevRealPlaceholder, "h_prev_real");

        var hPrevImagPlaceholder = new Tensor<T>(new int[] { 1, _innerDimension * _stateDimension });
        var hPrevImagNode = TensorOperations<T>.Variable(hPrevImagPlaceholder, "h_prev_imag");

        var dParamNode = TensorOperations<T>.Variable(_dParam, "D");
        var outProjWeightsNode = TensorOperations<T>.Variable(_outputProjectionWeights, "W_out");
        var outProjBiasNode = TensorOperations<T>.Variable(_outputProjectionBias, "b_out");

        inputNodes.Add(xNode);
        inputNodes.Add(hPrevRealNode);
        inputNodes.Add(hPrevImagNode);
        inputNodes.Add(dParamNode);
        inputNodes.Add(outProjWeightsNode);
        inputNodes.Add(outProjBiasNode);

        // Skip connection output: y = D * x
        var skipOutput = TensorOperations<T>.ElementwiseMultiply(xNode, dParamNode);

        // Output projection
        var outProjWeightsT = TensorOperations<T>.Transpose(outProjWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(skipOutput, outProjWeightsT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outProjBiasNode);

        return outputWithBias;
    }

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
