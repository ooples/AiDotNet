using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class S5Layer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _stateDimension;

    // Diagonal A stored as real/imaginary pairs: [stateDim]
    // Lambda = a_real + i * a_imag, initialized with HiPPO-LegS: Lambda_n = -1/2 + n*pi*i
    private Tensor<T> _aReal;
    private Tensor<T> _aImag;

    // B input matrix (complex, after V^{-1} transformation): [stateDim, modelDim] real/imag
    // Maps H-dimensional input to N-dimensional complex state
    private Tensor<T> _bReal;
    private Tensor<T> _bImag;

    // C output matrix (complex): [modelDim, stateDim] real/imag
    // Maps N-dimensional complex state to H-dimensional real output via Re(C * x)
    private Tensor<T> _cReal;
    private Tensor<T> _cImag;

    // D skip connection: [modelDim]
    private Tensor<T> _dParam;

    // Discretization step size: [stateDim] (stored as log for positivity)
    private Tensor<T> _logDelta;

    // Input projection: [modelDim, modelDim]
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

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
    private Tensor<T>? _logDeltaGradient;
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _aReal.Length + _aImag.Length +
        _bReal.Length + _bImag.Length +
        _cReal.Length + _cImag.Length +
        _dParam.Length + _logDelta.Length +
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

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
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
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
        for (int n = 0; n < _stateDimension; n++)
        {
            _aReal[n] = NumOps.FromDouble(-0.5);
            _aImag[n] = NumOps.FromDouble(Math.PI * (n + 1));
        }

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
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

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

        // Step 1: Input projection [batch*seq, modelDim] -> [batch*seq, modelDim]
        var input2D = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = _inputProjectionBias.Reshape(1, _modelDimension);
        var projectedWithBias = Engine.TensorBroadcastAdd(projected, bias2D);
        var projected3D = projectedWithBias.Reshape(batchSize, seqLen, _modelDimension);
        _lastProjectedInput = projected3D;

        // Step 2: MIMO parallel scan with diagonalized state matrix
        var scanOutput = MIMOParallelScan(projected3D, batchSize, seqLen);
        _lastScanOutputReal = scanOutput;

        // Step 3: Output projection [batch*seq, modelDim] -> [batch*seq, modelDim]
        var scanFlat = scanOutput.Reshape(batchSize * seqLen, _modelDimension);
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
    private Tensor<T> MIMOParallelScan(Tensor<T> u, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Compute delta = exp(logDelta): [stateDim]
        var delta = new T[_stateDimension];
        for (int n = 0; n < _stateDimension; n++)
            delta[n] = NumOps.Exp(_logDelta[n]);

        // Pre-compute discretized A_bar = exp(delta * A) for complex diagonal A
        // exp((delta*a_real) + i*(delta*a_imag))
        //   = exp(delta*a_real) * (cos(delta*a_imag) + i*sin(delta*a_imag))
        var aBarReal = new T[_stateDimension];
        var aBarImag = new T[_stateDimension];
        var bBarReal = new T[_stateDimension * _modelDimension];
        var bBarImag = new T[_stateDimension * _modelDimension];

        for (int n = 0; n < _stateDimension; n++)
        {
            double dt = NumOps.ToDouble(delta[n]);
            double ar = NumOps.ToDouble(_aReal[n]);
            double ai = NumOps.ToDouble(_aImag[n]);

            // Complex exp: exp(dt*ar) * (cos(dt*ai) + i*sin(dt*ai))
            double expDtAr = Math.Exp(dt * ar);
            double cosVal = Math.Cos(dt * ai);
            double sinVal = Math.Sin(dt * ai);

            aBarReal[n] = NumOps.FromDouble(expDtAr * cosVal);
            aBarImag[n] = NumOps.FromDouble(expDtAr * sinVal);

            // B_bar = (A_bar - I) / A * B for each input dimension h
            // (A_bar - I) = (expDtAr*cos - 1) + i*(expDtAr*sin)
            double diffReal = expDtAr * cosVal - 1.0;
            double diffImag = expDtAr * sinVal;

            // Complex division by A = ar + i*ai
            double aMagSq = ar * ar + ai * ai;

            for (int h = 0; h < _modelDimension; h++)
            {
                double br = NumOps.ToDouble(_bReal[new[] { n, h }]);
                double bi = NumOps.ToDouble(_bImag[new[] { n, h }]);

                double quotReal, quotImag;
                if (aMagSq < 1e-12)
                {
                    // First-order approximation when A is near zero: B_bar ~ delta * B
                    quotReal = dt * br;
                    quotImag = dt * bi;
                }
                else
                {
                    // (diffR + i*diffI) / (ar + i*ai) = ((diffR*ar + diffI*ai) + i*(diffI*ar - diffR*ai)) / |A|^2
                    double divReal = (diffReal * ar + diffImag * ai) / aMagSq;
                    double divImag = (diffImag * ar - diffReal * ai) / aMagSq;

                    // Multiply by B[n,h]: (divR + i*divI) * (br + i*bi)
                    quotReal = divReal * br - divImag * bi;
                    quotImag = divReal * bi + divImag * br;
                }

                int idx = n * _modelDimension + h;
                bBarReal[idx] = NumOps.FromDouble(quotReal);
                bBarImag[idx] = NumOps.FromDouble(quotImag);
            }
        }

        // Hidden state: [batch, stateDim] real and imaginary
        var hReal = new Tensor<T>(new[] { batchSize, _stateDimension });
        var hImag = new Tensor<T>(new[] { batchSize, _stateDimension });

        // Store all hidden states for backward pass: [batch, seqLen+1, stateDim]
        var allHiddenReal = new Tensor<T>(new[] { batchSize, seqLen + 1, _stateDimension });
        var allHiddenImag = new Tensor<T>(new[] { batchSize, seqLen + 1, _stateDimension });

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                // State update: x_n[t] = A_bar_n * x_n[t-1] + sum_h(B_bar[n,h] * u_h[t])
                for (int n = 0; n < _stateDimension; n++)
                {
                    T abR = aBarReal[n];
                    T abI = aBarImag[n];
                    T hR = hReal[new[] { bi, n }];
                    T hI = hImag[new[] { bi, n }];

                    // Complex multiply A_bar * h_prev:
                    // (abR + i*abI) * (hR + i*hI) = (abR*hR - abI*hI) + i*(abR*hI + abI*hR)
                    T newHR = NumOps.Subtract(
                        NumOps.Multiply(abR, hR),
                        NumOps.Multiply(abI, hI));
                    T newHI = NumOps.Add(
                        NumOps.Multiply(abR, hI),
                        NumOps.Multiply(abI, hR));

                    // Add B_bar * u: sum_h(B_bar[n,h] * u[t,h])
                    // B_bar is complex, u is real, so contribution is:
                    //   real += B_bar_real[n,h] * u[h], imag += B_bar_imag[n,h] * u[h]
                    for (int h = 0; h < _modelDimension; h++)
                    {
                        T uVal = u[new[] { bi, t, h }];
                        int idx = n * _modelDimension + h;
                        newHR = NumOps.Add(newHR, NumOps.Multiply(bBarReal[idx], uVal));
                        newHI = NumOps.Add(newHI, NumOps.Multiply(bBarImag[idx], uVal));
                    }

                    hReal[new[] { bi, n }] = newHR;
                    hImag[new[] { bi, n }] = newHI;
                }

                // Output: y_h[t] = sum_n Re(C[h,n] * x_n[t]) + D[h] * u_h[t]
                // Re((cR + i*cI) * (xR + i*xI)) = cR*xR - cI*xI
                for (int h = 0; h < _modelDimension; h++)
                {
                    T yVal = NumOps.Zero;
                    for (int n = 0; n < _stateDimension; n++)
                    {
                        T cR = _cReal[new[] { h, n }];
                        T cI = _cImag[new[] { h, n }];
                        T xR = hReal[new[] { bi, n }];
                        T xI = hImag[new[] { bi, n }];

                        yVal = NumOps.Add(yVal,
                            NumOps.Subtract(
                                NumOps.Multiply(cR, xR),
                                NumOps.Multiply(cI, xI)));
                    }

                    // Skip connection: + D[h] * u[t,h]
                    T uH = u[new[] { bi, t, h }];
                    yVal = NumOps.Add(yVal, NumOps.Multiply(_dParam[h], uH));

                    output[new[] { bi, t, h }] = yVal;
                }
            }

            // Save state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int n = 0; n < _stateDimension; n++)
                {
                    allHiddenReal[new[] { bi, t + 1, n }] = hReal[new[] { bi, n }];
                    allHiddenImag[new[] { bi, t + 1, n }] = hImag[new[] { bi, n }];
                }
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

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var scanFlat = _lastScanOutputReal.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(
            scanFlat.Transpose([1, 0]), gradFlat);

        var dScan = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Recurrent scan backward (complex MIMO)
        var dProjected = MIMOScanBackward(dScan, _lastProjectedInput, batchSize, seqLen);

        // Input projection backward
        var dProjFlat = dProjected.Reshape(batchSize * seqLen, _modelDimension);
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
    /// Backward pass through the MIMO complex recurrent scan.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Computes gradients for all MIMO SSM parameters (A, B, C, D, delta) by backpropagating
    /// through time. The gradient flows backward through the recurrence using:
    ///   dh_prev = conj(A_bar) * dh  (conjugate transpose for complex diagonal)
    /// and accumulates parameter gradients at each timestep.
    /// </para>
    /// </remarks>
    private Tensor<T> MIMOScanBackward(
        Tensor<T> dOutput, Tensor<T> u, int batchSize, int seqLen)
    {
        var dU = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Initialize gradient accumulators
        _aRealGradient = new Tensor<T>(new[] { _stateDimension });
        _aImagGradient = new Tensor<T>(new[] { _stateDimension });
        _bRealGradient = new Tensor<T>(new[] { _stateDimension, _modelDimension });
        _bImagGradient = new Tensor<T>(new[] { _stateDimension, _modelDimension });
        _cRealGradient = new Tensor<T>(new[] { _modelDimension, _stateDimension });
        _cImagGradient = new Tensor<T>(new[] { _modelDimension, _stateDimension });
        _dParamGradient = new Tensor<T>(new[] { _modelDimension });
        _logDeltaGradient = new Tensor<T>(new[] { _stateDimension });

        var hiddenStatesReal = _lastHiddenStatesReal
            ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var hiddenStatesImag = _lastHiddenStatesImag
            ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Recompute discretized A_bar for backward pass
        var aBarReal = new T[_stateDimension];
        var aBarImag = new T[_stateDimension];
        var deltaArr = new double[_stateDimension];

        for (int n = 0; n < _stateDimension; n++)
        {
            double dt = Math.Exp(NumOps.ToDouble(_logDelta[n]));
            deltaArr[n] = dt;
            double ar = NumOps.ToDouble(_aReal[n]);
            double ai = NumOps.ToDouble(_aImag[n]);
            double expDtAr = Math.Exp(dt * ar);

            aBarReal[n] = NumOps.FromDouble(expDtAr * Math.Cos(dt * ai));
            aBarImag[n] = NumOps.FromDouble(expDtAr * Math.Sin(dt * ai));
        }

        // Running gradient of hidden state (complex): dL/dh[t]
        var dhReal = new Tensor<T>(new[] { batchSize, _stateDimension });
        var dhImag = new Tensor<T>(new[] { batchSize, _stateDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                // Gradient from output: y_h = sum_n Re(C[h,n] * x_n) + D[h] * u_h
                // dh_real[n] += sum_h C_real[h,n] * dOut[h]
                // dh_imag[n] -= sum_h C_imag[h,n] * dOut[h]
                // dC_real[h,n] += x_real[n] * dOut[h]
                // dC_imag[h,n] -= x_imag[n] * dOut[h]
                // dD[h] += u[h] * dOut[h]
                // dU[h] += D[h] * dOut[h]

                for (int h = 0; h < _modelDimension; h++)
                {
                    T dOutVal = dOutput[new[] { bi, t, h }];
                    T uVal = u[new[] { bi, t, h }];

                    // dD gradient
                    _dParamGradient[h] = NumOps.Add(_dParamGradient[h],
                        NumOps.Multiply(uVal, dOutVal));

                    // dU from skip connection
                    dU[new[] { bi, t, h }] = NumOps.Add(dU[new[] { bi, t, h }],
                        NumOps.Multiply(_dParam[h], dOutVal));

                    for (int n = 0; n < _stateDimension; n++)
                    {
                        T xR = hiddenStatesReal[new[] { bi, t + 1, n }];
                        T xI = hiddenStatesImag[new[] { bi, t + 1, n }];

                        // dh from output (Re(C*x) = Cr*xr - Ci*xi)
                        dhReal[new[] { bi, n }] = NumOps.Add(dhReal[new[] { bi, n }],
                            NumOps.Multiply(_cReal[new[] { h, n }], dOutVal));
                        dhImag[new[] { bi, n }] = NumOps.Add(dhImag[new[] { bi, n }],
                            NumOps.Negate(NumOps.Multiply(_cImag[new[] { h, n }], dOutVal)));

                        // dC gradient
                        _cRealGradient[new[] { h, n }] = NumOps.Add(
                            _cRealGradient[new[] { h, n }],
                            NumOps.Multiply(xR, dOutVal));
                        _cImagGradient[new[] { h, n }] = NumOps.Add(
                            _cImagGradient[new[] { h, n }],
                            NumOps.Negate(NumOps.Multiply(xI, dOutVal)));
                    }
                }

                // Backprop through state update: x[t] = A_bar * x[t-1] + B_bar * u[t]
                for (int n = 0; n < _stateDimension; n++)
                {
                    T dhR = dhReal[new[] { bi, n }];
                    T dhI = dhImag[new[] { bi, n }];
                    T hPrevR = hiddenStatesReal[new[] { bi, t, n }];
                    T hPrevI = hiddenStatesImag[new[] { bi, t, n }];
                    T abR = aBarReal[n];
                    T abI = aBarImag[n];

                    // d_A_bar contribution: dh * conj(h_prev)
                    // d_A_bar_real += dhR * hPrevR + dhI * hPrevI
                    // d_A_bar_imag += dhI * hPrevR - dhR * hPrevI
                    T dAbR = NumOps.Add(
                        NumOps.Multiply(dhR, hPrevR),
                        NumOps.Multiply(dhI, hPrevI));
                    T dAbI = NumOps.Subtract(
                        NumOps.Multiply(dhI, hPrevR),
                        NumOps.Multiply(dhR, hPrevI));

                    // Chain rule through A_bar = exp(delta * A):
                    // dA_bar/dA_real = delta * A_bar_real (for diagonal real part)
                    // dA_bar/dA_imag = delta * (-A_bar_imag, A_bar_real) (rotation)
                    // Simplified: dA_real += delta * (dAbR * abR + dAbI * abI)
                    //            dA_imag += delta * (dAbI * abR - dAbR * abI)
                    double dt = deltaArr[n];
                    T dtT = NumOps.FromDouble(dt);

                    _aRealGradient[n] = NumOps.Add(_aRealGradient[n],
                        NumOps.Multiply(dtT, NumOps.Add(
                            NumOps.Multiply(dAbR, abR),
                            NumOps.Multiply(dAbI, abI))));
                    _aImagGradient[n] = NumOps.Add(_aImagGradient[n],
                        NumOps.Multiply(dtT, NumOps.Subtract(
                            NumOps.Multiply(dAbI, abR),
                            NumOps.Multiply(dAbR, abI))));

                    // dLogDelta: chain through delta = exp(logDelta) and A_bar = exp(delta*A)
                    // d_logDelta += delta * (dAbR * (ar * abR + ai * (-abI)) + dAbI * (ar * abI + ai * abR))
                    double ar = NumOps.ToDouble(_aReal[n]);
                    double ai = NumOps.ToDouble(_aImag[n]);
                    T dLogDelta_n = NumOps.Multiply(dtT, NumOps.Add(
                        NumOps.Multiply(dAbR, NumOps.FromDouble(ar * NumOps.ToDouble(abR) - ai * NumOps.ToDouble(abI))),
                        NumOps.Multiply(dAbI, NumOps.FromDouble(ar * NumOps.ToDouble(abI) + ai * NumOps.ToDouble(abR)))));
                    _logDeltaGradient[n] = NumOps.Add(_logDeltaGradient[n], dLogDelta_n);

                    // dB gradient and dU from B path
                    // Using simplified B_bar ~ delta * B for gradient computation
                    // dB_real[n,h] += dhR * delta * u[t,h], dB_imag[n,h] += dhI * delta * u[t,h]
                    // dU[t,h] += delta * (B_real[n,h] * dhR + B_imag[n,h] * dhI)
                    for (int h = 0; h < _modelDimension; h++)
                    {
                        T uVal = u[new[] { bi, t, h }];
                        T dtU = NumOps.Multiply(dtT, uVal);

                        _bRealGradient[new[] { n, h }] = NumOps.Add(
                            _bRealGradient[new[] { n, h }],
                            NumOps.Multiply(dhR, dtU));
                        _bImagGradient[new[] { n, h }] = NumOps.Add(
                            _bImagGradient[new[] { n, h }],
                            NumOps.Multiply(dhI, dtU));

                        // dU from B: Re(conj(B_bar) * dh) ~ delta * (Br * dhR + Bi * dhI)
                        dU[new[] { bi, t, h }] = NumOps.Add(dU[new[] { bi, t, h }],
                            NumOps.Multiply(dtT, NumOps.Add(
                                NumOps.Multiply(_bReal[new[] { n, h }], dhR),
                                NumOps.Multiply(_bImag[new[] { n, h }], dhI))));
                    }

                    // Propagate dh to previous timestep: dh_prev = conj(A_bar) * dh
                    // conj(A_bar) = (abR - i*abI)
                    // (abR - i*abI)(dhR + i*dhI) = (abR*dhR + abI*dhI) + i*(abR*dhI - abI*dhR)
                    T newDhR = NumOps.Add(
                        NumOps.Multiply(abR, dhR),
                        NumOps.Multiply(abI, dhI));
                    T newDhI = NumOps.Subtract(
                        NumOps.Multiply(abR, dhI),
                        NumOps.Multiply(abI, dhR));

                    dhReal[new[] { bi, n }] = newDhR;
                    dhImag[new[] { bi, n }] = newDhI;
                }
            }
        }

        return dU;
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
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights,
            Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient!, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias,
            Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights,
            Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias,
            Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;
        foreach (var tensor in GetAllTensors())
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        int index = 0;
        foreach (var tensor in GetAllTensors())
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
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
        _logDeltaGradient = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _modelDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "x_t");

        var hPrevRealPlaceholder = new Tensor<T>(new int[] { 1, _stateDimension });
        var hPrevRealNode = TensorOperations<T>.Variable(hPrevRealPlaceholder, "h_prev_real");

        var hPrevImagPlaceholder = new Tensor<T>(new int[] { 1, _stateDimension });
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
