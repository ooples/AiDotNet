using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Linear Recurrent Unit (LRU) layer from Orvieto et al., 2023.
/// </summary>
/// <remarks>
/// <para>
/// The Linear Recurrent Unit is a simple linear recurrence with diagonal complex-valued state
/// transitions that achieves competitive performance with more complex SSMs on long-range tasks.
/// It is the core recurrence used in Google's Griffin/Hawk architectures.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Input projection: u = W_in * input
///   2. Compute stable diagonal eigenvalues:
///      lambda = exp(-exp(nu) + i * exp(theta))
///      where nu (log magnitude) and theta (phase) are learnable parameters.
///      The exp(-exp(nu)) form guarantees |lambda| &lt; 1 (eigenvalues inside the unit circle),
///      ensuring the recurrence is always stable.
///   3. Diagonal state recurrence (using real-valued pairs for complex arithmetic):
///      x_t = lambda * x_{t-1} + B * u_t
///      where lambda is diagonal (each state dimension is independent).
///   4. Output: y_t = Re(C * x_t) + D * u_t
///   5. Output projection: output = W_out * y
/// </code>
/// </para>
/// <para>
/// The key insight of LRU is that diagonal linear recurrences are surprisingly powerful when
/// properly parameterized. The exp(-exp(nu)) parameterization for eigenvalue magnitudes ensures:
/// - Stability: all eigenvalues are strictly inside the unit circle
/// - Expressivity: the model can learn eigenvalues very close to 1 (long memory) or close to 0 (short memory)
/// - Smooth gradients: the double-exponential parameterization avoids gradient issues at the boundary
/// </para>
/// <para>
/// Since the state matrix is diagonal, each state dimension evolves independently, enabling
/// O(n) parallel scan computation for the entire sequence. This is in contrast to full-matrix
/// recurrences which require O(n * d^2) per step.
/// </para>
/// <para>
/// Complex values are represented as pairs of real numbers throughout, since the generic type T
/// is real-valued. Each complex state dimension uses two real numbers (real part + imaginary part),
/// and complex multiplication is performed using the identity (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
/// </para>
/// <para><b>For Beginners:</b> LRU is one of the simplest state space models that actually works well.
///
/// Think of it like a set of independent "memory cells", each vibrating at its own frequency:
/// - Each cell has a complex number (think: a spinning arrow) that controls how it decays and oscillates
/// - At each time step, the cell's state is multiplied by its complex eigenvalue (shrinks and rotates)
///   and then the new input is added in
/// - The output combines all these spinning memories back into real values
///
/// The critical trick is how the eigenvalues are parameterized:
/// - exp(-exp(nu)) ensures the arrow ALWAYS shrinks (stable)
/// - exp(theta) controls the rotation speed (frequency)
/// - The model learns which frequencies and decay rates are useful for the task
///
/// This is like having a bank of tunable resonators that can remember patterns at different
/// time scales, from very short (fast decay) to very long (slow decay, eigenvalue near 1).
///
/// Despite its simplicity, LRU matches or beats much more complex architectures on many
/// long-range benchmarks, showing that the parameterization matters more than the complexity
/// of the recurrence.
/// </para>
/// <para>
/// <b>Reference:</b> Orvieto et al., "Resurrecting Recurrent Neural Networks for Long Sequences", 2023.
/// https://arxiv.org/abs/2303.06349
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LinearRecurrentUnitLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _stateDimension;

    // Eigenvalue parameterization: lambda = exp(-exp(nu) + i * exp(theta))
    // nu (log magnitude): [stateDimension]
    private Tensor<T> _nu;
    // theta (phase): [stateDimension]
    private Tensor<T> _theta;

    // B input projection (complex): [stateDimension] real and imaginary parts
    // Maps scalar input per dimension to complex state
    private Tensor<T> _bReal;
    private Tensor<T> _bImag;

    // C output projection (complex): [stateDimension] real and imaginary parts
    // Maps complex state back to scalar output per dimension
    private Tensor<T> _cReal;
    private Tensor<T> _cImag;

    // D skip connection: [modelDimension]
    private Tensor<T> _dParam;

    // Input projection: [modelDimension, modelDimension]
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Output projection: [modelDimension, modelDimension]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedInput;
    private Tensor<T>? _lastHiddenStatesReal;
    private Tensor<T>? _lastHiddenStatesImag;
    private Tensor<T>? _lastRecurrenceOutput;
    private Tensor<T>? _lastLambdaReal;
    private Tensor<T>? _lastLambdaImag;
    private Tensor<T>? _lastLambdaMag;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _nuGradient;
    private Tensor<T>? _thetaGradient;
    private Tensor<T>? _bRealGradient;
    private Tensor<T>? _bImagGradient;
    private Tensor<T>? _cRealGradient;
    private Tensor<T>? _cImagGradient;
    private Tensor<T>? _dParamGradient;
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Gets the model dimension (d_model) of this LRU layer.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the state dimension (N) controlling the number of independent complex recurrence channels.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each state dimension is an independent "memory oscillator" with its
    /// own learned frequency and decay rate. More state dimensions means the model can track more
    /// different temporal patterns simultaneously. The original LRU paper uses values between 64 and 512.
    /// </para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _nu.Length + _theta.Length +
        _bReal.Length + _bImag.Length +
        _cReal.Length + _cImag.Length +
        _dParam.Length +
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Linear Recurrent Unit (LRU) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of the representation at each sequence position.
    /// This determines the input and output size of the layer.</para>
    /// </param>
    /// <param name="stateDimension">
    /// SSM state dimension (N). Default: 256.
    /// <para><b>For Beginners:</b> The number of independent complex oscillators in the recurrence.
    /// Each oscillator learns its own frequency and decay rate. The LRU paper shows that using
    /// a state dimension equal to or larger than the model dimension works best, unlike Mamba
    /// which uses a much smaller N=16. This is because LRU relies entirely on the linear recurrence
    /// for expressivity (no gating or selection mechanisms).</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public LinearRecurrentUnitLayer(
        int sequenceLength,
        int modelDimension = 256,
        int stateDimension = 256,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (sequenceLength <= 0)
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (stateDimension <= 0)
            throw new ArgumentException($"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));

        _modelDimension = modelDimension;
        _stateDimension = stateDimension;

        // Eigenvalue parameters
        _nu = new Tensor<T>([stateDimension]);
        _theta = new Tensor<T>([stateDimension]);

        // B (complex): input-to-state mapping per state dimension
        _bReal = new Tensor<T>([modelDimension, stateDimension]);
        _bImag = new Tensor<T>([modelDimension, stateDimension]);

        // C (complex): state-to-output mapping per state dimension
        _cReal = new Tensor<T>([stateDimension, modelDimension]);
        _cImag = new Tensor<T>([stateDimension, modelDimension]);

        // D: skip connection
        _dParam = new Tensor<T>([modelDimension]);

        // Input/output projections
        _inputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _inputProjectionBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Initialize nu so that exp(-exp(nu)) gives magnitudes uniformly distributed
        // in [0.9, 0.999] (the useful range for long-range dependencies).
        // |lambda| = exp(-exp(nu)), so nu = log(-log(|lambda|)).
        // For |lambda| in [0.9, 0.999]: nu in [log(-log(0.999)), log(-log(0.9))]
        //   = [log(0.001001), log(0.10536)] ~ [-6.907, -2.251]
        for (int n = 0; n < _stateDimension; n++)
        {
            double mag = 0.9 + Random.NextDouble() * 0.099; // [0.9, 0.999]
            double nuVal = Math.Log(-Math.Log(mag));
            _nu[n] = NumOps.FromDouble(nuVal);
        }

        // Initialize theta uniformly in [0, 2*pi) to cover all phase angles
        for (int n = 0; n < _stateDimension; n++)
        {
            _theta[n] = NumOps.FromDouble(Math.Log(Random.NextDouble() * 2.0 * Math.PI + 0.001));
        }

        // B: Xavier initialization
        InitializeTensor2D(_bReal);
        InitializeTensor2D(_bImag);

        // C: Xavier initialization
        InitializeTensor2D(_cReal);
        InitializeTensor2D(_cImag);

        // D: ones (identity skip connection)
        _dParam.Fill(NumOps.One);

        // Input/output projections: Xavier initialization
        InitializeTensor2D(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
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

        // Step 2: Compute diagonal recurrence with complex eigenvalues
        var recurrenceOutput = DiagonalComplexRecurrence(projected3D, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 3: Output projection [batch*seq, modelDim] -> [batch*seq, modelDim]
        var recFlat = recurrenceOutput.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorMatMul(recFlat, _outputProjectionWeights);
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
    /// Performs the LRU diagonal recurrence with complex-valued state transitions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The recurrence is computed element-wise for each state dimension n:
    ///   x_n[t] = lambda_n * x_n[t-1] + B_n * u[t]
    ///   y[t] = sum_n Re(C_n * x_n[t]) + D * u[t]
    ///
    /// where lambda_n = exp(-exp(nu_n) + i*exp(theta_n)) is the n-th diagonal eigenvalue.
    ///
    /// Complex arithmetic is performed using real pairs:
    ///   lambda_real = exp(-exp(nu)) * cos(exp(theta))
    ///   lambda_imag = exp(-exp(nu)) * sin(exp(theta))
    /// </para>
    /// </remarks>
    private Tensor<T> DiagonalComplexRecurrence(Tensor<T> u, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Compute lambda = exp(-exp(nu) + i*exp(theta))
        // |lambda| = exp(-exp(nu)), angle = exp(theta)
        // lambda_real = |lambda| * cos(angle)
        // lambda_imag = |lambda| * sin(angle)
        var lambdaMag = new Tensor<T>(new[] { _stateDimension });
        var lambdaReal = new Tensor<T>(new[] { _stateDimension });
        var lambdaImag = new Tensor<T>(new[] { _stateDimension });

        for (int n = 0; n < _stateDimension; n++)
        {
            double nuVal = NumOps.ToDouble(_nu[n]);
            double thetaVal = NumOps.ToDouble(_theta[n]);

            double mag = Math.Exp(-Math.Exp(nuVal));
            double angle = Math.Exp(thetaVal);

            lambdaMag[n] = NumOps.FromDouble(mag);
            lambdaReal[n] = NumOps.FromDouble(mag * Math.Cos(angle));
            lambdaImag[n] = NumOps.FromDouble(mag * Math.Sin(angle));
        }

        _lastLambdaMag = lambdaMag;
        _lastLambdaReal = lambdaReal;
        _lastLambdaImag = lambdaImag;

        // Hidden state: [batch, stateDimension] real and imaginary parts
        var hReal = new Tensor<T>(new[] { batchSize, _stateDimension });
        var hImag = new Tensor<T>(new[] { batchSize, _stateDimension });

        // Store hidden states for backward pass: [batch, seqLen+1, stateDimension]
        var allHiddenReal = new Tensor<T>(new[] { batchSize, seqLen + 1, _stateDimension });
        var allHiddenImag = new Tensor<T>(new[] { batchSize, seqLen + 1, _stateDimension });

        // D for skip connection: [1, modelDimension]
        var D2D = _dParam.Reshape(1, _modelDimension);

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                // For each state dimension n:
                //   x_n[t] = lambda_n * x_n[t-1] + sum_d(B[d,n] * u[t,d])
                for (int n = 0; n < _stateDimension; n++)
                {
                    T lR = lambdaReal[n];
                    T lI = lambdaImag[n];

                    // Complex multiply: lambda * h_prev
                    // (lR + i*lI) * (hR + i*hI) = (lR*hR - lI*hI) + i*(lR*hI + lI*hR)
                    T hR = hReal[new[] { bi, n }];
                    T hI = hImag[new[] { bi, n }];

                    T newHR = NumOps.Subtract(
                        NumOps.Multiply(lR, hR),
                        NumOps.Multiply(lI, hI));
                    T newHI = NumOps.Add(
                        NumOps.Multiply(lR, hI),
                        NumOps.Multiply(lI, hR));

                    // Add B * u[t]: sum over model dimensions
                    // B is complex: (bReal[d,n] + i*bImag[d,n]) * u[t,d] (u is real)
                    T bContribR = NumOps.Zero;
                    T bContribI = NumOps.Zero;
                    for (int d = 0; d < _modelDimension; d++)
                    {
                        T uVal = u[new[] { bi, t, d }];
                        bContribR = NumOps.Add(bContribR,
                            NumOps.Multiply(_bReal[new[] { d, n }], uVal));
                        bContribI = NumOps.Add(bContribI,
                            NumOps.Multiply(_bImag[new[] { d, n }], uVal));
                    }

                    newHR = NumOps.Add(newHR, bContribR);
                    newHI = NumOps.Add(newHI, bContribI);

                    hReal[new[] { bi, n }] = newHR;
                    hImag[new[] { bi, n }] = newHI;
                }

                // Compute output: y[t,d] = sum_n Re(C[n,d] * x_n[t]) + D[d] * u[t,d]
                // Re((cR + i*cI) * (hR + i*hI)) = cR*hR - cI*hI
                for (int d = 0; d < _modelDimension; d++)
                {
                    T yVal = NumOps.Zero;
                    for (int n = 0; n < _stateDimension; n++)
                    {
                        T cR = _cReal[new[] { n, d }];
                        T cI = _cImag[new[] { n, d }];
                        T xR = hReal[new[] { bi, n }];
                        T xI = hImag[new[] { bi, n }];

                        // Re(C * x) = cR * xR - cI * xI
                        yVal = NumOps.Add(yVal,
                            NumOps.Subtract(
                                NumOps.Multiply(cR, xR),
                                NumOps.Multiply(cI, xI)));
                    }

                    // Skip connection: + D[d] * u[t,d]
                    T uVal = u[new[] { bi, t, d }];
                    yVal = NumOps.Add(yVal, NumOps.Multiply(_dParam[d], uVal));

                    output[new[] { bi, t, d }] = yVal;
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
        if (_lastInput == null || _lastOutput == null || _lastProjectedInput == null ||
            _lastRecurrenceOutput == null || _lastHiddenStatesReal == null ||
            _lastHiddenStatesImag == null || _lastLambdaReal == null ||
            _lastLambdaImag == null || _lastLambdaMag == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var hiddenStatesReal = _lastHiddenStatesReal;
        var hiddenStatesImag = _lastHiddenStatesImag;
        var lambdaMag = _lastLambdaMag;

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var recFlat = _lastRecurrenceOutput.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(recFlat.Transpose([1, 0]), gradFlat);

        var dRecurrence = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Recurrence backward
        var dProjected = DiagonalComplexRecurrenceBackward(
            dRecurrence, _lastProjectedInput, batchSize, seqLen);

        // Input projection backward
        var dProjFlat = dProjected.Reshape(batchSize * seqLen, _modelDimension);
        _inputProjectionBiasGradient = Engine.ReduceSum(dProjected, new int[] { 0, 1 });

        var input2D = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        _inputProjectionWeightsGradient = Engine.TensorMatMul(input2D.Transpose([1, 0]), dProjFlat);

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
    /// Backward pass through the diagonal complex recurrence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Computes gradients for all recurrence parameters (nu, theta, B, C, D) by
    /// backpropagating through time. The gradient of lambda with respect to nu and theta
    /// requires the chain rule through the double-exponential parameterization:
    ///   d_lambda/d_nu = lambda * (-exp(nu))     (magnitude gradient)
    ///   d_lambda/d_theta = i * lambda * exp(theta) (phase gradient)
    /// </para>
    /// </remarks>
    private Tensor<T> DiagonalComplexRecurrenceBackward(
        Tensor<T> dOutput, Tensor<T> u, int batchSize, int seqLen)
    {
        var dU = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Initialize gradient accumulators
        _nuGradient = new Tensor<T>(new[] { _stateDimension });
        _thetaGradient = new Tensor<T>(new[] { _stateDimension });
        _bRealGradient = new Tensor<T>(new[] { _modelDimension, _stateDimension });
        _bImagGradient = new Tensor<T>(new[] { _modelDimension, _stateDimension });
        _cRealGradient = new Tensor<T>(new[] { _stateDimension, _modelDimension });
        _cImagGradient = new Tensor<T>(new[] { _stateDimension, _modelDimension });
        _dParamGradient = new Tensor<T>(new[] { _modelDimension });

        var lambdaReal = _lastLambdaReal
            ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lambdaImag = _lastLambdaImag
            ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var hiddenStatesReal = _lastHiddenStatesReal
            ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var hiddenStatesImag = _lastHiddenStatesImag
            ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Running gradient of hidden state (complex): dL/dh[t]
        var dhReal = new Tensor<T>(new[] { batchSize, _stateDimension });
        var dhImag = new Tensor<T>(new[] { batchSize, _stateDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            // Gradient from output: y[t,d] = sum_n Re(C[n,d] * x[t,n]) + D[d] * u[t,d]
            // dh_real[n] += sum_d C_real[n,d] * dOut[t,d]
            // dh_imag[n] -= sum_d C_imag[n,d] * dOut[t,d]  (from Re(C*h) = Cr*hr - Ci*hi)
            // dC_real[n,d] += h_real[t,n] * dOut[t,d]
            // dC_imag[n,d] -= h_imag[t,n] * dOut[t,d]
            // dD[d] += u[t,d] * dOut[t,d]
            // dU[t,d] += D[d] * dOut[t,d]

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T dOutVal = dOutput[new[] { bi, t, d }];
                    T uVal = u[new[] { bi, t, d }];

                    // dD gradient
                    _dParamGradient[d] = NumOps.Add(_dParamGradient[d],
                        NumOps.Multiply(uVal, dOutVal));

                    // dU from skip connection
                    dU[new[] { bi, t, d }] = NumOps.Add(dU[new[] { bi, t, d }],
                        NumOps.Multiply(_dParam[d], dOutVal));

                    for (int n = 0; n < _stateDimension; n++)
                    {
                        T hR = hiddenStatesReal[new[] { bi, t + 1, n }];
                        T hI = hiddenStatesImag[new[] { bi, t + 1, n }];

                        // dh from output
                        dhReal[new[] { bi, n }] = NumOps.Add(dhReal[new[] { bi, n }],
                            NumOps.Multiply(_cReal[new[] { n, d }], dOutVal));
                        dhImag[new[] { bi, n }] = NumOps.Add(dhImag[new[] { bi, n }],
                            NumOps.Negate(NumOps.Multiply(_cImag[new[] { n, d }], dOutVal)));

                        // dC gradient
                        _cRealGradient[new[] { n, d }] = NumOps.Add(
                            _cRealGradient[new[] { n, d }],
                            NumOps.Multiply(hR, dOutVal));
                        _cImagGradient[new[] { n, d }] = NumOps.Add(
                            _cImagGradient[new[] { n, d }],
                            NumOps.Negate(NumOps.Multiply(hI, dOutVal)));
                    }
                }

                // Backprop through state update: h[t] = lambda * h[t-1] + B * u[t]
                // dh_prev = conj(lambda) * dh  (conjugate transpose for complex)
                // d_lambda += dh * conj(h_prev)
                // dB += dh * u[t]  (complex B, real u)
                // dU[t] += Re(conj(B) * dh)  (chain rule back to real input)

                for (int n = 0; n < _stateDimension; n++)
                {
                    T dhR = dhReal[new[] { bi, n }];
                    T dhI = dhImag[new[] { bi, n }];
                    T hPrevR = hiddenStatesReal[new[] { bi, t, n }];
                    T hPrevI = hiddenStatesImag[new[] { bi, t, n }];
                    T lR = lambdaReal[n];
                    T lI = lambdaImag[n];

                    // d_lambda_real += dhR * hPrevR + dhI * hPrevI
                    // d_lambda_imag += dhI * hPrevR - dhR * hPrevI
                    T dLambdaR = NumOps.Add(
                        NumOps.Multiply(dhR, hPrevR),
                        NumOps.Multiply(dhI, hPrevI));
                    T dLambdaI = NumOps.Subtract(
                        NumOps.Multiply(dhI, hPrevR),
                        NumOps.Multiply(dhR, hPrevI));

                    // Chain rule: lambda = mag * (cos(angle) + i*sin(angle))
                    // where mag = exp(-exp(nu)), angle = exp(theta)
                    //
                    // d_nu: d_loss/d_nu = Re(d_loss/d_lambda * d_lambda/d_nu)
                    //   d_lambda/d_nu = lambda * (-exp(nu))
                    //   Re(dLambda^* * lambda * (-exp(nu)))
                    //   = -exp(nu) * Re(dLambda^* * lambda)
                    //   = -exp(nu) * (dLambdaR * lR + dLambdaI * lI)
                    double nuVal = NumOps.ToDouble(_nu[n]);
                    double expNu = Math.Exp(nuVal);
                    T nuChain = NumOps.Multiply(
                        NumOps.FromDouble(-expNu),
                        NumOps.Add(
                            NumOps.Multiply(dLambdaR, lR),
                            NumOps.Multiply(dLambdaI, lI)));
                    _nuGradient[n] = NumOps.Add(_nuGradient[n], nuChain);

                    // d_theta: d_loss/d_theta = Re(d_loss/d_lambda * d_lambda/d_theta)
                    //   d_lambda/d_theta = i * lambda * exp(theta)
                    //   Re(dLambda^* * i * lambda * exp(theta))
                    //   = exp(theta) * Re(dLambda^* * i * lambda)
                    //   = exp(theta) * (dLambdaR * (-lI) + dLambdaI * lR)
                    double thetaVal = NumOps.ToDouble(_theta[n]);
                    double expTheta = Math.Exp(thetaVal);
                    T thetaChain = NumOps.Multiply(
                        NumOps.FromDouble(expTheta),
                        NumOps.Add(
                            NumOps.Multiply(dLambdaR, NumOps.Negate(lI)),
                            NumOps.Multiply(dLambdaI, lR)));
                    _thetaGradient[n] = NumOps.Add(_thetaGradient[n], thetaChain);

                    // dB gradient and dU from B path
                    // h[t] += B * u, where B is complex and u is real
                    // dB_real[d,n] += dhR * u[t,d], dB_imag[d,n] += dhI * u[t,d]
                    // dU[t,d] += B_real[d,n] * dhR + B_imag[d,n] * dhI
                    for (int d = 0; d < _modelDimension; d++)
                    {
                        T uVal = u[new[] { bi, t, d }];

                        _bRealGradient[new[] { d, n }] = NumOps.Add(
                            _bRealGradient[new[] { d, n }],
                            NumOps.Multiply(dhR, uVal));
                        _bImagGradient[new[] { d, n }] = NumOps.Add(
                            _bImagGradient[new[] { d, n }],
                            NumOps.Multiply(dhI, uVal));

                        // dU from B: Re(conj(B) * dh) = B_real * dhR + B_imag * dhI
                        dU[new[] { bi, t, d }] = NumOps.Add(dU[new[] { bi, t, d }],
                            NumOps.Add(
                                NumOps.Multiply(_bReal[new[] { d, n }], dhR),
                                NumOps.Multiply(_bImag[new[] { d, n }], dhI)));
                    }

                    // Propagate dh to previous timestep through lambda multiplication
                    // dh_prev = conj(lambda) * dh
                    // conj(lambda) = (lR - i*lI)
                    // (lR - i*lI)(dhR + i*dhI) = (lR*dhR + lI*dhI) + i*(lR*dhI - lI*dhR)
                    T newDhR = NumOps.Add(
                        NumOps.Multiply(lR, dhR),
                        NumOps.Multiply(lI, dhI));
                    T newDhI = NumOps.Subtract(
                        NumOps.Multiply(lR, dhI),
                        NumOps.Multiply(lI, dhR));

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
        if (_nuGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _nu = Engine.TensorAdd(_nu, Engine.TensorMultiplyScalar(_nuGradient, negLR));
        _theta = Engine.TensorAdd(_theta, Engine.TensorMultiplyScalar(_thetaGradient!, negLR));
        _bReal = Engine.TensorAdd(_bReal, Engine.TensorMultiplyScalar(_bRealGradient!, negLR));
        _bImag = Engine.TensorAdd(_bImag, Engine.TensorMultiplyScalar(_bImagGradient!, negLR));
        _cReal = Engine.TensorAdd(_cReal, Engine.TensorMultiplyScalar(_cRealGradient!, negLR));
        _cImag = Engine.TensorAdd(_cImag, Engine.TensorMultiplyScalar(_cImagGradient!, negLR));
        _dParam = Engine.TensorAdd(_dParam, Engine.TensorMultiplyScalar(_dParamGradient!, negLR));
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights, Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient!, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias, Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
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
        _nu, _theta,
        _bReal, _bImag,
        _cReal, _cImag,
        _dParam,
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
        _lastRecurrenceOutput = null;
        _lastLambdaReal = null;
        _lastLambdaImag = null;
        _lastLambdaMag = null;
        _originalInputShape = null;
        _nuGradient = null;
        _thetaGradient = null;
        _bRealGradient = null;
        _bImagGradient = null;
        _cRealGradient = null;
        _cImagGradient = null;
        _dParamGradient = null;
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
        var outWeightsNode = TensorOperations<T>.Variable(_outputProjectionWeights, "W_out");
        var outBiasNode = TensorOperations<T>.Variable(_outputProjectionBias, "b_out");

        inputNodes.Add(xNode);
        inputNodes.Add(outWeightsNode);
        inputNodes.Add(outBiasNode);

        var outT = TensorOperations<T>.Transpose(outWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(xNode, outT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outBiasNode);

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
    /// Gets the nu (log magnitude) parameter for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The nu parameter controls the magnitude of each eigenvalue via |lambda_n| = exp(-exp(nu_n)).
    /// Smaller nu values mean eigenvalues closer to the unit circle (longer memory).
    /// </para>
    /// </remarks>
    public Tensor<T> GetNuParameter() => _nu;

    /// <summary>
    /// Gets the theta (phase) parameter for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The theta parameter controls the oscillation frequency of each eigenvalue via angle = exp(theta_n).
    /// Different theta values allow different state dimensions to resonate at different frequencies.
    /// </para>
    /// </remarks>
    public Tensor<T> GetThetaParameter() => _theta;

    /// <summary>
    /// Gets the D skip connection parameter for external inspection.
    /// </summary>
    public Tensor<T> GetDParameter() => _dParam;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;
}
