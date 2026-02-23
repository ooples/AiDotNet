using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the minLSTM layer from "Were RNNs All We Needed?" (Feng et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// minLSTM is a minimal LSTM variant that removes all hidden-state dependencies from the gates,
/// enabling fully parallelizable training via a parallel prefix scan. In a standard LSTM, the forget
/// gate f_t, input gate i_t, and output gate o_t all depend on the previous hidden state h_{t-1}.
/// minLSTM eliminates every one of these dependencies:
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <code>
///   Standard LSTM:  f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)   -- depends on h_{t-1}
///   minLSTM:        f_t = sigma(W_f * x_t + b_f)               -- input only!
///
///   1. Forget gate:      f_t = sigma(linear_f(x_t))
///   2. Input gate:       i_t = sigma(linear_i(x_t))
///   3. Normalize gates:  f'_t = f_t / (f_t + i_t),  i'_t = i_t / (f_t + i_t)
///      This ensures f'_t + i'_t = 1, making the cell update a convex combination.
///   4. Cell candidate:   c_tilde = linear_c(x_t)
///   5. Cell update:      c_t = f'_t * c_{t-1} + i'_t * c_tilde
///   6. Hidden state:     h_t = c_t    (no output gate, no tanh -- just the cell state)
/// </code>
/// </para>
/// <para>
/// <b>Gate Normalization:</b> The normalization f' = f/(f+i), i' = i/(f+i) is the key insight
/// that makes minLSTM mathematically equivalent to a linear recurrence. Because f' + i' = 1,
/// the cell update is a convex combination of the previous state and the new candidate. This
/// is directly analogous to minGRU's formulation where z and (1-z) play the same role.
/// This convex structure allows the recurrence to be computed via a parallel scan operation,
/// transforming training complexity from O(T) sequential to O(log T) parallel depth.
/// </para>
/// <para>
/// <b>Comparison with minGRU:</b>
/// <list type="bullet">
///   <item>minGRU uses a single gate z: h_t = (1-z)*h_{t-1} + z*tilde_h</item>
///   <item>minLSTM uses two gates (f, i) that are normalized: c_t = f'*c_{t-1} + i'*c_tilde</item>
///   <item>Both achieve input-only gating for parallel training</item>
///   <item>minLSTM's two separate gates provide more expressive control over the forget-vs-input
///   trade-off, whereas minGRU ties them together through a single z</item>
///   <item>Both outperform traditional LSTMs and GRUs on many sequence modeling benchmarks while
///   being significantly faster to train due to parallel scan</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> minLSTM is a drastically simplified LSTM that trains much faster.
///
/// Traditional LSTMs have a chicken-and-egg problem: to compute the gates at step t, you need
/// the hidden state from step t-1, which means you must process the sequence one step at a time.
/// minLSTM solves this by making gates depend only on the current input:
///
/// - Standard LSTM gates: "What should I remember?" depends on what I currently remember (h_{t-1})
/// - minLSTM gates: "What should I remember?" depends only on what I'm currently seeing (x_t)
///
/// The normalization trick (f' + i' = 1) makes the update a weighted average:
///   new_state = (weight_forget * old_state) + (weight_input * new_candidate)
///   where weight_forget + weight_input = 1
///
/// This simple change means the entire sequence can be processed in parallel during training,
/// just like a Transformer, while still maintaining efficient O(1)-per-step inference.
/// </para>
/// <para>
/// <b>Reference:</b> Feng et al., "Were RNNs All We Needed?", 2024.
/// https://arxiv.org/abs/2410.01201
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MinLSTMLayer<T> : LayerBase<T>
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _expandedDimension;

    // Input projection: [modelDim, expandedDim] -- projects input to internal dimension
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Forget gate projection: [expandedDim, expandedDim]
    // f_t = sigma(W_f * x_proj + b_f)
    private Tensor<T> _forgetGateWeights;
    private Tensor<T> _forgetGateBias;

    // Input gate projection: [expandedDim, expandedDim]
    // i_t = sigma(W_i * x_proj + b_i)
    private Tensor<T> _inputGateWeights;
    private Tensor<T> _inputGateBias;

    // Cell candidate projection: [expandedDim, expandedDim]
    // c_tilde = W_c * x_proj + b_c
    private Tensor<T> _cellCandidateWeights;
    private Tensor<T> _cellCandidateBias;

    // Output projection: [expandedDim, modelDim] -- projects back to model dimension
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values for backward
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedInput;
    private Tensor<T>? _lastForgetGateRaw;
    private Tensor<T>? _lastInputGateRaw;
    private Tensor<T>? _lastForgetGateSigmoid;
    private Tensor<T>? _lastInputGateSigmoid;
    private Tensor<T>? _lastForgetGateNorm;
    private Tensor<T>? _lastInputGateNorm;
    private Tensor<T>? _lastCellCandidate;
    private Tensor<T>? _lastCellStates;
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _forgetGateWeightsGradient;
    private Tensor<T>? _forgetGateBiasGradient;
    private Tensor<T>? _inputGateWeightsGradient;
    private Tensor<T>? _inputGateBiasGradient;
    private Tensor<T>? _cellCandidateWeightsGradient;
    private Tensor<T>? _cellCandidateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Gets the model dimension (input/output width).
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the expanded internal dimension.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The expansion factor controls how wide the internal hidden state is
    /// relative to the input/output. An expansion factor of 1 means the hidden state is the same width
    /// as the input. Larger values increase model capacity at the cost of more parameters.</para>
    /// </remarks>
    public int ExpandedDimension => _expandedDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _forgetGateWeights.Length + _forgetGateBias.Length +
        _inputGateWeights.Length + _inputGateBias.Length +
        _cellCandidateWeights.Length + _cellCandidateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new minLSTM layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// <para><b>For Beginners:</b> The maximum number of time steps or tokens in a single input sequence.</para>
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of the input and output vectors at each time step.</para>
    /// </param>
    /// <param name="expansionFactor">
    /// Expansion factor for the internal hidden dimension. Default: 1.
    /// <para><b>For Beginners:</b> Multiplies the model dimension to get the internal state width.
    /// A factor of 1 keeps the hidden state the same size as the input. A factor of 2 doubles it,
    /// which can increase the model's capacity to remember information but also doubles the
    /// parameter count for the gate and candidate projections.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public MinLSTMLayer(
        int sequenceLength,
        int modelDimension = 256,
        int expansionFactor = 1,
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
        if (expansionFactor <= 0)
            throw new ArgumentException($"Expansion factor ({expansionFactor}) must be positive.", nameof(expansionFactor));

        _modelDimension = modelDimension;
        _expandedDimension = modelDimension * expansionFactor;

        // Input projection: [modelDim, expandedDim]
        _inputProjectionWeights = new Tensor<T>([modelDimension, _expandedDimension]);
        _inputProjectionBias = new Tensor<T>([_expandedDimension]);

        // Forget gate: [expandedDim, expandedDim]
        _forgetGateWeights = new Tensor<T>([_expandedDimension, _expandedDimension]);
        _forgetGateBias = new Tensor<T>([_expandedDimension]);

        // Input gate: [expandedDim, expandedDim]
        _inputGateWeights = new Tensor<T>([_expandedDimension, _expandedDimension]);
        _inputGateBias = new Tensor<T>([_expandedDimension]);

        // Cell candidate: [expandedDim, expandedDim]
        _cellCandidateWeights = new Tensor<T>([_expandedDimension, _expandedDimension]);
        _cellCandidateBias = new Tensor<T>([_expandedDimension]);

        // Output projection: [expandedDim, modelDim]
        _outputProjectionWeights = new Tensor<T>([_expandedDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);

        InitializeTensor2D(_forgetGateWeights);
        // Bias forget gate slightly positive so sigmoid starts near 0.7 -> encourages remembering
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(1.0);

        InitializeTensor2D(_inputGateWeights);
        _inputGateBias.Fill(NumOps.Zero);

        InitializeTensor2D(_cellCandidateWeights);
        _cellCandidateBias.Fill(NumOps.Zero);

        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes a 2D tensor using Xavier/Glorot initialization.
    /// </summary>
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

        // Step 1: Input projection -- [batch*seq, modelDim] x [modelDim, expandedDim]
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var projected = Engine.TensorMatMul(inputFlat, _inputProjectionWeights);
        var projBias = _inputProjectionBias.Reshape(1, _expandedDimension);
        projected = Engine.TensorBroadcastAdd(projected, projBias);
        var projected3D = projected.Reshape(batchSize, seqLen, _expandedDimension);
        _lastProjectedInput = projected3D;

        // Step 2: Compute forget gate f_t = sigma(W_f * x_proj + b_f)
        var projFlat = projected3D.Reshape(batchSize * seqLen, _expandedDimension);
        var forgetRaw = Engine.TensorMatMul(projFlat, _forgetGateWeights);
        var fBias = _forgetGateBias.Reshape(1, _expandedDimension);
        forgetRaw = Engine.TensorBroadcastAdd(forgetRaw, fBias);
        var forgetRaw3D = forgetRaw.Reshape(batchSize, seqLen, _expandedDimension);
        var forgetSigmoid = Engine.Sigmoid(forgetRaw3D);
        _lastForgetGateRaw = forgetRaw3D;
        _lastForgetGateSigmoid = forgetSigmoid;

        // Step 3: Compute input gate i_t = sigma(W_i * x_proj + b_i)
        var inputGateRaw = Engine.TensorMatMul(projFlat, _inputGateWeights);
        var iBias = _inputGateBias.Reshape(1, _expandedDimension);
        inputGateRaw = Engine.TensorBroadcastAdd(inputGateRaw, iBias);
        var inputGateRaw3D = inputGateRaw.Reshape(batchSize, seqLen, _expandedDimension);
        var inputGateSigmoid = Engine.Sigmoid(inputGateRaw3D);
        _lastInputGateRaw = inputGateRaw3D;
        _lastInputGateSigmoid = inputGateSigmoid;

        // Step 4: Normalize gates -- f' = f/(f+i), i' = i/(f+i)
        var gateSum = Engine.TensorAdd(forgetSigmoid, inputGateSigmoid);
        // Add small epsilon for numerical stability to avoid division by zero
        var epsilon = new Tensor<T>(gateSum.Shape);
        epsilon.Fill(NumOps.FromDouble(1e-8));
        gateSum = Engine.TensorAdd(gateSum, epsilon);

        var forgetNorm = Engine.TensorDivide(forgetSigmoid, gateSum);
        var inputNorm = Engine.TensorDivide(inputGateSigmoid, gateSum);
        _lastForgetGateNorm = forgetNorm;
        _lastInputGateNorm = inputNorm;

        // Step 5: Cell candidate c_tilde = W_c * x_proj + b_c (no activation -- pure linear)
        var cellCandRaw = Engine.TensorMatMul(projFlat, _cellCandidateWeights);
        var cBias = _cellCandidateBias.Reshape(1, _expandedDimension);
        cellCandRaw = Engine.TensorBroadcastAdd(cellCandRaw, cBias);
        var cellCandidate = cellCandRaw.Reshape(batchSize, seqLen, _expandedDimension);
        _lastCellCandidate = cellCandidate;

        // Step 6: Gated recurrence -- c_t = f'_t * c_{t-1} + i'_t * c_tilde_t, h_t = c_t
        var recurrenceOutput = MinLSTMRecurrenceForward(
            forgetNorm, inputNorm, cellCandidate, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 7: Output projection -- [batch*seq, expandedDim] x [expandedDim, modelDim]
        var recFlat = recurrenceOutput.Reshape(batchSize * seqLen, _expandedDimension);
        var outputFlat = Engine.TensorMatMul(recFlat, _outputProjectionWeights);
        var outBias = _outputProjectionBias.Reshape(1, _modelDimension);
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias);
        var output3D = outputFlat.Reshape(batchSize, seqLen, _modelDimension);

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
    /// Implements the minLSTM recurrence: c_t = f'_t * c_{t-1} + i'_t * c_tilde_t.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Because f' + i' = 1, this is a convex combination. The cell state is a weighted average of
    /// the previous cell state and the new candidate, with weights controlled purely by the current
    /// input. The output is h_t = c_t (no output gate, no tanh).
    /// </para>
    /// <para>
    /// In a production implementation, this recurrence would be computed via a parallel prefix scan
    /// for O(log T) parallel depth. This sequential implementation is functionally equivalent and
    /// serves as a clear reference.
    /// </para>
    /// </remarks>
    private Tensor<T> MinLSTMRecurrenceForward(
        Tensor<T> forgetNorm, Tensor<T> inputNorm, Tensor<T> cellCandidate,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });
        // Cell state: [batch, expandedDim] -- initialized to zero
        var cellState = new Tensor<T>(new[] { batchSize, _expandedDimension });
        // Store all cell states for backward pass: [batch, seqLen+1, expandedDim]
        var allCellStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _expandedDimension });
        // Initial cell state (zeros) is already at t=0

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _expandedDimension; d++)
                {
                    T fPrime = forgetNorm[new[] { bi, t, d }];
                    T iPrime = inputNorm[new[] { bi, t, d }];
                    T cTilde = cellCandidate[new[] { bi, t, d }];
                    T cPrev = cellState[new[] { bi, d }];

                    // c_t = f'_t * c_{t-1} + i'_t * c_tilde_t
                    T cNew = NumOps.Add(
                        NumOps.Multiply(fPrime, cPrev),
                        NumOps.Multiply(iPrime, cTilde));

                    cellState[new[] { bi, d }] = cNew;
                    output[new[] { bi, t, d }] = cNew;
                }
            }

            // Save cell state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _expandedDimension; d++)
                    allCellStates[new[] { bi, t + 1, d }] = cellState[new[] { bi, d }];
        }

        _lastCellStates = allCellStates;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastProjectedInput == null ||
            _lastForgetGateRaw == null || _lastInputGateRaw == null ||
            _lastForgetGateSigmoid == null || _lastInputGateSigmoid == null ||
            _lastForgetGateNorm == null || _lastInputGateNorm == null ||
            _lastCellCandidate == null || _lastCellStates == null ||
            _lastRecurrenceOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // === Step 7 backward: Output projection ===
        // y = recOut * W_out + b_out
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var recOutFlat = _lastRecurrenceOutput.Reshape(batchSize * seqLen, _expandedDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(recOutFlat.Transpose([1, 0]), gradFlat);

        var dRecurrence = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _expandedDimension);

        // === Step 6 backward: Recurrence ===
        // c_t = f'_t * c_{t-1} + i'_t * c_tilde_t
        // h_t = c_t (output is directly the cell state)
        var dForgetNorm = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });
        var dInputNorm = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });
        var dCellCandidate = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });
        var dCellState = new Tensor<T>(new[] { batchSize, _expandedDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _expandedDimension; d++)
                {
                    // dL/dc_t from both the output gradient and the future cell state gradient
                    T dC = NumOps.Add(
                        dRecurrence[new[] { bi, t, d }],
                        dCellState[new[] { bi, d }]);

                    T fPrime = _lastForgetGateNorm[new[] { bi, t, d }];
                    T iPrime = _lastInputGateNorm[new[] { bi, t, d }];
                    T cTilde = _lastCellCandidate[new[] { bi, t, d }];
                    T cPrev = _lastCellStates[new[] { bi, t, d }]; // t=0 is initial state (zeros)

                    // dc/df' = c_{t-1}
                    dForgetNorm[new[] { bi, t, d }] = NumOps.Multiply(dC, cPrev);

                    // dc/di' = c_tilde
                    dInputNorm[new[] { bi, t, d }] = NumOps.Multiply(dC, cTilde);

                    // dc/dc_tilde = i'
                    dCellCandidate[new[] { bi, t, d }] = NumOps.Multiply(dC, iPrime);

                    // dc/dc_{t-1} = f' (propagate gradient to previous timestep)
                    dCellState[new[] { bi, d }] = NumOps.Multiply(dC, fPrime);
                }
            }
        }

        // === Steps 4-3 backward: Gate normalization -> sigmoid ===
        // f' = f / (f + i),  i' = i / (f + i)
        // Let s = f + i + eps
        // df'/df = i / s^2,  df'/di = -f / s^2
        // di'/df = -i / s^2, di'/di = f / s^2
        // Combined: dL/df_raw = dL/df * sig'(f_raw),  where dL/df = df'/df * dL/df' + di'/df * dL/di'
        var dForgetSigmoid = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });
        var dInputSigmoid = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int d = 0; d < _expandedDimension; d++)
                {
                    T f = _lastForgetGateSigmoid[new[] { bi, t, d }];
                    T inp = _lastInputGateSigmoid[new[] { bi, t, d }];
                    T s = NumOps.Add(NumOps.Add(f, inp), NumOps.FromDouble(1e-8));
                    T s2 = NumOps.Multiply(s, s);

                    T dFPrime = dForgetNorm[new[] { bi, t, d }];
                    T dIPrime = dInputNorm[new[] { bi, t, d }];

                    // dL/df = (i/s^2) * dL/df' + (-i/s^2) * dL/di'
                    //       = (i/s^2) * (dL/df' - dL/di')
                    T iOverS2 = NumOps.Divide(inp, s2);
                    T dF = NumOps.Multiply(iOverS2, NumOps.Subtract(dFPrime, dIPrime));

                    // dL/di = (-f/s^2) * dL/df' + (f/s^2) * dL/di'
                    //       = (f/s^2) * (dL/di' - dL/df')
                    T fOverS2 = NumOps.Divide(f, s2);
                    T dI = NumOps.Multiply(fOverS2, NumOps.Subtract(dIPrime, dFPrime));

                    dForgetSigmoid[new[] { bi, t, d }] = dF;
                    dInputSigmoid[new[] { bi, t, d }] = dI;
                }
            }
        }

        // Sigmoid derivative: sig'(x) = sig(x) * (1 - sig(x))
        var onesF = CreateOnesLike(_lastForgetGateSigmoid);
        var forgetSigDeriv = Engine.TensorMultiply(
            _lastForgetGateSigmoid,
            Engine.TensorSubtract(onesF, _lastForgetGateSigmoid));
        var dForgetRaw = Engine.TensorMultiply(dForgetSigmoid, forgetSigDeriv);

        var onesI = CreateOnesLike(_lastInputGateSigmoid);
        var inputSigDeriv = Engine.TensorMultiply(
            _lastInputGateSigmoid,
            Engine.TensorSubtract(onesI, _lastInputGateSigmoid));
        var dInputRaw = Engine.TensorMultiply(dInputSigmoid, inputSigDeriv);

        // === Step 5 backward: Cell candidate projection ===
        // cellCand = projFlat * W_c + b_c
        var projFlat = _lastProjectedInput.Reshape(batchSize * seqLen, _expandedDimension);
        var dCellCandFlat = dCellCandidate.Reshape(batchSize * seqLen, _expandedDimension);
        _cellCandidateWeightsGradient = Engine.TensorMatMul(projFlat.Transpose([1, 0]), dCellCandFlat);
        _cellCandidateBiasGradient = Engine.ReduceSum(dCellCandidate, new int[] { 0, 1 });

        // === Step 2-3 backward: Gate projections ===
        // forgetRaw = projFlat * W_f + b_f
        var dForgetRawFlat = dForgetRaw.Reshape(batchSize * seqLen, _expandedDimension);
        _forgetGateWeightsGradient = Engine.TensorMatMul(projFlat.Transpose([1, 0]), dForgetRawFlat);
        _forgetGateBiasGradient = Engine.ReduceSum(dForgetRaw, new int[] { 0, 1 });

        // inputGateRaw = projFlat * W_i + b_i
        var dInputRawFlat = dInputRaw.Reshape(batchSize * seqLen, _expandedDimension);
        _inputGateWeightsGradient = Engine.TensorMatMul(projFlat.Transpose([1, 0]), dInputRawFlat);
        _inputGateBiasGradient = Engine.ReduceSum(dInputRaw, new int[] { 0, 1 });

        // === Accumulate projected input gradient from all paths ===
        // Three paths feed back through projFlat: forget gate, input gate, cell candidate
        var dProjFromForget = Engine.TensorMatMul(dForgetRawFlat, _forgetGateWeights.Transpose([1, 0]));
        var dProjFromInput = Engine.TensorMatMul(dInputRawFlat, _inputGateWeights.Transpose([1, 0]));
        var dProjFromCell = Engine.TensorMatMul(dCellCandFlat, _cellCandidateWeights.Transpose([1, 0]));

        var dProjTotal = Engine.TensorAdd(dProjFromForget, dProjFromInput);
        dProjTotal = Engine.TensorAdd(dProjTotal, dProjFromCell);

        // === Step 1 backward: Input projection ===
        // projected = inputFlat * W_inp + b_inp
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        _inputProjectionWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dProjTotal);
        _inputProjectionBiasGradient = Engine.ReduceSum(
            dProjTotal.Reshape(batchSize, seqLen, _expandedDimension), new int[] { 0, 1 });

        var inputGradFlat = Engine.TensorMatMul(dProjTotal, _inputProjectionWeights.Transpose([1, 0]));
        var inputGrad3D = inputGradFlat.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return inputGrad3D.Reshape(_originalInputShape);

        return inputGrad3D;
    }

    /// <summary>
    /// Creates a tensor of ones with the same shape as the template tensor.
    /// </summary>
    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var ones = new Tensor<T>(template.Shape);
        for (int i = 0; i < ones.Length; i++) ones[i] = NumOps.One;
        return ones;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_inputProjectionWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights, Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias, Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _forgetGateWeights = Engine.TensorAdd(_forgetGateWeights, Engine.TensorMultiplyScalar(_forgetGateWeightsGradient!, negLR));
        _forgetGateBias = Engine.TensorAdd(_forgetGateBias, Engine.TensorMultiplyScalar(_forgetGateBiasGradient!, negLR));
        _inputGateWeights = Engine.TensorAdd(_inputGateWeights, Engine.TensorMultiplyScalar(_inputGateWeightsGradient!, negLR));
        _inputGateBias = Engine.TensorAdd(_inputGateBias, Engine.TensorMultiplyScalar(_inputGateBiasGradient!, negLR));
        _cellCandidateWeights = Engine.TensorAdd(_cellCandidateWeights, Engine.TensorMultiplyScalar(_cellCandidateWeightsGradient!, negLR));
        _cellCandidateBias = Engine.TensorAdd(_cellCandidateBias, Engine.TensorMultiplyScalar(_cellCandidateBiasGradient!, negLR));
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
        _inputProjectionWeights, _inputProjectionBias,
        _forgetGateWeights, _forgetGateBias,
        _inputGateWeights, _inputGateBias,
        _cellCandidateWeights, _cellCandidateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedInput = null;
        _lastForgetGateRaw = null;
        _lastInputGateRaw = null;
        _lastForgetGateSigmoid = null;
        _lastInputGateSigmoid = null;
        _lastForgetGateNorm = null;
        _lastInputGateNorm = null;
        _lastCellCandidate = null;
        _lastCellStates = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
        _cellCandidateWeightsGradient = null;
        _cellCandidateBiasGradient = null;
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
        var inProjWeightsNode = TensorOperations<T>.Variable(_inputProjectionWeights, "W_inproj");
        var outWeightsNode = TensorOperations<T>.Variable(_outputProjectionWeights, "W_out");
        var outBiasNode = TensorOperations<T>.Variable(_outputProjectionBias, "b_out");

        inputNodes.Add(xNode);
        inputNodes.Add(inProjWeightsNode);
        inputNodes.Add(outWeightsNode);
        inputNodes.Add(outBiasNode);

        // Simplified computation graph: input projection -> output projection
        var inProjT = TensorOperations<T>.Transpose(inProjWeightsNode);
        var projected = TensorOperations<T>.MatrixMultiply(xNode, inProjT);
        var outProjT = TensorOperations<T>.Transpose(outWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(projected, outProjT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outBiasNode);

        return outputWithBias;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["ExpandedDimension"] = _expandedDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the forget gate weights for external inspection.
    /// </summary>
    public Tensor<T> GetForgetGateWeights() => _forgetGateWeights;

    /// <summary>
    /// Gets the input gate weights for external inspection.
    /// </summary>
    public Tensor<T> GetInputGateWeights() => _inputGateWeights;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;
}
