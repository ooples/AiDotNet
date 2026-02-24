using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the minGRU layer from "Were RNNs All We Needed?" (Feng et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// minGRU is a minimal variant of the Gated Recurrent Unit that removes the hidden-state dependency
/// from the gate computation. This seemingly small change has a profound consequence: the recurrence
/// becomes a <b>linear recurrence</b> in log-space, enabling efficient parallel training via prefix
/// sum (parallel scan) algorithms that run in O(log n) sequential steps instead of O(n).
/// </para>
/// <para>
/// The standard GRU equations:
/// <code>
///   z_t = sigma(W_z * x_t + U_z * h_{t-1} + b_z)          -- gate depends on BOTH input AND hidden state
///   r_t = sigma(W_r * x_t + U_r * h_{t-1} + b_r)          -- reset gate (also depends on hidden state)
///   h_tilde = tanh(W_h * x_t + U_h * (r_t . h_{t-1}) + b_h)
///   h_t = (1 - z_t) . h_{t-1} + z_t . h_tilde
/// </code>
/// The minGRU simplification:
/// <code>
///   z_t = sigma(W_z * x_t + b_z)                            -- gate depends ONLY on input
///   h_tilde_t = W_h * x_t + b_h                             -- candidate depends ONLY on input
///   h_t = (1 - z_t) . h_{t-1} + z_t . h_tilde_t            -- simple gated interpolation
/// </code>
/// </para>
/// <para>
/// <b>Why this enables parallel training:</b> Because z_t and h_tilde_t depend only on x_t (not h_{t-1}),
/// they can be precomputed for all timesteps in parallel. The recurrence h_t = (1-z_t)*h_{t-1} + z_t*h_tilde_t
/// is then a linear first-order recurrence of the form h_t = a_t*h_{t-1} + b_t where a_t = (1-z_t) and
/// b_t = z_t*h_tilde_t are known constants. Linear recurrences can be solved with parallel prefix sum
/// (also called parallel scan) in O(log n) parallel time, compared to O(n) for sequential RNNs.
/// </para>
/// <para>
/// In log-space, the recurrence becomes numerically stable:
/// <code>
///   log(h_t) = log((1-z_t) * h_{t-1} + z_t * h_tilde_t)
/// </code>
/// which can be computed via the log-sum-exp trick in a parallel scan.
/// </para>
/// <para>
/// This implementation uses the sequential recurrence for correctness and clarity. For production
/// training on GPU, the parallel scan formulation would be used for O(log n) wall-clock time.
/// </para>
/// <para><b>For Beginners:</b> minGRU is a stripped-down version of the GRU that is much simpler yet
/// surprisingly powerful.
///
/// Imagine you are writing notes while listening to a lecture:
/// - At each moment, you decide how much of your old notes to keep vs. how much new content to write down.
/// - In a standard GRU, your decision depends on both the new content AND everything you have written so far.
///   This creates a chain of dependencies: step 1 must finish before step 2 can start.
/// - In minGRU, your decision depends ONLY on the new content. You can look at every slide in the lecture
///   in parallel, decide what is important, and then combine everything in one fast sweep.
///
/// This is like the difference between:
/// - Reading a book one page at a time, deciding what to remember based on what you read so far (slow, sequential)
/// - Skimming all pages at once, marking important parts, then combining them in a single organized pass (fast, parallel)
///
/// Despite being simpler, minGRU matches standard GRU and LSTM performance on most benchmarks,
/// showing that the hidden-state dependency in the gate was often unnecessary.
/// </para>
/// <para>
/// <b>Reference:</b> Feng et al., "Were RNNs All We Needed?", 2024.
/// https://arxiv.org/abs/2410.01201
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MinGRULayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _expandedDimension;
    private readonly int _expansionFactor;

    // Input projection: [modelDim, expandedDim] - projects input to expanded space
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Gate projection: z_t = sigma(W_z * x_t + b_z), [expandedDim, expandedDim]
    private Tensor<T> _gateWeights;
    private Tensor<T> _gateBias;

    // Candidate projection: h_tilde_t = W_h * x_t + b_h, [expandedDim, expandedDim]
    private Tensor<T> _candidateWeights;
    private Tensor<T> _candidateBias;

    // Output projection: [expandedDim, modelDim] - projects back to model dimension
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedInput;
    private Tensor<T>? _lastGatePreAct;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastCandidate;
    private Tensor<T>? _lastHiddenStates;
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _gateWeightsGradient;
    private Tensor<T>? _gateBiasGradient;
    private Tensor<T>? _candidateWeightsGradient;
    private Tensor<T>? _candidateBiasGradient;
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
    /// Gets the expanded internal dimension used for the recurrence.
    /// </summary>
    public int ExpandedDimension => _expandedDimension;

    /// <summary>
    /// Gets the expansion factor applied to the model dimension for the internal recurrence.
    /// </summary>
    public int ExpansionFactor => _expansionFactor;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _gateWeights.Length + _gateBias.Length +
        _candidateWeights.Length + _candidateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new minGRU layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length that this layer will process.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model), the width of the input and output tensors. Default: 256.
    /// <para><b>For Beginners:</b> This is how many features each position in the sequence has.
    /// Larger values let the model represent more information but use more memory.</para>
    /// </param>
    /// <param name="expansionFactor">
    /// Factor by which the internal recurrence dimension exceeds the model dimension. Default: 1.
    /// <para><b>For Beginners:</b> Setting this to 2 would double the internal width, giving the
    /// recurrence more capacity to remember information, at the cost of more parameters.</para>
    /// </param>
    /// <param name="activationFunction">
    /// Optional activation function applied to the final output. Defaults to identity (no activation).
    /// </param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public MinGRULayer(
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
        _expansionFactor = expansionFactor;
        _expandedDimension = modelDimension * expansionFactor;

        // Input projection: map from modelDim to expandedDim
        _inputProjectionWeights = new Tensor<T>([modelDimension, _expandedDimension]);
        _inputProjectionBias = new Tensor<T>([_expandedDimension]);

        // Gate projection: z_t = sigma(W_z * proj_t + b_z)
        _gateWeights = new Tensor<T>([_expandedDimension, _expandedDimension]);
        _gateBias = new Tensor<T>([_expandedDimension]);

        // Candidate projection: h_tilde_t = W_h * proj_t + b_h
        _candidateWeights = new Tensor<T>([_expandedDimension, _expandedDimension]);
        _candidateBias = new Tensor<T>([_expandedDimension]);

        // Output projection: map from expandedDim back to modelDim
        _outputProjectionWeights = new Tensor<T>([_expandedDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);

        InitializeTensor2D(_gateWeights);
        // Initialize gate bias slightly positive so sigmoid starts near 0.5-0.6,
        // encouraging the model to incorporate new information early in training.
        _gateBias.Fill(NumOps.FromDouble(0.5));

        InitializeTensor2D(_candidateWeights);
        _candidateBias.Fill(NumOps.Zero);

        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Xavier/Glorot uniform initialization for 2D weight tensors.
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

        // Step 1: Input projection from modelDim to expandedDim
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var projected = Engine.TensorMatMul(inputFlat, _inputProjectionWeights);
        var projBias = _inputProjectionBias.Reshape(1, _expandedDimension);
        projected = Engine.TensorBroadcastAdd(projected, projBias);
        var projected3D = projected.Reshape(batchSize, seqLen, _expandedDimension);
        _lastProjectedInput = projected3D;

        // Step 2: Gate computation: z_t = sigma(W_z * projected_t + b_z)
        var gatePreAct = Engine.TensorMatMul(projected.Reshape(batchSize * seqLen, _expandedDimension), _gateWeights);
        var gateBias2D = _gateBias.Reshape(1, _expandedDimension);
        gatePreAct = Engine.TensorBroadcastAdd(gatePreAct, gateBias2D);
        var gatePreAct3D = gatePreAct.Reshape(batchSize, seqLen, _expandedDimension);
        var gate = Engine.Sigmoid(gatePreAct3D);
        _lastGatePreAct = gatePreAct3D;
        _lastGate = gate;

        // Step 3: Candidate computation: h_tilde_t = W_h * projected_t + b_h
        var candidate = Engine.TensorMatMul(projected.Reshape(batchSize * seqLen, _expandedDimension), _candidateWeights);
        var candBias2D = _candidateBias.Reshape(1, _expandedDimension);
        candidate = Engine.TensorBroadcastAdd(candidate, candBias2D);
        var candidate3D = candidate.Reshape(batchSize, seqLen, _expandedDimension);
        _lastCandidate = candidate3D;

        // Step 4: Sequential recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
        var recurrenceOutput = MinGRURecurrenceForward(gate, candidate3D, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 5: Output projection from expandedDim back to modelDim
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
    /// Computes the minGRU recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t.
    /// </summary>
    /// <remarks>
    /// This is the sequential formulation. Because z_t and h_tilde_t are precomputed (no hidden-state
    /// dependency), this recurrence is a linear first-order system h_t = a_t * h_{t-1} + b_t where
    /// a_t = (1 - z_t) and b_t = z_t * h_tilde_t. On GPU, this would be replaced by a parallel scan
    /// for O(log n) wall-clock time.
    /// </remarks>
    private Tensor<T> MinGRURecurrenceForward(
        Tensor<T> gate, Tensor<T> candidate,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });

        // Store all hidden states including h_0 for backward pass: [batch, seqLen+1, expandedDim]
        var allHidden = new Tensor<T>(new[] { batchSize, seqLen + 1, _expandedDimension });
        // h_0 is initialized to zero (already default)

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _expandedDimension; d++)
                {
                    T z = gate[new[] { bi, t, d }];
                    T hTilde = candidate[new[] { bi, t, d }];
                    T hPrev = allHidden[new[] { bi, t, d }];

                    // h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
                    T oneMinusZ = NumOps.Subtract(NumOps.One, z);
                    T hNew = NumOps.Add(
                        NumOps.Multiply(oneMinusZ, hPrev),
                        NumOps.Multiply(z, hTilde));

                    allHidden[new[] { bi, t + 1, d }] = hNew;
                    output[new[] { bi, t, d }] = hNew;
                }
            }
        }

        _lastHiddenStates = allHidden;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastProjectedInput == null ||
            _lastGatePreAct == null || _lastGate == null || _lastCandidate == null ||
            _lastHiddenStates == null || _lastRecurrenceOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize gradients
        _inputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _expandedDimension]);
        _inputProjectionBiasGradient = new Tensor<T>([_expandedDimension]);
        _gateWeightsGradient = new Tensor<T>([_expandedDimension, _expandedDimension]);
        _gateBiasGradient = new Tensor<T>([_expandedDimension]);
        _candidateWeightsGradient = new Tensor<T>([_expandedDimension, _expandedDimension]);
        _candidateBiasGradient = new Tensor<T>([_expandedDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_expandedDimension, _modelDimension]);
        _outputProjectionBiasGradient = new Tensor<T>([_modelDimension]);

        // --- Step 5 backward: Output projection ---
        // y = recurrenceOutput * W_out + b_out
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var recFlat = _lastRecurrenceOutput.Reshape(batchSize * seqLen, _expandedDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(recFlat.Transpose([1, 0]), gradFlat);

        // Gradient flowing into recurrence output
        var dRecurrence = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _expandedDimension);

        // --- Step 4 backward: minGRU recurrence ---
        // h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
        // dh_{t-1} += dh_t * (1 - z_t)
        // dz_t = dh_t * (h_tilde_t - h_{t-1})
        // dh_tilde_t = dh_t * z_t
        var dGate = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });
        var dCandidate = new Tensor<T>(new[] { batchSize, seqLen, _expandedDimension });

        // Backward through time (BPTT) for the linear recurrence
        var dHidden = new Tensor<T>(new[] { batchSize, _expandedDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _expandedDimension; d++)
                {
                    // Total gradient at this timestep = from output + from future timestep
                    T dH = NumOps.Add(
                        dRecurrence[new[] { bi, t, d }],
                        dHidden[new[] { bi, d }]);

                    T z = _lastGate[new[] { bi, t, d }];
                    T hTilde = _lastCandidate[new[] { bi, t, d }];
                    T hPrev = _lastHiddenStates[new[] { bi, t, d }];

                    // dz_t = dh_t * (h_tilde_t - h_{t-1})
                    T dZ = NumOps.Multiply(dH, NumOps.Subtract(hTilde, hPrev));
                    dGate[new[] { bi, t, d }] = dZ;

                    // dh_tilde_t = dh_t * z_t
                    dCandidate[new[] { bi, t, d }] = NumOps.Multiply(dH, z);

                    // dh_{t-1} = dh_t * (1 - z_t)
                    T oneMinusZ = NumOps.Subtract(NumOps.One, z);
                    dHidden[new[] { bi, d }] = NumOps.Multiply(dH, oneMinusZ);
                }
            }
        }

        // --- Step 2-3 backward: Gate and candidate through sigmoid / linear ---

        // Gate: z = sigmoid(gatePreAct), dGatePreAct = dGate * sigmoid'(gatePreAct)
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) = z * (1 - z)
        var sigmoidDeriv = Engine.TensorMultiply(
            _lastGate,
            Engine.TensorSubtract(CreateOnesLike(_lastGate), _lastGate));
        var dGatePreAct = Engine.TensorMultiply(dGate, sigmoidDeriv);

        // Gate weight and bias gradients
        var projFlat = _lastProjectedInput.Reshape(batchSize * seqLen, _expandedDimension);
        var dGatePreActFlat = dGatePreAct.Reshape(batchSize * seqLen, _expandedDimension);
        _gateWeightsGradient = Engine.TensorMatMul(projFlat.Transpose([1, 0]), dGatePreActFlat);
        _gateBiasGradient = Engine.ReduceSum(dGatePreAct, new int[] { 0, 1 });

        // Candidate weight and bias gradients (candidate is linear, no activation)
        var dCandidateFlat = dCandidate.Reshape(batchSize * seqLen, _expandedDimension);
        _candidateWeightsGradient = Engine.TensorMatMul(projFlat.Transpose([1, 0]), dCandidateFlat);
        _candidateBiasGradient = Engine.ReduceSum(dCandidate, new int[] { 0, 1 });

        // Gradient flowing into the projected input from both gate and candidate paths
        var dProjectedFromGate = Engine.TensorMatMul(dGatePreActFlat, _gateWeights.Transpose([1, 0]));
        var dProjectedFromCandidate = Engine.TensorMatMul(dCandidateFlat, _candidateWeights.Transpose([1, 0]));
        var dProjectedFlat = Engine.TensorAdd(dProjectedFromGate, dProjectedFromCandidate);

        // --- Step 1 backward: Input projection ---
        // projected = input * W_inp + b_inp
        _inputProjectionBiasGradient = Engine.ReduceSum(
            dProjectedFlat.Reshape(batchSize, seqLen, _expandedDimension),
            new int[] { 0, 1 });

        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        _inputProjectionWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dProjectedFlat);

        // Gradient flowing into the input
        var dInput = Engine.TensorMatMul(dProjectedFlat, _inputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

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
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights,
            Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias,
            Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _gateWeights = Engine.TensorAdd(_gateWeights,
            Engine.TensorMultiplyScalar(_gateWeightsGradient!, negLR));
        _gateBias = Engine.TensorAdd(_gateBias,
            Engine.TensorMultiplyScalar(_gateBiasGradient!, negLR));
        _candidateWeights = Engine.TensorAdd(_candidateWeights,
            Engine.TensorMultiplyScalar(_candidateWeightsGradient!, negLR));
        _candidateBias = Engine.TensorAdd(_candidateBias,
            Engine.TensorMultiplyScalar(_candidateBiasGradient!, negLR));
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
        _inputProjectionWeights, _inputProjectionBias,
        _gateWeights, _gateBias,
        _candidateWeights, _candidateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedInput = null;
        _lastGatePreAct = null;
        _lastGate = null;
        _lastCandidate = null;
        _lastHiddenStates = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _gateWeightsGradient = null;
        _gateBiasGradient = null;
        _candidateWeightsGradient = null;
        _candidateBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Export a simplified single-step computation graph for JIT/export purposes.
        // The full recurrence is unrolled externally; this represents one step's output projection.
        var xPlaceholder = new Tensor<T>(new int[] { 1, _expandedDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "h_t");
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
        metadata["ExpandedDimension"] = _expandedDimension.ToString();
        metadata["ExpansionFactor"] = _expansionFactor.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the gate weights for external inspection.
    /// </summary>
    public Tensor<T> GetGateWeights() => _gateWeights;
}
