using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Real-Gated Linear Recurrence Unit (RG-LRU) from Google DeepMind's Griffin architecture.
/// </summary>
/// <remarks>
/// <para>
/// The RG-LRU is a gated linear recurrence that serves as the core sequence mixing mechanism in
/// the Griffin and Hawk architectures. It uses input-dependent gating to control both the recurrence
/// decay and the input contribution, providing selective memory similar to Mamba but through a
/// different mathematical formulation.
/// </para>
/// <para>
/// The recurrence is:
/// <code>
///   r_t = sigmoid(W_r * x_t + b_r)           // Recurrence gate
///   i_t = sigmoid(W_i * x_t + b_i)           // Input gate
///   a_t = diag(r_t) * diag(exp(-softplus(c))) // Gated decay (c is a learned parameter)
///   h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * (W_x * x_t))
///   y_t = h_t
/// </code>
/// The sqrt(1 - a_t^2) factor ensures the recurrence preserves signal magnitude, preventing
/// vanishing or exploding states.
/// </para>
/// <para>
/// Griffin combines RG-LRU with local attention in a hybrid architecture. This layer implements
/// the RG-LRU component which can be used standalone or as part of a hybrid.
/// </para>
/// <para><b>For Beginners:</b> The RG-LRU is like a learnable "leaky bucket" for information.
///
/// Imagine each position in your hidden state as a bucket:
/// - The recurrence gate (r) controls how much water leaks out each step (memory decay)
/// - The input gate (i) controls how much new water pours in
/// - The sqrt(1 - a^2) factor ensures the bucket never overflows or runs dry
///
/// This is simpler than Mamba (no Conv1D, no SSM parameters B/C) but surprisingly effective.
/// Google's RecurrentGemma models (2B, 9B) use this architecture and achieve competitive
/// performance with Transformer-based Gemma models.
/// </para>
/// <para>
/// <b>Reference:</b> De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", 2024.
/// https://arxiv.org/abs/2402.19427
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class RealGatedLinearRecurrenceLayer<T> : LayerBase<T>
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _recurrenceDimension;

    // Input projection: [modelDim, recurrenceDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputProjectionBias;

    // Recurrence gate: [recurrenceDim, recurrenceDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _recurrenceGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _recurrenceGateBias;

    // Input gate: [recurrenceDim, recurrenceDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputGateBias;

    // Value projection: [recurrenceDim, recurrenceDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueProjectionWeights;

    // Learned decay parameter: [recurrenceDim] (passed through softplus for positivity)
    private Tensor<T> _decayParam;

    // Output projection: [recurrenceDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedInput;
    private Tensor<T>? _lastRecurrenceGate;
    private Tensor<T>? _lastInputGate;
    private Tensor<T>? _lastHiddenStates;
    private Tensor<T>? _lastDecayFactors;
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _recurrenceGateWeightsGradient;
    private Tensor<T>? _recurrenceGateBiasGradient;
    private Tensor<T>? _inputGateWeightsGradient;
    private Tensor<T>? _inputGateBiasGradient;
    private Tensor<T>? _valueProjectionWeightsGradient;
    private Tensor<T>? _decayParamGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension (input/output width).
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the recurrence dimension (hidden state width).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The recurrence dimension controls the width of the hidden state.
    /// In Griffin, this is typically equal to the model dimension, but can be configured independently.</para>
    /// </remarks>
    public int RecurrenceDimension => _recurrenceDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override long ParameterCount =>
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _recurrenceGateWeights.Length + _recurrenceGateBias.Length +
        _inputGateWeights.Length + _inputGateBias.Length +
        _valueProjectionWeights.Length +
        _decayParam.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Real-Gated Linear Recurrence Unit (RG-LRU) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="recurrenceDimension">
    /// Recurrence state dimension. Default: -1 (same as modelDimension).
    /// <para><b>For Beginners:</b> Width of the hidden recurrence state. Using the same as modelDim
    /// is the standard configuration from the Griffin paper.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when modelDimension is not positive.</exception>
    public RealGatedLinearRecurrenceLayer(
        int sequenceLength,
        int modelDimension = 256,
        int recurrenceDimension = -1,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));

        _modelDimension = modelDimension;
        _recurrenceDimension = recurrenceDimension > 0 ? recurrenceDimension : modelDimension;

        _inputProjectionWeights = new Tensor<T>([modelDimension, _recurrenceDimension]);
        _inputProjectionBias = new Tensor<T>([_recurrenceDimension]);
        _recurrenceGateWeights = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _recurrenceGateBias = new Tensor<T>([_recurrenceDimension]);
        _inputGateWeights = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _inputGateBias = new Tensor<T>([_recurrenceDimension]);
        _valueProjectionWeights = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _decayParam = new Tensor<T>([_recurrenceDimension]);
        _outputProjectionWeights = new Tensor<T>([_recurrenceDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);
        InitializeTensor(_recurrenceGateWeights);
        _recurrenceGateBias.Fill(NumOps.Zero);
        InitializeTensor(_inputGateWeights);
        _inputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_valueProjectionWeights);

        // Initialize decay ~ 0.9 per step (softplus(2.2) ≈ 2.3, exp(-2.3) ≈ 0.1 -> 1-0.1=0.9)
        for (int i = 0; i < _recurrenceDimension; i++)
            _decayParam[i] = NumOps.FromDouble(2.2 + Random.NextDouble() * 0.5);

        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

        // Register ALL trainable tensors (in GetAllTensors order) so tape-based
        // training (GetTrainableParameters) trains the full layer. Previously only
        // _decayParam was registered, so the source generator exposed just that one
        // tensor to the tape optimizer and the input/gate/value/output projection
        // weights never received gradients under the tape path (the manual
        // Backward/UpdateParameters path trained them, but the tape path silently
        // did not). The ordering matches GetAllTensors / GetParameters so the flat
        // and tape parameter views stay consistent.
        RegisterTrainableParameter(_inputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_recurrenceGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_recurrenceGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_inputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_valueProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_decayParam, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        if (rank < 1)
            throw new ArgumentException(
                "RealGatedLinearRecurrenceLayer requires rank >= 1 input (got rank-0 scalar tensor).",
                nameof(input));

        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        // Reject zero-length sequences fast — the recurrence has no meaningful
        // output when there are no timesteps to process, and downstream
        // TensorAllocator.Rent / SetSlice / output Reshape paths all assume
        // seqLen >= 1. Without this guard a [B, 0, modelDim] input would
        // silently allocate empty tensors and surface as a confusing
        // out-of-bounds in GatedRecurrenceForward's slice indexing instead
        // of a clear input-validation error at the call boundary.
        if (seqLen < 1)
            throw new ArgumentException(
                $"RealGatedLinearRecurrenceLayer requires sequence length >= 1 " +
                $"(got seqLen={seqLen} from input shape [{string.Join(",", input.Shape)}]).",
                nameof(input));
        if (modelDim < 1)
            throw new ArgumentException(
                $"RealGatedLinearRecurrenceLayer requires modelDim >= 1 " +
                $"(got modelDim={modelDim} from input shape [{string.Join(",", input.Shape)}]).",
                nameof(input));
        // Reject input-width mismatches at the boundary instead of letting
        // them surface as a less actionable Engine.TensorMatMul shape error
        // deeper in the forward pass. The input projection's [_modelDimension,
        // _recurrenceDimension] weight matrix can only consume a tensor whose
        // last dim is _modelDimension, so any other width is a user contract
        // violation that's worth diagnosing here.
        if (modelDim != _modelDimension)
            throw new ArgumentException(
                $"RealGatedLinearRecurrenceLayer expected modelDim={_modelDimension}, " +
                $"but got modelDim={modelDim} from input shape [{string.Join(",", input.Shape)}].",
                nameof(input));

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, modelDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, modelDim });

        _lastInput = input3D;

        // Step 1: Input projection
        var input2D = Engine.Reshape(input3D, new[] { batchSize * seqLen, modelDim });
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var projBias = Engine.Reshape(_inputProjectionBias, new[] { 1, _recurrenceDimension });
        projected = Engine.TensorBroadcastAdd(projected, projBias);
        var projected3D = Engine.Reshape(projected, new[] { batchSize, seqLen, _recurrenceDimension });
        _lastProjectedInput = projected3D;

        // Step 2: Compute gates. Collect the per-time-step gates and assemble them
        // with a tape-connected Engine.TensorConcatenate — the previous SetSlice into
        // rented buffers detached the gates from the gate weights, so
        // _recurrenceGateWeights / _inputGateWeights never received a gradient.
        var recGateList = new System.Collections.Generic.List<Tensor<T>>(seqLen);
        var inpGateList = new System.Collections.Generic.List<Tensor<T>>(seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            var p_t = projected3D.GetSliceAlongDimension(t, 1);

            var rGate = Engine.Sigmoid(Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(p_t, _recurrenceGateWeights),
                Engine.Reshape(_recurrenceGateBias, new[] { 1, _recurrenceDimension })));
            var iGate = Engine.Sigmoid(Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(p_t, _inputGateWeights),
                Engine.Reshape(_inputGateBias, new[] { 1, _recurrenceDimension })));

            recGateList.Add(Engine.Reshape(rGate, new[] { batchSize, 1, _recurrenceDimension }));
            inpGateList.Add(Engine.Reshape(iGate, new[] { batchSize, 1, _recurrenceDimension }));
        }

        var recGate3D = Engine.TensorConcatenate(recGateList.ToArray(), axis: 1);
        var inpGate3D = Engine.TensorConcatenate(inpGateList.ToArray(), axis: 1);

        _lastRecurrenceGate = recGate3D;
        _lastInputGate = inpGate3D;

        // Step 3: Gated linear recurrence
        var output = GatedRecurrenceForward(projected3D, recGate3D, inpGate3D, batchSize, seqLen);
        _lastRecurrenceOutput = output;

        // Step 4: Output projection
        var outFlat = Engine.Reshape(output, new[] { batchSize * seqLen, _recurrenceDimension });
        var outputFlat = Engine.TensorMatMul(outFlat, _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias);
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 1)
            return Engine.Reshape(result, new[] { _modelDimension });
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
    /// Implements the gated linear recurrence with magnitude-preserving update.
    /// </summary>
    private Tensor<T> GatedRecurrenceForward(
        Tensor<T> x, Tensor<T> recGate, Tensor<T> inpGate,
        int batchSize, int seqLen)
    {
        var hiddenList = new System.Collections.Generic.List<Tensor<T>>(seqLen);
        var h = TensorAllocator.Rent<T>(new[] { batchSize, _recurrenceDimension });
        var allHidden = new Tensor<T>(new[] { batchSize, seqLen + 1, _recurrenceDimension });
        var allDecay = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _recurrenceDimension });

        // Pre-compute baseDecay[d] = exp(-softplus(c[d])) once per forward as a
        // [recurrenceDim] tensor. The closed form is
        //     exp(-softplus(c)) = exp(-log(1 + e^c)) = 1 / (1 + e^c) = sigmoid(-c),
        // so a single tape-connected Engine.Sigmoid(Engine.TensorNegate(_decayParam))
        // both (a) keeps _decayParam on the autodiff graph — the prior
        // NumOps.ToDouble/Math.Exp scalar loop detached it, so the learned decay
        // never received a gradient under tape-based training — and (b) stays
        // numerically stable, since sigmoid saturates gracefully for large |c|.
        // It is also a single SIMD-accelerated engine op rather than the
        // (recDim × batchSize) NumOps virtual dispatches per timestep the loop cost.
        var baseDecay = Engine.Sigmoid(Engine.TensorNegate(_decayParam));

        // A constant `ones` tensor for the tape-connected (1 - a²) computation.
        // Engine.TensorMultiplyScalar / TensorAddScalar do NOT propagate on the
        // autodiff tape (LayerTestBase's error message calls this out by name),
        // so we use TensorSubtract against this constant instead.
        var onesForMagnitude = Tensor<T>.CreateDefault(
            new[] { batchSize, _recurrenceDimension }, NumOps.One);

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);
            var r_t = recGate.GetSliceAlongDimension(t, 1);
            var i_t = inpGate.GetSliceAlongDimension(t, 1);

            // Value projection: v_t = x_t @ W_v   ([batch, recDim])
            var v_t = Engine.TensorMatMul(x_t, _valueProjectionWeights);

            // Decay: a_t = r_t * baseDecay (broadcast [batch, recDim] × [recDim])
            var a_t = Engine.TensorBroadcastMultiply(r_t, baseDecay);

            // Magnitude-preserving factor: sqrtFactor = sqrt(max(0, 1 - a²)),
            // composed from tape-connected element-wise tensor ops.
            var aSquared = Engine.TensorSquare(a_t);
            var oneMinusASquared = Engine.TensorSubtract(onesForMagnitude, aSquared);
            var clamped = Engine.TensorReLU(oneMinusASquared);
            var sqrtFactor = Engine.TensorSqrt(clamped);

            // h_t = a_t · h_{t-1} + sqrtFactor · (i_t · v_t)
            var iv = Engine.TensorMultiply(i_t, v_t);
            var weighted = Engine.TensorMultiply(sqrtFactor, iv);
            var aHPrev = Engine.TensorMultiply(a_t, h);
            var hNext = Engine.TensorAdd(aHPrev, weighted);

            // Update h in-place for next iteration. h is rented from the
            // allocator and we want to keep using the same buffer; bulk-copy
            // hNext's full storage into h via Engine.TensorCopy rather than
            // reassigning the local (which would drop the rented reference).
            // The previous element-wise write loop allocated 2 fresh int[]
            // index arrays per (batch, recurrenceDim) pair — at seqLen=1024,
            // batchSize=16, recurrenceDim=256 that's ~8 million per-call
            // allocations and erodes the SIMD gains from the gate/output
            // computation above. Engine.TensorCopy dispatches to the
            // SIMD-aware bulk path used by ConvLSTMLayer / Bidirectional
            // and friends.
            // Keep the tape node as the running hidden state so the time recurrence
            // stays on the autodiff graph. The previous Engine.TensorCopy into a
            // rented buffer detached h from hNext, breaking gradient flow through the
            // recurrence (and the SetSlice-assembled output) so the gate/value/decay
            // weights never received a gradient.
            h = hNext;
            hiddenList.Add(Engine.Reshape(h, new[] { batchSize, 1, _recurrenceDimension }));

            allDecay.SetSlice(1, t, a_t);       // caches for the manual backward path
            allHidden.SetSlice(1, t + 1, h);
        }

        // Assemble the [batch, seqLen, recDim] output on the tape so gradients flow
        // back through the recurrence to the weights.
        var output = Engine.TensorConcatenate(hiddenList.ToArray(), axis: 1);

        _lastHiddenStates = allHidden;
        _lastDecayFactors = allDecay;
        return output;
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
        _recurrenceGateWeights = Engine.TensorAdd(_recurrenceGateWeights, Engine.TensorMultiplyScalar(_recurrenceGateWeightsGradient!, negLR));
        _recurrenceGateBias = Engine.TensorAdd(_recurrenceGateBias, Engine.TensorMultiplyScalar(_recurrenceGateBiasGradient!, negLR));
        _inputGateWeights = Engine.TensorAdd(_inputGateWeights, Engine.TensorMultiplyScalar(_inputGateWeightsGradient!, negLR));
        _inputGateBias = Engine.TensorAdd(_inputGateBias, Engine.TensorMultiplyScalar(_inputGateBiasGradient!, negLR));
        _valueProjectionWeights = Engine.TensorAdd(_valueProjectionWeights, Engine.TensorMultiplyScalar(_valueProjectionWeightsGradient!, negLR));
        _decayParam = Engine.TensorAdd(_decayParam, Engine.TensorMultiplyScalar(_decayParamGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var tensor in GetAllTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");

        int index = 0;
        foreach (var tensor in GetAllTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _inputProjectionWeights, _inputProjectionBias,
        _recurrenceGateWeights, _recurrenceGateBias,
        _inputGateWeights, _inputGateBias,
        _valueProjectionWeights, _decayParam,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_inputProjectionWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_inputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_recurrenceGateWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_recurrenceGateBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputGateWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputGateBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_decayParamGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _inputProjectionWeightsGradient = null; _inputProjectionBiasGradient = null;
        _recurrenceGateWeightsGradient = null; _recurrenceGateBiasGradient = null;
        _inputGateWeightsGradient = null; _inputGateBiasGradient = null;
        _valueProjectionWeightsGradient = null; _decayParamGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedInput = null;
        _lastRecurrenceGate = null;
        _lastInputGate = null;
        _lastHiddenStates = null;
        _lastDecayFactors = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _recurrenceGateWeightsGradient = null;
        _recurrenceGateBiasGradient = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
        _valueProjectionWeightsGradient = null;
        _decayParamGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["RecurrenceDimension"] = _recurrenceDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the decay parameter for external inspection.
    /// </summary>
    public Tensor<T> GetDecayParameter() => _decayParam;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;
}
