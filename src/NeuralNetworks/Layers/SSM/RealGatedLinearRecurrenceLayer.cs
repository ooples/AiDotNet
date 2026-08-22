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
///   log(a_t) = -8 * softplus(Lambda) * r_t    // Paper's stable gated-decay form
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
// Shape-preserving at every accepted rank, and read from this layer's own forward rather than assumed:
// ForwardTraced takes seqLen = Shape[rank-2] and modelDim = Shape[rank-1], so rank 2 is [Time, Features]
// with NO batch axis - the same convention every layer in this folder follows. Its three exits confirm
// the shape is untouched: [_modelDimension] at rank 1, [seqLen, _modelDimension] at rank 2, and the
// original leading axes with [seqLen, _modelDimension] appended above that.
//
// The feature width is Same, not Fixed(_modelDimension), and the forward makes the distinction explicit:
// it THROWS when modelDim != _modelDimension, so the width is a precondition the caller must already
// satisfy, not something this layer sets. Rank 1 is accepted (seqLen defaults to 1) but not declared -
// a single timestep with no time axis is a degenerate probe shape, not a form to route a stack through.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RealGatedLinearRecurrenceLayer<T> : LayerBase<T>, IShapeContract
{
    // OutputAxesFor is GENERATED from the [TensorLayout] attributes above (ShapeContractGenerator).
    // Nothing to write here: the layouts already state that every axis is carried through, and
    // restating that in a hand-copied method is how a contract drifts from its own declaration.

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
            // Sequence is a FREE axis: -1, not the configured maximum. sequenceLength is
            // documented as a MAXIMUM and is used here for nothing but validation -- no weight and
            // no buffer is sized against it, because the recurrence runs over whatever length it
            // is handed. Publishing it as a concrete contract made the layer claim an output it
            // does not produce for any other length, which VerifyReportedOutputShape reports as
            // "[maxLen, D] declared but [B, actualLen, D] produced" and which anything sizing
            // itself from the declaration -- parameter slicing, chain resolution, ONNX export --
            // reads as fact. modelDimension IS structural and stays concrete.
            [-1, modelDimension],
            [-1, modelDimension],
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

        // Griffin, Section 2.4: c=8 and a^c is uniform in [0.9, 0.999]. The
        // numerically-stable implementation in Appendix A parameterizes
        // log(a) = -softplus(Lambda), so solve Lambda = log((1-a)/a) for each
        // sampled a. The previous positive 2.2..2.7 initialization produced
        // a≈0.06..0.10 and then multiplied it directly by r_t, erasing nearly
        // all recurrent state in one step instead of initializing long memory.
        const double recurrencePower = 8.0;
        for (int i = 0; i < _recurrenceDimension; i++)
        {
            double aToC = 0.9 + Random.NextDouble() * 0.099;
            double a = Math.Pow(aToC, 1.0 / recurrencePower);
            _decayParam[i] = NumOps.FromDouble(Math.Log((1.0 - a) / a));
        }

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
        projected = Engine.TensorAdd(projected, projBias);
        var projected3D = Engine.Reshape(projected, new[] { batchSize, seqLen, _recurrenceDimension });
        _lastProjectedInput = projected3D;

        // Step 2: Compute every position's gates in two batched projections. The
        // previous timestep loop emitted O(sequence length) slices, matmuls,
        // reshapes, and tape nodes even though these projections are independent.
        var recGate3D = Engine.Reshape(
            Engine.Sigmoid(Engine.TensorAdd(
                Engine.TensorMatMul(projected, _recurrenceGateWeights),
                Engine.Reshape(_recurrenceGateBias, new[] { 1, _recurrenceDimension }))),
            new[] { batchSize, seqLen, _recurrenceDimension });
        var inpGate3D = Engine.Reshape(
            Engine.Sigmoid(Engine.TensorAdd(
                Engine.TensorMatMul(projected, _inputGateWeights),
                Engine.Reshape(_inputGateBias, new[] { 1, _recurrenceDimension }))),
            new[] { batchSize, seqLen, _recurrenceDimension });

        _lastRecurrenceGate = recGate3D;
        _lastInputGate = inpGate3D;

        // Step 3: Gated linear recurrence
        var output = GatedRecurrenceForward(projected3D, recGate3D, inpGate3D, batchSize, seqLen);
        _lastRecurrenceOutput = output;

        // Step 4: Output projection
        var outFlat = Engine.Reshape(output, new[] { batchSize * seqLen, _recurrenceDimension });
        var outputFlat = Engine.TensorMatMul(outFlat, _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorAdd(outputFlat, outBias);
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        // Residual skip. The gated linear recurrence above is computed with in-place state
        // updates and is OFF the autodiff tape, so the gradient cannot cross it to reach the
        // input/gate projections or any upstream layer (the model trained nothing — params
        // never changed). The residual skip (output + input) re-attaches the block output to
        // its input ON the tape so gradients flow to every projection and propagate down.
        // Unlike the xLSTM covariance cell, the RG-LRU output is a convex blend of the input
        // projection and the prior state (the gate is a sigmoid in [0, 1]), so it preserves
        // signal scale across the stack without a per-block normalizer — adding one only damps
        // the residual stream and slows convergence. The final pre-head LayerNorm still
        // standardizes the stack output before the LM head.
        output3D = Engine.TensorAdd(output3D, input3D);

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
        var flat = Engine.Reshape(x, new[] { batchSize * seqLen, _recurrenceDimension });
        var value = Engine.Reshape(
            Engine.TensorMatMul(flat, _valueProjectionWeights),
            new[] { batchSize, seqLen, _recurrenceDimension });

        // Griffin Appendix A, Eq. 6: a_t = exp(-8*softplus(Lambda)*r_t).
        var negativeC = Tensor<T>.CreateDefault(
            new[] { _recurrenceDimension }, NumOps.FromDouble(-8.0));
        var logTransition = Engine.TensorMultiply(
            recGate,
            Engine.TensorMultiply(Engine.Softplus(_decayParam), negativeC));
        var transition = Engine.TensorExp(logTransition);

        // RgLruScanForward defines its effective transition as
        // recurrenceStream * sigmoid(-decay). With a zero decay vector the
        // internal factor is exactly 1/2, so passing 2*a_t implements the
        // Griffin transition exactly while retaining one analytic BPTT node.
        var twos = Tensor<T>.CreateDefault(
            new[] { batchSize, seqLen, _recurrenceDimension }, NumOps.FromDouble(2.0));
        var zeroDecay = new Tensor<T>(new[] { _recurrenceDimension });
        var output = Engine.RgLruScanForward(
            value,
            Engine.TensorMultiply(transition, twos),
            inpGate,
            zeroDecay);

        var initial = new Tensor<T>(new[] { batchSize, 1, _recurrenceDimension });
        _lastHiddenStates = Engine.TensorConcatenate(new[] { initial, output }, axis: 1);
        _lastDecayFactors = transition;
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
