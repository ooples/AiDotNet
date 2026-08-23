using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the ABC (Attention with Bounded-memory Control) layer from Peng et al., 2022.
/// </summary>
/// <remarks>
/// <para>
/// ABC uses a fixed-size set of memory "slots" with a competitive attention mechanism. Input tokens
/// compete for writing to slots via softmax attention scores, and a forget mechanism clears stale
/// slot content. This bounds memory usage regardless of sequence length while maintaining the ability
/// to selectively store and retrieve information.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute slot attention scores: score[t,s] = q_t^T * slot_key[s] / sqrt(d)
///   3. Competitive slot writing via softmax over slots:
///      write_weight[t,s] = softmax(score[t,s]) over slots dimension
///      slot[s] = forget_gate * slot[s] + sum_t(write_weight[t,s] * v_t)
///   4. Read from slots:
///      read_weight[t,s] = softmax(q_t^T * slot[s] / sqrt(d)) over slots
///      o_t = sum_s(read_weight[t,s] * slot[s])
///   5. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The key insight is competitive slot access: tokens compete to write into a bounded number of
/// memory slots via softmax. This naturally implements a form of memory management where the most
/// relevant information gets stored and stale information is gradually forgotten. Unlike unbounded
/// linear attention states, the fixed slot count guarantees constant memory.
/// </para>
/// <para><b>For Beginners:</b> ABC is like having a fixed number of filing cabinet drawers (slots)
/// for storing information as you read through a long document.
///
/// Imagine you have 32 drawers and you're reading a book:
/// - At each word, you decide which drawers are most relevant (via attention scores)
/// - You file information about the word into those drawers (competitive writing)
/// - Old information gradually fades from drawers (forget gate)
/// - When you need to answer a question, you look through the drawers (reading)
///
/// The "competitive" part is crucial: if many words want to use the same drawer,
/// softmax ensures the most relevant one gets priority. This is what "bounded-memory
/// control" means -- you never need more drawers than the fixed number, no matter
/// how long the book is.
///
/// Compare this to:
/// - Standard attention: You keep all words accessible (expensive for long books)
/// - Linear attention: You maintain a summary matrix (unbounded growth in rank)
/// - ABC: You maintain exactly numSlots drawers of information (bounded)
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "ABC: Attention with Bounded-memory Control", 2022.
/// https://arxiv.org/abs/2110.02488
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape relations DISCOVERED by probing (LayerShapeDiscoverySweepTests): every axis is carried
// through unchanged, at rank 2 and rank 3 alike. The ROLES, however, are read from the layer's own
// forward rather than from the probe - discovery recovers relations, never axis names, and here the
// positional stand-in would have been wrong. ForwardTraced computes
// seqLen = Shape[rank-2], modelDim = Shape[rank-1], batchSize = 1 when rank < 3, so a rank-2 input is
// [Time, Features] and NOT [Batch, Features]: the leading axis is sequence position, not batch.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Rank 3+ folds every leading axis into the batch; sequence and model dim stay last.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ABCLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numSlots;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Slot state is seeded from the slot keys scaled down by this factor, so the memory starts near
    // (but not at) zero. Not a value from the paper - it is this implementation's initialisation
    // choice, kept exactly as the previous scalar loop had it so the forward output is unchanged.
    private const double SlotInitScale = 0.1;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Slot key embeddings: [numHeads, numSlots, headDim]. Trainable like its q/k/v siblings:
    // the slot-competition scan reads it directly, and without the attribute it is neither
    // trained nor serialized even though _slotKeysGradient is already declared for it.
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _slotKeys;

    // Forget gate projection: [modelDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _forgetGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _forgetGateBias;

    // Output gate: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    [Scratch]
    private Tensor<T>? _lastInput;
    [Scratch]
    private Tensor<T>? _lastOutput;
    [Scratch]
    private Tensor<T>? _lastQuery;
    [Scratch]
    private Tensor<T>? _lastKey;
    [Scratch]
    private Tensor<T>? _lastValue;
    [Scratch]
    private Tensor<T>? _lastForgetGate;
    private Tensor<T>? _lastOutputGate;
    [Scratch]
    private Tensor<T>? _lastOutputGateRaw;
    [Scratch]
    private Tensor<T>? _lastSlotReadOutput;
    private int[]? _originalInputShape;

    // Gradients
    [Scratch]
    private Tensor<T>? _queryWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _slotKeysGradient;
    [Scratch]
    private Tensor<T>? _forgetGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _forgetGateBiasGradient;
    [Scratch]
    private Tensor<T>? _outputGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputGateBiasGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    /// <remarks>
    /// Shape-preserving at every rank it accepts: probing [4,4] and its per-axis variants produced an
    /// output identical to the input in both axes, and the rank-3 form behaves the same way. Only ranks
    /// 2 and 3 are declared because only those were MEASURED - claiming more would be the guess this
    /// system exists to remove, and a contract that over-claims is worse than one that declines.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        switch (inputRank)
        {
            case 2:
                return new[]
                {
                    new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                    new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
                };
            case 3:
                return new[]
                {
                    new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                    new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                    new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
                };
            default:
                return null;
        }
    }

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of memory slots per head.
    /// </summary>
    public int NumSlots => _numSlots;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// Creates a new ABC (Attention with Bounded-memory Control) layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each token's representation vector.</para>
    /// </param>
    /// <param name="numSlots">
    /// Number of memory slots per head. Default: 32.
    /// <para><b>For Beginners:</b> This bounds how much information each head can store. More slots
    /// allow richer memory but cost more compute. The ABC paper finds 32-64 slots work well,
    /// providing a good balance between memory capacity and efficiency.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 4.
    /// <para><b>For Beginners:</b> Each head has its own independent set of slots and can focus on
    /// different aspects of the input. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public ABCLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numSlots = 32,
        int numHeads = 4,
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
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numSlots <= 0)
            throw new ArgumentException($"Number of slots ({numSlots}) must be positive.", nameof(numSlots));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));

        _modelDimension = modelDimension;
        _numSlots = numSlots;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _slotKeys = new Tensor<T>([numHeads, numSlots, _headDimension]);
        _forgetGateWeights = new Tensor<T>([modelDimension, numHeads]);
        _forgetGateBias = new Tensor<T>([numHeads]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_queryWeights);
        InitializeTensor2D(_keyWeights);
        InitializeTensor2D(_valueWeights);
        InitializeSlotKeys();
        InitializeTensor2D(_forgetGateWeights);
        // Forget gate bias ~ 2 so sigmoid(2) ~ 0.88 -> strong initial memory retention
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(2.0);
        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    private void InitializeSlotKeys()
    {
        InitializeLayerWeights(_slotKeys, _numSlots, _headDimension);
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

        _lastInput = input3D;

        // Step 1: Q, K, V projections
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });
        var q = Engine.Reshape(Engine.TensorMatMul(inputFlat, _queryWeights), new[] { batchSize, seqLen, _modelDimension });
        var k = Engine.Reshape(Engine.TensorMatMul(inputFlat, _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var v = Engine.Reshape(Engine.TensorMatMul(inputFlat, _valueWeights), new[] { batchSize, seqLen, _modelDimension });
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Forget gate
        var forgetRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _forgetGateWeights),
            Engine.Reshape(_forgetGateBias, new[] { 1, _numHeads })), new[] { batchSize, seqLen, _numHeads });
        var forgetGate = Engine.Sigmoid(forgetRaw);
        _lastForgetGate = forgetGate;

        // Step 3: Output gate
        var gateRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var outputGate = Engine.Swish(gateRaw);
        _lastOutputGate = outputGate;
        _lastOutputGateRaw = gateRaw;

        // Step 4: Slot competition, write, and read.
        // One fused, differentiable op for the whole recurrence. The scalar loop this replaces was
        // detached from the tape, so q/k/v and both forget-gate parameters received NO gradient and
        // never learned - this layer has no Backward override, so the tape is its only gradient path.
        var slotOutput = Engine.AbcScanForward(
            q, k, v, forgetGate, _slotKeys, _numHeads, SlotInitScale);
        _lastSlotReadOutput = slotOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(slotOutput, outputGate);

        // Step 6: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorAdd(outputFlat, outBias);
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }


    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        var sig = Engine.Sigmoid(x);
        var oneMinusSig = Engine.ScalarMinusTensor(NumOps.One, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAddScalar(xTimesOneMinusSig, NumOps.One);
        return Engine.TensorMultiply(sig, onePlusXSig);
    }

    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var ones = new Tensor<T>(template._shape);
        ones.Fill(NumOps.One);
        return ones;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null ||
            _slotKeysGradient == null || _forgetGateWeightsGradient == null || _forgetGateBiasGradient == null ||
            _outputGateWeightsGradient == null || _outputGateBiasGradient == null ||
            _outputProjectionWeightsGradient == null || _outputProjectionBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient, negLR));
        _slotKeys = Engine.TensorAdd(_slotKeys, Engine.TensorMultiplyScalar(_slotKeysGradient, negLR));
        _forgetGateWeights = Engine.TensorAdd(_forgetGateWeights, Engine.TensorMultiplyScalar(_forgetGateWeightsGradient, negLR));
        _forgetGateBias = Engine.TensorAdd(_forgetGateBias, Engine.TensorMultiplyScalar(_forgetGateBiasGradient, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_forgetGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_forgetGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _keyWeights, _valueWeights,
        _slotKeys,
        _forgetGateWeights, _forgetGateBias,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_slotKeysGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_forgetGateWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_forgetGateBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _slotKeysGradient = null; _forgetGateWeightsGradient = null; _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastForgetGate = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _lastSlotReadOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _slotKeysGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumSlots"] = _numSlots.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the query weights for external inspection.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the slot key embeddings for external inspection.
    /// </summary>
    public Tensor<T> GetSlotKeys() => _slotKeys;
}
