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
public partial class ABCLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numSlots;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Slot key embeddings: [numHeads, numSlots, headDim]
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
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastForgetGate;
    private Tensor<T>? _lastWriteWeights;
    private Tensor<T>? _lastReadWeights;
    private Tensor<T>? _lastSlotStates;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private Tensor<T>? _lastSlotReadOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _slotKeysGradient;
    private Tensor<T>? _forgetGateWeightsGradient;
    private Tensor<T>? _forgetGateBiasGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

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

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _slotKeys.Length +
        _forgetGateWeights.Length + _forgetGateBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

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

        // Step 1: Q, K, V projections
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var q = Engine.TensorMatMul(inputFlat, _queryWeights).Reshape(batchSize, seqLen, _modelDimension);
        var k = Engine.TensorMatMul(inputFlat, _keyWeights).Reshape(batchSize, seqLen, _modelDimension);
        var v = Engine.TensorMatMul(inputFlat, _valueWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Forget gate
        var forgetRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _forgetGateWeights),
            _forgetGateBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var forgetGate = Engine.Sigmoid(forgetRaw);
        _lastForgetGate = forgetGate;

        // Step 3: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var outputGate = Engine.Swish(gateRaw);
        _lastOutputGate = outputGate;
        _lastOutputGateRaw = gateRaw;

        // Step 4: Slot competition, write, and read
        var slotOutput = SlotCompetitionForward(q, k, v, forgetGate, batchSize, seqLen);
        _lastSlotReadOutput = slotOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(slotOutput, outputGate);

        // Step 6: Output projection
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
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
    /// Competitive slot write and read mechanism.
    /// </summary>
    /// <remarks>
    /// For each timestep t and head h:
    ///   1. Write scores: score[s] = k_t^T * slotKey[s] / sqrt(d)
    ///   2. Write weights: w[s] = softmax(score) over slots
    ///   3. Slot update: slot[s] = forget * slot[s] + w[s] * v_t
    ///   4. Read scores: readScore[s] = q_t^T * slot[s] / sqrt(d)
    ///   5. Read weights: r[s] = softmax(readScore) over slots
    ///   6. Output: o_t = sum_s r[s] * slot[s]
    /// </remarks>
    private Tensor<T> SlotCompetitionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> forgetGate, int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Slot values: [batch, numHeads, numSlots, headDim] - the actual slot content
        var slotValues = TensorAllocator.Rent<T>(new[] { batchSize, _numHeads, _numSlots, _headDimension });
        // Initialize slot values to small random-like state (from slot keys scaled down)
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
                for (int si = 0; si < _numSlots; si++)
                    for (int di = 0; di < _headDimension; di++)
                        slotValues[new[] { bi, hi, si, di }] = NumOps.Multiply(
                            _slotKeys[new[] { hi, si, di }], NumOps.FromDouble(0.1));

        // Save all slot states for backward pass: [batch, seqLen+1, numHeads, numSlots, headDim]
        var allStates = TensorAllocator.Rent<T>(new[] { batchSize, seqLen + 1, _numHeads, _numSlots, _headDimension });
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
                for (int si = 0; si < _numSlots; si++)
                    for (int di = 0; di < _headDimension; di++)
                        allStates[new[] { bi, 0, hi, si, di }] = slotValues[new[] { bi, hi, si, di }];

        // Cache write and read weights for backward pass
        var writeWeights = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads, _numSlots });
        var readWeights = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads, _numSlots });

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T fGate = forgetGate[new[] { bi, t, hi }];

                    // Step 1: Compute write scores using key against slot keys
                    var wScores = new T[_numSlots];
                    T maxWScore = NumOps.MinValue;
                    for (int si = 0; si < _numSlots; si++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(k[new[] { bi, t, flatD }],
                                    _slotKeys[new[] { hi, si, d }]));
                        }
                        wScores[si] = NumOps.Multiply(dot, scale);
                        if (NumOps.GreaterThan(wScores[si], maxWScore))
                            maxWScore = wScores[si];
                    }

                    // Step 2: Softmax over slots for write weights
                    T sumExpW = NumOps.Zero;
                    var expW = new T[_numSlots];
                    for (int si = 0; si < _numSlots; si++)
                    {
                        expW[si] = NumOps.Exp(NumOps.Subtract(wScores[si], maxWScore));
                        sumExpW = NumOps.Add(sumExpW, expW[si]);
                    }
                    T sumExpWInv = NumOps.Divide(NumOps.One, NumOps.Add(sumExpW, NumOps.FromDouble(1e-10)));

                    // Step 3: Forget old content and write new content
                    for (int si = 0; si < _numSlots; si++)
                    {
                        T wWeight = NumOps.Multiply(expW[si], sumExpWInv);
                        writeWeights[new[] { bi, t, hi, si }] = wWeight;

                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            T prevSlot = slotValues[new[] { bi, hi, si, di }];
                            T vVal = v[new[] { bi, t, flatDi }];

                            // slot = forget * slot + writeWeight * value
                            T newSlot = NumOps.Add(
                                NumOps.Multiply(fGate, prevSlot),
                                NumOps.Multiply(wWeight, vVal));
                            slotValues[new[] { bi, hi, si, di }] = newSlot;
                        }
                    }

                    // Step 4: Compute read scores using query against current slot content
                    var rScores = new T[_numSlots];
                    T maxRScore = NumOps.MinValue;
                    for (int si = 0; si < _numSlots; si++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(q[new[] { bi, t, flatD }],
                                    slotValues[new[] { bi, hi, si, d }]));
                        }
                        rScores[si] = NumOps.Multiply(dot, scale);
                        if (NumOps.GreaterThan(rScores[si], maxRScore))
                            maxRScore = rScores[si];
                    }

                    // Step 5: Softmax for read weights
                    T sumExpR = NumOps.Zero;
                    var expR = new T[_numSlots];
                    for (int si = 0; si < _numSlots; si++)
                    {
                        expR[si] = NumOps.Exp(NumOps.Subtract(rScores[si], maxRScore));
                        sumExpR = NumOps.Add(sumExpR, expR[si]);
                    }
                    T sumExpRInv = NumOps.Divide(NumOps.One, NumOps.Add(sumExpR, NumOps.FromDouble(1e-10)));

                    // Step 6: Weighted read from slots
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int si = 0; si < _numSlots; si++)
                        {
                            T rWeight = NumOps.Multiply(expR[si], sumExpRInv);
                            readWeights[new[] { bi, t, hi, si }] = rWeight;
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(rWeight, slotValues[new[] { bi, hi, si, di }]));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }
                }
            }

            // Save slot state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int hi = 0; hi < _numHeads; hi++)
                    for (int si = 0; si < _numSlots; si++)
                        for (int di = 0; di < _headDimension; di++)
                            allStates[new[] { bi, t + 1, hi, si, di }] = slotValues[new[] { bi, hi, si, di }];
        }

        _lastSlotStates = allStates;
        _lastWriteWeights = writeWeights;
        _lastReadWeights = readWeights;
        return output;
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
        var ones = new Tensor<T>(template.Shape.ToArray());
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
        _queryWeights, _keyWeights, _valueWeights,
        _slotKeys,
        _forgetGateWeights, _forgetGateBias,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCount);
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
        _lastWriteWeights = null;
        _lastReadWeights = null;
        _lastSlotStates = null;
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
