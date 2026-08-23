using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Mixture of Memories (MoM) layer from Chou et al., 2025.
/// </summary>
/// <remarks>
/// <para>
/// Standard linear attention maintains a single key-value memory state S. As the sequence grows,
/// all information is compressed into this one matrix, leading to interference: unrelated key-value
/// associations overwrite each other. MoM addresses this by maintaining <b>multiple independent memory
/// states</b> (S_1, S_2, ..., S_M) and using a learned router to selectively read from and write to them.
/// </para>
/// <para>
/// The architecture at each timestep t:
/// <code>
///   1. Project input to Q, K, V
///   2. Router: compute routing weights for each memory
///      - Write weights w_i = softmax(R_write * x_t)_i  (which memories to write to)
///      - Read weights  r_i = softmax(R_read  * x_t)_i  (which memories to read from)
///      - Forget gates  g_i = sigmoid(R_gate  * x_t)_i  (how much to retain in each memory)
///
///   3. Write: selective update of each memory state
///      S_i[t] = g_i * S_i[t-1] + w_i * v_t * k_t^T
///      Only memories with high w_i receive the new key-value pair.
///      The forget gate g_i controls how much of the old state is retained.
///
///   4. Read: weighted combination across memories
///      o_t = sum_i  r_i * S_i[t] * q_t
///      Each memory contributes to the output proportionally to its read weight.
///
///   5. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The routing mechanism is the key innovation: by directing different tokens to different memories,
/// MoM prevents interference between unrelated information. This is analogous to how Mixture of Experts
/// (MoE) routes tokens to different expert networks, but applied to the memory states of a linear
/// attention model.
/// </para>
/// <para>
/// The forget gate per memory allows selective retention: some memories can maintain long-term state
/// (high g) while others are more transient (low g), naturally specializing into different timescales.
/// </para>
/// <para><b>For Beginners:</b> Think of this like having multiple filing cabinets (memories) instead of one:
///
/// Standard linear attention = one filing cabinet where all documents go.
/// Over time, the cabinet gets cluttered and finding specific documents is hard.
///
/// MoM = multiple filing cabinets, each for different topics:
/// - A router (like a librarian) decides which cabinet to file each new document in (write routing)
/// - When you need information, the librarian checks relevant cabinets and combines results (read routing)
/// - Each cabinet has its own retention policy: some keep documents forever, others regularly clean out (forget gate)
///
/// This prevents unrelated information from interfering with each other, which is the main weakness
/// of standard linear attention. The model learns to organize information across memories, much like
/// a well-organized library system.
///
/// The number of memories M is a key hyperparameter:
/// - More memories = less interference, more capacity, but more parameters
/// - Fewer memories = simpler model, but more compression needed
/// - 4-8 memories is typically a good balance
/// </para>
/// <para>
/// <b>Reference:</b> Chou et al., "MoM: Mixture of Memories for Linear Sequence Modeling", 2025.
/// https://arxiv.org/abs/2502.13685
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Memory)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving. Relations discovered by probing; roles read from the forward - this folder's
// convention is seqLen = Shape[rank-2], modelDim = Shape[rank-1], so rank 2 is [Time, Features].
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class MixtureOfMemoriesLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _numMemories;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _queryBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _keyBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _valueBias;

    // Router projections: [modelDim, numMemories]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _writeRouterWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _writeRouterBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _readRouterWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _readRouterBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _gateRouterWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _gateRouterBias;

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

    // Cached values for backward pass
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
    private Tensor<T>? _lastGate;
    [Scratch]
    private Tensor<T>? _lastGateRaw;
    [Scratch]
    private Tensor<T>? _lastMoMOutput;
    [Scratch]
    private Tensor<T>? _lastWriteWeights; // [batch, seqLen, numMemories]
    [Scratch]
    private Tensor<T>? _lastReadWeights;  // [batch, seqLen, numMemories]
    [Scratch]
    private Tensor<T>? _lastForgetGates;  // [batch, seqLen, numMemories]
    [Scratch]
    private Tensor<T>? _lastForgetGatesRaw;
    private int[]? _originalInputShape;

    // Gradients
    [Scratch]
    private Tensor<T>? _queryWeightsGradient;
    [Scratch]
    private Tensor<T>? _queryBiasGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyBiasGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueBiasGradient;
    [Scratch]
    private Tensor<T>? _writeRouterWeightsGradient;
    [Scratch]
    private Tensor<T>? _writeRouterBiasGradient;
    [Scratch]
    private Tensor<T>? _readRouterWeightsGradient;
    [Scratch]
    private Tensor<T>? _readRouterBiasGradient;
    [Scratch]
    private Tensor<T>? _gateRouterWeightsGradient;
    [Scratch]
    private Tensor<T>? _gateRouterBiasGradient;
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

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the number of memory states.
    /// </summary>
    public int NumMemories => _numMemories;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// Creates a new Mixture of Memories (MoM) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each token embedding.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own set of memory states.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="numMemories">
    /// Number of independent memory states. Default: 4.
    /// <para><b>For Beginners:</b> The number of "filing cabinets" for storing information.
    /// More memories reduce interference but increase computation. 4-8 is typical.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public MixtureOfMemoriesLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int numMemories = 4,
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
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        if (numMemories <= 0)
            throw new ArgumentException($"Number of memories ({numMemories}) must be positive.", nameof(numMemories));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _numMemories = numMemories;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);

        _writeRouterWeights = new Tensor<T>([modelDimension, numMemories]);
        _writeRouterBias = new Tensor<T>([numMemories]);
        _readRouterWeights = new Tensor<T>([modelDimension, numMemories]);
        _readRouterBias = new Tensor<T>([numMemories]);
        _gateRouterWeights = new Tensor<T>([modelDimension, numMemories]);
        _gateRouterBias = new Tensor<T>([numMemories]);

        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_queryWeights);
        _queryBias.Fill(NumOps.Zero);
        InitializeTensor2D(_keyWeights);
        _keyBias.Fill(NumOps.Zero);
        InitializeTensor2D(_valueWeights);
        _valueBias.Fill(NumOps.Zero);

        InitializeTensor2D(_writeRouterWeights);
        _writeRouterBias.Fill(NumOps.Zero);
        InitializeTensor2D(_readRouterWeights);
        _readRouterBias.Fill(NumOps.Zero);
        InitializeTensor2D(_gateRouterWeights);
        // Initialize gate bias high so sigmoid gives ~0.88 -> strong retention initially
        for (int i = 0; i < _numMemories; i++)
            _gateRouterBias[i] = NumOps.FromDouble(2.0);

        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
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

        _lastInput = input3D;

        // Step 1: Q, K, V projections
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });

        var q = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _queryWeights),
            Engine.Reshape(_queryBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });

        var k = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _keyWeights),
            Engine.Reshape(_keyBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });

        var v = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _valueWeights),
            Engine.Reshape(_valueBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Router computations
        var writeLogits = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _writeRouterWeights),
            Engine.Reshape(_writeRouterBias, new[] { 1, _numMemories })), new[] { batchSize, seqLen, _numMemories });
        var writeWeights = Engine.Softmax(writeLogits, axis: -1);

        var readLogits = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _readRouterWeights),
            Engine.Reshape(_readRouterBias, new[] { 1, _numMemories })), new[] { batchSize, seqLen, _numMemories });
        var readWeights = Engine.Softmax(readLogits, axis: -1);

        var forgetGatesRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _gateRouterWeights),
            Engine.Reshape(_gateRouterBias, new[] { 1, _numMemories })), new[] { batchSize, seqLen, _numMemories });
        var forgetGates = Engine.Sigmoid(forgetGatesRaw);

        _lastWriteWeights = writeWeights;
        _lastReadWeights = readWeights;
        _lastForgetGates = forgetGates;
        _lastForgetGatesRaw = forgetGatesRaw;

        // Step 3: Output gate
        var gateRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 4: MoM recurrence
        var momOutput = MoMForward(q, k, v, writeWeights, readWeights, forgetGates, batchSize, seqLen);
        _lastMoMOutput = momOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(momOutput, gate);

        // Step 6: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorAdd(
            Engine.TensorMatMul(gatedFlat, _outputProjectionWeights),
            Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension }));
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

    /// <summary>
    /// MoM forward: multi-memory state recurrence with routing.
    /// </summary>
    /// <remarks>
    /// <code>
    ///   S_m,t = g_m,t * S_m,t-1 + w_m,t * (v_t (x) k_t / sqrt(d))     per memory m
    ///   O_t   = sum_m r_m,t * (S_m,t * q_t)
    /// </code>
    /// Built from Engine ops so the recurrence stays on the autodiff tape. The NumOps scalar loop
    /// this replaces was detached from it, so q/k/v with their biases and all three routers - 12 of
    /// this layer's 16 trainable tensors, the worst of the SSM family - received no gradient and
    /// never learned. The layer has no Backward override, so the tape was its only gradient path.
    /// The write, read and forget weights are scalar per (batch, memory) and shared across heads,
    /// so they are laid out to the head-major [batch*numHeads, 1, 1] form before broadcasting.
    /// </remarks>
    private Tensor<T> MoMForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> writeWeights, Tensor<T> readWeights, Tensor<T> forgetGates,
        int batchSize, int seqLen)
    {
        int headBatch = batchSize * _numHeads;
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        var qHeads = ToHeadMajor(q, batchSize, seqLen);                          // [HB, S, D]
        var kHeads = Engine.TensorMultiplyScalar(ToHeadMajor(k, batchSize, seqLen), keyScale);
        var vHeads = ToHeadMajor(v, batchSize, seqLen);

        var memStates = new Tensor<T>[_numMemories];
        for (int mi = 0; mi < _numMemories; mi++)
            memStates[mi] = Tensor<T>.CreateDefault(
                new[] { headBatch, _headDimension, _headDimension }, NumOps.Zero);

        var outputs = new List<Tensor<T>>(seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            var qCol = Engine.Reshape(Engine.TensorSliceAxis(qHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var kRow = Engine.Reshape(Engine.TensorSliceAxis(kHeads, 1, t),
                new[] { headBatch, 1, _headDimension });
            var vCol = Engine.Reshape(Engine.TensorSliceAxis(vHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var outer = Engine.BatchMatMul(vCol, kRow);                          // [HB, D, D]

            // Write into every memory, gated per memory.
            for (int mi = 0; mi < _numMemories; mi++)
            {
                var gm = BroadcastPerMemory(forgetGates, t, mi, batchSize);
                var wm = BroadcastPerMemory(writeWeights, t, mi, batchSize);
                memStates[mi] = Engine.TensorAdd(
                    Engine.TensorMultiply(memStates[mi], gm),
                    Engine.TensorMultiply(outer, wm));
            }

            // Read from every memory, mixed by the read router. Memory 0 seeds the sum so the
            // accumulator is non-null by construction (there is always at least one memory).
            var combined = Engine.TensorMultiply(
                Engine.BatchMatMul(memStates[0], qCol),
                BroadcastPerMemory(readWeights, t, 0, batchSize));
            for (int mi = 1; mi < _numMemories; mi++)
            {
                combined = Engine.TensorAdd(combined, Engine.TensorMultiply(
                    Engine.BatchMatMul(memStates[mi], qCol),
                    BroadcastPerMemory(readWeights, t, mi, batchSize)));
            }
            outputs.Add(Engine.Reshape(combined, new[] { headBatch, 1, _headDimension }));
        }

        return FromHeadMajor(Engine.TensorConcatenate(outputs.ToArray(), 1), batchSize, seqLen);
    }

    /// <summary>
    /// Takes the scalar router value for memory <paramref name="memory"/> at timestep
    /// <paramref name="t"/> from a [batch, seqLen, numMemories] tensor and lays it out as
    /// [batch*numHeads, 1, 1]. The value is shared across heads, and head-major ordering is
    /// index = b*numHeads + h, so each batch value repeats numHeads times consecutively.
    /// </summary>
    private Tensor<T> BroadcastPerMemory(Tensor<T> routed, int t, int memory, int batchSize) =>
        Engine.Reshape(
            Engine.TensorTile(
                Engine.Reshape(
                    Engine.TensorSliceAxis(Engine.TensorSliceAxis(routed, 1, t), 1, memory),
                    new[] { batchSize, 1, 1 }),
                new[] { 1, _numHeads, 1 }),
            new[] { batchSize * _numHeads, 1, 1 });

    private Tensor<T> ToHeadMajor(Tensor<T> value, int batchSize, int seqLen) =>
        Engine.Reshape(Engine.TensorPermute(
            Engine.Reshape(value, new[] { batchSize, seqLen, _numHeads, _headDimension }),
            new[] { 0, 2, 1, 3 }),
            new[] { batchSize * _numHeads, seqLen, _headDimension });

    private Tensor<T> FromHeadMajor(Tensor<T> value, int batchSize, int seqLen) =>
        Engine.Reshape(Engine.TensorPermute(
            Engine.Reshape(value, new[] { batchSize, _numHeads, seqLen, _headDimension }),
            new[] { 0, 2, 1, 3 }),
            new[] { batchSize, seqLen, _modelDimension });

    /// <summary>
    /// Backward pass for softmax: dLogits[i] = softmax[i] * (dOutput[i] - sum_j(softmax[j]*dOutput[j]))
    /// </summary>
    private Tensor<T> SoftmaxBackward(
        Tensor<T> dOutput, Tensor<T> softmaxOutput,
        int batchSize, int seqLen, int dim)
    {
        var dLogits = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, dim });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                T dotProduct = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                    dotProduct = NumOps.Add(dotProduct,
                        NumOps.Multiply(softmaxOutput[new[] { bi, t, d }],
                            dOutput[new[] { bi, t, d }]));

                for (int d = 0; d < dim; d++)
                    dLogits[new[] { bi, t, d }] = NumOps.Multiply(
                        softmaxOutput[new[] { bi, t, d }],
                        NumOps.Subtract(dOutput[new[] { bi, t, d }], dotProduct));
            }
        }

        return dLogits;
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        var sig = Engine.Sigmoid(x);
        var oneMinusSig = Engine.ScalarMinusTensor(NumOps.One, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAddScalar(xTimesOneMinusSig, NumOps.One);
        return Engine.TensorMultiply(sig, onePlusXSig);
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _queryBias = Engine.TensorAdd(_queryBias, Engine.TensorMultiplyScalar(_queryBiasGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _keyBias = Engine.TensorAdd(_keyBias, Engine.TensorMultiplyScalar(_keyBiasGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _valueBias = Engine.TensorAdd(_valueBias, Engine.TensorMultiplyScalar(_valueBiasGradient!, negLR));
        _writeRouterWeights = Engine.TensorAdd(_writeRouterWeights, Engine.TensorMultiplyScalar(_writeRouterWeightsGradient!, negLR));
        _writeRouterBias = Engine.TensorAdd(_writeRouterBias, Engine.TensorMultiplyScalar(_writeRouterBiasGradient!, negLR));
        _readRouterWeights = Engine.TensorAdd(_readRouterWeights, Engine.TensorMultiplyScalar(_readRouterWeightsGradient!, negLR));
        _readRouterBias = Engine.TensorAdd(_readRouterBias, Engine.TensorMultiplyScalar(_readRouterBiasGradient!, negLR));
        _gateRouterWeights = Engine.TensorAdd(_gateRouterWeights, Engine.TensorMultiplyScalar(_gateRouterWeightsGradient!, negLR));
        _gateRouterBias = Engine.TensorAdd(_gateRouterBias, Engine.TensorMultiplyScalar(_gateRouterBiasGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_queryBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_writeRouterWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_writeRouterBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_readRouterWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_readRouterBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_gateRouterWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gateRouterBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _queryBias,
        _keyWeights, _keyBias,
        _valueWeights, _valueBias,
        _writeRouterWeights, _writeRouterBias,
        _readRouterWeights, _readRouterBias,
        _gateRouterWeights, _gateRouterBias,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_queryBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_writeRouterWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_writeRouterBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_readRouterWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_readRouterBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_gateRouterWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_gateRouterBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _queryBiasGradient = null; _keyWeightsGradient = null; _keyBiasGradient = null; _valueWeightsGradient = null; _valueBiasGradient = null; _writeRouterWeightsGradient = null; _writeRouterBiasGradient = null; _readRouterWeightsGradient = null; _readRouterBiasGradient = null; _gateRouterWeightsGradient = null; _gateRouterBiasGradient = null;
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
        _lastGate = null;
        _lastGateRaw = null;
        _lastMoMOutput = null;
        _lastWriteWeights = null;
        _lastReadWeights = null;
        _lastForgetGates = null;
        _lastForgetGatesRaw = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
        _writeRouterWeightsGradient = null;
        _writeRouterBiasGradient = null;
        _readRouterWeightsGradient = null;
        _readRouterBiasGradient = null;
        _gateRouterWeightsGradient = null;
        _gateRouterBiasGradient = null;
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
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        metadata["NumMemories"] = _numMemories.ToString();
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
    /// Gets the write router weights for external inspection.
    /// </summary>
    public Tensor<T> GetWriteRouterWeights() => _writeRouterWeights;

    /// <summary>
    /// Gets the read router weights for external inspection.
    /// </summary>
    public Tensor<T> GetReadRouterWeights() => _readRouterWeights;
}
