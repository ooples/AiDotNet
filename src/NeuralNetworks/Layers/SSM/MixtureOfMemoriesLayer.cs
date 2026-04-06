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
public partial class MixtureOfMemoriesLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _numMemories;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _queryBias;
    private Tensor<T> _keyWeights;
    private Tensor<T> _keyBias;
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;

    // Router projections: [modelDim, numMemories]
    private Tensor<T> _writeRouterWeights;
    private Tensor<T> _writeRouterBias;
    private Tensor<T> _readRouterWeights;
    private Tensor<T> _readRouterBias;
    private Tensor<T> _gateRouterWeights;
    private Tensor<T> _gateRouterBias;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastMoMOutput;
    private Tensor<T>? _lastWriteWeights; // [batch, seqLen, numMemories]
    private Tensor<T>? _lastReadWeights;  // [batch, seqLen, numMemories]
    private Tensor<T>? _lastForgetGates;  // [batch, seqLen, numMemories]
    private Tensor<T>? _lastForgetGatesRaw;
    private Tensor<T>? _lastStates;       // [batch, seqLen+1, numMemories, numHeads, headDim, headDim]
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
    private Tensor<T>? _writeRouterWeightsGradient;
    private Tensor<T>? _writeRouterBiasGradient;
    private Tensor<T>? _readRouterWeightsGradient;
    private Tensor<T>? _readRouterBiasGradient;
    private Tensor<T>? _gateRouterWeightsGradient;
    private Tensor<T>? _gateRouterBiasGradient;
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

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _writeRouterWeights.Length + _writeRouterBias.Length +
        _readRouterWeights.Length + _readRouterBias.Length +
        _gateRouterWeights.Length + _gateRouterBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

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

        var q = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _queryWeights),
            _queryBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        var k = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _keyWeights),
            _keyBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        var v = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _valueWeights),
            _valueBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Router computations
        var writeLogits = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _writeRouterWeights),
            _writeRouterBias.Reshape(1, _numMemories)).Reshape(batchSize, seqLen, _numMemories);
        var writeWeights = SoftmaxLastDim(writeLogits, batchSize, seqLen, _numMemories);

        var readLogits = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _readRouterWeights),
            _readRouterBias.Reshape(1, _numMemories)).Reshape(batchSize, seqLen, _numMemories);
        var readWeights = SoftmaxLastDim(readLogits, batchSize, seqLen, _numMemories);

        var forgetGatesRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _gateRouterWeights),
            _gateRouterBias.Reshape(1, _numMemories)).Reshape(batchSize, seqLen, _numMemories);
        var forgetGates = Engine.Sigmoid(forgetGatesRaw);

        _lastWriteWeights = writeWeights;
        _lastReadWeights = readWeights;
        _lastForgetGates = forgetGates;
        _lastForgetGatesRaw = forgetGatesRaw;

        // Step 3: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 4: MoM recurrence
        var momOutput = MoMForward(q, k, v, writeWeights, readWeights, forgetGates, batchSize, seqLen);
        _lastMoMOutput = momOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(momOutput, gate);

        // Step 6: Output projection
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(gatedFlat, _outputProjectionWeights),
            _outputProjectionBias.Reshape(1, _modelDimension));
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
    /// Computes softmax along the last dimension.
    /// </summary>
    private Tensor<T> SoftmaxLastDim(Tensor<T> logits, int batchSize, int seqLen, int dim)
    {
        var result = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, dim });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Find max for numerical stability
                T maxVal = logits[new[] { bi, t, 0 }];
                for (int d = 1; d < dim; d++)
                {
                    T val = logits[new[] { bi, t, d }];
                    if (NumOps.GreaterThan(val, maxVal))
                        maxVal = val;
                }

                T sumExp = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                {
                    T expVal = NumOps.Exp(NumOps.Subtract(logits[new[] { bi, t, d }], maxVal));
                    result[new[] { bi, t, d }] = expVal;
                    sumExp = NumOps.Add(sumExp, expVal);
                }

                for (int d = 0; d < dim; d++)
                    result[new[] { bi, t, d }] = NumOps.Divide(result[new[] { bi, t, d }], sumExp);
            }
        }

        return result;
    }

    /// <summary>
    /// MoM forward: multi-memory state recurrence with routing.
    /// </summary>
    private Tensor<T> MoMForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> writeWeights, Tensor<T> readWeights, Tensor<T> forgetGates,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });

        // Memory states: [batch, numMemories, numHeads, headDim, headDim]
        var memStates = new T[batchSize, _numMemories, _numHeads, _headDimension, _headDimension];

        // Save all states for backward: [batch, seqLen+1, numMemories, numHeads, headDim, headDim]
        // This would be very large, so we only save the final states per timestep
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numMemories, _numHeads, _headDimension, _headDimension });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Extract key, value for this head
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        kHead[di] = NumOps.Multiply(k[new[] { bi, t, dimStart + di }], keyScale);
                        vHead[di] = v[new[] { bi, t, dimStart + di }];
                    }

                    // Write to each memory: S_m[t] = g_m * S_m[t-1] + w_m * v * k^T
                    for (int mi = 0; mi < _numMemories; mi++)
                    {
                        T wm = writeWeights[new[] { bi, t, mi }];
                        T gm = forgetGates[new[] { bi, t, mi }];

                        for (int di = 0; di < _headDimension; di++)
                        {
                            for (int dj = 0; dj < _headDimension; dj++)
                            {
                                T prevState = memStates[bi, mi, hi, di, dj];
                                T writeVal = NumOps.Multiply(wm,
                                    NumOps.Multiply(vHead[di], kHead[dj]));
                                memStates[bi, mi, hi, di, dj] = NumOps.Add(
                                    NumOps.Multiply(gm, prevState), writeVal);
                            }
                        }
                    }

                    // Read from all memories: o = sum_m r_m * S_m * q
                    var qHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                        qHead[di] = q[new[] { bi, t, dimStart + di }];

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;

                        for (int mi = 0; mi < _numMemories; mi++)
                        {
                            T rm = readWeights[new[] { bi, t, mi }];
                            T smq = NumOps.Zero;
                            for (int dj = 0; dj < _headDimension; dj++)
                                smq = NumOps.Add(smq,
                                    NumOps.Multiply(memStates[bi, mi, hi, di, dj], qHead[dj]));

                            oVal = NumOps.Add(oVal, NumOps.Multiply(rm, smq));
                        }

                        output[new[] { bi, t, flatDi }] = oVal;
                    }

                    // Save states for backward
                    for (int mi = 0; mi < _numMemories; mi++)
                        for (int di = 0; di < _headDimension; di++)
                            for (int dj = 0; dj < _headDimension; dj++)
                                allStates[new[] { bi, t + 1, mi, hi, di, dj }] = memStates[bi, mi, hi, di, dj];
                }
            }
        }

        _lastStates = allStates;
        return output;
    }

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
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCount);
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
        _lastStates = null;
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
