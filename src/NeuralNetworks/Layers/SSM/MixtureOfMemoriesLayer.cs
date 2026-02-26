using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class MixtureOfMemoriesLayer<T> : LayerBase<T>
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
        var result = new Tensor<T>(new[] { batchSize, seqLen, dim });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Find max for numerical stability
                T maxVal = logits[new[] { bi, t, 0 }];
                for (int d = 1; d < dim; d++)
                {
                    T val = logits[new[] { bi, t, d }];
                    if (NumOps.ToDouble(val) > NumOps.ToDouble(maxVal))
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
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

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

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var lastInput = _lastInput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutput = _lastOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastQuery = _lastQuery ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastKey = _lastKey ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastValue = _lastValue ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGate = _lastGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGateRaw = _lastGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastMoMOutput = _lastMoMOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastWriteWeights = _lastWriteWeights ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastReadWeights = _lastReadWeights ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastForgetGates = _lastForgetGates ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastForgetGatesRaw = _lastForgetGatesRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastStates = _lastStates ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = lastInput.Shape[0];
        int seqLen = lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(lastOutput, grad3D);

        // Initialize gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryBiasGradient = new Tensor<T>([_modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyBiasGradient = new Tensor<T>([_modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueBiasGradient = new Tensor<T>([_modelDimension]);
        _writeRouterWeightsGradient = new Tensor<T>([_modelDimension, _numMemories]);
        _writeRouterBiasGradient = new Tensor<T>([_numMemories]);
        _readRouterWeightsGradient = new Tensor<T>([_modelDimension, _numMemories]);
        _readRouterBiasGradient = new Tensor<T>([_numMemories]);
        _gateRouterWeightsGradient = new Tensor<T>([_modelDimension, _numMemories]);
        _gateRouterBiasGradient = new Tensor<T>([_numMemories]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = new Tensor<T>([_modelDimension]);

        // Step 6 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var gatedFlat = Engine.TensorMultiply(lastMoMOutput, lastGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 5 backward: gating
        var dMoMOut = Engine.TensorMultiply(dGated, lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, lastMoMOutput);
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(lastGateRaw));

        var inputFlat = lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 4 backward: MoM recurrence
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dWriteWeights = new Tensor<T>(new[] { batchSize, seqLen, _numMemories });
        var dReadWeights = new Tensor<T>(new[] { batchSize, seqLen, _numMemories });
        var dForgetGates = new Tensor<T>(new[] { batchSize, seqLen, _numMemories });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // State gradients: dS[batch, numMemories, numHeads, headDim, headDim]
        var dState = new T[batchSize, _numMemories, _numHeads, _headDimension, _headDimension];

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    var qHead = new T[_headDimension];
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        qHead[di] = lastQuery[new[] { bi, t, dimStart + di }];
                        kHead[di] = NumOps.Multiply(lastKey[new[] { bi, t, dimStart + di }], keyScale);
                        vHead[di] = lastValue[new[] { bi, t, dimStart + di }];
                    }

                    // Read backward: o = sum_m r_m * S_m * q
                    for (int mi = 0; mi < _numMemories; mi++)
                    {
                        T rm = lastReadWeights[new[] { bi, t, mi }];

                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            T dO = dMoMOut[new[] { bi, t, flatDi }];

                            // dS_m[di,dj] += r_m * dO[di] * q[dj]
                            for (int dj = 0; dj < _headDimension; dj++)
                            {
                                dState[bi, mi, hi, di, dj] = NumOps.Add(
                                    dState[bi, mi, hi, di, dj],
                                    NumOps.Multiply(rm, NumOps.Multiply(dO, qHead[dj])));
                            }

                            // dQ[dj] += r_m * S_m[di,dj] * dO[di]
                            for (int dj = 0; dj < _headDimension; dj++)
                            {
                                int flatDj = dimStart + dj;
                                T sVal = lastStates[new[] { bi, t + 1, mi, hi, di, dj }];
                                dQ[new[] { bi, t, flatDj }] = NumOps.Add(
                                    dQ[new[] { bi, t, flatDj }],
                                    NumOps.Multiply(rm, NumOps.Multiply(sVal, dO)));
                            }

                            // dReadWeights[mi] += S_m * q . dO
                            T smq = NumOps.Zero;
                            for (int dj = 0; dj < _headDimension; dj++)
                                smq = NumOps.Add(smq,
                                    NumOps.Multiply(lastStates[new[] { bi, t + 1, mi, hi, di, dj }], qHead[dj]));
                            dReadWeights[new[] { bi, t, mi }] = NumOps.Add(
                                dReadWeights[new[] { bi, t, mi }],
                                NumOps.Multiply(dO, smq));
                        }
                    }

                    // Write backward: S_m[t] = g_m * S_m[t-1] + w_m * v * k^T
                    for (int mi = 0; mi < _numMemories; mi++)
                    {
                        T wm = lastWriteWeights[new[] { bi, t, mi }];
                        T gm = lastForgetGates[new[] { bi, t, mi }];

                        // dForgetGate[mi] += sum(dS_m[di,dj] * S_m[t-1][di,dj])
                        T dGateAccum = NumOps.Zero;
                        // dWriteWeight[mi] += sum(dS_m[di,dj] * v[di]*k[dj])
                        T dWriteAccum = NumOps.Zero;

                        for (int di = 0; di < _headDimension; di++)
                        {
                            for (int dj = 0; dj < _headDimension; dj++)
                            {
                                T dS = dState[bi, mi, hi, di, dj];
                                T sPrev = lastStates[new[] { bi, t, mi, hi, di, dj }];

                                dGateAccum = NumOps.Add(dGateAccum,
                                    NumOps.Multiply(dS, sPrev));
                                dWriteAccum = NumOps.Add(dWriteAccum,
                                    NumOps.Multiply(dS, NumOps.Multiply(vHead[di], kHead[dj])));

                                // dV[di] += dS[di,dj] * w_m * k[dj]
                                dV[new[] { bi, t, dimStart + di }] = NumOps.Add(
                                    dV[new[] { bi, t, dimStart + di }],
                                    NumOps.Multiply(dS, NumOps.Multiply(wm, kHead[dj])));

                                // dK[dj] += dS[di,dj] * w_m * v[di] * keyScale
                                dK[new[] { bi, t, dimStart + dj }] = NumOps.Add(
                                    dK[new[] { bi, t, dimStart + dj }],
                                    NumOps.Multiply(dS,
                                        NumOps.Multiply(wm,
                                            NumOps.Multiply(vHead[di], keyScale))));

                                // Propagate dS to previous timestep: dS_prev += g_m * dS
                                dState[bi, mi, hi, di, dj] = NumOps.Multiply(gm, dS);
                            }
                        }

                        dForgetGates[new[] { bi, t, mi }] = NumOps.Add(
                            dForgetGates[new[] { bi, t, mi }], dGateAccum);
                        dWriteWeights[new[] { bi, t, mi }] = NumOps.Add(
                            dWriteWeights[new[] { bi, t, mi }], dWriteAccum);
                    }
                }
            }
        }

        // Router gradient backward (softmax for write/read, sigmoid for forget)
        // Write: softmax backward
        var dWriteLogits = SoftmaxBackward(dWriteWeights, lastWriteWeights, batchSize, seqLen, _numMemories);
        // Read: softmax backward
        var dReadLogits = SoftmaxBackward(dReadWeights, lastReadWeights, batchSize, seqLen, _numMemories);
        // Forget: sigmoid backward: dLogits = dGates * sigmoid * (1 - sigmoid)
        var dForgetLogits = new Tensor<T>(new[] { batchSize, seqLen, _numMemories });
        for (int bi = 0; bi < batchSize; bi++)
            for (int t = 0; t < seqLen; t++)
                for (int mi = 0; mi < _numMemories; mi++)
                {
                    T sig = lastForgetGates[new[] { bi, t, mi }];
                    T sigDeriv = NumOps.Multiply(sig, NumOps.Subtract(NumOps.One, sig));
                    dForgetLogits[new[] { bi, t, mi }] = NumOps.Multiply(
                        dForgetGates[new[] { bi, t, mi }], sigDeriv);
                }

        // Router weight gradients
        var dWriteFlat = dWriteLogits.Reshape(batchSize * seqLen, _numMemories);
        var dReadFlat = dReadLogits.Reshape(batchSize * seqLen, _numMemories);
        var dForgetFlat = dForgetLogits.Reshape(batchSize * seqLen, _numMemories);

        _writeRouterWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dWriteFlat);
        _writeRouterBiasGradient = Engine.ReduceSum(dWriteLogits, new int[] { 0, 1 });
        _readRouterWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dReadFlat);
        _readRouterBiasGradient = Engine.ReduceSum(dReadLogits, new int[] { 0, 1 });
        _gateRouterWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dForgetFlat);
        _gateRouterBiasGradient = Engine.ReduceSum(dForgetLogits, new int[] { 0, 1 });

        // Q, K, V projection gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _queryBiasGradient = Engine.ReduceSum(dQ, new int[] { 0, 1 });
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _keyBiasGradient = Engine.ReduceSum(dK, new int[] { 0, 1 });
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);
        _valueBiasGradient = Engine.ReduceSum(dV, new int[] { 0, 1 });

        // Input gradient from all paths
        var dInputFromProj = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInputFromProj = Engine.TensorAdd(dInputFromProj,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dInputFromProj = Engine.TensorAdd(dInputFromProj,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));

        // Router input gradients
        var dInputFromRouters = Engine.TensorMatMul(dWriteFlat, _writeRouterWeights.Transpose([1, 0]));
        dInputFromRouters = Engine.TensorAdd(dInputFromRouters,
            Engine.TensorMatMul(dReadFlat, _readRouterWeights.Transpose([1, 0])));
        dInputFromRouters = Engine.TensorAdd(dInputFromRouters,
            Engine.TensorMatMul(dForgetFlat, _gateRouterWeights.Transpose([1, 0])));

        var dInputTotal = Engine.TensorAdd(dInputFromProj, dInputFromGate);
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromRouters);
        var dInput3D = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    /// <summary>
    /// Backward pass for softmax: dLogits[i] = softmax[i] * (dOutput[i] - sum_j(softmax[j]*dOutput[j]))
    /// </summary>
    private Tensor<T> SoftmaxBackward(
        Tensor<T> dOutput, Tensor<T> softmaxOutput,
        int batchSize, int seqLen, int dim)
    {
        var dLogits = new Tensor<T>(new[] { batchSize, seqLen, dim });

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
        var ones = new Tensor<T>(x.Shape);
        for (int i = 0; i < ones.Length; i++) ones[i] = NumOps.One;
        var oneMinusSig = Engine.TensorSubtract(ones, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAdd(ones, xTimesOneMinusSig);
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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
