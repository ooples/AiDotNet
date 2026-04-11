using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the DeltaFormer layer from "An Associative Memory Perspective on Transformers and DeltaNet"
/// (Li and Papailiopoulos, 2025, arXiv:2505.19488).
/// </summary>
/// <remarks>
/// <para>
/// DeltaFormer views transformers through an associative memory lens, proposing a hybrid architecture
/// that alternates between standard softmax attention layers and delta rule layers. The attention layers
/// handle retrieval of stored associations, while the delta rule layers handle memory consolidation by
/// writing only the correction needed to update the fast weight matrix.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Input projection to Q, K, V for both attention and delta rule paths
///   2. For attention steps (even layers):
///      output = softmax(Q * K^T / sqrt(d)) * V — standard scaled dot-product attention
///   3. For delta rule steps (odd layers):
///      S_t = S_{t-1} + (v_t - S_{t-1} * k_t) * k_t^T — delta update for consolidation
///      output_t = S_t * q_t
///   4. The useDeltaRule flag selects which mode this layer operates in
///   5. Output gating and projection
/// </code>
/// </para>
/// <para>
/// The key insight is that attention and delta rule are complementary: attention is excellent at
/// one-shot retrieval (given a query, find the best match in context), while the delta rule excels at
/// memory consolidation (incrementally building a reusable association table). Alternating them gets
/// the best of both worlds: consolidated memories that are efficiently retrievable.
/// </para>
/// <para><b>For Beginners:</b> DeltaFormer combines two different ways of processing information,
/// alternating between them like two specialized workers on an assembly line.
///
/// Imagine studying for an exam:
/// - The "delta rule" worker is the note-taker: they read through material and update their notes,
///   only writing down what's NEW or DIFFERENT from what they already have. This is the "delta" —
///   the correction needed to update existing knowledge.
/// - The "attention" worker is the test-taker: when asked a question, they search through all
///   available information to find the best answer.
///
/// By alternating these two operations:
/// 1. Delta rule layers consolidate information into compact, reusable memories
/// 2. Attention layers retrieve from those consolidated memories efficiently
///
/// This is more effective than using either approach alone. Pure attention has no persistent memory
/// between queries; pure delta rule has less flexible retrieval.
/// </para>
/// <para>
/// <b>Reference:</b> Li and Papailiopoulos, "An Associative Memory Perspective on Transformers and DeltaNet", 2025.
/// https://arxiv.org/abs/2505.19488
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class DeltaFormerLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly bool _useDeltaRule;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

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
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastAttentionWeights;
    private Tensor<T>? _lastStates;
    private Tensor<T>? _lastMechanismOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Gets the model dimension.</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of attention heads.</summary>
    public int NumHeads => _numHeads;

    /// <summary>Gets the dimension per head.</summary>
    public int HeadDimension => _headDimension;

    /// <summary>Gets whether this layer uses the delta rule (true) or standard attention (false).</summary>
    public bool UseDeltaRule => _useDeltaRule;

    /// <inheritdoc />
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new DeltaFormer layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own independent attention or delta rule state,
    /// allowing the model to attend to different aspects of the input simultaneously.</para>
    /// </param>
    /// <param name="useDeltaRule">
    /// If true, this layer uses the delta rule for memory consolidation.
    /// If false, this layer uses standard softmax attention for retrieval.
    /// Default: true.
    /// <para><b>For Beginners:</b> In a DeltaFormer model, you alternate layers:
    /// layer 0 = attention, layer 1 = delta rule, layer 2 = attention, etc.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public DeltaFormerLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        bool useDeltaRule = true,
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

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _useDeltaRule = useDeltaRule;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
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

        // Step 2: Output gate
        var gateRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Sigmoid(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 3: Either delta rule or standard attention
        Tensor<T> mechanismOutput;
        if (_useDeltaRule)
        {
            mechanismOutput = DeltaRuleForward(q, k, v, batchSize, seqLen);
        }
        else
        {
            mechanismOutput = SoftmaxAttentionForward(q, k, v, batchSize, seqLen);
        }
        _lastMechanismOutput = mechanismOutput;

        // Step 4: Gated output
        var gatedOutput = Engine.TensorMultiply(mechanismOutput, gate);

        // Step 5: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias);
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
    /// Standard softmax attention: softmax(Q*K^T / sqrt(d)) * V.
    /// Used for the retrieval steps in the DeltaFormer architecture.
    /// </summary>
    private Tensor<T> SoftmaxAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Store attention weights for backward pass
        _lastAttentionWeights = TensorAllocator.Rent<T>(new[] { batchSize, _numHeads, seqLen, seqLen });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                // Compute attention scores: Q * K^T / sqrt(d)
                for (int ti = 0; ti < seqLen; ti++)
                {
                    // Find max for numerical stability
                    T maxScore = NumOps.FromDouble(-1e9);
                    var scores = new T[seqLen];

                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T dot = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(q[new[] { bi, ti, flatDi }], k[new[] { bi, tj, flatDi }]));
                        }
                        scores[tj] = NumOps.Multiply(dot, scale);
                        double scoreVal = NumOps.ToDouble(scores[tj]);
                        double maxVal = NumOps.ToDouble(maxScore);
                        if (scoreVal > maxVal)
                            maxScore = scores[tj];
                    }

                    // Softmax
                    T sumExp = NumOps.Zero;
                    var expScores = new T[seqLen];
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        expScores[tj] = NumOps.Exp(NumOps.Subtract(scores[tj], maxScore));
                        sumExp = NumOps.Add(sumExp, expScores[tj]);
                    }

                    T sumExpSafe = NumOps.Add(sumExp, NumOps.FromDouble(1e-10));
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T weight = NumOps.Divide(expScores[tj], sumExpSafe);
                        _lastAttentionWeights[new[] { bi, hi, ti, tj }] = weight;
                    }

                    // Weighted sum of values
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int tj = 0; tj < seqLen; tj++)
                        {
                            T weight = _lastAttentionWeights[new[] { bi, hi, ti, tj }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(weight, v[new[] { bi, tj, flatDi }]));
                        }
                        output[new[] { bi, ti, flatDi }] = oVal;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Delta rule forward: S_t = S_{t-1} + (v_t - S_{t-1} * k_t) * k_t^T.
    /// Used for the memory consolidation steps in the DeltaFormer architecture.
    /// </summary>
    private Tensor<T> DeltaRuleForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });

        // State matrix per head: [batch, numHeads, headDim, headDim]
        var state = TensorAllocator.Rent<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allStates = TensorAllocator.Rent<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Retrieve current state's prediction: S * k
                    var sK = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        sK[di] = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);
                            sK[di] = NumOps.Add(sK[di],
                                NumOps.Multiply(state[new[] { bi, hi, di, ki }], kVal));
                        }
                    }

                    // Delta: v - S*k (the correction term)
                    var delta = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        delta[di] = NumOps.Subtract(v[new[] { bi, t, flatDi }], sK[di]);
                    }

                    // State update: S = S + delta * k^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);
                            T prevS = state[new[] { bi, hi, di, ki }];
                            T update = NumOps.Multiply(delta[di], kVal);
                            state[new[] { bi, hi, di, ki }] = NumOps.Add(prevS, update);
                        }
                    }

                    // Output: o = S * q
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(state[new[] { bi, hi, di, ki }], q[new[] { bi, t, flatKi }]));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }
                }
            }

            // Save state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int hi2 = 0; hi2 < _numHeads; hi2++)
                    for (int di = 0; di < _headDimension; di++)
                        for (int ki = 0; ki < _headDimension; ki++)
                            allStates[new[] { bi, t + 1, hi2, di, ki }] = state[new[] { bi, hi2, di, ki }];
        }

        _lastStates = allStates;
        return output;
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
            _outputGateWeightsGradient == null || _outputGateBiasGradient == null ||
            _outputProjectionWeightsGradient == null || _outputProjectionBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
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
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null;
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
        _lastAttentionWeights = null;
        _lastStates = null;
        _lastMechanismOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
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
        metadata["UseDeltaRule"] = _useDeltaRule.ToString();
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
}
