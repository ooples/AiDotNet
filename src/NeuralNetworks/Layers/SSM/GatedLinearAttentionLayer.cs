using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Gated Linear Attention (GLA) layer from Yang et al., 2023.
/// </summary>
/// <remarks>
/// <para>
/// GLA materializes the linear attention mechanism as a gated linear RNN with data-dependent
/// gating. It bridges the gap between linear attention (which can be computed as an RNN) and
/// traditional gated RNNs (which have data-dependent transitions), resulting in a model that
/// is both expressive and hardware-efficient.
/// </para>
/// <para>
/// The computation:
/// <code>
///   Q = x * W_q, K = x * W_k, V = x * W_v  // Query, Key, Value projections
///   G = sigmoid(x * W_g)                      // Data-dependent gate (per-head)
///
///   For each head h:
///     S_t = G_t * S_{t-1} + K_t^T * V_t      // Gated state update (KV outer product)
///     O_t = Q_t * S_t                          // Output from state
///
///   output = concat(O_1, ..., O_H) * W_o      // Multi-head output projection
/// </code>
/// The key insight is that the gate G controls how quickly the state forgets old information,
/// making the effective attention window data-dependent.
/// </para>
/// <para>
/// GLA supports a hardware-efficient chunked computation mode where the sequence is processed
/// in blocks, with intra-chunk computation parallelized as matrix multiplications.
/// </para>
/// <para><b>For Beginners:</b> GLA is a way to make attention fast by adding a "memory gate."
///
/// Standard attention: "Look at everything" -> O(n^2) cost
/// Linear attention: "Keep a running summary" -> O(n) cost, but forgets old info uniformly
/// GLA: "Keep a running summary with a gate that controls forgetting" -> O(n) cost, smart forgetting
///
/// The gate lets the model decide: "This information is important, keep it longer" or
/// "This is no longer relevant, forget it faster." This simple addition makes linear attention
/// much more competitive with standard attention.
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training", 2024.
/// https://arxiv.org/abs/2312.06635
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
internal partial class GatedLinearAttentionLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _keyDimension;

    // Q, K, V projections: [modelDim, numHeads * headDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Gate projection: [modelDim, numHeads * headDim]
    private Tensor<T> _gateWeights;
    private Tensor<T> _gateBias;

    // Output projection: [numHeads * headDim, modelDim]
    private Tensor<T> _outputWeights;
    private Tensor<T> _outputBias;

    // Cached values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastAttnOutput; // Pre-output-projection attention output
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _gateWeightsGradient;
    private Tensor<T>? _gateBiasGradient;
    private Tensor<T>? _outputWeightsGradient;
    private Tensor<T>? _outputBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

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
        _gateWeights.Length + _gateBias.Length +
        _outputWeights.Length + _outputBias.Length;

    /// <summary>
    /// Creates a new Gated Linear Attention layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head independently maintains its own state matrix.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public GatedLinearAttentionLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
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
        _keyDimension = _headDimension;

        int totalDim = numHeads * _headDimension;

        _queryWeights = new Tensor<T>([modelDimension, totalDim]);
        _keyWeights = new Tensor<T>([modelDimension, totalDim]);
        _valueWeights = new Tensor<T>([modelDimension, totalDim]);
        _gateWeights = new Tensor<T>([modelDimension, totalDim]);
        _gateBias = new Tensor<T>([totalDim]);
        _outputWeights = new Tensor<T>([totalDim, modelDimension]);
        _outputBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor(_queryWeights);
        InitializeTensor(_keyWeights);
        InitializeTensor(_valueWeights);
        InitializeTensor(_gateWeights);
        _gateBias.Fill(NumOps.Zero);
        InitializeTensor(_outputWeights);
        _outputBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
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

        int totalDim = _numHeads * _headDimension;

        // Project Q, K, V, G
        var input2D = input3D.Reshape(batchSize * seqLen, modelDim);
        var q = Engine.TensorMatMul(input2D, _queryWeights).Reshape(batchSize, seqLen, totalDim);
        var k = Engine.TensorMatMul(input2D, _keyWeights).Reshape(batchSize, seqLen, totalDim);
        var v = Engine.TensorMatMul(input2D, _valueWeights).Reshape(batchSize, seqLen, totalDim);

        var gFlat = Engine.TensorMatMul(input2D, _gateWeights);
        var gBias = _gateBias.Reshape(1, totalDim);
        gFlat = Engine.TensorBroadcastAdd(gFlat, gBias);
        var gate = Engine.Sigmoid(gFlat.Reshape(batchSize, seqLen, totalDim));

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;
        _lastGate = gate;

        // Gated linear attention recurrence per head
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, totalDim });

        for (int hi = 0; hi < _numHeads; hi++)
        {
            int dimStart = hi * _headDimension;

            // State matrix: [batch, headDim, keyDim] (KV outer product accumulator)
            var state = TensorAllocator.Rent<T>(new[] { batchSize, _headDimension, _keyDimension });

            for (int t = 0; t < seqLen; t++)
            {
                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Gate value for this head (use first element per head as scalar gate)
                    T gateVal = gate[new[] { bi, t, dimStart }];

                    // Gated state update: S = gate * S + K^T * V (outer product of K and V)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _keyDimension; ki++)
                        {
                            int flatK = dimStart + ki;
                            int flatD = dimStart + di;
                            T kVal = k[new[] { bi, t, flatK }];
                            T vVal = v[new[] { bi, t, flatD }];

                            T prevState = state[new[] { bi, di, ki }];
                            T kvOuter = NumOps.Multiply(kVal, vVal);
                            T newState = NumOps.Add(
                                NumOps.Multiply(gateVal, prevState), kvOuter);
                            state[new[] { bi, di, ki }] = newState;
                        }
                    }

                    // Output: O = Q * S → O[k] = sum_d Q[d] * S[d, k]
                    for (int ki = 0; ki < _keyDimension; ki++)
                    {
                        int flatK = dimStart + ki;
                        T sum = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                        {
                            T qVal = q[new[] { bi, t, dimStart + di }];
                            sum = NumOps.Add(sum, NumOps.Multiply(qVal, state[new[] { bi, di, ki }]));
                        }
                        output[new[] { bi, t, flatK }] = NumOps.Add(
                            output[new[] { bi, t, flatK }], sum);
                    }
                }
            }
        }

        _lastAttnOutput = output;

        // Output projection
        var outFlat = output.Reshape(batchSize * seqLen, totalDim);
        var outputFlat = Engine.TensorMatMul(outFlat, _outputWeights);
        var outBias2D = _outputBias.Reshape(1, _modelDimension);
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias2D);
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

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient is null || _keyWeightsGradient is null ||
            _valueWeightsGradient is null || _gateWeightsGradient is null ||
            _gateBiasGradient is null || _outputWeightsGradient is null ||
            _outputBiasGradient is null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient, negLR));
        _gateWeights = Engine.TensorAdd(_gateWeights, Engine.TensorMultiplyScalar(_gateWeightsGradient, negLR));
        _gateBias = Engine.TensorAdd(_gateBias, Engine.TensorMultiplyScalar(_gateBiasGradient, negLR));
        _outputWeights = Engine.TensorAdd(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGradient, negLR));
        _outputBias = Engine.TensorAdd(_outputBias, Engine.TensorMultiplyScalar(_outputBiasGradient, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);

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
        _gateWeights, _gateBias,
        _outputWeights, _outputBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient!.ToArray()),
            new Vector<T>(_keyWeightsGradient!.ToArray()),
            new Vector<T>(_valueWeightsGradient!.ToArray()),
            new Vector<T>(_gateWeightsGradient!.ToArray()),
            new Vector<T>(_gateBiasGradient!.ToArray()),
            new Vector<T>(_outputWeightsGradient?.ToArray() ?? new T[_outputWeights.Length]),
            new Vector<T>(_outputBiasGradient?.ToArray() ?? new T[_outputBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _gateWeightsGradient = null; _gateBiasGradient = null;
        _outputWeightsGradient = null; _outputBiasGradient = null;
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
        _lastAttnOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _gateWeightsGradient = null;
        _gateBiasGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets a copy of the query projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights.Clone();

    /// <summary>
    /// Gets a copy of the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputWeights() => _outputWeights.Clone();
}
