using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class GatedLinearAttentionLayer<T> : LayerBase<T>
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
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
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
        var output = new Tensor<T>(new[] { batchSize, seqLen, totalDim });

        for (int hi = 0; hi < _numHeads; hi++)
        {
            int dimStart = hi * _headDimension;

            // State matrix: [batch, headDim, keyDim] (KV outer product accumulator)
            var state = new Tensor<T>(new[] { batchSize, _headDimension, _keyDimension });

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

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastAttnOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null || _lastGate == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];
        int totalDim = _numHeads * _headDimension;

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Use cached attention output for output weight gradient
        var attnOutFlat = _lastAttnOutput.Reshape(batchSize * seqLen, totalDim);
        _outputWeightsGradient = Engine.TensorMatMul(attnOutFlat.Transpose([1, 0]), gradFlat);

        // Gradient through output projection to attention output
        var dAttn = Engine.TensorMatMul(gradFlat, _outputWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, totalDim);

        // Compute Q, K, V, gate gradients through the GLA recurrence
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, totalDim });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, totalDim });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, totalDim });
        var dGate = new Tensor<T>(new[] { batchSize, seqLen, totalDim });

        for (int hi = 0; hi < _numHeads; hi++)
        {
            int dimStart = hi * _headDimension;

            // Recompute forward states for this head (needed for backward)
            var states = new Tensor<T>[seqLen + 1];
            states[0] = new Tensor<T>(new[] { batchSize, _headDimension, _keyDimension });

            for (int t = 0; t < seqLen; t++)
            {
                states[t + 1] = new Tensor<T>(new[] { batchSize, _headDimension, _keyDimension });
                for (int bi = 0; bi < batchSize; bi++)
                {
                    T gateVal = _lastGate[new[] { bi, t, dimStart }];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _keyDimension; ki++)
                        {
                            int flatK = dimStart + ki;
                            int flatD = dimStart + di;
                            T kVal = _lastKey[new[] { bi, t, flatK }];
                            T vVal = _lastValue[new[] { bi, t, flatD }];
                            T prevState = states[t][new[] { bi, di, ki }];
                            states[t + 1][new[] { bi, di, ki }] = NumOps.Add(
                                NumOps.Multiply(gateVal, prevState),
                                NumOps.Multiply(kVal, vVal));
                        }
                    }
                }
            }

            // Backward through recurrence (reverse time)
            var dState = new Tensor<T>(new[] { batchSize, _headDimension, _keyDimension });

            for (int t = seqLen - 1; t >= 0; t--)
            {
                for (int bi = 0; bi < batchSize; bi++)
                {
                    T gateVal = _lastGate[new[] { bi, t, dimStart }];

                    // dO/dQ: dQ[d] += sum_k dAttn[k] * S[d,k]
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;
                        T dqSum = NumOps.Zero;
                        for (int ki = 0; ki < _keyDimension; ki++)
                        {
                            int flatK = dimStart + ki;
                            T dOut = dAttn[new[] { bi, t, flatK }];
                            T stateVal = states[t + 1][new[] { bi, di, ki }];
                            dqSum = NumOps.Add(dqSum, NumOps.Multiply(dOut, stateVal));
                        }
                        dQ[new[] { bi, t, flatD }] = NumOps.Add(dQ[new[] { bi, t, flatD }], dqSum);
                    }

                    // dO/dS: dS[d,k] += Q[d] * dAttn[k]
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T qVal = _lastQuery[new[] { bi, t, dimStart + di }];
                        for (int ki = 0; ki < _keyDimension; ki++)
                        {
                            int flatK = dimStart + ki;
                            T dOut = dAttn[new[] { bi, t, flatK }];
                            dState[new[] { bi, di, ki }] = NumOps.Add(
                                dState[new[] { bi, di, ki }],
                                NumOps.Multiply(qVal, dOut));
                        }
                    }

                    // dS/dGate: dGate += sum_{d,k} dState[d,k] * S_{t-1}[d,k]
                    T dGateSum = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _keyDimension; ki++)
                        {
                            T ds = dState[new[] { bi, di, ki }];
                            T prevState = states[t][new[] { bi, di, ki }];
                            dGateSum = NumOps.Add(dGateSum, NumOps.Multiply(ds, prevState));
                        }
                    }
                    // Sigmoid derivative: g * (1 - g)
                    T g = gateVal;
                    T sigmoidDeriv = NumOps.Multiply(g, NumOps.Subtract(NumOps.One, g));
                    T dGateVal = NumOps.Multiply(dGateSum, sigmoidDeriv);
                    dGate[new[] { bi, t, dimStart }] = NumOps.Add(
                        dGate[new[] { bi, t, dimStart }], dGateVal);

                    // dS/dK and dS/dV: dK[k] += sum_d dState[d,k] * V[d], dV[d] += sum_k dState[d,k] * K[k]
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;
                        T vVal = _lastValue[new[] { bi, t, flatD }];
                        for (int ki = 0; ki < _keyDimension; ki++)
                        {
                            int flatK = dimStart + ki;
                            T ds = dState[new[] { bi, di, ki }];
                            T kVal = _lastKey[new[] { bi, t, flatK }];
                            dK[new[] { bi, t, flatK }] = NumOps.Add(
                                dK[new[] { bi, t, flatK }], NumOps.Multiply(ds, vVal));
                            dV[new[] { bi, t, flatD }] = NumOps.Add(
                                dV[new[] { bi, t, flatD }], NumOps.Multiply(ds, kVal));
                        }
                    }

                    // Propagate dState backward through gate: dState_{t-1} = gate * dState_t
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _keyDimension; ki++)
                        {
                            dState[new[] { bi, di, ki }] = NumOps.Multiply(
                                gateVal, dState[new[] { bi, di, ki }]);
                        }
                    }
                }
            }
        }

        // Compute weight gradients: W_grad = input^T * dProjection
        var input2D = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dQFlat = dQ.Reshape(batchSize * seqLen, totalDim);
        var dKFlat = dK.Reshape(batchSize * seqLen, totalDim);
        var dVFlat = dV.Reshape(batchSize * seqLen, totalDim);
        var dGateFlat = dGate.Reshape(batchSize * seqLen, totalDim);

        _queryWeightsGradient = Engine.TensorMatMul(input2D.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(input2D.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(input2D.Transpose([1, 0]), dVFlat);
        _gateWeightsGradient = Engine.TensorMatMul(input2D.Transpose([1, 0]), dGateFlat);
        _gateBiasGradient = Engine.ReduceSum(dGate, new int[] { 0, 1 });

        // Input gradient: sum of all projection gradients flowing back
        var inputGradFlat = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        inputGradFlat = Engine.TensorAdd(inputGradFlat,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        inputGradFlat = Engine.TensorAdd(inputGradFlat,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        inputGradFlat = Engine.TensorAdd(inputGradFlat,
            Engine.TensorMatMul(dGateFlat, _gateWeights.Transpose([1, 0])));
        var inputGrad3D = inputGradFlat.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return inputGrad3D.Reshape(_originalInputShape);

        return inputGrad3D;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _gateWeights = Engine.TensorAdd(_gateWeights, Engine.TensorMultiplyScalar(_gateWeightsGradient!, negLR));
        _gateBias = Engine.TensorAdd(_gateBias, Engine.TensorMultiplyScalar(_gateBiasGradient!, negLR));
        _outputWeights = Engine.TensorAdd(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGradient!, negLR));
        _outputBias = Engine.TensorAdd(_outputBias, Engine.TensorMultiplyScalar(_outputBiasGradient!, negLR));
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _modelDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "x_t");
        var qWeightsNode = TensorOperations<T>.Variable(_queryWeights, "W_q");
        var outWeightsNode = TensorOperations<T>.Variable(_outputWeights, "W_out");
        var outBiasNode = TensorOperations<T>.Variable(_outputBias, "b_out");

        inputNodes.Add(xNode);
        inputNodes.Add(qWeightsNode);
        inputNodes.Add(outWeightsNode);
        inputNodes.Add(outBiasNode);

        var qWeightsT = TensorOperations<T>.Transpose(qWeightsNode);
        var query = TensorOperations<T>.MatrixMultiply(xNode, qWeightsT);
        var outWeightsT = TensorOperations<T>.Transpose(outWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(query, outWeightsT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outBiasNode);

        return outputWithBias;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the query projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputWeights() => _outputWeights;
}
