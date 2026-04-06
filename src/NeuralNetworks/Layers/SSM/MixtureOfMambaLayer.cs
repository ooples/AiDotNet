using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Mixture-of-Mamba layer from Jiang et al., 2025 (arXiv:2501.16295).
/// </summary>
/// <remarks>
/// <para>
/// Mixture-of-Mamba combines Mamba's selective state space model (SSM) with Mixture of Experts (MoE)
/// sparsity. Instead of a single monolithic SSM, the layer maintains multiple "expert" SSM blocks,
/// each with its own A, B, C, D parameters. A learned router selects the top-K experts for each
/// token, allowing different tokens to be processed by different specialized SSM pathways.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Input projection
///   2. Router: softmax(W_router * x) -> top-k selection of experts
///   3. For each active expert e:
///      - Project x through expert-specific B_e, C_e matrices
///      - Selective scan with expert-specific A_e, D_e:
///        h_t = A_e * h_{t-1} + B_e * x_t
///        y_t = C_e * h_t + D_e * x_t
///   4. Combine expert outputs weighted by router scores
///   5. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The key insight is that different types of sequential patterns benefit from different SSM
/// dynamics. By routing tokens to specialized experts, the model can learn distinct temporal
/// patterns without interference. For example, one expert might specialize in short-range
/// dependencies (fast-decaying A), while another handles long-range dependencies (slow-decaying A).
/// The sparse routing means only top-K experts are active per token, maintaining efficiency.
/// </para>
/// <para><b>For Beginners:</b> Mixture-of-Mamba is like having a team of specialized assistants
/// instead of one generalist.
///
/// Imagine you're processing a long document:
/// - You have 8 assistants (experts), each good at different things:
///   - Expert 1 might be great at tracking names and entities
///   - Expert 2 might excel at following numerical patterns
///   - Expert 3 might specialize in temporal/causal relationships
///   - etc.
/// - For each word, a "router" decides which 2 assistants (top-K=2) should handle it
/// - Each chosen assistant processes the word using their own specialized memory (SSM state)
/// - The final answer combines the two assistants' outputs, weighted by confidence
///
/// This is more efficient than one giant assistant because:
/// - Each expert can specialize in different patterns (divide and conquer)
/// - Only 2 out of 8 experts run per token (sparse computation saves resources)
/// - Different tokens automatically get routed to the most relevant experts
///
/// The "Mamba" part refers to the selective scan mechanism each expert uses, which is a
/// very efficient way to process sequences with learned, input-dependent state transitions.
/// </para>
/// <para>
/// <b>Reference:</b> Jiang et al., "Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Mixture of Experts", 2025.
/// https://arxiv.org/abs/2501.16295
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.MixtureOfExperts)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.Routing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class MixtureOfMambaLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numExperts;
    private readonly int _topK;
    private readonly int _stateDimension;

    // Router: [modelDim, numExperts]
    private Tensor<T> _routerWeights;
    private Tensor<T> _routerBias;

    // Per-expert SSM parameters
    // A (state transition): [numExperts, stateDim] — diagonal approximation
    private Tensor<T> _expertA;
    // B (input projection): [numExperts, stateDim, modelDim]
    private Tensor<T> _expertB;
    // C (output projection): [numExperts, modelDim, stateDim]
    private Tensor<T> _expertC;
    // D (skip connection): [numExperts, modelDim]
    private Tensor<T> _expertD;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastRouterLogits;
    private Tensor<T>? _lastRouterWeightsResult;
    private int[,]? _lastTopKIndices;
    private Tensor<T>? _lastExpertOutputs;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastMoEOutput;
    private Tensor<T>? _lastExpertStates;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _routerWeightsGradient;
    private Tensor<T>? _routerBiasGradient;
    private Tensor<T>? _expertAGradient;
    private Tensor<T>? _expertBGradient;
    private Tensor<T>? _expertCGradient;
    private Tensor<T>? _expertDGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Gets the model dimension.</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of experts.</summary>
    public int NumExperts => _numExperts;

    /// <summary>Gets the number of active experts per token (top-K).</summary>
    public int TopK => _topK;

    /// <summary>Gets the SSM state dimension per expert.</summary>
    public int StateDimension => _stateDimension;

    /// <inheritdoc />
    public override int ParameterCount =>
        _routerWeights.Length + _routerBias.Length +
        _expertA.Length + _expertB.Length + _expertC.Length + _expertD.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Mixture-of-Mamba layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numExperts">
    /// Total number of expert SSM blocks. Default: 8.
    /// <para><b>For Beginners:</b> More experts = more specialization but more parameters.
    /// Typical values are 4, 8, or 16.</para>
    /// </param>
    /// <param name="topK">
    /// Number of experts activated per token. Default: 2.
    /// <para><b>For Beginners:</b> Each token is only processed by topK experts, not all of them.
    /// This keeps computation sparse and efficient. Usually 1 or 2.</para>
    /// </param>
    /// <param name="stateDimension">
    /// Dimension of each expert's SSM hidden state. Default: 16.
    /// <para><b>For Beginners:</b> This is the size of the "memory" each expert maintains
    /// as it processes the sequence. Larger = more memory capacity but more computation.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public MixtureOfMambaLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numExperts = 8,
        int topK = 2,
        int stateDimension = 16,
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
        if (numExperts <= 0)
            throw new ArgumentException($"Number of experts ({numExperts}) must be positive.", nameof(numExperts));
        if (topK <= 0)
            throw new ArgumentException($"Top-K ({topK}) must be positive.", nameof(topK));
        if (topK > numExperts)
            throw new ArgumentException($"Top-K ({topK}) cannot exceed number of experts ({numExperts}).", nameof(topK));
        if (stateDimension <= 0)
            throw new ArgumentException($"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));

        _modelDimension = modelDimension;
        _numExperts = numExperts;
        _topK = topK;
        _stateDimension = stateDimension;

        _routerWeights = new Tensor<T>([modelDimension, numExperts]);
        _routerBias = new Tensor<T>([numExperts]);
        _expertA = new Tensor<T>([numExperts, stateDimension]);
        _expertB = new Tensor<T>([numExperts, stateDimension, modelDimension]);
        _expertC = new Tensor<T>([numExperts, modelDimension, stateDimension]);
        _expertD = new Tensor<T>([numExperts, modelDimension]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_routerWeights);
        _routerBias.Fill(NumOps.Zero);

        // Initialize A with negative values (exponential decay) in [-4, -1] range
        // This ensures stable state dynamics: exp(A) in [exp(-4), exp(-1)] ~ [0.018, 0.368]
        for (int e = 0; e < _numExperts; e++)
        {
            for (int s = 0; s < _stateDimension; s++)
            {
                double aVal = -1.0 - 3.0 * Random.NextDouble(); // Range [-4, -1]
                _expertA[new[] { e, s }] = NumOps.FromDouble(aVal);
            }
        }

        // Initialize B and C with Xavier/Glorot
        InitializeTensor3D(_expertB, _stateDimension, _modelDimension);
        InitializeTensor3D(_expertC, _modelDimension, _stateDimension);

        // Initialize D to 1 (identity skip connection)
        for (int e = 0; e < _numExperts; e++)
            for (int d = 0; d < _modelDimension; d++)
                _expertD[new[] { e, d }] = NumOps.One;

        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    private void InitializeTensor3D(Tensor<T> tensor, int fanIn, int fanOut)
    {
        InitializeLayerWeights(tensor, fanIn, fanOut);
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

        // Step 1: Router - compute expert selection per token
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var routerLogits = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _routerWeights),
            _routerBias.Reshape(1, _numExperts));
        _lastRouterLogits = routerLogits;

        // Softmax over experts for each token, then top-K selection
        int totalTokens = batchSize * seqLen;
        var routerWeightsResult = new Tensor<T>(new[] { totalTokens, _numExperts });
        var topKIndices = new int[totalTokens, _topK];
        ComputeTopKSoftmax(routerLogits, routerWeightsResult, topKIndices, totalTokens);
        _lastRouterWeightsResult = routerWeightsResult;
        _lastTopKIndices = topKIndices;

        // Step 2: Run selective scan for each active expert per token
        var expertOutputs = new Tensor<T>(new[] { totalTokens, _topK, _modelDimension });
        var expertStates = TensorAllocator.Rent<T>(new[] { batchSize, _numExperts, seqLen + 1, _stateDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int tokenIdx = bi * seqLen + t;

                for (int ki = 0; ki < _topK; ki++)
                {
                    int expertIdx = topKIndices[tokenIdx, ki];

                    // SSM scan: h_t = diag(exp(A)) * h_{t-1} + B * x_t
                    //           y_t = C * h_t + D * x_t
                    for (int si = 0; si < _stateDimension; si++)
                    {
                        // A is diagonal: h_t[s] = exp(A[e,s]) * h_{t-1}[s] + (B*x)[s]
                        T aVal = NumOps.Exp(_expertA[new[] { expertIdx, si }]);
                        T prevH = t > 0
                            ? expertStates[new[] { bi, expertIdx, t, si }]
                            : NumOps.Zero;

                        // B * x: sum over modelDim
                        T bx = NumOps.Zero;
                        for (int di = 0; di < _modelDimension; di++)
                        {
                            T xVal = input3D[new[] { bi, t, di }];
                            bx = NumOps.Add(bx, NumOps.Multiply(_expertB[new[] { expertIdx, si, di }], xVal));
                        }

                        T newH = NumOps.Add(NumOps.Multiply(aVal, prevH), bx);
                        expertStates[new[] { bi, expertIdx, t + 1, si }] = newH;
                    }

                    // Output: y = C * h + D * x
                    for (int di = 0; di < _modelDimension; di++)
                    {
                        // C * h
                        T ch = NumOps.Zero;
                        for (int si = 0; si < _stateDimension; si++)
                        {
                            T hVal = expertStates[new[] { bi, expertIdx, t + 1, si }];
                            ch = NumOps.Add(ch, NumOps.Multiply(_expertC[new[] { expertIdx, di, si }], hVal));
                        }

                        // D * x (element-wise skip)
                        T dx = NumOps.Multiply(_expertD[new[] { expertIdx, di }], input3D[new[] { bi, t, di }]);

                        expertOutputs[new[] { tokenIdx, ki, di }] = NumOps.Add(ch, dx);
                    }
                }
            }
        }
        _lastExpertOutputs = expertOutputs;
        _lastExpertStates = expertStates;

        // Step 3: Combine expert outputs weighted by router scores
        var moeOutput = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        for (int tokenIdx = 0; tokenIdx < totalTokens; tokenIdx++)
        {
            int bi = tokenIdx / seqLen;
            int t = tokenIdx % seqLen;

            for (int di = 0; di < _modelDimension; di++)
            {
                T combined = NumOps.Zero;
                for (int ki = 0; ki < _topK; ki++)
                {
                    int expertIdx = topKIndices[tokenIdx, ki];
                    T weight = routerWeightsResult[new[] { tokenIdx, expertIdx }];
                    T expertOut = expertOutputs[new[] { tokenIdx, ki, di }];
                    combined = NumOps.Add(combined, NumOps.Multiply(weight, expertOut));
                }
                moeOutput[new[] { bi, t, di }] = combined;
            }
        }
        _lastMoEOutput = moeOutput;

        // Step 4: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(moeOutput, gate);

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
    /// Computes softmax over experts and selects top-K for each token.
    /// </summary>
    private void ComputeTopKSoftmax(
        Tensor<T> logits, Tensor<T> weights, int[,] topKIndices, int totalTokens)
    {
        for (int tokenIdx = 0; tokenIdx < totalTokens; tokenIdx++)
        {
            // Find max for numerical stability
            T maxLogit = NumOps.FromDouble(-1e9);
            for (int e = 0; e < _numExperts; e++)
            {
                T val = logits[new[] { tokenIdx, e }];
                double valD = NumOps.ToDouble(val);
                double maxD = NumOps.ToDouble(maxLogit);
                if (valD > maxD) maxLogit = val;
            }

            // Compute exp and sum
            var expValues = new T[_numExperts];
            T sumExp = NumOps.Zero;
            for (int e = 0; e < _numExperts; e++)
            {
                expValues[e] = NumOps.Exp(NumOps.Subtract(logits[new[] { tokenIdx, e }], maxLogit));
                sumExp = NumOps.Add(sumExp, expValues[e]);
            }

            T sumExpSafe = NumOps.Add(sumExp, NumOps.FromDouble(1e-10));

            // Softmax probabilities
            var probs = new T[_numExperts];
            for (int e = 0; e < _numExperts; e++)
                probs[e] = NumOps.Divide(expValues[e], sumExpSafe);

            // Top-K selection by finding K largest probabilities
            var selected = new bool[_numExperts];
            for (int ki = 0; ki < _topK; ki++)
            {
                int bestIdx = -1;
                double bestVal = -1.0;
                for (int e = 0; e < _numExperts; e++)
                {
                    if (!selected[e])
                    {
                        double probVal = NumOps.ToDouble(probs[e]);
                        if (probVal > bestVal)
                        {
                            bestVal = probVal;
                            bestIdx = e;
                        }
                    }
                }
                topKIndices[tokenIdx, ki] = bestIdx;
                selected[bestIdx] = true;
            }

            // Renormalize selected experts' weights
            T topKSum = NumOps.Zero;
            for (int ki = 0; ki < _topK; ki++)
            {
                int expertIdx = topKIndices[tokenIdx, ki];
                topKSum = NumOps.Add(topKSum, probs[expertIdx]);
            }

            T topKSumSafe = NumOps.Add(topKSum, NumOps.FromDouble(1e-10));
            for (int e = 0; e < _numExperts; e++)
                weights[new[] { tokenIdx, e }] = NumOps.Zero;
            for (int ki = 0; ki < _topK; ki++)
            {
                int expertIdx = topKIndices[tokenIdx, ki];
                weights[new[] { tokenIdx, expertIdx }] = NumOps.Divide(probs[expertIdx], topKSumSafe);
            }
        }
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
        if (_routerWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _routerWeights = Engine.TensorAdd(_routerWeights, Engine.TensorMultiplyScalar(_routerWeightsGradient, negLR));
        _routerBias = Engine.TensorAdd(_routerBias, Engine.TensorMultiplyScalar(_routerBiasGradient!, negLR));
        _expertA = Engine.TensorAdd(_expertA, Engine.TensorMultiplyScalar(_expertAGradient!, negLR));
        _expertB = Engine.TensorAdd(_expertB, Engine.TensorMultiplyScalar(_expertBGradient!, negLR));
        _expertC = Engine.TensorAdd(_expertC, Engine.TensorMultiplyScalar(_expertCGradient!, negLR));
        _expertD = Engine.TensorAdd(_expertD, Engine.TensorMultiplyScalar(_expertDGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_routerWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_routerBias, PersistentTensorRole.Biases);
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
        _routerWeights, _routerBias,
        _expertA, _expertB, _expertC, _expertD,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_routerWeightsGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_routerWeightsGradient!.ToArray()),
            new Vector<T>(_routerBiasGradient!.ToArray()),
            new Vector<T>(_expertAGradient!.ToArray()),
            new Vector<T>(_expertBGradient!.ToArray()),
            new Vector<T>(_expertCGradient!.ToArray()),
            new Vector<T>(_expertDGradient!.ToArray()),
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _routerWeightsGradient = null; _routerBiasGradient = null; _expertAGradient = null; _expertBGradient = null; _expertCGradient = null; _expertDGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastRouterLogits = null;
        _lastRouterWeightsResult = null;
        _lastTopKIndices = null;
        _lastExpertOutputs = null;
        _lastGate = null;
        _lastGateRaw = null;
        _lastMoEOutput = null;
        _lastExpertStates = null;
        _originalInputShape = null;
        _routerWeightsGradient = null;
        _routerBiasGradient = null;
        _expertAGradient = null;
        _expertBGradient = null;
        _expertCGradient = null;
        _expertDGradient = null;
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
        metadata["NumExperts"] = _numExperts.ToString();
        metadata["TopK"] = _topK.ToString();
        metadata["StateDimension"] = _stateDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the router weights for external inspection.
    /// </summary>
    public Tensor<T> GetRouterWeights() => _routerWeights;

    /// <summary>
    /// Gets the expert A (state transition) parameters for external inspection.
    /// </summary>
    public Tensor<T> GetExpertA() => _expertA;
}
