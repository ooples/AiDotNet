using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class MixtureOfMambaLayer<T> : LayerBase<T>
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
    }

    private void InitializeTensor3D(Tensor<T> tensor, int fanIn, int fanOut)
    {
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
        var expertStates = new Tensor<T>(new[] { batchSize, _numExperts, seqLen + 1, _stateDimension });

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
        var moeOutput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
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

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var lastInput = _lastInput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutput = _lastOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastRouterWeightsResult = _lastRouterWeightsResult ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastTopKIndices = _lastTopKIndices ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastExpertOutputs = _lastExpertOutputs ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGate = _lastGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGateRaw = _lastGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastMoEOutput = _lastMoEOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastExpertStates = _lastExpertStates ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = lastInput.Shape[0];
        int seqLen = lastInput.Shape[1];
        int totalTokens = batchSize * seqLen;

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(lastOutput, grad3D);

        // Initialize all gradients
        _routerWeightsGradient = new Tensor<T>([_modelDimension, _numExperts]);
        _routerBiasGradient = new Tensor<T>([_numExperts]);
        _expertAGradient = new Tensor<T>([_numExperts, _stateDimension]);
        _expertBGradient = new Tensor<T>([_numExperts, _stateDimension, _modelDimension]);
        _expertCGradient = new Tensor<T>([_numExperts, _modelDimension, _stateDimension]);
        _expertDGradient = new Tensor<T>([_numExperts, _modelDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Output projection backward
        var gradFlat = activationGrad.Reshape(totalTokens, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(lastMoEOutput, lastGate)
            .Reshape(totalTokens, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Gate backward (Swish)
        var dMoEOutput = Engine.TensorMultiply(dGated, lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, lastMoEOutput);
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(lastGateRaw));

        var inputFlat = lastInput.Reshape(totalTokens, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(totalTokens, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // MoE combination backward
        var dExpertOutputs = new Tensor<T>(new[] { totalTokens, _topK, _modelDimension });
        var dRouterWeights = new Tensor<T>(new[] { totalTokens, _numExperts });

        for (int tokenIdx = 0; tokenIdx < totalTokens; tokenIdx++)
        {
            int bi = tokenIdx / seqLen;
            int t = tokenIdx % seqLen;

            for (int ki = 0; ki < _topK; ki++)
            {
                int expertIdx = lastTopKIndices[tokenIdx, ki];
                T weight = lastRouterWeightsResult[new[] { tokenIdx, expertIdx }];

                for (int di = 0; di < _modelDimension; di++)
                {
                    T dMoE = dMoEOutput[new[] { bi, t, di }];
                    T expertOut = lastExpertOutputs[new[] { tokenIdx, ki, di }];

                    // dExpertOutput = weight * dMoE
                    dExpertOutputs[new[] { tokenIdx, ki, di }] = NumOps.Multiply(weight, dMoE);

                    // dWeight = expertOutput * dMoE
                    dRouterWeights[new[] { tokenIdx, expertIdx }] = NumOps.Add(
                        dRouterWeights[new[] { tokenIdx, expertIdx }],
                        NumOps.Multiply(expertOut, dMoE));
                }
            }
        }

        // Expert SSM backward
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            // Per-expert state gradient: dH[expert, t, stateDim]
            var dH = new T[_numExperts, _stateDimension];

            for (int t = seqLen - 1; t >= 0; t--)
            {
                int tokenIdx = bi * seqLen + t;

                for (int ki = 0; ki < _topK; ki++)
                {
                    int expertIdx = lastTopKIndices[tokenIdx, ki];

                    // y = C * h + D * x, backward
                    for (int di = 0; di < _modelDimension; di++)
                    {
                        T dY = dExpertOutputs[new[] { tokenIdx, ki, di }];

                        // dD
                        T xVal = lastInput[new[] { bi, t, di }];
                        _expertDGradient[new[] { expertIdx, di }] = NumOps.Add(
                            _expertDGradient[new[] { expertIdx, di }],
                            NumOps.Multiply(dY, xVal));

                        // dInput from D*x
                        dInput[new[] { bi, t, di }] = NumOps.Add(
                            dInput[new[] { bi, t, di }],
                            NumOps.Multiply(dY, _expertD[new[] { expertIdx, di }]));

                        // dC and dH from C * h
                        for (int si = 0; si < _stateDimension; si++)
                        {
                            T hVal = lastExpertStates[new[] { bi, expertIdx, t + 1, si }];
                            _expertCGradient[new[] { expertIdx, di, si }] = NumOps.Add(
                                _expertCGradient[new[] { expertIdx, di, si }],
                                NumOps.Multiply(dY, hVal));

                            dH[expertIdx, si] = NumOps.Add(dH[expertIdx, si],
                                NumOps.Multiply(dY, _expertC[new[] { expertIdx, di, si }]));
                        }
                    }

                    // h_t = exp(A) * h_{t-1} + B * x, backward
                    for (int si = 0; si < _stateDimension; si++)
                    {
                        T dHVal = dH[expertIdx, si];
                        T aVal = NumOps.Exp(_expertA[new[] { expertIdx, si }]);
                        T prevH = t > 0
                            ? lastExpertStates[new[] { bi, expertIdx, t, si }]
                            : NumOps.Zero;

                        // dA: d/dA[exp(A) * h_{t-1}] = exp(A) * h_{t-1} * dH
                        _expertAGradient[new[] { expertIdx, si }] = NumOps.Add(
                            _expertAGradient[new[] { expertIdx, si }],
                            NumOps.Multiply(NumOps.Multiply(aVal, prevH), dHVal));

                        // dB and dInput from B * x
                        for (int di = 0; di < _modelDimension; di++)
                        {
                            T xVal = lastInput[new[] { bi, t, di }];
                            _expertBGradient[new[] { expertIdx, si, di }] = NumOps.Add(
                                _expertBGradient[new[] { expertIdx, si, di }],
                                NumOps.Multiply(dHVal, xVal));

                            dInput[new[] { bi, t, di }] = NumOps.Add(
                                dInput[new[] { bi, t, di }],
                                NumOps.Multiply(dHVal, _expertB[new[] { expertIdx, si, di }]));
                        }

                        // Propagate dH to previous timestep: dH_{t-1} += exp(A) * dH_t
                        dH[expertIdx, si] = NumOps.Multiply(aVal, dHVal);
                    }
                }
            }
        }

        // Router backward (simplified: gradient through softmax top-K)
        // We approximate by passing gradient through the softmax for selected experts
        for (int tokenIdx = 0; tokenIdx < totalTokens; tokenIdx++)
        {
            // Softmax backward for router
            T sumWD = NumOps.Zero;
            for (int ki = 0; ki < _topK; ki++)
            {
                int expertIdx = lastTopKIndices[tokenIdx, ki];
                T w = lastRouterWeightsResult[new[] { tokenIdx, expertIdx }];
                T dW = dRouterWeights[new[] { tokenIdx, expertIdx }];
                sumWD = NumOps.Add(sumWD, NumOps.Multiply(w, dW));
            }

            for (int ki = 0; ki < _topK; ki++)
            {
                int expertIdx = lastTopKIndices[tokenIdx, ki];
                T w = lastRouterWeightsResult[new[] { tokenIdx, expertIdx }];
                T dW = dRouterWeights[new[] { tokenIdx, expertIdx }];
                T dLogit = NumOps.Multiply(w, NumOps.Subtract(dW, sumWD));

                // Router weight gradient
                for (int di = 0; di < _modelDimension; di++)
                {
                    T xVal = inputFlat[new[] { tokenIdx, di }];
                    _routerWeightsGradient[new[] { di, expertIdx }] = NumOps.Add(
                        _routerWeightsGradient[new[] { di, expertIdx }],
                        NumOps.Multiply(xVal, dLogit));

                    // dInput from router
                    int bi = tokenIdx / seqLen;
                    int t = tokenIdx % seqLen;
                    dInput[new[] { bi, t, di }] = NumOps.Add(
                        dInput[new[] { bi, t, di }],
                        NumOps.Multiply(_routerWeights[new[] { di, expertIdx }], dLogit));
                }

                _routerBiasGradient[expertIdx] = NumOps.Add(
                    _routerBiasGradient[expertIdx], dLogit);
            }
        }

        // Add gate input gradient
        var dInputFlat = dInput.Reshape(totalTokens, _modelDimension);
        dInputFlat = Engine.TensorAdd(dInputFlat, dInputFromGate);
        dInput = dInputFlat.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
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
