using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Gated DeltaProduct layer from "DeltaProduct: Increasing the Expressivity of DeltaNet
/// Through Products of Householders" (Siems et al., 2025).
/// </summary>
/// <remarks>
/// <para>
/// Gated DeltaProduct combines the Householder product state transitions of DeltaProduct with
/// output gating (similar to how GatedDeltaNet adds gating to DeltaNet). The forget gate (alpha)
/// provides an additional scalar decay before the Householder rotation, and the output gate
/// controls information flow to the final output.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute gates:
///      - alpha (forget gate via sigmoid): scalar decay per head
///      - beta (write strength via sigmoid): controls write magnitude
///      - gate (output gate via Swish): modulates the output
///   3. For each timestep, compute M Householder vectors {u_1, ..., u_M}
///   4. State update with gated Householder product:
///      H_t = (I - 2*u_M*u_M^T/||u_M||^2) * ... * (I - 2*u_1*u_1^T/||u_1||^2)
///      S_t = alpha_t * H_t * S_{t-1} + beta_t * v_t * k_t^T
///   5. Output: O_t = gate_t * (S_t * q_t)
///   6. Output projection
/// </code>
/// </para>
/// <para>
/// The combination of gating and Householder products provides maximum expressivity:
/// - Alpha gate: Controls how much old state to retain (like GatedDeltaNet)
/// - Householder product: ROTATES the retained state (like DeltaProduct)
/// - Beta gate: Controls how strongly to write new information
/// - Output gate: Controls what to expose from the state
///
/// This means the model can: fade old information (alpha), rearrange what remains (Householder),
/// write new information selectively (beta), and filter what to output (gate).
/// </para>
/// <para><b>For Beginners:</b> Gated DeltaProduct is the fully-featured version that combines
/// everything from both GatedDeltaNet and DeltaProduct.
///
/// Think of managing a library:
/// - Alpha (forget gate): "Remove 10% of books from each shelf" (uniform fading)
/// - Householder product: "Rearrange remaining books by topic instead of author" (rotation)
/// - Beta (write gate): "How many new books to add to the collection" (write strength)
/// - Output gate: "Which shelves to show to the current visitor" (output filtering)
///
/// Without gating (plain DeltaProduct): you can rearrange and add books, but can't thin out
/// the collection. Without Householder (plain GatedDeltaNet): you can thin out and add, but
/// can't rearrange. Gated DeltaProduct does all four operations at each step.
///
/// This makes it the most expressive variant in the DeltaNet family, at the cost of slightly
/// more compute per step.
/// </para>
/// <para>
/// <b>Reference:</b> Siems et al., "DeltaProduct: Increasing the Expressivity of DeltaNet Through
/// Products of Householders", 2025. https://arxiv.org/abs/2502.10297
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GatedDeltaProductLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _numHouseholders;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Beta (write strength) projection: [modelDim, numHeads]
    private Tensor<T> _betaWeights;
    private Tensor<T> _betaBias;

    // Alpha (forget gate) projection: [modelDim, numHeads]
    private Tensor<T> _alphaWeights;
    private Tensor<T> _alphaBias;

    // Householder vector projections: [numHouseholders, modelDim, headDim]
    private Tensor<T> _householderWeights;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastBeta;
    private Tensor<T>? _lastAlpha;
    private Tensor<T>? _lastHouseholderVecs;
    private Tensor<T>? _lastStates;
    private Tensor<T>? _lastRecurrenceOutput;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _betaWeightsGradient;
    private Tensor<T>? _betaBiasGradient;
    private Tensor<T>? _alphaWeightsGradient;
    private Tensor<T>? _alphaBiasGradient;
    private Tensor<T>? _householderWeightsGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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
    /// Gets the number of Householder reflections per timestep.
    /// </summary>
    public int NumHouseholders => _numHouseholders;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _betaWeights.Length + _betaBias.Length +
        _alphaWeights.Length + _alphaBias.Length +
        _householderWeights.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Gated DeltaProduct layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own state matrix and Householder
    /// reflections. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="numHouseholders">
    /// Number of Householder reflections per timestep. Default: 4.
    /// <para><b>For Beginners:</b> More reflections give more expressive state rotations.
    /// 2-4 reflections capture most of the benefit in practice.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public GatedDeltaProductLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int numHouseholders = 4,
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
        if (numHouseholders <= 0)
            throw new ArgumentException($"Number of Householder reflections ({numHouseholders}) must be positive.", nameof(numHouseholders));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _numHouseholders = numHouseholders;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _betaWeights = new Tensor<T>([modelDimension, numHeads]);
        _betaBias = new Tensor<T>([numHeads]);
        _alphaWeights = new Tensor<T>([modelDimension, numHeads]);
        _alphaBias = new Tensor<T>([numHeads]);
        _householderWeights = new Tensor<T>([numHouseholders, modelDimension, _headDimension]);
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
        InitializeTensor2D(_betaWeights);
        _betaBias.Fill(NumOps.FromDouble(0.1));
        InitializeTensor2D(_alphaWeights);
        // Alpha bias ~ 2 so sigmoid(2) ~ 0.88 -> strong initial memory retention
        for (int i = 0; i < _alphaBias.Length; i++)
            _alphaBias[i] = NumOps.FromDouble(2.0);
        InitializeHouseholderWeights();
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

    private void InitializeHouseholderWeights()
    {
        int fanIn = _modelDimension;
        int fanOut = _headDimension;
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < _householderWeights.Length; i++)
            _householderWeights[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);

        // Step 1: Q, K, V projections
        var q = Engine.TensorMatMul(inputFlat, _queryWeights).Reshape(batchSize, seqLen, _modelDimension);
        var k = Engine.TensorMatMul(inputFlat, _keyWeights).Reshape(batchSize, seqLen, _modelDimension);
        var v = Engine.TensorMatMul(inputFlat, _valueWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Gates
        var betaRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _betaWeights),
            _betaBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var beta = Engine.Sigmoid(betaRaw);
        _lastBeta = beta;

        var alphaRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _alphaWeights),
            _alphaBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var alpha = Engine.Sigmoid(alphaRaw);
        _lastAlpha = alpha;

        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastOutputGate = gate;
        _lastOutputGateRaw = gateRaw;

        // Step 3: Compute Householder vectors
        var hVecs = ComputeHouseholderVectors(inputFlat, batchSize, seqLen);
        _lastHouseholderVecs = hVecs;

        // Step 4: Gated DeltaProduct recurrence
        var recOutput = GatedDeltaProductRecurrence(q, k, v, alpha, beta, hVecs, batchSize, seqLen);
        _lastRecurrenceOutput = recOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(recOutput, gate);

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
    /// Computes Householder vectors from input for each timestep.
    /// Returns shape [batch * seqLen, numHouseholders, numHeads, headDim].
    /// </summary>
    private Tensor<T> ComputeHouseholderVectors(Tensor<T> inputFlat, int batchSize, int seqLen)
    {
        int total = batchSize * seqLen;
        var hVecs = new Tensor<T>(new[] { total, _numHouseholders, _numHeads, _headDimension });

        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            for (int pos = 0; pos < total; pos++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    for (int d = 0; d < _headDimension; d++)
                    {
                        T val = NumOps.Zero;
                        for (int j = 0; j < _modelDimension; j++)
                        {
                            val = NumOps.Add(val,
                                NumOps.Multiply(
                                    inputFlat[new[] { pos, j }],
                                    _householderWeights[new[] { mi, j, d }]));
                        }
                        hVecs[new[] { pos, mi, hi, d }] = val;
                    }
                }
            }
        }

        return hVecs;
    }

    /// <summary>
    /// Applies the product of M Householder reflections to the state, then scales by alpha.
    /// S <- alpha * H * S  where H = H_M * ... * H_1.
    /// </summary>
    private void ApplyGatedHouseholderProduct(
        Tensor<T> state, Tensor<T> hVecs, T alphaVal,
        int bi, int hi, int posFlat)
    {
        // First apply Householder reflections: S <- H * S
        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            T normSq = NumOps.Zero;
            for (int d = 0; d < _headDimension; d++)
            {
                T u = hVecs[new[] { posFlat, mi, hi, d }];
                normSq = NumOps.Add(normSq, NumOps.Multiply(u, u));
            }
            T eps = NumOps.FromDouble(1e-8);
            normSq = NumOps.Add(normSq, eps);
            T twoOverNormSq = NumOps.Divide(NumOps.FromDouble(2.0), normSq);

            for (int j = 0; j < _headDimension; j++)
            {
                T dot = NumOps.Zero;
                for (int d = 0; d < _headDimension; d++)
                {
                    T u = hVecs[new[] { posFlat, mi, hi, d }];
                    dot = NumOps.Add(dot, NumOps.Multiply(u, state[new[] { bi, hi, d, j }]));
                }
                T factor = NumOps.Multiply(twoOverNormSq, dot);

                for (int d = 0; d < _headDimension; d++)
                {
                    T u = hVecs[new[] { posFlat, mi, hi, d }];
                    state[new[] { bi, hi, d, j }] = NumOps.Subtract(
                        state[new[] { bi, hi, d, j }],
                        NumOps.Multiply(factor, u));
                }
            }
        }

        // Then scale by alpha: S <- alpha * S
        for (int di = 0; di < _headDimension; di++)
        {
            for (int ki = 0; ki < _headDimension; ki++)
            {
                state[new[] { bi, hi, di, ki }] = NumOps.Multiply(
                    alphaVal, state[new[] { bi, hi, di, ki }]);
            }
        }
    }

    /// <summary>
    /// Gated DeltaProduct recurrence:
    /// S_t = alpha_t * H_t * S_{t-1} + beta_t * v_t * k_t^T
    /// O_t = S_t * q_t
    /// </summary>
    private Tensor<T> GatedDeltaProductRecurrence(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> alpha, Tensor<T> beta,
        Tensor<T> hVecs, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    int posFlat = bi * seqLen + t;
                    T alphaVal = alpha[new[] { bi, t, hi }];
                    T betaVal = beta[new[] { bi, t, hi }];

                    // Apply gated Householder product: S <- alpha * H * S
                    ApplyGatedHouseholderProduct(state, hVecs, alphaVal, bi, hi, posFlat);

                    // Add outer product: S += beta * v * k^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);
                            T update = NumOps.Multiply(betaVal,
                                NumOps.Multiply(v[new[] { bi, t, flatDi }], kVal));
                            state[new[] { bi, hi, di, ki }] = NumOps.Add(
                                state[new[] { bi, hi, di, ki }], update);
                        }
                    }

                    // Output: O = S * q
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T qVal = q[new[] { bi, t, flatKi }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(state[new[] { bi, hi, di, ki }], qVal));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }
                }
            }

            // Save state snapshot
            for (int bi = 0; bi < batchSize; bi++)
                for (int hi = 0; hi < _numHeads; hi++)
                    for (int di = 0; di < _headDimension; di++)
                        for (int ki = 0; ki < _headDimension; ki++)
                            allStates[new[] { bi, t + 1, hi, di, ki }] = state[new[] { bi, hi, di, ki }];
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
        var lastBeta = _lastBeta ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastAlpha = _lastAlpha ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastHouseholderVecs = _lastHouseholderVecs ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastStates = _lastStates ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastRecurrenceOutput = _lastRecurrenceOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutputGate = _lastOutputGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutputGateRaw = _lastOutputGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = lastInput.Shape[0];
        int seqLen = lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(lastOutput, grad3D);

        // Initialize gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _betaWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _betaBiasGradient = new Tensor<T>([_numHeads]);
        _alphaWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _alphaBiasGradient = new Tensor<T>([_numHeads]);
        _householderWeightsGradient = new Tensor<T>([_numHouseholders, _modelDimension, _headDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 6 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(lastRecurrenceOutput, lastOutputGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 5 backward: gating
        var dRecOutput = Engine.TensorMultiply(dGated, lastOutputGate);
        var dGateSwish = Engine.TensorMultiply(dGated, lastRecurrenceOutput);
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(lastOutputGateRaw));

        var inputFlat = lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 4 backward: Gated DeltaProduct recurrence (backward through time)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dAlpha = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        var dBeta = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        var dHVecs = new Tensor<T>(new[] { batchSize * seqLen, _numHouseholders, _numHeads, _headDimension });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    int posFlat = bi * seqLen + t;
                    T alphaVal = lastAlpha[new[] { bi, t, hi }];
                    T betaVal = lastBeta[new[] { bi, t, hi }];

                    // O_t = S_t * q_t -> dS += dO * q^T, dQ += S^T * dO
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dRecOutput[new[] { bi, t, flatDi }];

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T qVal = lastQuery[new[] { bi, t, flatKi }];
                            T sVal = lastStates[new[] { bi, t + 1, hi, di, ki }];

                            dState[new[] { bi, hi, di, ki }] = NumOps.Add(
                                dState[new[] { bi, hi, di, ki }],
                                NumOps.Multiply(dO, qVal));

                            dQ[new[] { bi, t, flatKi }] = NumOps.Add(
                                dQ[new[] { bi, t, flatKi }],
                                NumOps.Multiply(dO, sVal));
                        }
                    }

                    // S_t = alpha * H * S_{t-1} + beta * v * k^T
                    // dBeta += sum(dS * v * k^T), dV += beta * dS * k, dK += beta * v^T * dS
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T dS = dState[new[] { bi, hi, di, ki }];
                            T kVal = NumOps.Multiply(lastKey[new[] { bi, t, flatKi }], keyScale);
                            T vVal = lastValue[new[] { bi, t, flatDi }];

                            dBeta[new[] { bi, t, hi }] = NumOps.Add(
                                dBeta[new[] { bi, t, hi }],
                                NumOps.Multiply(dS, NumOps.Multiply(vVal, kVal)));

                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(NumOps.Multiply(betaVal, dS), kVal));

                            dK[new[] { bi, t, flatKi }] = NumOps.Add(
                                dK[new[] { bi, t, flatKi }],
                                NumOps.Multiply(NumOps.Multiply(betaVal, vVal),
                                    NumOps.Multiply(dS, keyScale)));
                        }
                    }

                    // Backward through alpha * H * S_{t-1}:
                    // dAlpha += dS_after_householder . S_prev_after_householder (element-wise sum)
                    // Then dS_prev = alpha * H^T * dS (H is symmetric so H^T = H)
                    // We need the state BEFORE the outer product was added, which is alpha*H*S_{t-1}
                    // First accumulate dAlpha from the alpha scaling
                    // dAlpha = sum_{d,k} dS[d,k] * (H * S_{t-1})[d,k]
                    // We approximate by noting: S_t - beta*v*k^T = alpha*H*S_{t-1}
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T dS = dState[new[] { bi, hi, di, ki }];
                            T kVal = NumOps.Multiply(lastKey[new[] { bi, t, flatKi }], keyScale);
                            T vVal = lastValue[new[] { bi, t, flatDi }];

                            // alpha*H*S_{t-1} = S_t - beta*v*k^T
                            T sTVal = lastStates[new[] { bi, t + 1, hi, di, ki }];
                            T outerProd = NumOps.Multiply(betaVal, NumOps.Multiply(vVal, kVal));
                            T alphaHS = NumOps.Subtract(sTVal, outerProd);

                            // dAlpha += dS * alphaHS / alpha (chain rule through alpha scaling)
                            T safeAlpha = NumOps.Add(alphaVal, NumOps.FromDouble(1e-10));
                            T hsPrev = NumOps.Divide(alphaHS, safeAlpha);
                            dAlpha[new[] { bi, t, hi }] = NumOps.Add(
                                dAlpha[new[] { bi, t, hi }],
                                NumOps.Multiply(dS, hsPrev));

                            // Propagate dState through alpha scaling
                            dState[new[] { bi, hi, di, ki }] = NumOps.Multiply(alphaVal, dS);
                        }
                    }

                    // Backward through Householder product and accumulate Householder gradients
                    BackwardHouseholderProduct(dState, lastHouseholderVecs, dHVecs, bi, hi, posFlat);
                }
            }
        }

        // Alpha and beta through sigmoid derivative
        var alphaSigDeriv = Engine.TensorMultiply(lastAlpha,
            Engine.TensorSubtract(CreateOnesLike(lastAlpha), lastAlpha));
        var dAlphaRaw = Engine.TensorMultiply(dAlpha, alphaSigDeriv);

        var betaSigDeriv = Engine.TensorMultiply(lastBeta,
            Engine.TensorSubtract(CreateOnesLike(lastBeta), lastBeta));
        var dBetaRaw = Engine.TensorMultiply(dBeta, betaSigDeriv);

        var dAlphaFlat = dAlphaRaw.Reshape(batchSize * seqLen, _numHeads);
        _alphaWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dAlphaFlat);
        _alphaBiasGradient = Engine.ReduceSum(dAlphaRaw, new int[] { 0, 1 });

        var dBetaFlat = dBetaRaw.Reshape(batchSize * seqLen, _numHeads);
        _betaWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dBetaFlat);
        _betaBiasGradient = Engine.ReduceSum(dBetaRaw, new int[] { 0, 1 });

        // Q, K, V weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Householder weight gradients
        AccumulateHouseholderWeightGradients(dHVecs, inputFlat, batchSize, seqLen);

        // Input gradient from all paths
        var dInput = Engine.TensorAdd(dInputFromGate,
            Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dAlphaFlat, _alphaWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dBetaFlat, _betaWeights.Transpose([1, 0])));

        var dInputFromH = ComputeHouseholderInputGradient(dHVecs, batchSize, seqLen);
        dInput = Engine.TensorAdd(dInput, dInputFromH);

        var dInput3D = dInput.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    /// <summary>
    /// Backward through Householder product: apply reflections in reverse and accumulate gradients.
    /// </summary>
    private void BackwardHouseholderProduct(
        Tensor<T> dState, Tensor<T> hVecs, Tensor<T> dHVecs,
        int bi, int hi, int posFlat)
    {
        for (int mi = _numHouseholders - 1; mi >= 0; mi--)
        {
            T normSq = NumOps.Zero;
            for (int d = 0; d < _headDimension; d++)
            {
                T u = hVecs[new[] { posFlat, mi, hi, d }];
                normSq = NumOps.Add(normSq, NumOps.Multiply(u, u));
            }
            T eps = NumOps.FromDouble(1e-8);
            normSq = NumOps.Add(normSq, eps);
            T twoOverNormSq = NumOps.Divide(NumOps.FromDouble(2.0), normSq);

            for (int j = 0; j < _headDimension; j++)
            {
                T dot = NumOps.Zero;
                for (int d = 0; d < _headDimension; d++)
                {
                    T u = hVecs[new[] { posFlat, mi, hi, d }];
                    dot = NumOps.Add(dot, NumOps.Multiply(u, dState[new[] { bi, hi, d, j }]));
                }
                T factor = NumOps.Multiply(twoOverNormSq, dot);

                for (int d = 0; d < _headDimension; d++)
                {
                    T u = hVecs[new[] { posFlat, mi, hi, d }];

                    dHVecs[new[] { posFlat, mi, hi, d }] = NumOps.Add(
                        dHVecs[new[] { posFlat, mi, hi, d }],
                        NumOps.Negate(NumOps.Multiply(factor, dState[new[] { bi, hi, d, j }])));

                    dState[new[] { bi, hi, d, j }] = NumOps.Subtract(
                        dState[new[] { bi, hi, d, j }],
                        NumOps.Multiply(factor, u));
                }
            }
        }
    }

    /// <summary>
    /// Accumulates Householder weight gradients from per-position gradients.
    /// </summary>
    private void AccumulateHouseholderWeightGradients(
        Tensor<T> dHVecs, Tensor<T> inputFlat, int batchSize, int seqLen)
    {
        int total = batchSize * seqLen;
        var gradient = _householderWeightsGradient ?? throw new InvalidOperationException("Gradients not initialized.");

        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            for (int pos = 0; pos < total; pos++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    for (int d = 0; d < _headDimension; d++)
                    {
                        T dH = dHVecs[new[] { pos, mi, hi, d }];
                        for (int j = 0; j < _modelDimension; j++)
                        {
                            gradient[new[] { mi, j, d }] = NumOps.Add(
                                gradient[new[] { mi, j, d }],
                                NumOps.Multiply(dH, inputFlat[new[] { pos, j }]));
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes input gradient contribution from Householder vectors.
    /// </summary>
    private Tensor<T> ComputeHouseholderInputGradient(
        Tensor<T> dHVecs, int batchSize, int seqLen)
    {
        int total = batchSize * seqLen;
        var dInput = new Tensor<T>(new[] { total, _modelDimension });

        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            for (int pos = 0; pos < total; pos++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    for (int d = 0; d < _headDimension; d++)
                    {
                        T dH = dHVecs[new[] { pos, mi, hi, d }];
                        for (int j = 0; j < _modelDimension; j++)
                        {
                            dInput[new[] { pos, j }] = NumOps.Add(
                                dInput[new[] { pos, j }],
                                NumOps.Multiply(dH, _householderWeights[new[] { mi, j, d }]));
                        }
                    }
                }
            }
        }

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

    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var ones = new Tensor<T>(template.Shape);
        for (int i = 0; i < ones.Length; i++) ones[i] = NumOps.One;
        return ones;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var queryWeightsGrad = _queryWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var keyWeightsGrad = _keyWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var valueWeightsGrad = _valueWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var betaWeightsGrad = _betaWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var betaBiasGrad = _betaBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var alphaWeightsGrad = _alphaWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var alphaBiasGrad = _alphaBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var householderWeightsGrad = _householderWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputGateWeightsGrad = _outputGateWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputGateBiasGrad = _outputGateBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionWeightsGrad = _outputProjectionWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionBiasGrad = _outputProjectionBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(queryWeightsGrad, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(keyWeightsGrad, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(valueWeightsGrad, negLR));
        _betaWeights = Engine.TensorAdd(_betaWeights, Engine.TensorMultiplyScalar(betaWeightsGrad, negLR));
        _betaBias = Engine.TensorAdd(_betaBias, Engine.TensorMultiplyScalar(betaBiasGrad, negLR));
        _alphaWeights = Engine.TensorAdd(_alphaWeights, Engine.TensorMultiplyScalar(alphaWeightsGrad, negLR));
        _alphaBias = Engine.TensorAdd(_alphaBias, Engine.TensorMultiplyScalar(alphaBiasGrad, negLR));
        _householderWeights = Engine.TensorAdd(_householderWeights, Engine.TensorMultiplyScalar(householderWeightsGrad, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(outputGateWeightsGrad, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(outputGateBiasGrad, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(outputProjectionWeightsGrad, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(outputProjectionBiasGrad, negLR));
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
        _betaWeights, _betaBias,
        _alphaWeights, _alphaBias,
        _householderWeights,
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
        _lastBeta = null;
        _lastAlpha = null;
        _lastHouseholderVecs = null;
        _lastStates = null;
        _lastRecurrenceOutput = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _betaWeightsGradient = null;
        _betaBiasGradient = null;
        _alphaWeightsGradient = null;
        _alphaBiasGradient = null;
        _householderWeightsGradient = null;
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
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        metadata["NumHouseholders"] = _numHouseholders.ToString();
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
    /// Gets the Householder projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetHouseholderWeights() => _householderWeights;
}
