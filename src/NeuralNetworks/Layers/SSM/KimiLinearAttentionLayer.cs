using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Kimi KDA (Key-Value Driven Gated Linear Attention) layer from the
/// "Kimi-VL Technical Report" (Kimi Team, 2025, arXiv:2510.26692).
/// </summary>
/// <remarks>
/// <para>
/// Kimi KDA is a gated linear attention mechanism where the gate is computed from the interaction
/// between keys and values rather than from a separate learned projection. The gate signal
/// g_t = sigma(k_t^T * v_t + bias) captures whether the current key-value pair contains
/// information that conflicts with or reinforces the current state. This KV-driven gating
/// replaces the typical input-driven gating found in other linear attention variants.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute KV-driven gate: g_t = sigmoid(k_t^T * v_t + bias)
///      - This measures key-value coherence/conflict
///   3. Gated linear attention recurrence:
///      S_t = g_t * S_{t-1} + k_t * v_t^T
///      - The gate controls how much old state is retained
///      - New key-value outer product is always added
///   4. Output: o_t = S_t * q_t
///   5. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The key insight is that the gate should reflect the CONTENT of what's being stored, not just
/// where in the sequence we are. When k_t and v_t are highly aligned (high dot product), the
/// information is coherent and can be safely accumulated. When they conflict with stored patterns,
/// the gate should allow partial forgetting. This KV-driven approach automatically detects when
/// new information should overwrite old information.
/// </para>
/// <para><b>For Beginners:</b> Kimi KDA is a way to read through a sequence and build up a
/// memory of what you've seen, with a smart forgetting mechanism.
///
/// Think of it like taking notes while reading a book:
/// - At each word, you have a "key" (what topic this is about) and a "value" (what it says)
/// - The gate is like asking: "Does this new information FIT with what I already know?"
///   - If key and value are very aligned (k^T * v is large): "This is strong, clear info" -> gate near 1
///   - If they're not aligned: "This is ambiguous or contradictory" -> gate lower
/// - When the gate is high: keep most of old notes (g * S) and add new info (k * v^T)
/// - When the gate is low: forget more old notes, making room for new patterns
///
/// This is different from other approaches that gate based on the INPUT position:
/// - Position-based gate: "I'm at word 50, so I should forget some old stuff" (doesn't know WHAT to forget)
/// - KV-driven gate: "This new key-value pair conflicts with stored patterns, so selectively forget" (content-aware)
///
/// The result is more intelligent memory management that adapts to the actual information content.
/// </para>
/// <para>
/// <b>Reference:</b> Kimi Team, "Kimi-VL Technical Report", 2025.
/// https://arxiv.org/abs/2510.26692
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class KimiLinearAttentionLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // KV gate bias: [numHeads] (the gate is computed from k^T*v, plus this bias)
    private Tensor<T> _gateKVBias;

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
    private Tensor<T>? _lastKVGate;
    private Tensor<T>? _lastKVGateRaw;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private Tensor<T>? _lastStates;
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _gateKVBiasGradient;
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

    /// <summary>Gets the number of heads.</summary>
    public int NumHeads => _numHeads;

    /// <summary>Gets the dimension per head.</summary>
    public int HeadDimension => _headDimension;

    /// <inheritdoc />
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _gateKVBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Kimi KDA (Key-Value Driven Gated Linear Attention) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own independent state matrix S,
    /// allowing the model to track multiple types of associations simultaneously.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public KimiLinearAttentionLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
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

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _gateKVBias = new Tensor<T>([numHeads]);
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
        // Initialize gate bias to ~2.0 so sigmoid(2)~0.88 -> strong initial retention
        for (int i = 0; i < _gateKVBias.Length; i++)
            _gateKVBias[i] = NumOps.FromDouble(2.0);
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
        var q = Engine.TensorMatMul(inputFlat, _queryWeights).Reshape(batchSize, seqLen, _modelDimension);
        var k = Engine.TensorMatMul(inputFlat, _keyWeights).Reshape(batchSize, seqLen, _modelDimension);
        var v = Engine.TensorMatMul(inputFlat, _valueWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var outputGate = Engine.Swish(gateRaw);
        _lastOutputGate = outputGate;
        _lastOutputGateRaw = gateRaw;

        // Step 3: KV-driven gated linear attention recurrence
        var recurrenceOutput = KVGatedRecurrence(q, k, v, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 4: Gated output
        var gatedOutput = Engine.TensorMultiply(recurrenceOutput, outputGate);

        // Step 5: Output projection
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
    /// KV-driven gated linear attention recurrence.
    /// Gate: g_t = sigmoid(k_t^T * v_t + bias)
    /// State: S_t = g_t * S_{t-1} + k_t * v_t^T
    /// Output: o_t = S_t * q_t
    /// </summary>
    private Tensor<T> KVGatedRecurrence(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State: [batch, numHeads, headDim, headDim]
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });

        // Store KV gate values
        _lastKVGate = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        _lastKVGateRaw = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Compute KV-driven gate: g_t = sigmoid(k_t^T * v_t + bias)
                    T kvDot = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T kVal = NumOps.Multiply(k[new[] { bi, t, flatDi }], keyScale);
                        T vVal = v[new[] { bi, t, flatDi }];
                        kvDot = NumOps.Add(kvDot, NumOps.Multiply(kVal, vVal));
                    }

                    T gateRawVal = NumOps.Add(kvDot, _gateKVBias[hi]);
                    _lastKVGateRaw[new[] { bi, t, hi }] = gateRawVal;

                    // Sigmoid activation for gate
                    T gateVal = SigmoidScalar(gateRawVal);
                    _lastKVGate[new[] { bi, t, hi }] = gateVal;

                    // State update: S_t = g_t * S_{t-1} + k_t * v_t^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T vVal = v[new[] { bi, t, flatDi }];
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);
                            T prevS = state[new[] { bi, hi, di, ki }];
                            T kvOuter = NumOps.Multiply(kVal, vVal);
                            T newS = NumOps.Add(NumOps.Multiply(gateVal, prevS), kvOuter);
                            state[new[] { bi, hi, di, ki }] = newS;
                        }
                    }

                    // Output: o_t = S_t * q_t
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

            // Save state snapshot
            for (int bi = 0; bi < batchSize; bi++)
                for (int hi2 = 0; hi2 < _numHeads; hi2++)
                    for (int di = 0; di < _headDimension; di++)
                        for (int ki = 0; ki < _headDimension; ki++)
                            allStates[new[] { bi, t + 1, hi2, di, ki }] = state[new[] { bi, hi2, di, ki }];
        }

        _lastStates = allStates;
        return output;
    }

    /// <summary>
    /// Computes sigmoid for a scalar value: 1 / (1 + exp(-x)).
    /// </summary>
    private T SigmoidScalar(T x)
    {
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var lastInput = _lastInput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutput = _lastOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastQuery = _lastQuery ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastKey = _lastKey ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastValue = _lastValue ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastKVGate = _lastKVGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutputGate = _lastOutputGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutputGateRaw = _lastOutputGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastRecurrenceOutput = _lastRecurrenceOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastStates = _lastStates ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

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
        _gateKVBiasGradient = new Tensor<T>([_numHeads]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(lastRecurrenceOutput, lastOutputGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Output gate backward (Swish)
        var dRecurrenceOut = Engine.TensorMultiply(dGated, lastOutputGate);
        var dGateSwish = Engine.TensorMultiply(dGated, lastRecurrenceOutput);
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(lastOutputGateRaw));

        var inputFlat = lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // KV-gated recurrence backward
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T gateVal = lastKVGate[new[] { bi, t, hi }];

                    // Output backward: o_t = S_t * q_t -> dS, dQ
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dRecurrenceOut[new[] { bi, t, flatDi }];

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

                    // State update backward: S_t = g_t * S_{t-1} + k_t * v_t^T
                    // dG from gate: dG += sum(dS * S_{t-1})
                    T dGateScalar = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T dS = dState[new[] { bi, hi, di, ki }];
                            T sPrev = lastStates[new[] { bi, t, hi, di, ki }];

                            // dGate
                            dGateScalar = NumOps.Add(dGateScalar, NumOps.Multiply(dS, sPrev));

                            // dK and dV from k * v^T term
                            T kVal = NumOps.Multiply(lastKey[new[] { bi, t, flatKi }], keyScale);
                            T vVal = lastValue[new[] { bi, t, flatDi }];
                            dK[new[] { bi, t, flatKi }] = NumOps.Add(
                                dK[new[] { bi, t, flatKi }],
                                NumOps.Multiply(dS, vVal));
                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(dS, kVal));

                            // Propagate dState to previous timestep
                            dState[new[] { bi, hi, di, ki }] = NumOps.Multiply(gateVal, dS);
                        }
                    }

                    // Gate sigmoid derivative: g * (1 - g)
                    T sigDeriv = NumOps.Multiply(gateVal, NumOps.Subtract(NumOps.One, gateVal));
                    T dGateRawScalar = NumOps.Multiply(dGateScalar, sigDeriv);

                    // Gate bias gradient
                    _gateKVBiasGradient[hi] = NumOps.Add(_gateKVBiasGradient[hi], dGateRawScalar);

                    // dK and dV from gate computation: g = sigmoid(k^T * v + bias)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T kVal = NumOps.Multiply(lastKey[new[] { bi, t, flatDi }], keyScale);
                        T vVal = lastValue[new[] { bi, t, flatDi }];
                        dK[new[] { bi, t, flatDi }] = NumOps.Add(
                            dK[new[] { bi, t, flatDi }],
                            NumOps.Multiply(dGateRawScalar, vVal));
                        dV[new[] { bi, t, flatDi }] = NumOps.Add(
                            dV[new[] { bi, t, flatDi }],
                            NumOps.Multiply(dGateRawScalar, kVal));
                    }
                }
            }
        }

        // Projection weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]),
            Engine.TensorMultiplyScalar(dKFlat, keyScale));
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradient from all paths
        var dInputTotal = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(Engine.TensorMultiplyScalar(dKFlat, keyScale), _keyWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromGate);

        var dInput3D = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
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
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _gateKVBias = Engine.TensorAdd(_gateKVBias, Engine.TensorMultiplyScalar(_gateKVBiasGradient!, negLR));
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
        _queryWeights, _keyWeights, _valueWeights,
        _gateKVBias,
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
        _lastKVGate = null;
        _lastKVGateRaw = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _lastStates = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _gateKVBiasGradient = null;
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
