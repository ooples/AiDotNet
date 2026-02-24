using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class DeltaFormerLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly bool _useDeltaRule;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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
    /// Standard softmax attention: softmax(Q*K^T / sqrt(d)) * V.
    /// Used for the retrieval steps in the DeltaFormer architecture.
    /// </summary>
    private Tensor<T> SoftmaxAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Store attention weights for backward pass
        _lastAttentionWeights = new Tensor<T>(new[] { batchSize, _numHeads, seqLen, seqLen });

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
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State matrix per head: [batch, numHeads, headDim, headDim]
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
        var lastMechanismOutput = _lastMechanismOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

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
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(lastMechanismOutput, lastGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Gate backward
        var dMechanismOut = Engine.TensorMultiply(dGated, lastGate);
        var dGateSigmoid = Engine.TensorMultiply(dGated, lastMechanismOutput);

        // Sigmoid derivative: sigma(x) * (1 - sigma(x))
        var sigDeriv = Engine.TensorMultiply(lastGate,
            Engine.TensorSubtract(CreateOnesLike(lastGate), lastGate));
        var dGateRaw = Engine.TensorMultiply(dGateSigmoid, sigDeriv);

        var inputFlat = lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Mechanism backward (delta rule or attention)
        Tensor<T> dQ, dK, dV;
        if (_useDeltaRule)
        {
            DeltaRuleBackward(dMechanismOut, lastQuery, lastKey, lastValue, batchSize, seqLen,
                out dQ, out dK, out dV);
        }
        else
        {
            AttentionBackward(dMechanismOut, lastQuery, lastKey, lastValue, batchSize, seqLen,
                out dQ, out dK, out dV);
        }

        // Projection weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradient from all paths
        var dInputTotal = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
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

    private void DeltaRuleBackward(
        Tensor<T> dOutput, Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen,
        out Tensor<T> dQ, out Tensor<T> dK, out Tensor<T> dV)
    {
        dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var lastStates = _lastStates ?? throw new InvalidOperationException("States not computed.");

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // dS += dO * q^T (from output = S * q)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dOutput[new[] { bi, t, flatDi }];
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T qVal = q[new[] { bi, t, flatKi }];
                            T sVal = lastStates[new[] { bi, t + 1, hi, di, ki }];

                            dState[new[] { bi, hi, di, ki }] = NumOps.Add(
                                dState[new[] { bi, hi, di, ki }],
                                NumOps.Multiply(dO, qVal));

                            dQ[new[] { bi, t, flatKi }] = NumOps.Add(
                                dQ[new[] { bi, t, flatKi }],
                                NumOps.Multiply(dO, sVal));
                        }
                    }

                    // State update backward: S = S_prev + delta * k^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T vVal = v[new[] { bi, t, flatDi }];

                        // Recompute delta[di]
                        T sK_di = NumOps.Zero;
                        for (int ki2 = 0; ki2 < _headDimension; ki2++)
                        {
                            int flatKi2 = dimStart + ki2;
                            T kVal2 = NumOps.Multiply(k[new[] { bi, t, flatKi2 }], keyScale);
                            T sPrev = lastStates[new[] { bi, t, hi, di, ki2 }];
                            sK_di = NumOps.Add(sK_di, NumOps.Multiply(sPrev, kVal2));
                        }
                        T deltaDi = NumOps.Subtract(vVal, sK_di);

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);
                            T dS = dState[new[] { bi, hi, di, ki }];

                            // dV: gradient flows through delta = v - S*k
                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(dS, kVal));

                            // dK: gradient from delta * k^T
                            dK[new[] { bi, t, flatKi }] = NumOps.Add(
                                dK[new[] { bi, t, flatKi }],
                                NumOps.Multiply(deltaDi, dS));

                            // dState propagates to previous timestep unchanged (no forget gate)
                            // plus dS from -S_prev*k term in delta
                        }
                    }
                }
            }
        }
    }

    private void AttentionBackward(
        Tensor<T> dOutput, Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen,
        out Tensor<T> dQ, out Tensor<T> dK, out Tensor<T> dV)
    {
        dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        var lastAttnWeights = _lastAttentionWeights ?? throw new InvalidOperationException("Attention weights not computed.");

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int ti = 0; ti < seqLen; ti++)
                {
                    // dV: dV[tj] += attn_weight[ti, tj] * dO[ti]
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T weight = lastAttnWeights[new[] { bi, hi, ti, tj }];
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            T dO = dOutput[new[] { bi, ti, flatDi }];
                            dV[new[] { bi, tj, flatDi }] = NumOps.Add(
                                dV[new[] { bi, tj, flatDi }],
                                NumOps.Multiply(weight, dO));
                        }
                    }

                    // dAttnWeights[ti, tj] = sum_d(dO[ti,d] * V[tj,d])
                    var dAttnWeights = new T[seqLen];
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T dAW = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            dAW = NumOps.Add(dAW,
                                NumOps.Multiply(dOutput[new[] { bi, ti, flatDi }], v[new[] { bi, tj, flatDi }]));
                        }
                        dAttnWeights[tj] = dAW;
                    }

                    // Softmax backward: dScore = attn * (dAttn - sum(attn * dAttn))
                    T sumAttnDAttn = NumOps.Zero;
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T w = lastAttnWeights[new[] { bi, hi, ti, tj }];
                        sumAttnDAttn = NumOps.Add(sumAttnDAttn, NumOps.Multiply(w, dAttnWeights[tj]));
                    }

                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T w = lastAttnWeights[new[] { bi, hi, ti, tj }];
                        T dScore = NumOps.Multiply(w, NumOps.Subtract(dAttnWeights[tj], sumAttnDAttn));
                        dScore = NumOps.Multiply(dScore, scale);

                        // dQ, dK from score = q^T * k
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            dQ[new[] { bi, ti, flatDi }] = NumOps.Add(
                                dQ[new[] { bi, ti, flatDi }],
                                NumOps.Multiply(dScore, k[new[] { bi, tj, flatDi }]));
                            dK[new[] { bi, tj, flatDi }] = NumOps.Add(
                                dK[new[] { bi, tj, flatDi }],
                                NumOps.Multiply(dScore, q[new[] { bi, ti, flatDi }]));
                        }
                    }
                }
            }
        }
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
        if (_queryWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
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
