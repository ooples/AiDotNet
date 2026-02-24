using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the TransNormerLLM layer from "TransNormerLLM: A Faster and Better LLM" (Qin et al., 2023).
/// </summary>
/// <remarks>
/// <para>
/// TransNormerLLM uses "Lightning Attention" -- a linear attention variant with exponential decay and
/// efficient normalization. Unlike standard Transformers that use softmax attention (O(n^2)), Lightning
/// Attention achieves linear complexity O(n) by combining linear attention with a decay factor and
/// RMSNorm-based normalization.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from the input
///   2. Apply RMSNorm to Q and K (key innovation -- stabilizes linear attention)
///   3. Linear attention with exponential decay (recurrent form):
///      S_t = gamma * S_{t-1} + k_t * v_t^T    (running state matrix)
///      z_t = gamma * z_{t-1} + k_t             (running normalizer)
///      o_t = RMSNorm(S_t * q_t)                (normalized output)
///   4. Output gate: y = swish(X W_g + b_g) * o
///   5. Output projection: output = y W_o + b_o
/// </code>
/// </para>
/// <para>
/// The key innovations over standard linear attention:
/// - RMSNorm on Q and K prevents the magnitude explosion that plagues linear attention
/// - Exponential decay gamma provides a soft causal bias (like RetNet) without rotary PE
/// - Per-output RMSNorm stabilizes the attention output, preventing training instability
/// These simple modifications make linear attention competitive with softmax attention for LLMs.
/// </para>
/// <para><b>For Beginners:</b> TransNormerLLM makes "linear attention" actually work well for language models.
///
/// Standard linear attention has a known problem: it tends to become numerically unstable during training
/// because the accumulated state matrix can grow without bound. TransNormerLLM fixes this with two tricks:
///
/// 1. RMSNorm on Q and K: Before computing attention, the queries and keys are normalized. This is like
///    making sure all "questions" and "answers" have similar magnitude, preventing any single token from
///    dominating the accumulated state.
///
/// 2. Exponential decay: Old information naturally fades away (controlled by gamma), preventing the state
///    from accumulating indefinitely. This is similar to RetNet's approach but simpler.
///
/// Together, these allow TransNormerLLM to match or exceed Transformer quality while being much faster
/// for long sequences (linear vs quadratic complexity).
/// </para>
/// <para>
/// <b>Reference:</b> Qin et al., "TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer", 2023.
/// https://arxiv.org/abs/2307.14995
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TransNormerLLMLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly double _decayRate;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // RMSNorm parameters for Q and K: [numHeads, headDim]
    private Tensor<T> _queryNormScale;
    private Tensor<T> _keyNormScale;

    // Per-head decay parameters (gammas): [numHeads]
    private Tensor<T> _gammas;

    // Output RMSNorm: [numHeads, headDim]
    private Tensor<T> _outputNormScale;

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
    private Tensor<T>? _lastQueryNormed;
    private Tensor<T>? _lastKeyNormed;
    private Tensor<T>? _lastQueryRmsInv;
    private Tensor<T>? _lastKeyRmsInv;
    private Tensor<T>? _lastAttnRaw;
    private Tensor<T>? _lastAttnNormed;
    private Tensor<T>? _lastAttnRmsInv;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastStates;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _queryNormScaleGradient;
    private Tensor<T>? _keyNormScaleGradient;
    private Tensor<T>? _gammasGradient;
    private Tensor<T>? _outputNormScaleGradient;
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
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the decay rate.
    /// </summary>
    public double DecayRate => _decayRate;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _queryNormScale.Length + _keyNormScale.Length +
        _gammas.Length +
        _outputNormScale.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new TransNormerLLM layer with lightning attention.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of the vector representing each token.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head can focus on different patterns. More heads give
    /// more diversity but each head has a smaller dimension. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="decayRate">
    /// Exponential decay rate gamma. Default: 0.99.
    /// <para><b>For Beginners:</b> Controls how quickly old information fades. A value of 0.99 means
    /// about 1% of information is lost per step, so the effective context window is about 100 steps.
    /// Higher values (closer to 1.0) give longer memory but may be harder to train.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public TransNormerLLMLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        double decayRate = 0.99,
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
        if (decayRate <= 0.0 || decayRate >= 1.0)
            throw new ArgumentException($"Decay rate ({decayRate}) must be in (0, 1).", nameof(decayRate));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _decayRate = decayRate;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);

        _queryNormScale = new Tensor<T>([numHeads, _headDimension]);
        _keyNormScale = new Tensor<T>([numHeads, _headDimension]);

        _gammas = new Tensor<T>([numHeads]);

        _outputNormScale = new Tensor<T>([numHeads, _headDimension]);

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

        // RMSNorm scales initialized to 1.0
        for (int i = 0; i < _queryNormScale.Length; i++)
            _queryNormScale[i] = NumOps.One;
        for (int i = 0; i < _keyNormScale.Length; i++)
            _keyNormScale[i] = NumOps.One;
        for (int i = 0; i < _outputNormScale.Length; i++)
            _outputNormScale[i] = NumOps.One;

        // Initialize gammas with multi-scale decay rates centered around the specified decay rate
        // Heads get slightly different rates for multi-scale context capture
        for (int h = 0; h < _numHeads; h++)
        {
            double spread = 0.02;
            double offset = (h - (_numHeads - 1.0) / 2.0) / Math.Max(1, _numHeads - 1) * spread;
            double gamma = Math.Max(0.9, Math.Min(0.9999, _decayRate + offset));
            _gammas[h] = NumOps.FromDouble(gamma);
        }

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

    /// <summary>
    /// Applies RMSNorm per head dimension and returns the inverse RMS for backward.
    /// </summary>
    private void ApplyRMSNorm(
        Tensor<T> input, Tensor<T> scale, Tensor<T> output, Tensor<T> rmsInv,
        int batchSize, int seqLen)
    {
        T eps = NumOps.FromDouble(1e-6);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    int dimStart = h * _headDimension;

                    // Compute RMS
                    T sumSq = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T val = input[new[] { bi, t, flatD }];
                        sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));
                    }
                    T meanSq = NumOps.Divide(sumSq, NumOps.FromDouble(_headDimension));
                    T rms = NumOps.Sqrt(NumOps.Add(meanSq, eps));
                    T invRms = NumOps.Divide(NumOps.One, rms);

                    rmsInv[new[] { bi, t, h }] = invRms;

                    // Normalize and scale
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T normalized = NumOps.Multiply(input[new[] { bi, t, flatD }], invRms);
                        output[new[] { bi, t, flatD }] = NumOps.Multiply(normalized, scale[new[] { h, d }]);
                    }
                }
            }
        }
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

        // Step 2: RMSNorm on Q and K
        var qNormed = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var kNormed = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var qRmsInv = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        var kRmsInv = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        ApplyRMSNorm(q, _queryNormScale, qNormed, qRmsInv, batchSize, seqLen);
        ApplyRMSNorm(k, _keyNormScale, kNormed, kRmsInv, batchSize, seqLen);
        _lastQueryNormed = qNormed;
        _lastKeyNormed = kNormed;
        _lastQueryRmsInv = qRmsInv;
        _lastKeyRmsInv = kRmsInv;

        // Step 3: Compute output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGateRaw = gateRaw;
        _lastGate = gate;

        // Step 4: Lightning attention with decay (recurrent form)
        var attnRaw = LightningAttentionForward(qNormed, kNormed, v, batchSize, seqLen);
        _lastAttnRaw = attnRaw;

        // Step 5: RMSNorm on attention output
        var attnNormed = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var attnRmsInv = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        ApplyRMSNorm(attnRaw, _outputNormScale, attnNormed, attnRmsInv, batchSize, seqLen);
        _lastAttnNormed = attnNormed;
        _lastAttnRmsInv = attnRmsInv;

        // Step 6: Gated output
        var gatedOutput = Engine.TensorMultiply(gate, attnNormed);

        // Step 7: Output projection
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
    /// Lightning attention forward pass: linear attention with exponential decay.
    /// S_t = gamma * S_{t-1} + k_t * v_t^T
    /// o_t = S_t * q_t
    /// </summary>
    private Tensor<T> LightningAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State matrix per head: [batch, numHeads, headDim, headDim]
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;
                    T gamma = _gammas[hi];

                    // State update: S = gamma * S + k * v^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDi = dimStart + di;
                            int flatDj = dimStart + dj;

                            T prevS = state[new[] { bi, hi, di, dj }];
                            T kVal = k[new[] { bi, t, flatDi }];
                            T vVal = v[new[] { bi, t, flatDj }];

                            T newS = NumOps.Add(
                                NumOps.Multiply(gamma, prevS),
                                NumOps.Multiply(kVal, vVal));
                            state[new[] { bi, hi, di, dj }] = newS;
                        }
                    }

                    // Output: o = S * q
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDj = dimStart + dj;
                            T qVal = q[new[] { bi, t, flatDj }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(state[new[] { bi, hi, di, dj }], qVal));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }

                    // Save state snapshot
                    for (int di = 0; di < _headDimension; di++)
                        for (int dj = 0; dj < _headDimension; dj++)
                            allStates[new[] { bi, t + 1, hi, di, dj }] = state[new[] { bi, hi, di, dj }];
                }
            }
        }

        _lastStates = allStates;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastQueryNormed == null || _lastKeyNormed == null ||
            _lastQueryRmsInv == null || _lastKeyRmsInv == null ||
            _lastAttnRaw == null || _lastAttnNormed == null || _lastAttnRmsInv == null ||
            _lastGate == null || _lastGateRaw == null || _lastStates == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryNormScaleGradient = new Tensor<T>([_numHeads, _headDimension]);
        _keyNormScaleGradient = new Tensor<T>([_numHeads, _headDimension]);
        _gammasGradient = new Tensor<T>([_numHeads]);
        _outputNormScaleGradient = new Tensor<T>([_numHeads, _headDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 7 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedOutput = Engine.TensorMultiply(_lastGate, _lastAttnNormed);
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 6 backward: gating
        var dAttnNormed = Engine.TensorMultiply(dGated, _lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, _lastAttnNormed);

        // Swish derivative
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(_lastGateRaw));

        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 5 backward: RMSNorm on attention output
        var dAttnRaw = RMSNormBackward(
            dAttnNormed, _lastAttnRaw, _outputNormScale, _lastAttnRmsInv,
            _outputNormScaleGradient, batchSize, seqLen);

        // Step 4 backward: lightning attention recurrence
        var dQNormed = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dKNormed = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;
                    T gamma = _gammas[hi];

                    // Output: o_di = sum_dj S[di,dj] * q[dj]
                    // dS[di,dj] += dO[di] * q[dj]
                    // dQ[dj] += sum_di dO[di] * S[di,dj]
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dAttnRaw[new[] { bi, t, flatDi }];

                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDj = dimStart + dj;
                            T qVal = _lastQueryNormed[new[] { bi, t, flatDj }];
                            T sVal = _lastStates[new[] { bi, t + 1, hi, di, dj }];

                            dState[new[] { bi, hi, di, dj }] = NumOps.Add(
                                dState[new[] { bi, hi, di, dj }],
                                NumOps.Multiply(dO, qVal));

                            dQNormed[new[] { bi, t, flatDj }] = NumOps.Add(
                                dQNormed[new[] { bi, t, flatDj }],
                                NumOps.Multiply(dO, sVal));
                        }
                    }

                    // State update: S = gamma * S_prev + k * v^T
                    // dS_prev += gamma * dS
                    // dGamma += sum_{di,dj} dS[di,dj] * S_prev[di,dj]
                    // dK[di] += sum_dj dS[di,dj] * v[dj]
                    // dV[dj] += sum_di dS[di,dj] * k[di]
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDj = dimStart + dj;
                            T dS = dState[new[] { bi, hi, di, dj }];
                            T prevS = _lastStates[new[] { bi, t, hi, di, dj }];

                            _gammasGradient![hi] = NumOps.Add(
                                _gammasGradient[hi],
                                NumOps.Multiply(dS, prevS));

                            T kVal = _lastKeyNormed[new[] { bi, t, flatDi }];
                            T vVal = _lastValue[new[] { bi, t, flatDj }];

                            dKNormed[new[] { bi, t, flatDi }] = NumOps.Add(
                                dKNormed[new[] { bi, t, flatDi }],
                                NumOps.Multiply(dS, vVal));

                            dV[new[] { bi, t, flatDj }] = NumOps.Add(
                                dV[new[] { bi, t, flatDj }],
                                NumOps.Multiply(dS, kVal));

                            // Propagate dState to previous timestep
                            dState[new[] { bi, hi, di, dj }] = NumOps.Multiply(gamma, dS);
                        }
                    }
                }
            }
        }

        // Step 2 backward: RMSNorm on Q and K
        var dQ = RMSNormBackward(
            dQNormed, _lastQuery, _queryNormScale, _lastQueryRmsInv,
            _queryNormScaleGradient, batchSize, seqLen);
        var dK = RMSNormBackward(
            dKNormed, _lastKey, _keyNormScale, _lastKeyRmsInv,
            _keyNormScaleGradient, batchSize, seqLen);

        // Step 1 backward: projection weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradients from all paths
        var dInput = Engine.TensorAdd(dInputFromGate,
            Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));

        var dInput3D = dInput.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    /// <summary>
    /// Backward pass through RMSNorm.
    /// </summary>
    private Tensor<T> RMSNormBackward(
        Tensor<T> dOutput, Tensor<T> input, Tensor<T> scale, Tensor<T> rmsInv,
        Tensor<T> scaleGradient, int batchSize, int seqLen)
    {
        var dInput = new Tensor<T>(input.Shape);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    int dimStart = h * _headDimension;
                    T invRms = rmsInv[new[] { bi, t, h }];

                    // Compute intermediate sums for the RMSNorm gradient
                    T sumDyXhat = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T xNorm = NumOps.Multiply(input[new[] { bi, t, flatD }], invRms);
                        T dy = dOutput[new[] { bi, t, flatD }];
                        T s = scale[new[] { h, d }];

                        // Scale gradient
                        scaleGradient[new[] { h, d }] = NumOps.Add(
                            scaleGradient[new[] { h, d }],
                            NumOps.Multiply(dy, xNorm));

                        T dyScaled = NumOps.Multiply(dy, s);
                        sumDyXhat = NumOps.Add(sumDyXhat,
                            NumOps.Multiply(dyScaled, xNorm));
                    }

                    T invHD = NumOps.FromDouble(1.0 / _headDimension);
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T xNorm = NumOps.Multiply(input[new[] { bi, t, flatD }], invRms);
                        T dy = dOutput[new[] { bi, t, flatD }];
                        T s = scale[new[] { h, d }];
                        T dyScaled = NumOps.Multiply(dy, s);

                        // dInput = invRms * (dyScaled - xNorm * invHD * sumDyXhat)
                        T correction = NumOps.Multiply(xNorm,
                            NumOps.Multiply(invHD, sumDyXhat));
                        dInput[new[] { bi, t, flatD }] = NumOps.Multiply(
                            invRms, NumOps.Subtract(dyScaled, correction));
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
        _queryNormScale = Engine.TensorAdd(_queryNormScale, Engine.TensorMultiplyScalar(_queryNormScaleGradient!, negLR));
        _keyNormScale = Engine.TensorAdd(_keyNormScale, Engine.TensorMultiplyScalar(_keyNormScaleGradient!, negLR));
        _gammas = Engine.TensorAdd(_gammas, Engine.TensorMultiplyScalar(_gammasGradient!, negLR));
        _outputNormScale = Engine.TensorAdd(_outputNormScale, Engine.TensorMultiplyScalar(_outputNormScaleGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

        // Clamp gammas to valid range
        for (int h = 0; h < _numHeads; h++)
        {
            double gVal = NumOps.ToDouble(_gammas[h]);
            if (gVal <= 0.0) _gammas[h] = NumOps.FromDouble(1e-6);
            else if (gVal >= 1.0) _gammas[h] = NumOps.FromDouble(1.0 - 1e-6);
        }
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
        _queryNormScale, _keyNormScale,
        _gammas,
        _outputNormScale,
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
        _lastQueryNormed = null;
        _lastKeyNormed = null;
        _lastQueryRmsInv = null;
        _lastKeyRmsInv = null;
        _lastAttnRaw = null;
        _lastAttnNormed = null;
        _lastAttnRmsInv = null;
        _lastGateRaw = null;
        _lastGate = null;
        _lastStates = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _queryNormScaleGradient = null;
        _keyNormScaleGradient = null;
        _gammasGradient = null;
        _outputNormScaleGradient = null;
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
        metadata["DecayRate"] = _decayRate.ToString("F4");

        var gammaValues = new string[_numHeads];
        for (int h = 0; h < _numHeads; h++)
            gammaValues[h] = NumOps.ToDouble(_gammas[h]).ToString("F6");
        metadata["PerHeadDecayRates"] = string.Join(", ", gammaValues);

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
    /// Gets the per-head decay rates (gammas) for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each head has its own decay rate gamma in (0, 1). These are initialized near the specified
    /// decay rate but with slight offsets to create multi-scale context capture.
    /// </para>
    /// <para><b>For Beginners:</b> These control how quickly each head forgets old information.
    /// Different decay rates allow different heads to focus on different time scales.</para>
    /// </remarks>
    public Tensor<T> GetDecayRates() => _gammas;
}
