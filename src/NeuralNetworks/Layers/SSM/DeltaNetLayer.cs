using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the DeltaNet layer from "Linear Transformers with Learnable Kernel Functions" (Yang et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// DeltaNet applies the delta rule to linear attention, replacing the naive accumulation of key-value
/// outer products with an error-corrective update. This produces a linear-complexity recurrent model
/// that is significantly more expressive than standard linear attention.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute beta (write strength) per head via learned projection + sigmoid
///   3. Delta rule state update per head:
///      S_t = S_{t-1} + beta_t * (v_t - S_{t-1} * k_t) outer_product k_t
///      The (v - S*k) term is the "delta rule": it only writes the DIFFERENCE
///      between the target v and what the state would currently retrieve for key k.
///   4. Output: o_t = S_t * q_t
///   5. Output projection
/// </code>
/// </para>
/// <para>
/// This is the "ungated" version of GatedDeltaNet: there is no alpha forget gate, no output gate,
/// and no short convolution. The state is purely additive (S_{t-1} carries forward with weight 1),
/// and the only learned control is beta which modulates how strongly corrections are written.
/// </para>
/// <para>
/// The delta rule update is key: instead of blindly accumulating K*V outer products (like linear
/// attention), it computes the error (V - S*K) first and updates accordingly. This is exactly the
/// Widrow-Hoff / delta rule from neural network learning theory, applied to a fast weight matrix
/// at each timestep.
/// </para>
/// <para><b>For Beginners:</b> DeltaNet is a simpler, foundational variant of GatedDeltaNet.
///
/// Think of the state matrix S as a "lookup table" that maps keys to values:
/// - Linear attention: "Just add every key-value pair to the table" -> entries pile up, old ones never corrected
/// - Delta rule: "Before adding, check what S already predicts for this key. Only write the correction."
///
/// This is like the difference between:
/// - Writing every flashcard answer on top of the previous one (linear attention -> messy)
/// - Erasing only the wrong part and writing the correction (delta rule -> clean)
///
/// The beta parameter controls how much of the correction to actually apply:
/// - beta near 0: "I trust the existing memory, barely update"
/// - beta near 1: "Fully overwrite whatever was stored for this key"
///
/// Because there is no forget gate (alpha) or output gate, this model is simpler and faster than
/// GatedDeltaNet, but may underperform on tasks that require selective forgetting or output gating.
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "Linear Transformers with Learnable Kernel Functions", 2024.
/// https://arxiv.org/abs/2406.06484
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DeltaNetLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _queryBias;
    private Tensor<T> _keyWeights;
    private Tensor<T> _keyBias;
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;

    // Beta (write strength) projection: [modelDim, numHeads]
    private Tensor<T> _betaWeights;
    private Tensor<T> _betaBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastBeta;
    private Tensor<T>? _lastStates;
    private Tensor<T>? _lastDeltaRuleOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
    private Tensor<T>? _betaWeightsGradient;
    private Tensor<T>? _betaBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Gets the model dimension (d_model).
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head (modelDimension / numHeads).
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _betaWeights.Length + _betaBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new DeltaNet layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length this layer will process.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of each token's representation vector.
    /// Larger values capture more information but require more computation.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own fast weight matrix S.
    /// Must evenly divide modelDimension. More heads let the model attend to different
    /// aspects of the input simultaneously.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public DeltaNetLayer(
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
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);
        _betaWeights = new Tensor<T>([modelDimension, numHeads]);
        _betaBias = new Tensor<T>([numHeads]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes all trainable parameters using Xavier/Glorot initialization for weight matrices
    /// and appropriate constants for biases.
    /// </summary>
    private void InitializeParameters()
    {
        InitializeTensor2D(_queryWeights);
        _queryBias.Fill(NumOps.Zero);
        InitializeTensor2D(_keyWeights);
        _keyBias.Fill(NumOps.Zero);
        InitializeTensor2D(_valueWeights);
        _valueBias.Fill(NumOps.Zero);
        InitializeTensor2D(_betaWeights);
        // Beta bias ~ 0.1 so sigmoid(0.1) ~ 0.52 -> moderate initial write strength
        _betaBias.Fill(NumOps.FromDouble(0.1));
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Applies Xavier/Glorot uniform initialization to a 2D weight tensor.
    /// </summary>
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

        var qFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _queryWeights),
            _queryBias.Reshape(1, _modelDimension));
        var q = qFlat.Reshape(batchSize, seqLen, _modelDimension);

        var kFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _keyWeights),
            _keyBias.Reshape(1, _modelDimension));
        var k = kFlat.Reshape(batchSize, seqLen, _modelDimension);

        var vFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _valueWeights),
            _valueBias.Reshape(1, _modelDimension));
        var v = vFlat.Reshape(batchSize, seqLen, _modelDimension);

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Beta (write strength) via sigmoid
        var betaRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _betaWeights),
            _betaBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var beta = Engine.Sigmoid(betaRaw);
        _lastBeta = beta;

        // Step 3: Delta rule recurrence per head
        var output = DeltaRuleForward(q, k, v, beta, batchSize, seqLen);
        _lastDeltaRuleOutput = output;

        // Step 4: Output projection
        var outputFlat = Engine.TensorMatMul(
            output.Reshape(batchSize * seqLen, _modelDimension),
            _outputProjectionWeights);
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
    /// Delta rule forward: error-corrective fast weight update without gating.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For each timestep t and head h, the recurrence is:
    /// <code>
    ///   S_t = S_{t-1} + beta_t * (v_t - S_{t-1} * k_t) outer k_t
    ///   o_t = S_t * q_t
    /// </code>
    /// Note the implicit alpha = 1 (no forgetting). The state S accumulates indefinitely,
    /// with the delta rule correction preventing unbounded growth by only writing errors.
    /// </para>
    /// </remarks>
    private Tensor<T> DeltaRuleForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> beta,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State matrix per head: [batch, numHeads, headDim, headDim]
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        // Save all states for backward pass: [batch, seqLen+1, numHeads, headDim, headDim]
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T betaVal = beta[new[] { bi, t, hi }];

                    // Retrieve current state's prediction for this key: S * k
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

                    // Delta: v - S*k (the error/correction term)
                    var delta = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        delta[di] = NumOps.Subtract(v[new[] { bi, t, flatDi }], sK[di]);
                    }

                    // State update: S = S + beta * delta * k^T  (no alpha, implicit alpha = 1)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);

                            T prevS = state[new[] { bi, hi, di, ki }];
                            T update = NumOps.Multiply(betaVal,
                                NumOps.Multiply(delta[di], kVal));
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
                            T qVal = q[new[] { bi, t, flatKi }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(state[new[] { bi, hi, di, ki }], qVal));
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
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastBeta == null || _lastDeltaRuleOutput == null || _lastStates == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryBiasGradient = new Tensor<T>([_modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyBiasGradient = new Tensor<T>([_modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueBiasGradient = new Tensor<T>([_modelDimension]);
        _betaWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _betaBiasGradient = new Tensor<T>([_numHeads]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 4 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var deltaRuleOutFlat = _lastDeltaRuleOutput.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(
            deltaRuleOutFlat.Transpose([1, 0]), gradFlat);

        var dDeltaRuleOut = Engine.TensorMatMul(
            gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 3 backward: delta rule recurrence (reverse time)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dBeta = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T betaVal = _lastBeta[new[] { bi, t, hi }];

                    // --- Backward through: o = S_t * q ---
                    // dS_t += dO outer q, dQ += S_t^T * dO
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dDeltaRuleOut[new[] { bi, t, flatDi }];

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T qVal = _lastQuery[new[] { bi, t, flatKi }];
                            T sVal = _lastStates[new[] { bi, t + 1, hi, di, ki }];

                            dState[new[] { bi, hi, di, ki }] = NumOps.Add(
                                dState[new[] { bi, hi, di, ki }],
                                NumOps.Multiply(dO, qVal));

                            dQ[new[] { bi, t, flatKi }] = NumOps.Add(
                                dQ[new[] { bi, t, flatKi }],
                                NumOps.Multiply(dO, sVal));
                        }
                    }

                    // --- Backward through: S_t = S_{t-1} + beta * delta * k^T ---
                    // where delta[di] = v[di] - (S_{t-1} * k)[di]
                    //
                    // dS_{t-1} += dS_t  (alpha = 1 implicitly, so dS propagates unchanged)
                    //           - beta * (k^T * dS_t * k) contribution from delta dependency on S_{t-1}
                    // dBeta += sum over (di,ki) of dS[di,ki] * delta[di] * k[ki]
                    // dV[di] += beta * sum_ki(dS[di,ki] * k[ki])
                    // dK[ki] += beta * sum_di(delta[di] * dS[di,ki])
                    //         - beta * sum_di(dS[di,ki'] * S_{t-1}[di,ki] * ...) (from S*k in delta)

                    // Recompute delta for this timestep
                    var deltaDi = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T vVal = _lastValue[new[] { bi, t, flatDi }];
                        T sK_di = NumOps.Zero;
                        for (int ki2 = 0; ki2 < _headDimension; ki2++)
                        {
                            int flatKi2 = dimStart + ki2;
                            T kVal2 = NumOps.Multiply(_lastKey[new[] { bi, t, flatKi2 }], keyScale);
                            T sPrev = _lastStates[new[] { bi, t, hi, di, ki2 }];
                            sK_di = NumOps.Add(sK_di, NumOps.Multiply(sPrev, kVal2));
                        }
                        deltaDi[di] = NumOps.Subtract(vVal, sK_di);
                    }

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(_lastKey[new[] { bi, t, flatKi }], keyScale);
                            T dS = dState[new[] { bi, hi, di, ki }];
                            T sPrev = _lastStates[new[] { bi, t, hi, di, ki }];

                            // dBeta: gradient through beta * delta[di] * k[ki]
                            dBeta[new[] { bi, t, hi }] = NumOps.Add(
                                dBeta[new[] { bi, t, hi }],
                                NumOps.Multiply(dS, NumOps.Multiply(deltaDi[di], kVal)));

                            // dV: delta = v - S*k, so d(delta)/dv = I
                            // dV[di] += beta * dS[di,ki] * k[ki]
                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(NumOps.Multiply(betaVal, dS), kVal));

                            // dK: from beta * delta[di] * k[ki] term, dK += beta * delta[di] * dS[di,ki]
                            // and from delta = v - S*k, dK -= beta * S_prev^T * (dS * k) contribution
                            dK[new[] { bi, t, flatKi }] = NumOps.Add(
                                dK[new[] { bi, t, flatKi }],
                                NumOps.Multiply(NumOps.Multiply(betaVal, deltaDi[di]), dS));

                            // Propagate dState to previous timestep
                            // S_t = S_{t-1} + beta * delta * k^T
                            // dS_{t-1} = dS_t (from direct carry-forward, alpha=1)
                            //          - beta * (beta * dS * k * k^T contribution from delta's dependency on S_{t-1})
                            // The S_{t-1} appears inside delta = v - S_{t-1}*k, so
                            // d/dS_{t-1}[di',ki'] of (beta * (v[di] - sum_j S[di,j]*k[j]) * k[ki])
                            // = -beta * k[ki'] * k[ki] if di'==di, else 0
                            // This means: dS_{t-1}[di,ki'] -= beta * k[ki'] * sum_ki(dS[di,ki] * k[ki])
                            // but that double-sum is expensive. We handle it below after the ki loop.

                            // For now, the direct carry-forward part: dS_{t-1} = dS_t
                            // (we will apply the correction term after this inner loop)
                        }
                    }

                    // Apply corrections for the S_{t-1} dependency through delta = v - S_{t-1}*k
                    // For each di: let c[di] = sum_ki(dS_t[di,ki] * k_scaled[ki])
                    //
                    // dState correction: dS_{t-1}[di,ki'] -= beta * c[di] * k_scaled[ki']
                    // dK correction: dK[ki'] -= beta * sum_di(c[di] * S_{t-1}[di,ki'])
                    //   This is the missing "-beta * S_{t-1}^T * ..." term from delta's k dependency.
                    for (int di = 0; di < _headDimension; di++)
                    {
                        // Compute c[di] = sum_ki(dS[di,ki] * k_scaled[ki])
                        T cDi = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(_lastKey[new[] { bi, t, flatKi }], keyScale);
                            cDi = NumOps.Add(cDi,
                                NumOps.Multiply(dState[new[] { bi, hi, di, ki }], kVal));
                        }

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(_lastKey[new[] { bi, t, flatKi }], keyScale);

                            // dState correction: dS_{t-1}[di,ki'] -= beta * c[di] * k_scaled[ki']
                            T correction = NumOps.Multiply(betaVal, NumOps.Multiply(cDi, kVal));
                            dState[new[] { bi, hi, di, ki }] = NumOps.Subtract(
                                dState[new[] { bi, hi, di, ki }], correction);

                            // dK correction: dK[ki'] -= beta * c[di] * S_{t-1}[di,ki']
                            // This accounts for k appearing inside delta = v - S_{t-1}*k
                            T sPrevDiKi = _lastStates[new[] { bi, t, hi, di, ki }];
                            dK[new[] { bi, t, flatKi }] = NumOps.Subtract(
                                dK[new[] { bi, t, flatKi }],
                                NumOps.Multiply(betaVal, NumOps.Multiply(cDi, sPrevDiKi)));
                        }
                    }
                }
            }
        }

        // Step 2 backward: beta through sigmoid derivative
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        var betaSigDeriv = Engine.TensorMultiply(_lastBeta,
            Engine.TensorSubtract(CreateOnesLike(_lastBeta), _lastBeta));
        var dBetaRaw = Engine.TensorMultiply(dBeta, betaSigDeriv);

        // Step 1 backward: projection weight gradients
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);

        var dBetaFlat = dBetaRaw.Reshape(batchSize * seqLen, _numHeads);
        _betaWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dBetaFlat);
        _betaBiasGradient = Engine.ReduceSum(dBetaRaw, new int[] { 0, 1 });

        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _queryBiasGradient = Engine.ReduceSum(dQ, new int[] { 0, 1 });
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]),
            Engine.TensorMultiplyScalar(dKFlat, keyScale));
        _keyBiasGradient = Engine.ReduceSum(
            Engine.TensorMultiplyScalar(dK, keyScale), new int[] { 0, 1 });
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);
        _valueBiasGradient = Engine.ReduceSum(dV, new int[] { 0, 1 });

        // Input gradient: sum contributions from all projection paths
        var dInput = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(Engine.TensorMultiplyScalar(dKFlat, keyScale),
                _keyWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dBetaFlat, _betaWeights.Transpose([1, 0])));

        var dInput3D = dInput.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    /// <summary>
    /// Creates a tensor of ones with the same shape as the template tensor.
    /// </summary>
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
        _queryBias = Engine.TensorAdd(_queryBias, Engine.TensorMultiplyScalar(_queryBiasGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _keyBias = Engine.TensorAdd(_keyBias, Engine.TensorMultiplyScalar(_keyBiasGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _valueBias = Engine.TensorAdd(_valueBias, Engine.TensorMultiplyScalar(_valueBiasGradient!, negLR));
        _betaWeights = Engine.TensorAdd(_betaWeights, Engine.TensorMultiplyScalar(_betaWeightsGradient!, negLR));
        _betaBias = Engine.TensorAdd(_betaBias, Engine.TensorMultiplyScalar(_betaBiasGradient!, negLR));
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

    /// <summary>
    /// Returns all trainable parameter tensors in a consistent order for serialization.
    /// </summary>
    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _queryBias,
        _keyWeights, _keyBias,
        _valueWeights, _valueBias,
        _betaWeights, _betaBias,
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
        _lastStates = null;
        _lastDeltaRuleOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
        _betaWeightsGradient = null;
        _betaBiasGradient = null;
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
    /// Gets the output projection weights for external inspection or analysis.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the query weights for external inspection or analysis.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;
}
