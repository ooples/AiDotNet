using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the ReBased linear attention layer from "Linearizing Large Language Models" (Bick et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// ReBased replaces standard softmax attention with a linear attention mechanism that uses improved
/// polynomial feature maps. The key idea is to use squared ReLU features to approximate softmax attention,
/// achieving sub-quadratic complexity while maintaining competitive quality.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Apply polynomial kernel feature map: phi(x) = ReLU(x)^2 / ||ReLU(x)^2||
///      The squaring of ReLU features provides a better approximation of the softmax kernel
///      than first-order feature maps (like ELU+1 used in earlier linear attention work).
///   3. Causal linear attention recurrence:
///      S_t = S_{t-1} + phi(k_t) * v_t^T    (accumulate key-value outer products)
///      z_t = z_{t-1} + phi(k_t)             (accumulate normalizer)
///      o_t = S_t * phi(q_t) / (z_t^T * phi(q_t) + eps)
///   4. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The squared ReLU kernel is central: compared to first-order maps like phi(x) = ELU(x) + 1,
/// the quadratic term ReLU(x)^2 creates richer feature interactions that more closely approximate
/// the exponential kernel exp(q*k) used in softmax attention. The L2 normalization prevents the
/// squared features from growing too large, stabilizing training.
/// </para>
/// <para><b>For Beginners:</b> ReBased is a way to make attention much faster for long sequences.
///
/// Standard attention compares every token with every other token, which takes O(n^2) time.
/// ReBased replaces this with a clever trick:
/// - Instead of comparing all pairs, it maintains a running "summary" matrix S
/// - At each step, it updates the summary with the current key-value pair
/// - To compute output, it just multiplies the summary by the query
///
/// The "squared ReLU" feature map is what makes this work well:
/// - ReLU(x) = max(0, x) keeps only positive values
/// - Squaring these positive values (ReLU(x)^2) creates quadratic interactions
/// - This is a much better approximation of what softmax attention does internally
/// - Normalizing by the L2 norm keeps values from exploding
///
/// Think of it like summarizing a book: instead of re-reading the whole book for each question
/// (softmax attention), you maintain a running summary (the state matrix S) and just look up
/// the answer from the summary (multiply by query). The squared ReLU trick makes the summary
/// much more informative than simpler approaches.
/// </para>
/// <para>
/// <b>Reference:</b> Bick et al., "Linearizing Large Language Models", 2024.
/// https://arxiv.org/abs/2402.10644
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RebasedLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

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

    // Cached forward pass values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastPhiQ;
    private Tensor<T>? _lastPhiK;
    private Tensor<T>? _lastPhiQNorm;
    private Tensor<T>? _lastPhiKNorm;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private Tensor<T>? _lastLinearAttnOutput;
    private Tensor<T>? _lastDenominators;
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
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new ReBased linear attention layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each token's representation vector.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head independently computes linear attention with its own
    /// state matrix. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public RebasedLayer(
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

    /// <summary>
    /// Applies the squared ReLU feature map with L2 normalization: phi(x) = ReLU(x)^2 / ||ReLU(x)^2||.
    /// </summary>
    /// <remarks>
    /// The squared ReLU kernel creates richer quadratic feature interactions compared to linear
    /// feature maps. The L2 normalization prevents feature magnitudes from growing unboundedly
    /// as the head dimension increases, stabilizing the linear attention recurrence.
    /// </remarks>
    private void ApplySquaredReluFeatureMap(T[] headVector, T[] featureOutput)
    {
        // Step 1: ReLU(x)^2
        T normSq = NumOps.Zero;
        for (int d = 0; d < _headDimension; d++)
        {
            T val = headVector[d];
            // ReLU: max(0, x)
            T reluVal = NumOps.ToDouble(val) > 0.0 ? val : NumOps.Zero;
            // Square the ReLU output
            T squared = NumOps.Multiply(reluVal, reluVal);
            featureOutput[d] = squared;
            normSq = NumOps.Add(normSq, NumOps.Multiply(squared, squared));
        }

        // Step 2: L2 normalize
        T norm = NumOps.Sqrt(NumOps.Add(normSq, NumOps.FromDouble(1e-8)));
        T normInv = NumOps.Divide(NumOps.One, norm);
        for (int d = 0; d < _headDimension; d++)
            featureOutput[d] = NumOps.Multiply(featureOutput[d], normInv);
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

        // Step 3: Apply squared ReLU feature map and run linear attention recurrence
        var linearOutput = SquaredReluLinearAttentionForward(q, k, v, batchSize, seqLen);
        _lastLinearAttnOutput = linearOutput;

        // Step 4: Gated output
        var gatedOutput = Engine.TensorMultiply(linearOutput, outputGate);

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
    /// Causal linear attention with squared ReLU feature maps.
    /// For each position t:
    ///   S_t = S_{t-1} + phi(k_t) * v_t^T
    ///   z_t = z_{t-1} + phi(k_t)
    ///   o_t = S_t * phi(q_t) / (z_t^T * phi(q_t) + epsilon)
    /// </summary>
    private Tensor<T> SquaredReluLinearAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T epsilon = NumOps.FromDouble(1e-6);

        // Cache feature maps for backward pass
        var phiQCache = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _headDimension });
        var phiKCache = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _headDimension });
        var phiQNormCache = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        var phiKNormCache = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        var denomCache = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        // Per-head state: S[featureDim, headDim] and z[featureDim]
        var stateS = new T[batchSize, _numHeads, _headDimension, _headDimension];
        var stateZ = new T[batchSize, _numHeads, _headDimension];

        var phiQ = new T[_headDimension];
        var phiK = new T[_headDimension];

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Extract head vectors
                    var qHead = new T[_headDimension];
                    var kHead = new T[_headDimension];

                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        qHead[d] = q[new[] { bi, t, flatD }];
                        kHead[d] = k[new[] { bi, t, flatD }];
                    }

                    // Apply squared ReLU feature map: phi(x) = ReLU(x)^2 / ||ReLU(x)^2||
                    ApplySquaredReluFeatureMap(qHead, phiQ);
                    ApplySquaredReluFeatureMap(kHead, phiK);

                    // Cache for backward pass
                    T qNormSq = NumOps.Zero;
                    T kNormSq = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        phiQCache[new[] { bi, t, hi, d }] = phiQ[d];
                        phiKCache[new[] { bi, t, hi, d }] = phiK[d];
                        qNormSq = NumOps.Add(qNormSq, NumOps.Multiply(phiQ[d], phiQ[d]));
                        kNormSq = NumOps.Add(kNormSq, NumOps.Multiply(phiK[d], phiK[d]));
                    }
                    phiQNormCache[new[] { bi, t, hi }] = NumOps.Sqrt(NumOps.Add(qNormSq, NumOps.FromDouble(1e-8)));
                    phiKNormCache[new[] { bi, t, hi }] = NumOps.Sqrt(NumOps.Add(kNormSq, NumOps.FromDouble(1e-8)));

                    // Update state: S += phi(k) * v^T
                    for (int fi = 0; fi < _headDimension; fi++)
                    {
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatD = dimStart + di;
                            T vVal = v[new[] { bi, t, flatD }];
                            stateS[bi, hi, fi, di] = NumOps.Add(
                                stateS[bi, hi, fi, di],
                                NumOps.Multiply(phiK[fi], vVal));
                        }
                        // Update normalizer: z += phi(k)
                        stateZ[bi, hi, fi] = NumOps.Add(stateZ[bi, hi, fi], phiK[fi]);
                    }

                    // Compute output: o = S^T * phi(q), denominator = z^T * phi(q)
                    T denom = epsilon;
                    for (int fi = 0; fi < _headDimension; fi++)
                        denom = NumOps.Add(denom, NumOps.Multiply(stateZ[bi, hi, fi], phiQ[fi]));
                    denomCache[new[] { bi, t, hi }] = denom;

                    for (int di = 0; di < _headDimension; di++)
                    {
                        T oVal = NumOps.Zero;
                        for (int fi = 0; fi < _headDimension; fi++)
                            oVal = NumOps.Add(oVal, NumOps.Multiply(stateS[bi, hi, fi, di], phiQ[fi]));

                        int flatD = dimStart + di;
                        output[new[] { bi, t, flatD }] = NumOps.Divide(oVal, denom);
                    }
                }
            }
        }

        _lastPhiQ = phiQCache;
        _lastPhiK = phiKCache;
        _lastPhiQNorm = phiQNormCache;
        _lastPhiKNorm = phiKNormCache;
        _lastDenominators = denomCache;

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastOutputGate == null || _lastOutputGateRaw == null ||
            _lastLinearAttnOutput == null || _lastPhiQ == null ||
            _lastPhiK == null || _lastDenominators == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 5 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(_lastLinearAttnOutput, _lastOutputGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 4 backward: gating - gatedOutput = linearAttnOutput * outputGate
        var dLinearAttn = Engine.TensorMultiply(dGated, _lastOutputGate);
        var dGateSwish = Engine.TensorMultiply(dGated, _lastLinearAttnOutput);

        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(_lastOutputGateRaw));

        // Output gate weight gradients
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 3 backward: linear attention with squared ReLU features
        var (dQ, dK, dV) = LinearAttentionBackward(dLinearAttn, batchSize, seqLen);

        // Projection weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradient: accumulate from all paths
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
    /// Backward pass for causal linear attention with squared ReLU feature maps.
    /// Uses reverse-mode recurrence to propagate gradients through the running state.
    /// </summary>
    private (Tensor<T> dQ, Tensor<T> dK, Tensor<T> dV) LinearAttentionBackward(
        Tensor<T> dOutput, int batchSize, int seqLen)
    {
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T epsilon = NumOps.FromDouble(1e-6);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                // Recompute forward states for backward
                var states = new T[seqLen + 1, _headDimension, _headDimension];
                var norms = new T[seqLen + 1, _headDimension];

                for (int t = 0; t < seqLen; t++)
                {
                    // Copy previous state
                    for (int fi = 0; fi < _headDimension; fi++)
                    {
                        norms[t + 1, fi] = norms[t, fi];
                        for (int di = 0; di < _headDimension; di++)
                            states[t + 1, fi, di] = states[t, fi, di];
                    }

                    // Update with phi(k) * v^T
                    for (int fi = 0; fi < _headDimension; fi++)
                    {
                        T phiKVal = _lastPhiK![new[] { bi, t, hi, fi }];
                        norms[t + 1, fi] = NumOps.Add(norms[t + 1, fi], phiKVal);
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatD = dimStart + di;
                            states[t + 1, fi, di] = NumOps.Add(
                                states[t + 1, fi, di],
                                NumOps.Multiply(phiKVal, _lastValue![new[] { bi, t, flatD }]));
                        }
                    }
                }

                // Backward: reverse accumulation
                var dS = new T[_headDimension, _headDimension];
                var dZ = new T[_headDimension];

                for (int t = seqLen - 1; t >= 0; t--)
                {
                    T denom = _lastDenominators![new[] { bi, t, hi }];
                    T denomSq = NumOps.Multiply(denom, denom);

                    // Gradient of phi(q) from output computation
                    var dPhiQ = new T[_headDimension];

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;
                        T dO = dOutput[new[] { bi, t, flatD }];

                        // Numerator for this dimension
                        T numVal = NumOps.Zero;
                        for (int fi = 0; fi < _headDimension; fi++)
                            numVal = NumOps.Add(numVal,
                                NumOps.Multiply(states[t + 1, fi, di],
                                    _lastPhiQ![new[] { bi, t, hi, fi }]));

                        for (int fi = 0; fi < _headDimension; fi++)
                        {
                            T phiQVal = _lastPhiQ![new[] { bi, t, hi, fi }];

                            // dS[fi,di] += dO * phiQ[fi] / denom
                            dS[fi, di] = NumOps.Add(dS[fi, di],
                                NumOps.Divide(NumOps.Multiply(dO, phiQVal), denom));

                            // dPhiQ[fi] += dO * (S[fi,di]/denom - numVal*z[fi]/denomSq)
                            T term1 = NumOps.Divide(
                                NumOps.Multiply(dO, states[t + 1, fi, di]), denom);
                            T term2 = NumOps.Divide(
                                NumOps.Multiply(NumOps.Multiply(dO, numVal),
                                    norms[t + 1, fi]), denomSq);
                            dPhiQ[fi] = NumOps.Add(dPhiQ[fi], NumOps.Subtract(term1, term2));
                        }
                    }

                    // dZ contribution
                    for (int fi = 0; fi < _headDimension; fi++)
                    {
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatD = dimStart + di;
                            T dO = dOutput[new[] { bi, t, flatD }];
                            T numVal = NumOps.Zero;
                            for (int fj = 0; fj < _headDimension; fj++)
                                numVal = NumOps.Add(numVal,
                                    NumOps.Multiply(states[t + 1, fj, di],
                                        _lastPhiQ![new[] { bi, t, hi, fj }]));

                            T phiQVal = _lastPhiQ![new[] { bi, t, hi, fi }];
                            dZ[fi] = NumOps.Subtract(dZ[fi],
                                NumOps.Divide(
                                    NumOps.Multiply(NumOps.Multiply(dO, numVal), phiQVal),
                                    denomSq));
                        }
                    }

                    // Propagate through state update
                    var dPhiK = new T[_headDimension];
                    for (int fi = 0; fi < _headDimension; fi++)
                    {
                        dPhiK[fi] = dZ[fi];
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatD = dimStart + di;
                            dPhiK[fi] = NumOps.Add(dPhiK[fi],
                                NumOps.Multiply(dS[fi, di], _lastValue![new[] { bi, t, flatD }]));

                            T phiKVal = _lastPhiK![new[] { bi, t, hi, fi }];
                            dV[new[] { bi, t, flatD }] = NumOps.Add(
                                dV[new[] { bi, t, flatD }],
                                NumOps.Multiply(dS[fi, di], phiKVal));
                        }
                    }

                    // Propagate through squared ReLU feature map with L2 normalization:
                    // phi(x) = ReLU(x)^2 / ||ReLU(x)^2||
                    // Let u = ReLU(x)^2, n = ||u||, phi = u/n
                    // dphi/du = (I - phi*phi^T) / n  (Jacobian of L2 normalization)
                    // du/dx = 2*ReLU(x) for x > 0, 0 otherwise
                    // Chain: dphi/dx = dphi/du * du/dx

                    // Query normalization gradient
                    T qNorm = _lastPhiQNorm![new[] { bi, t, hi }];
                    T qNormInv = NumOps.Divide(NumOps.One, qNorm);

                    // Compute dot(dPhiQ, phiQ) for the normalization correction
                    T dotQ = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                        dotQ = NumOps.Add(dotQ, NumOps.Multiply(dPhiQ[d], _lastPhiQ![new[] { bi, t, hi, d }]));

                    // Key normalization gradient
                    T kNorm = _lastPhiKNorm![new[] { bi, t, hi }];
                    T kNormInv = NumOps.Divide(NumOps.One, kNorm);

                    T dotK = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                        dotK = NumOps.Add(dotK, NumOps.Multiply(dPhiK[d], _lastPhiK![new[] { bi, t, hi, d }]));

                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T qVal = _lastQuery![new[] { bi, t, flatD }];
                        T kVal = _lastKey![new[] { bi, t, flatD }];

                        // dphi/du = (dPhiQ - phiQ * dot(dPhiQ, phiQ)) / norm
                        T phiQd = _lastPhiQ![new[] { bi, t, hi, d }];
                        T dU_Q = NumOps.Multiply(NumOps.Subtract(dPhiQ[d], NumOps.Multiply(phiQd, dotQ)), qNormInv);

                        T phiKd = _lastPhiK![new[] { bi, t, hi, d }];
                        T dU_K = NumOps.Multiply(NumOps.Subtract(dPhiK[d], NumOps.Multiply(phiKd, dotK)), kNormInv);

                        // du/dx = 2*ReLU(x) for x > 0, 0 otherwise
                        T qRelu = NumOps.ToDouble(qVal) > 0.0 ? qVal : NumOps.Zero;
                        T kRelu = NumOps.ToDouble(kVal) > 0.0 ? kVal : NumOps.Zero;

                        T dQ_d = NumOps.Multiply(dU_Q, NumOps.Multiply(NumOps.FromDouble(2.0), qRelu));
                        T dK_d = NumOps.Multiply(dU_K, NumOps.Multiply(NumOps.FromDouble(2.0), kRelu));

                        dQ[new[] { bi, t, flatD }] = NumOps.Add(
                            dQ[new[] { bi, t, flatD }], dQ_d);
                        dK[new[] { bi, t, flatD }] = NumOps.Add(
                            dK[new[] { bi, t, flatD }], dK_d);
                    }
                }
            }
        }

        return (dQ, dK, dV);
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
        _lastPhiQ = null;
        _lastPhiK = null;
        _lastPhiQNorm = null;
        _lastPhiKNorm = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _lastLinearAttnOutput = null;
        _lastDenominators = null;
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
