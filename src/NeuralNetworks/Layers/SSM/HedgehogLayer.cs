using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Hedgehog layer from "The Hedgehog and the Porcupine: Expressive Linear Attentions
/// with Softmax Mimicry" (Zhang et al., 2024, ICLR 2024).
/// </summary>
/// <remarks>
/// <para>
/// Hedgehog replaces the fixed feature map in linear attention with a small trainable MLP that learns
/// to approximate softmax attention from data. Standard linear attention uses simple feature maps like
/// ELU(x)+1 or polynomial expansions, which poorly approximate softmax and lead to degraded quality.
/// Hedgehog instead trains the feature map end-to-end, achieving much better softmax approximation.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Apply trainable feature map MLP to Q and K:
///      phi(x) = W2 * GELU(W1 * x + b1) + b2
///      where W1: [headDim, hiddenDim], W2: [hiddenDim, headDim]
///      This small MLP is trained to make phi(q)^T phi(k) approximate softmax(q^T k / sqrt(d))
///   3. Causal linear attention with the learned features:
///      S_t = S_{t-1} + phi(k_t) * v_t^T        (running state matrix)
///      z_t = z_{t-1} + phi(k_t)                 (running normalizer)
///      o_t = (S_t * phi(q_t)) / (z_t^T * phi(q_t) + eps)
///   4. Output gate: y = swish(X W_g + b_g) * o
///   5. Output projection: output = y W_o + b_o
/// </code>
/// </para>
/// <para>
/// The key insight is that softmax attention can be decomposed as a kernel: softmax(QK^T) = phi(Q) phi(K)^T
/// for some (unknown) feature map phi. Rather than using a fixed approximation, Hedgehog learns phi directly.
/// The MLP is small (typically 64 hidden units) so the overhead is minimal, but the quality improvement
/// over fixed feature maps is substantial -- closing most of the gap to full softmax attention.
/// </para>
/// <para><b>For Beginners:</b> Hedgehog makes linear attention much better by learning HOW to pay attention.
///
/// In standard linear attention, there's a mathematical trick (feature map) that converts the expensive
/// softmax operation into a cheaper form. But the standard tricks (like adding 1 and clipping negatives)
/// are crude approximations. It's like trying to draw a circle using only straight lines -- you can
/// approximate it, but it's never quite right.
///
/// Hedgehog says: "Instead of using a fixed approximation, let's LEARN the best feature map from data."
/// It uses a small neural network (just two layers) that is trained alongside the main model. This learned
/// feature map can capture much more nuanced attention patterns than any fixed formula.
///
/// The result: Hedgehog gets close to the quality of full softmax attention while keeping the speed
/// advantage of linear attention (O(n) instead of O(n^2)).
///
/// Think of it this way:
/// - Standard linear attention: "I'll use this simple formula to decide what's important"
/// - Hedgehog: "I'll learn the BEST formula for deciding what's important from the data itself"
/// </para>
/// <para>
/// <b>Reference:</b> Zhang et al., "The Hedgehog and the Porcupine: Expressive Linear Attentions with Softmax Mimicry", ICLR 2024.
/// https://arxiv.org/abs/2402.04347
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HedgehogLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _featureMapHiddenDim;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Trainable feature map MLP per head: phi(x) = W2 * GELU(W1 * x + b1) + b2
    // W1: [numHeads, headDim, hiddenDim], W2: [numHeads, hiddenDim, headDim]
    // b1: [numHeads, hiddenDim], b2: [numHeads, headDim]
    private Tensor<T> _featureMapW1;
    private Tensor<T> _featureMapB1;
    private Tensor<T> _featureMapW2;
    private Tensor<T> _featureMapB2;

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
    private Tensor<T>? _lastPhiQHidden;
    private Tensor<T>? _lastPhiKHidden;
    private Tensor<T>? _lastPhiQPreActivation;
    private Tensor<T>? _lastPhiKPreActivation;
    private Tensor<T>? _lastAttnOutput;
    private Tensor<T>? _lastAttnDenominators;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastGate;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _featureMapW1Gradient;
    private Tensor<T>? _featureMapB1Gradient;
    private Tensor<T>? _featureMapW2Gradient;
    private Tensor<T>? _featureMapB2Gradient;
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
    /// Gets the hidden dimension of the feature map MLP.
    /// </summary>
    public int FeatureMapHiddenDim => _featureMapHiddenDim;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _featureMapW1.Length + _featureMapB1.Length +
        _featureMapW2.Length + _featureMapB2.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Hedgehog layer with trainable feature maps for linear attention.
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
    /// <para><b>For Beginners:</b> Each head has its own learned feature map MLP, allowing
    /// different heads to develop different attention patterns. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="featureMapHiddenDim">
    /// Hidden dimension of the feature map MLP. Default: 64.
    /// <para><b>For Beginners:</b> The size of the intermediate layer in the small neural network
    /// that learns the feature map. Larger values give more expressive feature maps but add
    /// a small compute overhead. The paper finds 64 works well in most settings.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public HedgehogLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int featureMapHiddenDim = 64,
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
        if (featureMapHiddenDim <= 0)
            throw new ArgumentException($"Feature map hidden dimension ({featureMapHiddenDim}) must be positive.", nameof(featureMapHiddenDim));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _featureMapHiddenDim = featureMapHiddenDim;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);

        _featureMapW1 = new Tensor<T>([numHeads, _headDimension, featureMapHiddenDim]);
        _featureMapB1 = new Tensor<T>([numHeads, featureMapHiddenDim]);
        _featureMapW2 = new Tensor<T>([numHeads, featureMapHiddenDim, _headDimension]);
        _featureMapB2 = new Tensor<T>([numHeads, _headDimension]);

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

        // Initialize feature map MLP weights with small values
        InitializeFeatureMapWeights(_featureMapW1, _headDimension, _featureMapHiddenDim);
        _featureMapB1.Fill(NumOps.Zero);
        InitializeFeatureMapWeights(_featureMapW2, _featureMapHiddenDim, _headDimension);
        _featureMapB2.Fill(NumOps.Zero);

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

    private void InitializeFeatureMapWeights(Tensor<T> tensor, int fanIn, int fanOut)
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
    }

    /// <summary>
    /// Applies the trainable feature map MLP: phi(x) = W2 * GELU(W1 * x + b1) + b2
    /// </summary>
    private void ApplyFeatureMap(
        T[] headVector, int headIndex, T[] featureOutput,
        T[] preActivation, T[] hiddenOutput)
    {
        // Layer 1: hidden = GELU(W1 * x + b1)
        for (int hi = 0; hi < _featureMapHiddenDim; hi++)
        {
            T sum = _featureMapB1[new[] { headIndex, hi }];
            for (int di = 0; di < _headDimension; di++)
            {
                sum = NumOps.Add(sum,
                    NumOps.Multiply(headVector[di], _featureMapW1[new[] { headIndex, di, hi }]));
            }
            preActivation[hi] = sum;

            // GELU approximation: x * sigmoid(1.702 * x)
            T scaled = NumOps.Multiply(NumOps.FromDouble(1.702), sum);
            T expNeg = NumOps.Exp(NumOps.Negate(scaled));
            T sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));
            hiddenOutput[hi] = NumOps.Multiply(sum, sigmoid);
        }

        // Layer 2: output = W2 * hidden + b2
        for (int di = 0; di < _headDimension; di++)
        {
            T sum = _featureMapB2[new[] { headIndex, di }];
            for (int hi = 0; hi < _featureMapHiddenDim; hi++)
            {
                sum = NumOps.Add(sum,
                    NumOps.Multiply(hiddenOutput[hi], _featureMapW2[new[] { headIndex, hi, di }]));
            }
            featureOutput[di] = sum;
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

        // Step 2: Apply trainable feature map to Q and K
        var phiQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var phiK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var phiQHidden = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _featureMapHiddenDim });
        var phiKHidden = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _featureMapHiddenDim });
        var phiQPreAct = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _featureMapHiddenDim });
        var phiKPreAct = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _featureMapHiddenDim });

        var headVec = new T[_headDimension];
        var featureOut = new T[_headDimension];
        var preAct = new T[_featureMapHiddenDim];
        var hiddenOut = new T[_featureMapHiddenDim];

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Apply feature map to Q
                    for (int d = 0; d < _headDimension; d++)
                        headVec[d] = q[new[] { bi, t, dimStart + d }];

                    ApplyFeatureMap(headVec, hi, featureOut, preAct, hiddenOut);

                    for (int d = 0; d < _headDimension; d++)
                        phiQ[new[] { bi, t, dimStart + d }] = featureOut[d];
                    for (int fi = 0; fi < _featureMapHiddenDim; fi++)
                    {
                        phiQHidden[new[] { bi, t, hi, fi }] = hiddenOut[fi];
                        phiQPreAct[new[] { bi, t, hi, fi }] = preAct[fi];
                    }

                    // Apply feature map to K
                    for (int d = 0; d < _headDimension; d++)
                        headVec[d] = k[new[] { bi, t, dimStart + d }];

                    ApplyFeatureMap(headVec, hi, featureOut, preAct, hiddenOut);

                    for (int d = 0; d < _headDimension; d++)
                        phiK[new[] { bi, t, dimStart + d }] = featureOut[d];
                    for (int fi = 0; fi < _featureMapHiddenDim; fi++)
                    {
                        phiKHidden[new[] { bi, t, hi, fi }] = hiddenOut[fi];
                        phiKPreAct[new[] { bi, t, hi, fi }] = preAct[fi];
                    }
                }
            }
        }

        _lastPhiQ = phiQ;
        _lastPhiK = phiK;
        _lastPhiQHidden = phiQHidden;
        _lastPhiKHidden = phiKHidden;
        _lastPhiQPreActivation = phiQPreAct;
        _lastPhiKPreActivation = phiKPreAct;

        // Step 3: Compute output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGateRaw = gateRaw;
        _lastGate = gate;

        // Step 4: Causal linear attention with learned features
        var attnOutput = LinearAttentionForward(phiQ, phiK, v, batchSize, seqLen);
        _lastAttnOutput = attnOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(gate, attnOutput);

        // Step 6: Output projection
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
    /// Causal linear attention with learned feature maps.
    /// S_t = S_{t-1} + phi(k_t) * v_t^T, z_t = z_{t-1} + phi(k_t)
    /// o_t = (S_t * phi(q_t)) / (z_t^T * phi(q_t) + eps)
    /// </summary>
    private Tensor<T> LinearAttentionForward(
        Tensor<T> phiQ, Tensor<T> phiK, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T epsilon = NumOps.FromDouble(1e-6);

        // Store denominators for backward pass
        var denominators = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        // Per-head state: S[headDim, headDim] and z[headDim]
        var stateS = new T[batchSize, _numHeads, _headDimension, _headDimension];
        var stateZ = new T[batchSize, _numHeads, _headDimension];

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Update state: S += phi(k) * v^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T phiKVal = phiK[new[] { bi, t, dimStart + di }];
                        stateZ[bi, hi, di] = NumOps.Add(stateZ[bi, hi, di], phiKVal);

                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T vVal = v[new[] { bi, t, dimStart + dj }];
                            stateS[bi, hi, di, dj] = NumOps.Add(
                                stateS[bi, hi, di, dj],
                                NumOps.Multiply(phiKVal, vVal));
                        }
                    }

                    // Compute normalizer: z^T * phi(q)
                    T denom = epsilon;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T phiQVal = phiQ[new[] { bi, t, dimStart + di }];
                        denom = NumOps.Add(denom, NumOps.Multiply(stateZ[bi, hi, di], phiQVal));
                    }
                    denominators[new[] { bi, t, hi }] = denom;

                    // Compute output: o = S * phi(q) / denom
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T oVal = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T phiQVal = phiQ[new[] { bi, t, dimStart + dj }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(stateS[bi, hi, di, dj], phiQVal));
                        }
                        output[new[] { bi, t, dimStart + di }] = NumOps.Divide(oVal, denom);
                    }
                }
            }
        }

        _lastAttnDenominators = denominators;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastPhiQ == null || _lastPhiK == null ||
            _lastPhiQHidden == null || _lastPhiKHidden == null ||
            _lastPhiQPreActivation == null || _lastPhiKPreActivation == null ||
            _lastAttnOutput == null || _lastAttnDenominators == null ||
            _lastGate == null || _lastGateRaw == null)
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
        _featureMapW1Gradient = new Tensor<T>([_numHeads, _headDimension, _featureMapHiddenDim]);
        _featureMapB1Gradient = new Tensor<T>([_numHeads, _featureMapHiddenDim]);
        _featureMapW2Gradient = new Tensor<T>([_numHeads, _featureMapHiddenDim, _headDimension]);
        _featureMapB2Gradient = new Tensor<T>([_numHeads, _headDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 6 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedOutput = Engine.TensorMultiply(_lastGate, _lastAttnOutput);
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 5 backward: gating
        var dAttnOutput = Engine.TensorMultiply(dGated, _lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, _lastAttnOutput);

        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(_lastGateRaw));
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 4 backward: linear attention
        // We need dPhiQ, dPhiK, dV from the linear attention backward
        var dPhiQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dPhiK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T epsilon = NumOps.FromDouble(1e-6);

        // Reverse-mode recurrence for linear attention backward
        var dS = new T[batchSize, _numHeads, _headDimension, _headDimension];
        var dZ = new T[batchSize, _numHeads, _headDimension];

        // Recompute forward states
        var stateS = new T[batchSize, _numHeads, _headDimension, _headDimension];
        var stateZ = new T[batchSize, _numHeads, _headDimension];

        // Forward pass to rebuild states at each timestep
        var statesAtT = new T[seqLen + 1, batchSize, _numHeads, _headDimension, _headDimension];
        var normsAtT = new T[seqLen + 1, batchSize, _numHeads, _headDimension];

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Copy previous state
                    for (int di = 0; di < _headDimension; di++)
                    {
                        normsAtT[t + 1, bi, hi, di] = normsAtT[t, bi, hi, di];
                        for (int dj = 0; dj < _headDimension; dj++)
                            statesAtT[t + 1, bi, hi, di, dj] = statesAtT[t, bi, hi, di, dj];
                    }

                    for (int di = 0; di < _headDimension; di++)
                    {
                        T phiKVal = _lastPhiK[new[] { bi, t, dimStart + di }];
                        normsAtT[t + 1, bi, hi, di] = NumOps.Add(normsAtT[t + 1, bi, hi, di], phiKVal);
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T vVal = _lastValue[new[] { bi, t, dimStart + dj }];
                            statesAtT[t + 1, bi, hi, di, dj] = NumOps.Add(
                                statesAtT[t + 1, bi, hi, di, dj],
                                NumOps.Multiply(phiKVal, vVal));
                        }
                    }
                }
            }
        }

        // Backward through time
        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;
                    T denom = _lastAttnDenominators[new[] { bi, t, hi }];
                    T denomSq = NumOps.Multiply(denom, denom);

                    // o_di = sum_dj(S[di,dj] * phiQ[dj]) / denom
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T dO = dAttnOutput[new[] { bi, t, dimStart + di }];

                        // Numerator for this dim
                        T numVal = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T phiQVal = _lastPhiQ[new[] { bi, t, dimStart + dj }];
                            numVal = NumOps.Add(numVal,
                                NumOps.Multiply(statesAtT[t + 1, bi, hi, di, dj], phiQVal));
                        }

                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T phiQVal = _lastPhiQ[new[] { bi, t, dimStart + dj }];

                            // dS[di,dj] += dO * phiQ[dj] / denom
                            dS[bi, hi, di, dj] = NumOps.Add(dS[bi, hi, di, dj],
                                NumOps.Divide(NumOps.Multiply(dO, phiQVal), denom));

                            // dPhiQ[dj] += dO * (S[di,dj]/denom - numVal*z[dj]/denomSq)
                            T term1 = NumOps.Divide(
                                NumOps.Multiply(dO, statesAtT[t + 1, bi, hi, di, dj]), denom);
                            T term2 = NumOps.Divide(
                                NumOps.Multiply(NumOps.Multiply(dO, numVal),
                                    normsAtT[t + 1, bi, hi, dj]), denomSq);
                            dPhiQ[new[] { bi, t, dimStart + dj }] = NumOps.Add(
                                dPhiQ[new[] { bi, t, dimStart + dj }],
                                NumOps.Subtract(term1, term2));
                        }
                    }

                    // Propagate through S += phiK * v^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T dSVal = dS[bi, hi, di, dj];
                            T phiKVal = _lastPhiK[new[] { bi, t, dimStart + di }];
                            T vVal = _lastValue[new[] { bi, t, dimStart + dj }];

                            dPhiK[new[] { bi, t, dimStart + di }] = NumOps.Add(
                                dPhiK[new[] { bi, t, dimStart + di }],
                                NumOps.Multiply(dSVal, vVal));

                            dV[new[] { bi, t, dimStart + dj }] = NumOps.Add(
                                dV[new[] { bi, t, dimStart + dj }],
                                NumOps.Multiply(dSVal, phiKVal));
                        }

                        // dZ contribution from normalizer
                        T dZVal = NumOps.Zero;
                        for (int di2 = 0; di2 < _headDimension; di2++)
                        {
                            T dO = dAttnOutput[new[] { bi, t, dimStart + di2 }];
                            T numVal2 = NumOps.Zero;
                            for (int dj = 0; dj < _headDimension; dj++)
                                numVal2 = NumOps.Add(numVal2,
                                    NumOps.Multiply(statesAtT[t + 1, bi, hi, di2, dj],
                                        _lastPhiQ[new[] { bi, t, dimStart + dj }]));

                            T phiQi = _lastPhiQ[new[] { bi, t, dimStart + di }];
                            dZVal = NumOps.Subtract(dZVal,
                                NumOps.Divide(
                                    NumOps.Multiply(NumOps.Multiply(dO, numVal2), phiQi),
                                    denomSq));
                        }

                        dPhiK[new[] { bi, t, dimStart + di }] = NumOps.Add(
                            dPhiK[new[] { bi, t, dimStart + di }], dZVal);
                    }
                }
            }
        }

        // Step 2 backward: feature map MLP backward for both phi(Q) and phi(K)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Backward through phi(Q) feature map
                    FeatureMapBackward(
                        dPhiQ, _lastQuery!, _lastPhiQHidden!, _lastPhiQPreActivation!,
                        dQ, bi, t, hi, dimStart);

                    // Backward through phi(K) feature map
                    FeatureMapBackward(
                        dPhiK, _lastKey!, _lastPhiKHidden!, _lastPhiKPreActivation!,
                        dK, bi, t, hi, dimStart);
                }
            }
        }

        // Step 1 backward: Q, K, V projection weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Accumulate input gradients
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
    /// Backward through the feature map MLP: phi(x) = W2 * GELU(W1 * x + b1) + b2.
    /// Accumulates gradients for W1, b1, W2, b2 and propagates to input.
    /// </summary>
    private void FeatureMapBackward(
        Tensor<T> dPhi, Tensor<T> inputQK, Tensor<T> hidden, Tensor<T> preActivation,
        Tensor<T> dInput, int bi, int t, int hi, int dimStart)
    {
        // dPhi -> W2 backward: dHidden = W2^T * dPhi, dW2 += dPhi * hidden^T
        var dHidden = new T[_featureMapHiddenDim];

        for (int fi = 0; fi < _featureMapHiddenDim; fi++)
        {
            T dH = NumOps.Zero;
            for (int di = 0; di < _headDimension; di++)
            {
                T dPhiVal = dPhi[new[] { bi, t, dimStart + di }];
                T hVal = hidden[new[] { bi, t, hi, fi }];

                // dW2[hi, fi, di] += dPhiVal * hVal
                _featureMapW2Gradient![new[] { hi, fi, di }] = NumOps.Add(
                    _featureMapW2Gradient[new[] { hi, fi, di }],
                    NumOps.Multiply(dPhiVal, hVal));

                // dB2[hi, di] += dPhiVal
                _featureMapB2Gradient![new[] { hi, di }] = NumOps.Add(
                    _featureMapB2Gradient[new[] { hi, di }], dPhiVal);

                dH = NumOps.Add(dH,
                    NumOps.Multiply(dPhiVal, _featureMapW2[new[] { hi, fi, di }]));
            }
            dHidden[fi] = dH;
        }

        // GELU backward: d(GELU)/dx = sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
        var dPreAct = new T[_featureMapHiddenDim];
        for (int fi = 0; fi < _featureMapHiddenDim; fi++)
        {
            T x = preActivation[new[] { bi, t, hi, fi }];
            T scaled = NumOps.Multiply(NumOps.FromDouble(1.702), x);
            T expNeg = NumOps.Exp(NumOps.Negate(scaled));
            T sig = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));
            T sigDeriv = NumOps.Multiply(sig, NumOps.Subtract(NumOps.One, sig));
            T geluDeriv = NumOps.Add(sig,
                NumOps.Multiply(x, NumOps.Multiply(NumOps.FromDouble(1.702), sigDeriv)));
            dPreAct[fi] = NumOps.Multiply(dHidden[fi], geluDeriv);

            // dB1[hi, fi] += dPreAct
            _featureMapB1Gradient![new[] { hi, fi }] = NumOps.Add(
                _featureMapB1Gradient[new[] { hi, fi }], dPreAct[fi]);
        }

        // W1 backward: dInput += W1^T * dPreAct, dW1 += dPreAct * input^T
        for (int di = 0; di < _headDimension; di++)
        {
            T dIn = NumOps.Zero;
            T inputVal = inputQK[new[] { bi, t, dimStart + di }];

            for (int fi = 0; fi < _featureMapHiddenDim; fi++)
            {
                // dW1[hi, di, fi] += dPreAct[fi] * input[di]
                _featureMapW1Gradient![new[] { hi, di, fi }] = NumOps.Add(
                    _featureMapW1Gradient[new[] { hi, di, fi }],
                    NumOps.Multiply(dPreAct[fi], inputVal));

                dIn = NumOps.Add(dIn,
                    NumOps.Multiply(dPreAct[fi], _featureMapW1[new[] { hi, di, fi }]));
            }

            dInput[new[] { bi, t, dimStart + di }] = NumOps.Add(
                dInput[new[] { bi, t, dimStart + di }], dIn);
        }
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
        _featureMapW1 = Engine.TensorAdd(_featureMapW1, Engine.TensorMultiplyScalar(_featureMapW1Gradient!, negLR));
        _featureMapB1 = Engine.TensorAdd(_featureMapB1, Engine.TensorMultiplyScalar(_featureMapB1Gradient!, negLR));
        _featureMapW2 = Engine.TensorAdd(_featureMapW2, Engine.TensorMultiplyScalar(_featureMapW2Gradient!, negLR));
        _featureMapB2 = Engine.TensorAdd(_featureMapB2, Engine.TensorMultiplyScalar(_featureMapB2Gradient!, negLR));
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
        _featureMapW1, _featureMapB1,
        _featureMapW2, _featureMapB2,
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
        _lastPhiQHidden = null;
        _lastPhiKHidden = null;
        _lastPhiQPreActivation = null;
        _lastPhiKPreActivation = null;
        _lastAttnOutput = null;
        _lastAttnDenominators = null;
        _lastGateRaw = null;
        _lastGate = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _featureMapW1Gradient = null;
        _featureMapB1Gradient = null;
        _featureMapW2Gradient = null;
        _featureMapB2Gradient = null;
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
        metadata["FeatureMapHiddenDim"] = _featureMapHiddenDim.ToString();
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
    /// Gets the feature map W1 weights for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// W1 is the first layer of the feature map MLP: hidden = GELU(W1 * x + b1).
    /// Shape: [numHeads, headDim, featureMapHiddenDim].
    /// </para>
    /// <para><b>For Beginners:</b> This is the first layer of the small network that learns
    /// how to transform queries and keys for the attention computation.</para>
    /// </remarks>
    public Tensor<T> GetFeatureMapW1() => _featureMapW1;

    /// <summary>
    /// Gets the feature map W2 weights for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// W2 is the second layer of the feature map MLP: output = W2 * hidden + b2.
    /// Shape: [numHeads, featureMapHiddenDim, headDim].
    /// </para>
    /// <para><b>For Beginners:</b> This is the second layer that produces the final
    /// feature-mapped representation used for linear attention.</para>
    /// </remarks>
    public Tensor<T> GetFeatureMapW2() => _featureMapW2;
}
