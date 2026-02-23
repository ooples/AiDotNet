using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the MEGA (Moving Average Equipped Gated Attention) layer from Ma et al., 2023.
/// </summary>
/// <remarks>
/// <para>
/// MEGA combines an exponential moving average (EMA) with gated single-head attention to capture
/// both position-aware local smoothing and content-based global mixing. It achieves strong results
/// on sequence modeling benchmarks while being significantly more efficient than multi-head attention.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Multi-dimensional EMA:
///      h_t = alpha * h_{t-1} + (1 - alpha) * x_t
///      where alpha is a learned per-dimension decay parameter in (0, 1).
///      This provides position-aware smoothing of the input sequence.
///
///   2. Gated Attention:
///      - Project EMA output to Q, K, and the original input to V
///      - Compute single-head attention: A = softmax(Q K^T / sqrt(d))
///      - Weighted sum: attention_output = A V
///      - Output gate: y = gate * attention_output
///
///   3. Output projection maps the gated attention output back to model dimension.
/// </code>
/// </para>
/// <para>
/// The EMA acts as a learned positional prior: it smooths the input with different decay rates
/// per dimension, so some dimensions capture very local context (fast decay) while others retain
/// long-range information (slow decay). The attention mechanism then operates on these position-aware
/// representations, making it easier to learn both local and global dependencies.
/// </para>
/// <para><b>For Beginners:</b> MEGA is like having two complementary systems working together:
///
/// 1. The EMA is like a set of "smoothing filters" applied to the input sequence. Imagine running
///    your finger along a line in a graph to smooth out bumps -- each dimension has its own smoothing
///    strength. Some dimensions smooth heavily (capturing broad trends), others barely smooth at all
///    (preserving sharp details).
///
/// 2. The attention mechanism then looks at these smoothed representations and decides which positions
///    are important for each output. Because the input has already been smoothed, the attention can
///    focus on content similarity rather than having to also learn positional patterns.
///
/// Together, EMA handles "where to look" (position-aware) and attention handles "what's relevant"
/// (content-aware), dividing the labor efficiently.
/// </para>
/// <para>
/// <b>Reference:</b> Ma et al., "MEGA: Moving Average Equipped Gated Attention", ICLR 2023.
/// https://arxiv.org/abs/2209.10655
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MEGALayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _emaDimension;

    // EMA parameters: learned decay alpha per EMA dimension [emaDimension]
    // Stored as logit so sigmoid(alphaLogit) = alpha in (0,1)
    private Tensor<T> _emaAlphaLogit;

    // EMA projection: input -> EMA space [modelDim, emaDimension]
    private Tensor<T> _emaProjectInWeights;
    private Tensor<T> _emaProjectInBias;

    // EMA projection: EMA space -> model [emaDimension, modelDim]
    private Tensor<T> _emaProjectOutWeights;
    private Tensor<T> _emaProjectOutBias;

    // Q, K projections from EMA output: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _queryBias;
    private Tensor<T> _keyWeights;
    private Tensor<T> _keyBias;

    // V projection from original input: [modelDim, modelDim]
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastEmaInput;
    private Tensor<T>? _lastEmaStates;
    private Tensor<T>? _lastEmaProjected;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastAttnScores;
    private Tensor<T>? _lastAttnOutput;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastGate;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _emaAlphaLogitGradient;
    private Tensor<T>? _emaProjectInWeightsGradient;
    private Tensor<T>? _emaProjectInBiasGradient;
    private Tensor<T>? _emaProjectOutWeightsGradient;
    private Tensor<T>? _emaProjectOutBiasGradient;
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
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
    /// Gets the EMA dimension.
    /// </summary>
    public int EmaDimension => _emaDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _emaAlphaLogit.Length +
        _emaProjectInWeights.Length + _emaProjectInBias.Length +
        _emaProjectOutWeights.Length + _emaProjectOutBias.Length +
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new MEGA (Moving Average Equipped Gated Attention) layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of the vector representing each token.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 4.
    /// <para><b>For Beginners:</b> Each head can focus on different aspects of the input.
    /// MEGA typically uses fewer heads than standard Transformers because the EMA already
    /// provides positional information. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="emaDimension">
    /// Dimension of the EMA state. Default: 16.
    /// <para><b>For Beginners:</b> How many different "smoothing channels" the EMA uses.
    /// Each channel learns its own decay rate, creating a multi-scale smoothing effect.
    /// Larger values capture more nuanced position patterns but use more memory.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public MEGALayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 4,
        int emaDimension = 16,
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
        if (emaDimension <= 0)
            throw new ArgumentException($"EMA dimension ({emaDimension}) must be positive.", nameof(emaDimension));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _emaDimension = emaDimension;

        _emaAlphaLogit = new Tensor<T>([emaDimension]);
        _emaProjectInWeights = new Tensor<T>([modelDimension, emaDimension]);
        _emaProjectInBias = new Tensor<T>([emaDimension]);
        _emaProjectOutWeights = new Tensor<T>([emaDimension, modelDimension]);
        _emaProjectOutBias = new Tensor<T>([modelDimension]);

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);

        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Initialize alpha logits so sigmoid gives a range of decay rates
        // Logit ~ 1.0 -> sigmoid ~ 0.73 (moderate decay)
        // Spread logits so different dimensions have different decay speeds
        for (int i = 0; i < _emaDimension; i++)
        {
            double logit = 0.5 + 2.0 * i / _emaDimension;
            _emaAlphaLogit[i] = NumOps.FromDouble(logit);
        }

        InitializeTensor2D(_emaProjectInWeights);
        _emaProjectInBias.Fill(NumOps.Zero);
        InitializeTensor2D(_emaProjectOutWeights);
        _emaProjectOutBias.Fill(NumOps.Zero);
        InitializeTensor2D(_queryWeights);
        _queryBias.Fill(NumOps.Zero);
        InitializeTensor2D(_keyWeights);
        _keyBias.Fill(NumOps.Zero);
        InitializeTensor2D(_valueWeights);
        _valueBias.Fill(NumOps.Zero);
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

        // Step 1: Project input to EMA space
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var emaInput = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _emaProjectInWeights),
            _emaProjectInBias.Reshape(1, _emaDimension)).Reshape(batchSize, seqLen, _emaDimension);
        _lastEmaInput = emaInput;

        // Step 2: Apply multi-dimensional EMA: h_t = alpha * h_{t-1} + (1 - alpha) * x_t
        var emaOutput = EmaForward(emaInput, batchSize, seqLen);

        // Step 3: Project EMA output back to model dimension
        var emaFlat = emaOutput.Reshape(batchSize * seqLen, _emaDimension);
        var emaProjected = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(emaFlat, _emaProjectOutWeights),
            _emaProjectOutBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        _lastEmaProjected = emaProjected;

        // Step 4: Compute Q, K from EMA output, V from original input
        var emaFlatFull = emaProjected.Reshape(batchSize * seqLen, _modelDimension);
        var q = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(emaFlatFull, _queryWeights),
            _queryBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var k = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(emaFlatFull, _keyWeights),
            _keyBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var v = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _valueWeights),
            _valueBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 5: Compute gated attention
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Sigmoid(gateRaw);
        _lastGateRaw = gateRaw;
        _lastGate = gate;

        // Step 6: Multi-head causal attention
        var attnOutput = CausalAttentionForward(q, k, v, batchSize, seqLen);
        _lastAttnOutput = attnOutput;

        // Step 7: Gated output
        var gatedOutput = Engine.TensorMultiply(gate, attnOutput);

        // Step 8: Output projection
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
    /// Multi-dimensional EMA forward pass.
    /// h_t = alpha * h_{t-1} + (1 - alpha) * x_t for each dimension independently.
    /// </summary>
    private Tensor<T> EmaForward(Tensor<T> input, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _emaDimension });

        // Store all states for backward: [batch, seqLen+1, emaDim]
        var states = new Tensor<T>(new[] { batchSize, seqLen + 1, _emaDimension });
        // States at t=0 are zero (already initialized)

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _emaDimension; d++)
                {
                    // alpha = sigmoid(alphaLogit) -> in (0, 1)
                    T alphaLogit = _emaAlphaLogit[d];
                    T expNeg = NumOps.Exp(NumOps.Negate(alphaLogit));
                    T alpha = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));
                    T oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);

                    T prevState = states[new[] { bi, t, d }];
                    T xVal = input[new[] { bi, t, d }];

                    T newState = NumOps.Add(
                        NumOps.Multiply(alpha, prevState),
                        NumOps.Multiply(oneMinusAlpha, xVal));

                    states[new[] { bi, t + 1, d }] = newState;
                    output[new[] { bi, t, d }] = newState;
                }
            }
        }

        _lastEmaStates = states;
        return output;
    }

    /// <summary>
    /// Multi-head causal softmax attention.
    /// </summary>
    private Tensor<T> CausalAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T headScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Store attention scores: [batch, numHeads, seqLen, seqLen]
        var attnScores = new Tensor<T>(new[] { batchSize, _numHeads, seqLen, seqLen });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                // Compute causal attention scores and apply softmax row-wise
                for (int i = 0; i < seqLen; i++)
                {
                    // Compute Q_i * K_j^T for j <= i
                    var rawScores = new T[i + 1];
                    T maxScore = NumOps.FromDouble(double.NegativeInfinity);

                    for (int j = 0; j <= i; j++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(q[new[] { bi, i, flatD }], k[new[] { bi, j, flatD }]));
                        }
                        rawScores[j] = NumOps.Multiply(dot, headScale);
                        if (NumOps.ToDouble(rawScores[j]) > NumOps.ToDouble(maxScore))
                            maxScore = rawScores[j];
                    }

                    // Softmax
                    T sumExp = NumOps.Zero;
                    var expScores = new T[i + 1];
                    for (int j = 0; j <= i; j++)
                    {
                        expScores[j] = NumOps.Exp(NumOps.Subtract(rawScores[j], maxScore));
                        sumExp = NumOps.Add(sumExp, expScores[j]);
                    }

                    T sumExpInv = NumOps.Divide(NumOps.One, NumOps.Add(sumExp, NumOps.FromDouble(1e-10)));
                    for (int j = 0; j <= i; j++)
                    {
                        T weight = NumOps.Multiply(expScores[j], sumExpInv);
                        attnScores[new[] { bi, hi, i, j }] = weight;
                    }

                    // Weighted sum of values
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T val = NumOps.Zero;
                        for (int j = 0; j <= i; j++)
                        {
                            val = NumOps.Add(val,
                                NumOps.Multiply(attnScores[new[] { bi, hi, i, j }],
                                    v[new[] { bi, j, flatD }]));
                        }
                        output[new[] { bi, i, flatD }] = val;
                    }
                }
            }
        }

        _lastAttnScores = attnScores;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastEmaInput == null ||
            _lastEmaStates == null || _lastEmaProjected == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastAttnScores == null || _lastAttnOutput == null ||
            _lastGate == null || _lastGateRaw == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize gradients
        _emaAlphaLogitGradient = new Tensor<T>([_emaDimension]);
        _emaProjectInWeightsGradient = new Tensor<T>([_modelDimension, _emaDimension]);
        _emaProjectInBiasGradient = new Tensor<T>([_emaDimension]);
        _emaProjectOutWeightsGradient = new Tensor<T>([_emaDimension, _modelDimension]);
        _emaProjectOutBiasGradient = new Tensor<T>([_modelDimension]);
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryBiasGradient = new Tensor<T>([_modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyBiasGradient = new Tensor<T>([_modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueBiasGradient = new Tensor<T>([_modelDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 8 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedOutput = Engine.TensorMultiply(_lastGate, _lastAttnOutput);
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 7 backward: gating
        var dAttnOutput = Engine.TensorMultiply(dGated, _lastGate);
        var dGateSigmoid = Engine.TensorMultiply(dGated, _lastAttnOutput);

        // Sigmoid derivative: sig * (1 - sig)
        var sigDeriv = Engine.TensorMultiply(_lastGate,
            Engine.TensorSubtract(CreateOnesLike(_lastGate), _lastGate));
        var dGateRaw = Engine.TensorMultiply(dGateSigmoid, sigDeriv);

        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 6 backward: causal attention
        T headScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int i = 0; i < seqLen; i++)
                {
                    // dScore[i,j] = sum_d dAttnOutput[i,d] * V[j,d]
                    var dScores = new T[i + 1];
                    for (int j = 0; j <= i; j++)
                    {
                        T ds = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            ds = NumOps.Add(ds,
                                NumOps.Multiply(dAttnOutput[new[] { bi, i, flatD }],
                                    _lastValue[new[] { bi, j, flatD }]));
                        }
                        dScores[j] = ds;
                    }

                    // Softmax backward
                    T dotAS = NumOps.Zero;
                    for (int j = 0; j <= i; j++)
                    {
                        T attnW = _lastAttnScores[new[] { bi, hi, i, j }];
                        dotAS = NumOps.Add(dotAS, NumOps.Multiply(attnW, dScores[j]));
                    }

                    for (int j = 0; j <= i; j++)
                    {
                        T attnW = _lastAttnScores[new[] { bi, hi, i, j }];
                        T dRaw = NumOps.Multiply(attnW, NumOps.Subtract(dScores[j], dotAS));
                        T dRawScaled = NumOps.Multiply(dRaw, headScale);

                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dQ[new[] { bi, i, flatD }] = NumOps.Add(
                                dQ[new[] { bi, i, flatD }],
                                NumOps.Multiply(dRawScaled, _lastKey[new[] { bi, j, flatD }]));
                            dK[new[] { bi, j, flatD }] = NumOps.Add(
                                dK[new[] { bi, j, flatD }],
                                NumOps.Multiply(dRawScaled, _lastQuery[new[] { bi, i, flatD }]));
                        }

                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dV[new[] { bi, j, flatD }] = NumOps.Add(
                                dV[new[] { bi, j, flatD }],
                                NumOps.Multiply(attnW, dAttnOutput[new[] { bi, i, flatD }]));
                        }
                    }
                }
            }
        }

        // Steps 4-5 backward: Q,K,V projection weight gradients
        var emaFlatFull = _lastEmaProjected.Reshape(batchSize * seqLen, _modelDimension);
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(emaFlatFull.Transpose([1, 0]), dQFlat);
        _queryBiasGradient = Engine.ReduceSum(dQ, new int[] { 0, 1 });
        _keyWeightsGradient = Engine.TensorMatMul(emaFlatFull.Transpose([1, 0]), dKFlat);
        _keyBiasGradient = Engine.ReduceSum(dK, new int[] { 0, 1 });
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);
        _valueBiasGradient = Engine.ReduceSum(dV, new int[] { 0, 1 });

        // Input gradient from V path
        var dInputFromV = Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0]));

        // Gradient through EMA projected -> Q,K weights -> EMA projected
        var dEmaProjected = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dEmaProjected = Engine.TensorAdd(dEmaProjected,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));

        // Step 3 backward: EMA project-out
        var dEmaProjected3D = dEmaProjected.Reshape(batchSize, seqLen, _modelDimension);
        var emaOutputFlat = new Tensor<T>(new[] { batchSize * seqLen, _emaDimension });
        for (int bi = 0; bi < batchSize; bi++)
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < _emaDimension; d++)
                    emaOutputFlat[new[] { bi * seqLen + t, d }] = _lastEmaStates[new[] { bi, t + 1, d }];

        _emaProjectOutWeightsGradient = Engine.TensorMatMul(
            emaOutputFlat.Transpose([1, 0]), dEmaProjected);
        _emaProjectOutBiasGradient = Engine.ReduceSum(dEmaProjected3D, new int[] { 0, 1 });

        var dEmaOutput = Engine.TensorMatMul(dEmaProjected, _emaProjectOutWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _emaDimension);

        // Step 2 backward: EMA recurrence (reverse pass)
        var dEmaInput = EmaBackward(dEmaOutput, batchSize, seqLen);

        // Step 1 backward: EMA project-in
        var dEmaInput3D = dEmaInput;
        var dEmaInputFlat = dEmaInput3D.Reshape(batchSize * seqLen, _emaDimension);
        _emaProjectInWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dEmaInputFlat);
        _emaProjectInBiasGradient = Engine.ReduceSum(dEmaInput3D, new int[] { 0, 1 });

        var dInputFromEma = Engine.TensorMatMul(dEmaInputFlat, _emaProjectInWeights.Transpose([1, 0]));

        // Sum all input gradients
        var dInputTotal = Engine.TensorAdd(dInputFromGate, dInputFromV);
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromEma);

        var dInput = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

    /// <summary>
    /// Backward pass through the EMA recurrence.
    /// </summary>
    private Tensor<T> EmaBackward(Tensor<T> dOutput, int batchSize, int seqLen)
    {
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _emaDimension });

        for (int d = 0; d < _emaDimension; d++)
        {
            T alphaLogit = _emaAlphaLogit[d];
            T expNeg = NumOps.Exp(NumOps.Negate(alphaLogit));
            T alpha = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));
            T oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);
            // sigmoid derivative: alpha * (1 - alpha)
            T alphaDeriv = NumOps.Multiply(alpha, oneMinusAlpha);

            for (int bi = 0; bi < batchSize; bi++)
            {
                T dState = NumOps.Zero;

                for (int t = seqLen - 1; t >= 0; t--)
                {
                    T dOut = dOutput[new[] { bi, t, d }];
                    // h_t = alpha * h_{t-1} + (1-alpha) * x_t
                    // dh_t = dOut + alpha * dh_{t+1}   (from future)
                    T dH = NumOps.Add(dOut, dState);

                    // dx_t = dH * (1 - alpha)
                    dInput[new[] { bi, t, d }] = NumOps.Multiply(dH, oneMinusAlpha);

                    // dAlpha += dH * (h_{t-1} - x_t)
                    T prevState = _lastEmaStates![new[] { bi, t, d }];
                    T xVal = _lastEmaInput![new[] { bi, t, d }];
                    T dAlphaContrib = NumOps.Multiply(dH,
                        NumOps.Multiply(NumOps.Subtract(prevState, xVal), alphaDeriv));
                    _emaAlphaLogitGradient![d] = NumOps.Add(
                        _emaAlphaLogitGradient[d], dAlphaContrib);

                    // Propagate to previous timestep
                    dState = NumOps.Multiply(alpha, dH);
                }
            }
        }

        return dInput;
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
        _emaAlphaLogit = Engine.TensorAdd(_emaAlphaLogit, Engine.TensorMultiplyScalar(_emaAlphaLogitGradient!, negLR));
        _emaProjectInWeights = Engine.TensorAdd(_emaProjectInWeights, Engine.TensorMultiplyScalar(_emaProjectInWeightsGradient!, negLR));
        _emaProjectInBias = Engine.TensorAdd(_emaProjectInBias, Engine.TensorMultiplyScalar(_emaProjectInBiasGradient!, negLR));
        _emaProjectOutWeights = Engine.TensorAdd(_emaProjectOutWeights, Engine.TensorMultiplyScalar(_emaProjectOutWeightsGradient!, negLR));
        _emaProjectOutBias = Engine.TensorAdd(_emaProjectOutBias, Engine.TensorMultiplyScalar(_emaProjectOutBiasGradient!, negLR));
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _queryBias = Engine.TensorAdd(_queryBias, Engine.TensorMultiplyScalar(_queryBiasGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _keyBias = Engine.TensorAdd(_keyBias, Engine.TensorMultiplyScalar(_keyBiasGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _valueBias = Engine.TensorAdd(_valueBias, Engine.TensorMultiplyScalar(_valueBiasGradient!, negLR));
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
        _emaAlphaLogit,
        _emaProjectInWeights, _emaProjectInBias,
        _emaProjectOutWeights, _emaProjectOutBias,
        _queryWeights, _queryBias,
        _keyWeights, _keyBias,
        _valueWeights, _valueBias,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastEmaInput = null;
        _lastEmaStates = null;
        _lastEmaProjected = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastAttnScores = null;
        _lastAttnOutput = null;
        _lastGateRaw = null;
        _lastGate = null;
        _originalInputShape = null;
        _emaAlphaLogitGradient = null;
        _emaProjectInWeightsGradient = null;
        _emaProjectInBiasGradient = null;
        _emaProjectOutWeightsGradient = null;
        _emaProjectOutBiasGradient = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
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
        metadata["EmaDimension"] = _emaDimension.ToString();
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
    /// Gets the EMA alpha logit parameters for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The actual decay rates are sigmoid(alphaLogit). Values closer to 1 mean slower decay
    /// (longer memory), while values closer to 0 mean faster decay (shorter memory).
    /// </para>
    /// <para><b>For Beginners:</b> These control how much each EMA dimension "remembers" from the past.
    /// The raw logit values are transformed by sigmoid to keep them between 0 and 1.</para>
    /// </remarks>
    public Tensor<T> GetEmaAlphaLogit() => _emaAlphaLogit;
}
