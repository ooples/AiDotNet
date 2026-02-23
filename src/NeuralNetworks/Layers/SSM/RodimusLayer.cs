using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Rodimus layer from "Rodimus: Breaking the Accuracy-Efficiency Trade-Off
/// with Efficient Attentions" (He et al., 2025).
/// </summary>
/// <remarks>
/// <para>
/// Rodimus combines a data-dependent tempered selection mechanism with gated linear recurrence
/// to achieve both high quality and linear-time efficiency. The key innovation is using a
/// temperature-scaled softmax for selective state updates, allowing the model to dynamically
/// control the sharpness of its attention/selection mechanism.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute data-dependent temperature: tau_t = softplus(W_temp * x_t + b_temp)
///   3. Compute selection weights via tempered softmax:
///      score_t = q_t^T * k_t / (sqrt(d) * tau_t)
///      weight_t = softmax(score_t / tau_t) over the key dimension
///   4. Gated linear recurrence with tempered selection:
///      forget_gate = sigmoid(W_f * x_t + b_f)
///      S_t = forget_gate * S_{t-1} + weight_t * (k_t * v_t^T)
///      The tempered weights control how selectively the state is updated.
///   5. Output: o_t = S_t * q_t
///   6. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The temperature parameter tau is crucial: it controls the "sharpness" of attention.
/// - Low temperature (tau near 0): Very selective, focuses on the best-matching key (like argmax)
/// - High temperature (tau near infinity): Uniform attention, treats all keys equally
/// - Data-dependent: The model learns when to be selective vs. when to spread attention
///
/// This allows Rodimus to adaptively decide: "Should I focus sharply on one specific key-value pair
/// (low temp), or should I aggregate broadly (high temp)?" This breaks the typical accuracy-efficiency
/// trade-off because the model can be precise when needed and efficient otherwise.
/// </para>
/// <para><b>For Beginners:</b> Rodimus is a smart attention mechanism that can adjust its "focus level"
/// depending on the input.
///
/// Think of reading a textbook:
/// - Sometimes you need to focus sharply on one specific definition (low temperature = very selective)
/// - Other times you need to understand the general theme of a paragraph (high temperature = broad focus)
/// - A good reader adjusts their focus level based on what they're reading
///
/// Rodimus does exactly this:
/// - It learns a "temperature" for each position that controls focus sharpness
/// - Low temperature = laser focus on the most relevant information
/// - High temperature = broad survey of all available information
/// - The temperature is "data-dependent" meaning it adjusts based on the input itself
///
/// Combined with a gated recurrence (like LSTM but for a matrix-valued state), this gives
/// Rodimus the quality of Transformer attention with the efficiency of linear recurrence.
/// The gated recurrence maintains a running state matrix that gets selectively updated
/// at each step, avoiding the O(n^2) cost of full attention.
/// </para>
/// <para>
/// <b>Reference:</b> He et al., "Rodimus: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions", 2025.
/// https://arxiv.org/abs/2410.06577
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RodimusLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly double _baseTemperature;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Temperature projection: [modelDim, numHeads] -> produces per-head temperature
    private Tensor<T> _temperatureWeights;
    private Tensor<T> _temperatureBias;

    // Forget gate projection: [modelDim, numHeads]
    private Tensor<T> _forgetGateWeights;
    private Tensor<T> _forgetGateBias;

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
    private Tensor<T>? _lastTemperature;
    private Tensor<T>? _lastTemperatureRaw;
    private Tensor<T>? _lastForgetGate;
    private Tensor<T>? _lastSelectionWeights;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private Tensor<T>? _lastRecurrenceOutput;
    private Tensor<T>? _lastStates;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _temperatureWeightsGradient;
    private Tensor<T>? _temperatureBiasGradient;
    private Tensor<T>? _forgetGateWeightsGradient;
    private Tensor<T>? _forgetGateBiasGradient;
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
    /// Gets the base temperature value.
    /// </summary>
    public double BaseTemperature => _baseTemperature;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _temperatureWeights.Length + _temperatureBias.Length +
        _forgetGateWeights.Length + _forgetGateBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Rodimus layer with data-dependent tempered selection.
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
    /// <para><b>For Beginners:</b> Each head independently computes tempered selection with its own
    /// temperature. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="temperature">
    /// Base temperature for the tempered softmax. Default: 1.0.
    /// <para><b>For Beginners:</b> This is the starting temperature before the learned
    /// data-dependent adjustment. Lower values make the model more selective initially.
    /// The model will learn to adjust this per-head and per-position.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public RodimusLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        double temperature = 1.0,
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
        if (temperature <= 0)
            throw new ArgumentException($"Temperature ({temperature}) must be positive.", nameof(temperature));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _baseTemperature = temperature;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _temperatureWeights = new Tensor<T>([modelDimension, numHeads]);
        _temperatureBias = new Tensor<T>([numHeads]);
        _forgetGateWeights = new Tensor<T>([modelDimension, numHeads]);
        _forgetGateBias = new Tensor<T>([numHeads]);
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
        InitializeTensor2D(_temperatureWeights);
        // Temperature bias initialized so softplus(bias) ~ baseTemperature
        // softplus(x) = ln(1 + exp(x)), so x ~ ln(exp(temp) - 1)
        T tempBiasVal = NumOps.FromDouble(Math.Log(Math.Exp(_baseTemperature) - 1.0 + 1e-8));
        for (int i = 0; i < _temperatureBias.Length; i++)
            _temperatureBias[i] = tempBiasVal;
        InitializeTensor2D(_forgetGateWeights);
        // Forget gate bias ~ 2 so sigmoid(2) ~ 0.88 -> strong initial memory retention
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(2.0);
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
    /// Computes softplus activation: softplus(x) = ln(1 + exp(x)).
    /// This ensures the temperature is always positive.
    /// </summary>
    private T Softplus(T x)
    {
        double xVal = NumOps.ToDouble(x);
        // For numerical stability: if x > 20, softplus(x) ~ x
        if (xVal > 20.0)
            return x;
        return NumOps.FromDouble(Math.Log(1.0 + Math.Exp(xVal)));
    }

    /// <summary>
    /// Computes the derivative of softplus: softplus'(x) = sigmoid(x) = 1 / (1 + exp(-x)).
    /// </summary>
    private T SoftplusDerivative(T x)
    {
        double xVal = NumOps.ToDouble(x);
        return NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-xVal)));
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

        // Step 2: Data-dependent temperature via softplus
        var tempRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _temperatureWeights),
            _temperatureBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        _lastTemperatureRaw = tempRaw;

        var temperature = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        for (int i = 0; i < temperature.Length; i++)
            temperature[i] = Softplus(tempRaw[i]);
        _lastTemperature = temperature;

        // Step 3: Forget gate
        var forgetRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _forgetGateWeights),
            _forgetGateBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var forgetGate = Engine.Sigmoid(forgetRaw);
        _lastForgetGate = forgetGate;

        // Step 4: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var outputGate = Engine.Swish(gateRaw);
        _lastOutputGate = outputGate;
        _lastOutputGateRaw = gateRaw;

        // Step 5: Tempered selection with gated linear recurrence
        var recurrenceOutput = TemperedRecurrenceForward(q, k, v, temperature, forgetGate, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 6: Gated output
        var gatedOutput = Engine.TensorMultiply(recurrenceOutput, outputGate);

        // Step 7: Output projection
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
    /// Tempered selection with gated linear recurrence.
    /// </summary>
    /// <remarks>
    /// For each timestep t and head h:
    ///   1. Compute tempered selection score: score_i = (q_t dot k_t[i]) / (sqrt(d) * tau_t)
    ///   2. Apply softmax over key dimensions to get selection weight
    ///   3. State update: S_t = forget * S_{t-1} + selectionWeight * (k_t * v_t^T)
    ///   4. Output: o_t = S_t * q_t
    ///
    /// The temperature tau controls selection sharpness:
    /// - tau near 0: Only the highest-scoring key-value pair updates the state (very selective)
    /// - tau large: All key-value pairs contribute equally (uniform update)
    /// </remarks>
    private Tensor<T> TemperedRecurrenceForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> temperature, Tensor<T> forgetGate,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T baseScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // State matrix per head: [batch, numHeads, headDim, headDim]
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });
        var selectionWeightsCache = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _headDimension });

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T tau = temperature[new[] { bi, t, hi }];
                    T fGate = forgetGate[new[] { bi, t, hi }];

                    // Compute tempered selection scores: score_i = (q dot k_i) / (sqrt(d) * tau)
                    // Here k_i are the individual key dimension values, and the softmax
                    // distributes attention across the head dimensions
                    var scores = new T[_headDimension];
                    T maxScore = NumOps.FromDouble(double.NegativeInfinity);
                    for (int ki = 0; ki < _headDimension; ki++)
                    {
                        int flatKi = dimStart + ki;
                        // Score is the product of q and k scaled by temperature
                        T qVal = q[new[] { bi, t, flatKi }];
                        T kVal = k[new[] { bi, t, flatKi }];
                        T score = NumOps.Divide(
                            NumOps.Multiply(NumOps.Multiply(qVal, kVal), baseScale), tau);
                        scores[ki] = score;
                        if (NumOps.ToDouble(score) > NumOps.ToDouble(maxScore))
                            maxScore = score;
                    }

                    // Softmax over key dimensions for selection weights
                    T sumExp = NumOps.Zero;
                    var expScores = new T[_headDimension];
                    for (int ki = 0; ki < _headDimension; ki++)
                    {
                        expScores[ki] = NumOps.Exp(NumOps.Subtract(scores[ki], maxScore));
                        sumExp = NumOps.Add(sumExp, expScores[ki]);
                    }
                    T sumExpInv = NumOps.Divide(NumOps.One, NumOps.Add(sumExp, NumOps.FromDouble(1e-10)));

                    var selWeights = new T[_headDimension];
                    for (int ki = 0; ki < _headDimension; ki++)
                    {
                        selWeights[ki] = NumOps.Multiply(expScores[ki], sumExpInv);
                        selectionWeightsCache[new[] { bi, t, hi, ki }] = selWeights[ki];
                    }

                    // Gated state update: S_t = forget * S_{t-1} + selWeight * (k * v^T)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            int flatDi = dimStart + di;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], baseScale);
                            T vVal = v[new[] { bi, t, flatDi }];

                            T prevS = state[new[] { bi, hi, di, ki }];
                            // Tempered update: selectionWeight scales the outer product
                            T update = NumOps.Multiply(selWeights[ki],
                                NumOps.Multiply(kVal, vVal));
                            T newS = NumOps.Add(NumOps.Multiply(fGate, prevS), update);
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
        _lastSelectionWeights = selectionWeightsCache;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastTemperature == null || _lastTemperatureRaw == null ||
            _lastForgetGate == null || _lastSelectionWeights == null ||
            _lastOutputGate == null || _lastOutputGateRaw == null ||
            _lastRecurrenceOutput == null || _lastStates == null)
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
        _temperatureWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _temperatureBiasGradient = new Tensor<T>([_numHeads]);
        _forgetGateWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _forgetGateBiasGradient = new Tensor<T>([_numHeads]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 7 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(_lastRecurrenceOutput, _lastOutputGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 6 backward: gating
        var dRecurrence = Engine.TensorMultiply(dGated, _lastOutputGate);
        var dGateSwish = Engine.TensorMultiply(dGated, _lastRecurrenceOutput);

        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(_lastOutputGateRaw));

        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 5 backward: tempered recurrence (backward through time)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dForgetGate = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        var dTemperature = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        T baseScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T fGate = _lastForgetGate[new[] { bi, t, hi }];
                    T tau = _lastTemperature[new[] { bi, t, hi }];

                    // Backward through output: o_t[di] = sum_ki S_t[di,ki] * q[ki]
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dRecurrence[new[] { bi, t, flatDi }];

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

                    // Backward through state update:
                    // S_t[di,ki] = fGate * S_{t-1}[di,ki] + selWeight[ki] * k_scaled[ki] * v[di]
                    var dSelWeights = new T[_headDimension];

                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            int flatDi = dimStart + di;
                            T dS = dState[new[] { bi, hi, di, ki }];
                            T sPrev = _lastStates[new[] { bi, t, hi, di, ki }];
                            T kVal = NumOps.Multiply(_lastKey[new[] { bi, t, flatKi }], baseScale);
                            T vVal = _lastValue[new[] { bi, t, flatDi }];
                            T selW = _lastSelectionWeights[new[] { bi, t, hi, ki }];

                            // dForgetGate
                            dForgetGate[new[] { bi, t, hi }] = NumOps.Add(
                                dForgetGate[new[] { bi, t, hi }],
                                NumOps.Multiply(dS, sPrev));

                            // dSelWeight[ki] += sum_di dS[di,ki] * k_scaled[ki] * v[di]
                            dSelWeights[ki] = NumOps.Add(dSelWeights[ki],
                                NumOps.Multiply(dS, NumOps.Multiply(kVal, vVal)));

                            // dK_scaled[ki] += sum_di dS[di,ki] * selWeight[ki] * v[di]
                            dK[new[] { bi, t, flatKi }] = NumOps.Add(
                                dK[new[] { bi, t, flatKi }],
                                NumOps.Multiply(dS,
                                    NumOps.Multiply(selW, NumOps.Multiply(vVal, baseScale))));

                            // dV[di] += sum_ki dS[di,ki] * selWeight[ki] * k_scaled[ki]
                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(dS, NumOps.Multiply(selW, kVal)));

                            // Propagate to previous timestep
                            dState[new[] { bi, hi, di, ki }] = NumOps.Multiply(fGate, dS);
                        }
                    }

                    // Backward through tempered softmax selection weights
                    // selWeight[ki] = softmax(score[ki] / tau)
                    // dScore[ki] = (selWeight[ki] * (dSelWeight[ki] - dot)) / tau
                    T dotSW = NumOps.Zero;
                    for (int ki = 0; ki < _headDimension; ki++)
                    {
                        T selW = _lastSelectionWeights[new[] { bi, t, hi, ki }];
                        dotSW = NumOps.Add(dotSW, NumOps.Multiply(selW, dSelWeights[ki]));
                    }

                    T dTauAccum = NumOps.Zero;
                    for (int ki = 0; ki < _headDimension; ki++)
                    {
                        int flatKi = dimStart + ki;
                        T selW = _lastSelectionWeights[new[] { bi, t, hi, ki }];
                        T dSelW = NumOps.Multiply(selW, NumOps.Subtract(dSelWeights[ki], dotSW));

                        // dScore = dSelW / tau (since score was divided by tau for softmax)
                        T dScore = NumOps.Divide(dSelW, tau);

                        // score = q * k * baseScale / tau
                        // dQ += dScore * k * baseScale / tau
                        // dK += dScore * q * baseScale / tau
                        T qVal = _lastQuery[new[] { bi, t, flatKi }];
                        T kVal = _lastKey[new[] { bi, t, flatKi }];

                        T scaledDScore = NumOps.Divide(NumOps.Multiply(dScore, baseScale), tau);
                        dQ[new[] { bi, t, flatKi }] = NumOps.Add(
                            dQ[new[] { bi, t, flatKi }],
                            NumOps.Multiply(scaledDScore, kVal));
                        dK[new[] { bi, t, flatKi }] = NumOps.Add(
                            dK[new[] { bi, t, flatKi }],
                            NumOps.Multiply(scaledDScore, qVal));

                        // dTau: score was score_raw / tau, so d(score)/d(tau) = -score_raw/tau^2
                        T scoreRaw = NumOps.Multiply(NumOps.Multiply(qVal, kVal), baseScale);
                        T tauSq = NumOps.Multiply(tau, tau);
                        dTauAccum = NumOps.Subtract(dTauAccum,
                            NumOps.Divide(NumOps.Multiply(dSelW, scoreRaw), tauSq));
                    }

                    dTemperature[new[] { bi, t, hi }] = dTauAccum;
                }
            }
        }

        // Temperature through softplus derivative: dTempRaw = dTemp * sigmoid(tempRaw)
        var dTempRaw = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        for (int i = 0; i < dTempRaw.Length; i++)
            dTempRaw[i] = NumOps.Multiply(dTemperature[i], SoftplusDerivative(_lastTemperatureRaw[i]));

        var dTempFlat = dTempRaw.Reshape(batchSize * seqLen, _numHeads);
        _temperatureWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dTempFlat);
        _temperatureBiasGradient = Engine.ReduceSum(dTempRaw, new int[] { 0, 1 });

        // Forget gate through sigmoid derivative
        var forgetSigDeriv = Engine.TensorMultiply(_lastForgetGate,
            Engine.TensorSubtract(CreateOnesLike(_lastForgetGate), _lastForgetGate));
        var dForgetGateRaw = Engine.TensorMultiply(dForgetGate, forgetSigDeriv);

        var dForgetFlat = dForgetGateRaw.Reshape(batchSize * seqLen, _numHeads);
        _forgetGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dForgetFlat);
        _forgetGateBiasGradient = Engine.ReduceSum(dForgetGateRaw, new int[] { 0, 1 });

        // Q, K, V weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradient from all paths
        var dInput = Engine.TensorAdd(dInputFromGate,
            Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dTempFlat, _temperatureWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dForgetFlat, _forgetGateWeights.Transpose([1, 0])));

        var dInput3D = dInput.Reshape(batchSize, seqLen, _modelDimension);

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
        _temperatureWeights = Engine.TensorAdd(_temperatureWeights, Engine.TensorMultiplyScalar(_temperatureWeightsGradient!, negLR));
        _temperatureBias = Engine.TensorAdd(_temperatureBias, Engine.TensorMultiplyScalar(_temperatureBiasGradient!, negLR));
        _forgetGateWeights = Engine.TensorAdd(_forgetGateWeights, Engine.TensorMultiplyScalar(_forgetGateWeightsGradient!, negLR));
        _forgetGateBias = Engine.TensorAdd(_forgetGateBias, Engine.TensorMultiplyScalar(_forgetGateBiasGradient!, negLR));
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
        _temperatureWeights, _temperatureBias,
        _forgetGateWeights, _forgetGateBias,
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
        _lastTemperature = null;
        _lastTemperatureRaw = null;
        _lastForgetGate = null;
        _lastSelectionWeights = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _lastRecurrenceOutput = null;
        _lastStates = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _temperatureWeightsGradient = null;
        _temperatureBiasGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
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
        metadata["BaseTemperature"] = _baseTemperature.ToString();
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
    /// Gets the temperature weights for external inspection.
    /// </summary>
    public Tensor<T> GetTemperatureWeights() => _temperatureWeights;
}
