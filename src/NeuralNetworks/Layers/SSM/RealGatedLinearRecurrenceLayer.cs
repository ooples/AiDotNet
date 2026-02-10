using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Real-Gated Linear Recurrence Unit (RG-LRU) from Google DeepMind's Griffin architecture.
/// </summary>
/// <remarks>
/// <para>
/// The RG-LRU is a gated linear recurrence that serves as the core sequence mixing mechanism in
/// the Griffin and Hawk architectures. It uses input-dependent gating to control both the recurrence
/// decay and the input contribution, providing selective memory similar to Mamba but through a
/// different mathematical formulation.
/// </para>
/// <para>
/// The recurrence is:
/// <code>
///   r_t = sigmoid(W_r * x_t + b_r)           // Recurrence gate
///   i_t = sigmoid(W_i * x_t + b_i)           // Input gate
///   a_t = diag(r_t) * diag(exp(-softplus(c))) // Gated decay (c is a learned parameter)
///   h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * (W_x * x_t))
///   y_t = h_t
/// </code>
/// The sqrt(1 - a_t^2) factor ensures the recurrence preserves signal magnitude, preventing
/// vanishing or exploding states.
/// </para>
/// <para>
/// Griffin combines RG-LRU with local attention in a hybrid architecture. This layer implements
/// the RG-LRU component which can be used standalone or as part of a hybrid.
/// </para>
/// <para><b>For Beginners:</b> The RG-LRU is like a learnable "leaky bucket" for information.
///
/// Imagine each position in your hidden state as a bucket:
/// - The recurrence gate (r) controls how much water leaks out each step (memory decay)
/// - The input gate (i) controls how much new water pours in
/// - The sqrt(1 - a^2) factor ensures the bucket never overflows or runs dry
///
/// This is simpler than Mamba (no Conv1D, no SSM parameters B/C) but surprisingly effective.
/// Google's RecurrentGemma models (2B, 9B) use this architecture and achieve competitive
/// performance with Transformer-based Gemma models.
/// </para>
/// <para>
/// <b>Reference:</b> De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", 2024.
/// https://arxiv.org/abs/2402.19427
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RealGatedLinearRecurrenceLayer<T> : LayerBase<T>
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _recurrenceDimension;

    // Input projection: [modelDim, recurrenceDim]
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Recurrence gate: [recurrenceDim, recurrenceDim]
    private Tensor<T> _recurrenceGateWeights;
    private Tensor<T> _recurrenceGateBias;

    // Input gate: [recurrenceDim, recurrenceDim]
    private Tensor<T> _inputGateWeights;
    private Tensor<T> _inputGateBias;

    // Value projection: [recurrenceDim, recurrenceDim]
    private Tensor<T> _valueProjectionWeights;

    // Learned decay parameter: [recurrenceDim] (passed through softplus for positivity)
    private Tensor<T> _decayParam;

    // Output projection: [recurrenceDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedInput;
    private Tensor<T>? _lastRecurrenceGate;
    private Tensor<T>? _lastInputGate;
    private Tensor<T>? _lastHiddenStates;
    private Tensor<T>? _lastDecayFactors;
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _recurrenceGateWeightsGradient;
    private Tensor<T>? _recurrenceGateBiasGradient;
    private Tensor<T>? _inputGateWeightsGradient;
    private Tensor<T>? _inputGateBiasGradient;
    private Tensor<T>? _valueProjectionWeightsGradient;
    private Tensor<T>? _decayParamGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension (input/output width).
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the recurrence dimension (hidden state width).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The recurrence dimension controls the width of the hidden state.
    /// In Griffin, this is typically equal to the model dimension, but can be configured independently.</para>
    /// </remarks>
    public int RecurrenceDimension => _recurrenceDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _recurrenceGateWeights.Length + _recurrenceGateBias.Length +
        _inputGateWeights.Length + _inputGateBias.Length +
        _valueProjectionWeights.Length +
        _decayParam.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Real-Gated Linear Recurrence Unit (RG-LRU) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="recurrenceDimension">
    /// Recurrence state dimension. Default: -1 (same as modelDimension).
    /// <para><b>For Beginners:</b> Width of the hidden recurrence state. Using the same as modelDim
    /// is the standard configuration from the Griffin paper.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when modelDimension is not positive.</exception>
    public RealGatedLinearRecurrenceLayer(
        int sequenceLength,
        int modelDimension = 256,
        int recurrenceDimension = -1,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));

        _modelDimension = modelDimension;
        _recurrenceDimension = recurrenceDimension > 0 ? recurrenceDimension : modelDimension;

        _inputProjectionWeights = new Tensor<T>([modelDimension, _recurrenceDimension]);
        _inputProjectionBias = new Tensor<T>([_recurrenceDimension]);
        _recurrenceGateWeights = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _recurrenceGateBias = new Tensor<T>([_recurrenceDimension]);
        _inputGateWeights = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _inputGateBias = new Tensor<T>([_recurrenceDimension]);
        _valueProjectionWeights = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _decayParam = new Tensor<T>([_recurrenceDimension]);
        _outputProjectionWeights = new Tensor<T>([_recurrenceDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);
        InitializeTensor(_recurrenceGateWeights);
        _recurrenceGateBias.Fill(NumOps.Zero);
        InitializeTensor(_inputGateWeights);
        _inputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_valueProjectionWeights);

        // Initialize decay ~ 0.9 per step (softplus(2.2) ≈ 2.3, exp(-2.3) ≈ 0.1 -> 1-0.1=0.9)
        for (int i = 0; i < _recurrenceDimension; i++)
            _decayParam[i] = NumOps.FromDouble(2.2 + Random.NextDouble() * 0.5);

        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
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

        // Step 1: Input projection
        var input2D = input3D.Reshape(batchSize * seqLen, modelDim);
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var projBias = _inputProjectionBias.Reshape(1, _recurrenceDimension);
        projected = Engine.TensorBroadcastAdd(projected, projBias);
        var projected3D = projected.Reshape(batchSize, seqLen, _recurrenceDimension);
        _lastProjectedInput = projected3D;

        // Step 2: Compute gates and value projection
        var recGate3D = new Tensor<T>(new[] { batchSize, seqLen, _recurrenceDimension });
        var inpGate3D = new Tensor<T>(new[] { batchSize, seqLen, _recurrenceDimension });

        for (int t = 0; t < seqLen; t++)
        {
            var p_t = projected3D.GetSliceAlongDimension(t, 1);

            var rGate = Engine.Sigmoid(Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(p_t, _recurrenceGateWeights),
                _recurrenceGateBias.Reshape(1, _recurrenceDimension)));
            var iGate = Engine.Sigmoid(Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(p_t, _inputGateWeights),
                _inputGateBias.Reshape(1, _recurrenceDimension)));

            recGate3D.SetSlice(1, t, rGate);
            inpGate3D.SetSlice(1, t, iGate);
        }

        _lastRecurrenceGate = recGate3D;
        _lastInputGate = inpGate3D;

        // Step 3: Gated linear recurrence
        var output = GatedRecurrenceForward(projected3D, recGate3D, inpGate3D, batchSize, seqLen);
        _lastRecurrenceOutput = output;

        // Step 4: Output projection
        var outFlat = output.Reshape(batchSize * seqLen, _recurrenceDimension);
        var outputFlat = Engine.TensorMatMul(outFlat, _outputProjectionWeights);
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
    /// Implements the gated linear recurrence with magnitude-preserving update.
    /// </summary>
    private Tensor<T> GatedRecurrenceForward(
        Tensor<T> x, Tensor<T> recGate, Tensor<T> inpGate,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _recurrenceDimension });
        var h = new Tensor<T>(new[] { batchSize, _recurrenceDimension });
        var allHidden = new Tensor<T>(new[] { batchSize, seqLen + 1, _recurrenceDimension });
        var allDecay = new Tensor<T>(new[] { batchSize, seqLen, _recurrenceDimension });

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);
            var r_t = recGate.GetSliceAlongDimension(t, 1);
            var i_t = inpGate.GetSliceAlongDimension(t, 1);

            // Value projection
            var v_t = Engine.TensorMatMul(x_t, _valueProjectionWeights);

            // Compute decay: a_t = r_t * exp(-softplus(c))
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _recurrenceDimension; d++)
                {
                    double cVal = NumOps.ToDouble(_decayParam[d]);
                    // Numerically stable softplus: for large x, softplus(x) ≈ x
                    double softplusC = cVal > 20.0 ? cVal : Math.Log(1.0 + Math.Exp(cVal));
                    double baseDecay = Math.Exp(-softplusC);
                    double rVal = NumOps.ToDouble(r_t[new[] { bi, d }]);
                    double a = rVal * baseDecay;

                    // Magnitude-preserving: sqrt(1 - a^2)
                    double sqrtFactor = Math.Sqrt(Math.Max(0, 1.0 - a * a));

                    double iVal = NumOps.ToDouble(i_t[new[] { bi, d }]);
                    double vVal = NumOps.ToDouble(v_t[new[] { bi, d }]);
                    double hPrev = NumOps.ToDouble(h[new[] { bi, d }]);

                    // h_t = a * h_{t-1} + sqrt(1 - a^2) * (i * v)
                    double hNew = a * hPrev + sqrtFactor * (iVal * vVal);
                    h[new[] { bi, d }] = NumOps.FromDouble(hNew);
                    allDecay[new[] { bi, t, d }] = NumOps.FromDouble(a);
                }
            }

            allHidden.SetSlice(1, t + 1, h);
            output.SetSlice(1, t, h);
        }

        _lastHiddenStates = allHidden;
        _lastDecayFactors = allDecay;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Output projection backward: y = recOut * Wout + bout
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var recOutFlat = _lastRecurrenceOutput!.Reshape(batchSize * seqLen, _recurrenceDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(recOutFlat.Transpose([1, 0]), gradFlat);

        var dRecurrence = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _recurrenceDimension);

        // Initialize all gradients
        _inputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _recurrenceDimension]);
        _inputProjectionBiasGradient = new Tensor<T>([_recurrenceDimension]);
        _recurrenceGateWeightsGradient = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _recurrenceGateBiasGradient = new Tensor<T>([_recurrenceDimension]);
        _inputGateWeightsGradient = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _inputGateBiasGradient = new Tensor<T>([_recurrenceDimension]);
        _valueProjectionWeightsGradient = new Tensor<T>([_recurrenceDimension, _recurrenceDimension]);
        _decayParamGradient = new Tensor<T>([_recurrenceDimension]);

        // Recurrence backward through time
        // h_t = a_t * h_{t-1} + sqrt(1-a_t^2) * (i_t * v_t)
        var dh = new Tensor<T>(new[] { batchSize, _recurrenceDimension });
        var dProjected = new Tensor<T>(new[] { batchSize, seqLen, _recurrenceDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            var dOut_t = dRecurrence.GetSliceAlongDimension(t, 1);
            dh = Engine.TensorAdd(dh, dOut_t);

            var p_t = _lastProjectedInput!.GetSliceAlongDimension(t, 1);
            var r_t = _lastRecurrenceGate!.GetSliceAlongDimension(t, 1);
            var i_t = _lastInputGate!.GetSliceAlongDimension(t, 1);
            var v_t = Engine.TensorMatMul(p_t, _valueProjectionWeights);

            var dRGate = new Tensor<T>(new[] { batchSize, _recurrenceDimension });
            var dIGate = new Tensor<T>(new[] { batchSize, _recurrenceDimension });
            var dV = new Tensor<T>(new[] { batchSize, _recurrenceDimension });
            var dhPrev = new Tensor<T>(new[] { batchSize, _recurrenceDimension });

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _recurrenceDimension; d++)
                {
                    double dhVal = NumOps.ToDouble(dh[new[] { bi, d }]);
                    double cVal = NumOps.ToDouble(_decayParam[d]);
                    double softplusC = cVal > 20.0 ? cVal : Math.Log(1.0 + Math.Exp(cVal));
                    double baseDecay = Math.Exp(-softplusC);
                    double rVal = NumOps.ToDouble(r_t[new[] { bi, d }]);
                    double aVal = rVal * baseDecay;
                    double sqrtFactor = Math.Sqrt(Math.Max(0, 1.0 - aVal * aVal));
                    double iVal = NumOps.ToDouble(i_t[new[] { bi, d }]);
                    double vVal = NumOps.ToDouble(v_t[new[] { bi, d }]);
                    double hPrev = NumOps.ToDouble(_lastHiddenStates![new[] { bi, t, d }]);

                    // dh/dh_prev = a_t
                    dhPrev[new[] { bi, d }] = NumOps.FromDouble(dhVal * aVal);

                    // dh/da = h_prev + (-a / sqrt(1-a^2)) * (i*v) if sqrtFactor > 0
                    double dhda = dhVal * hPrev;
                    if (sqrtFactor > 1e-10)
                        dhda += dhVal * (-aVal / sqrtFactor) * (iVal * vVal);

                    // da/dr = baseDecay
                    dRGate[new[] { bi, d }] = NumOps.FromDouble(dhda * baseDecay);

                    // da/dc: chain through softplus and exp
                    // a = r * exp(-softplus(c)), da/dc = -r * exp(-softplus(c)) * sigmoid(c)
                    double sigC = 1.0 / (1.0 + Math.Exp(-cVal));
                    double dadC = -rVal * baseDecay * sigC;
                    _decayParamGradient[d] = NumOps.Add(_decayParamGradient[d],
                        NumOps.FromDouble(dhda * dadC));

                    // dh/d(i*v) = sqrtFactor
                    double dIV = dhVal * sqrtFactor;
                    dIGate[new[] { bi, d }] = NumOps.FromDouble(dIV * vVal);
                    dV[new[] { bi, d }] = NumOps.FromDouble(dIV * iVal);
                }
            }

            // Propagate dh to previous timestep
            dh = dhPrev;

            // Value projection gradient: v = p * Wv
            _valueProjectionWeightsGradient = Engine.TensorAdd(_valueProjectionWeightsGradient,
                Engine.TensorMatMul(p_t.Transpose([1, 0]), dV));
            var dPFromV = Engine.TensorMatMul(dV, _valueProjectionWeights.Transpose([1, 0]));

            // Gate gradients through sigmoid: d/dx sig(x) = sig(x)*(1-sig(x))
            var ones = new Tensor<T>(r_t.Shape);
            for (int idx = 0; idx < ones.Length; idx++) ones[idx] = NumOps.One;

            var rSigDeriv = Engine.TensorMultiply(r_t, Engine.TensorSubtract(ones, r_t));
            var dRGateRaw = Engine.TensorMultiply(dRGate, rSigDeriv);

            var iSigDeriv = Engine.TensorMultiply(i_t, Engine.TensorSubtract(ones, i_t));
            var dIGateRaw = Engine.TensorMultiply(dIGate, iSigDeriv);

            // Gate weight gradients: gate = sigmoid(p * W + b)
            _recurrenceGateWeightsGradient = Engine.TensorAdd(_recurrenceGateWeightsGradient,
                Engine.TensorMatMul(p_t.Transpose([1, 0]), dRGateRaw));
            _recurrenceGateBiasGradient = Engine.TensorAdd(_recurrenceGateBiasGradient,
                Engine.ReduceSum(dRGateRaw, new int[] { 0 }));

            _inputGateWeightsGradient = Engine.TensorAdd(_inputGateWeightsGradient,
                Engine.TensorMatMul(p_t.Transpose([1, 0]), dIGateRaw));
            _inputGateBiasGradient = Engine.TensorAdd(_inputGateBiasGradient,
                Engine.ReduceSum(dIGateRaw, new int[] { 0 }));

            // Projected input gradient from all gate paths
            var dP_t = Engine.TensorMatMul(dRGateRaw, _recurrenceGateWeights.Transpose([1, 0]));
            dP_t = Engine.TensorAdd(dP_t,
                Engine.TensorMatMul(dIGateRaw, _inputGateWeights.Transpose([1, 0])));
            dP_t = Engine.TensorAdd(dP_t, dPFromV);

            dProjected.SetSlice(1, t, dP_t);
        }

        // Input projection backward: projected = input * Winp + binp
        var input2D = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dProjFlat = dProjected.Reshape(batchSize * seqLen, _recurrenceDimension);
        _inputProjectionWeightsGradient = Engine.TensorMatMul(input2D.Transpose([1, 0]), dProjFlat);
        _inputProjectionBiasGradient = Engine.ReduceSum(dProjected, new int[] { 0, 1 });

        var inputGradFlat = Engine.TensorMatMul(dProjFlat, _inputProjectionWeights.Transpose([1, 0]));
        var inputGrad3D = inputGradFlat.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return inputGrad3D.Reshape(_originalInputShape);

        return inputGrad3D;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_inputProjectionWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights, Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias, Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _recurrenceGateWeights = Engine.TensorAdd(_recurrenceGateWeights, Engine.TensorMultiplyScalar(_recurrenceGateWeightsGradient!, negLR));
        _recurrenceGateBias = Engine.TensorAdd(_recurrenceGateBias, Engine.TensorMultiplyScalar(_recurrenceGateBiasGradient!, negLR));
        _inputGateWeights = Engine.TensorAdd(_inputGateWeights, Engine.TensorMultiplyScalar(_inputGateWeightsGradient!, negLR));
        _inputGateBias = Engine.TensorAdd(_inputGateBias, Engine.TensorMultiplyScalar(_inputGateBiasGradient!, negLR));
        _valueProjectionWeights = Engine.TensorAdd(_valueProjectionWeights, Engine.TensorMultiplyScalar(_valueProjectionWeightsGradient!, negLR));
        _decayParam = Engine.TensorAdd(_decayParam, Engine.TensorMultiplyScalar(_decayParamGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var tensor in GetAllTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");

        int index = 0;
        foreach (var tensor in GetAllTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _inputProjectionWeights, _inputProjectionBias,
        _recurrenceGateWeights, _recurrenceGateBias,
        _inputGateWeights, _inputGateBias,
        _valueProjectionWeights, _decayParam,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedInput = null;
        _lastRecurrenceGate = null;
        _lastInputGate = null;
        _lastHiddenStates = null;
        _lastDecayFactors = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _recurrenceGateWeightsGradient = null;
        _recurrenceGateBiasGradient = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
        _valueProjectionWeightsGradient = null;
        _decayParamGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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

        var inProjWeightsNode = TensorOperations<T>.Variable(_inputProjectionWeights, "W_in");
        inputNodes.Add(inProjWeightsNode);

        var inProjT = TensorOperations<T>.Transpose(inProjWeightsNode);
        var projected = TensorOperations<T>.MatrixMultiply(xNode, inProjT);
        var outProjT = TensorOperations<T>.Transpose(outWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(projected, outProjT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outBiasNode);

        return outputWithBias;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["RecurrenceDimension"] = _recurrenceDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the decay parameter for external inspection.
    /// </summary>
    public Tensor<T> GetDecayParameter() => _decayParam;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;
}
