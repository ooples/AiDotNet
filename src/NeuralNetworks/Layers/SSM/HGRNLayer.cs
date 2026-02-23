using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Hierarchically Gated Recurrent Neural Network (HGRN) layer from NeurIPS 2023.
/// </summary>
/// <remarks>
/// <para>
/// HGRN uses hierarchical gating to achieve multi-scale temporal processing with linear O(n) complexity.
/// Each layer performs a simple element-wise gated recurrence:
/// <code>
///   f_t = sigmoid(W_f * x_t + b_f)    // Forget gate (controls memory retention)
///   i_t = sigmoid(W_i * x_t + b_i)    // Input gate (controls new information)
///   h_t = f_t * h_{t-1} + i_t * x_t   // Element-wise gated recurrence (no matrix state)
///   y_t = W_out * h_t + b_out          // Output projection
/// </code>
/// </para>
/// <para>
/// The "hierarchical" aspect is the key insight: when stacking multiple HGRN layers, each layer
/// uses a different forget gate bias. Lower layers use higher forget gate bias values (sigmoid output
/// closer to 1), creating slow-decaying memory that captures long-range dependencies. Upper layers
/// use lower bias values (sigmoid output closer to 0), creating fast-decaying memory for short-range
/// local patterns. This naturally creates a multi-scale temporal hierarchy where different layers
/// specialize in different time scales, similar to how wavelets decompose signals at multiple
/// resolutions.
/// </para>
/// <para>
/// Unlike GatedDeltaNet or Mamba which maintain matrix-valued states, HGRN maintains only a
/// vector-valued hidden state h_t (same dimension as the input). This makes it extremely
/// memory-efficient and fast, while the hierarchical gating compensates for the simpler state
/// by distributing temporal modeling across layers.
/// </para>
/// <para><b>For Beginners:</b> HGRN is like a stack of simple "memory filters" that work at different speeds.
///
/// Imagine you're listening to music:
/// - One filter (lower layer, high forget bias) remembers the overall melody for a long time
///   -- it barely forgets anything, holding onto the big picture
/// - Another filter (upper layer, low forget bias) tracks the current beat and rhythm
///   -- it quickly forgets old beats to focus on what's happening right now
/// - Together, they understand both the long melody AND the short rhythm simultaneously
///
/// Each filter is extremely simple: at every step, it just does:
///   new_memory = (forget_amount * old_memory) + (input_amount * new_input)
///
/// This is much simpler than Transformers (which compare every position to every other position)
/// or even Mamba (which maintains matrix-valued states). Yet by stacking these simple filters
/// with different forgetting speeds, HGRN achieves competitive performance with linear O(n)
/// complexity -- it processes a sequence of length n in time proportional to n, not n^2.
/// </para>
/// <para>
/// <b>Reference:</b> Qin et al., "Hierarchically Gated Recurrent Neural Network for Sequence Modeling", NeurIPS 2023.
/// https://arxiv.org/abs/2311.04823
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HGRNLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly double _forgetBias;

    // Input projection: [modelDim, modelDim] - projects input before gating
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Forget gate: [modelDim, modelDim]
    private Tensor<T> _forgetGateWeights;
    private Tensor<T> _forgetGateBias;

    // Input gate: [modelDim, modelDim]
    private Tensor<T> _inputGateWeights;
    private Tensor<T> _inputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedInput;
    private Tensor<T>? _lastForgetGate;
    private Tensor<T>? _lastInputGate;
    private Tensor<T>? _lastHiddenStates;
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _forgetGateWeightsGradient;
    private Tensor<T>? _forgetGateBiasGradient;
    private Tensor<T>? _inputGateWeightsGradient;
    private Tensor<T>? _inputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension (input/output width).
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the forget gate bias value used for hierarchical gating.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how much the layer "remembers" by default.
    /// Higher values (e.g., 3.0) mean the layer retains information longer (slow decay, long memory).
    /// Lower values (e.g., 0.0) mean the layer forgets faster (fast decay, short memory).
    /// In a hierarchical stack, lower layers should use higher forget bias for long-range patterns,
    /// and upper layers should use lower forget bias for short-range patterns.</para>
    /// </remarks>
    public double ForgetBias => _forgetBias;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _forgetGateWeights.Length + _forgetGateBias.Length +
        _inputGateWeights.Length + _inputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new HGRN (Hierarchically Gated Recurrent Neural Network) layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// <para><b>For Beginners:</b> The longest sequence this layer can process. For example, if working
    /// with sentences, this is the maximum number of words.</para>
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of the hidden representation. Larger values can capture
    /// more complex patterns but use more memory and compute.</para>
    /// </param>
    /// <param name="forgetBias">
    /// Initial bias for the forget gate. Default: 1.0.
    /// <para><b>For Beginners:</b> This is the key parameter for hierarchical gating. When stacking
    /// multiple HGRN layers, use different forget bias values:
    /// <list type="bullet">
    /// <item>Lower layers: high bias (e.g., 2.0-4.0) for long-range memory</item>
    /// <item>Upper layers: low bias (e.g., 0.0-1.0) for short-range memory</item>
    /// </list>
    /// A bias of 1.0 means sigmoid(1.0) ~ 0.73, retaining about 73% of state per step.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public HGRNLayer(
        int sequenceLength,
        int modelDimension = 256,
        double forgetBias = 1.0,
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

        _modelDimension = modelDimension;
        _forgetBias = forgetBias;

        _inputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _inputProjectionBias = new Tensor<T>([modelDimension]);
        _forgetGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _forgetGateBias = new Tensor<T>([modelDimension]);
        _inputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _inputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);

        InitializeTensor2D(_forgetGateWeights);
        // Initialize forget gate bias to the configured forgetBias value.
        // Higher values -> sigmoid closer to 1 -> slower forgetting -> longer memory.
        // This is the core of hierarchical gating: different layers get different biases.
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(_forgetBias);

        InitializeTensor2D(_inputGateWeights);
        _inputGateBias.Fill(NumOps.Zero);

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

        // Step 1: Input projection
        var input2D = input3D.Reshape(batchSize * seqLen, modelDim);
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var projBias = _inputProjectionBias.Reshape(1, _modelDimension);
        projected = Engine.TensorBroadcastAdd(projected, projBias);
        var projected3D = projected.Reshape(batchSize, seqLen, _modelDimension);
        _lastProjectedInput = projected3D;

        // Step 2: Compute forget and input gates from projected input
        var forgetGate3D = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var inputGate3D = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        var fBias = _forgetGateBias.Reshape(1, _modelDimension);
        var iBias = _inputGateBias.Reshape(1, _modelDimension);

        for (int t = 0; t < seqLen; t++)
        {
            var p_t = projected3D.GetSliceAlongDimension(t, 1);

            // f_t = sigmoid(W_f * p_t + b_f)
            var fRaw = Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(p_t, _forgetGateWeights), fBias);
            var f_t = Engine.Sigmoid(fRaw);

            // i_t = sigmoid(W_i * p_t + b_i)
            var iRaw = Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(p_t, _inputGateWeights), iBias);
            var i_t = Engine.Sigmoid(iRaw);

            forgetGate3D.SetSlice(1, t, f_t);
            inputGate3D.SetSlice(1, t, i_t);
        }

        _lastForgetGate = forgetGate3D;
        _lastInputGate = inputGate3D;

        // Step 3: Element-wise gated recurrence
        // h_t = f_t * h_{t-1} + i_t * x_t (where x_t is the projected input)
        var recurrenceOutput = GatedRecurrenceForward(
            projected3D, forgetGate3D, inputGate3D, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 4: Output projection
        var outFlat = recurrenceOutput.Reshape(batchSize * seqLen, _modelDimension);
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
    /// Implements the element-wise gated recurrence: h_t = f_t * h_{t-1} + i_t * x_t.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the core of HGRN. Unlike matrix-valued states in GatedDeltaNet or Mamba,
    /// the hidden state h is a simple vector of the same dimension as the input. The forget
    /// gate f_t and input gate i_t control how much of the previous state is retained and
    /// how much new information is incorporated.
    /// </para>
    /// <para><b>For Beginners:</b> This is the simplest possible recurrence with gating.
    /// At each step, the hidden state is a weighted blend of the old state (scaled by the forget gate)
    /// and the new input (scaled by the input gate). Despite its simplicity, hierarchical bias
    /// initialization makes this surprisingly powerful across layers.</para>
    /// </remarks>
    private Tensor<T> GatedRecurrenceForward(
        Tensor<T> x, Tensor<T> forgetGate, Tensor<T> inputGate,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var h = new Tensor<T>(new[] { batchSize, _modelDimension });

        // Store all hidden states for backward pass: [batch, seqLen+1, modelDim]
        // Index 0 is the initial zero state, indices 1..seqLen are states after each step
        var allHidden = new Tensor<T>(new[] { batchSize, seqLen + 1, _modelDimension });

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T fVal = forgetGate[new[] { bi, t, d }];
                    T iVal = inputGate[new[] { bi, t, d }];
                    T xVal = x[new[] { bi, t, d }];
                    T hPrev = h[new[] { bi, d }];

                    // h_t = f_t * h_{t-1} + i_t * x_t
                    T hNew = NumOps.Add(
                        NumOps.Multiply(fVal, hPrev),
                        NumOps.Multiply(iVal, xVal));

                    h[new[] { bi, d }] = hNew;
                    output[new[] { bi, t, d }] = hNew;
                }
            }

            // Save hidden state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _modelDimension; d++)
                    allHidden[new[] { bi, t + 1, d }] = h[new[] { bi, d }];
        }

        _lastHiddenStates = allHidden;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastProjectedInput == null ||
            _lastForgetGate == null || _lastInputGate == null ||
            _lastHiddenStates == null || _lastRecurrenceOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _inputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _inputProjectionBiasGradient = new Tensor<T>([_modelDimension]);
        _forgetGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _forgetGateBiasGradient = new Tensor<T>([_modelDimension]);
        _inputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _inputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = new Tensor<T>([_modelDimension]);

        // Step 4 backward: output projection y = recOut * W_out + b_out
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var recOutFlat = _lastRecurrenceOutput.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(
            recOutFlat.Transpose([1, 0]), gradFlat);

        var dRecurrence = Engine.TensorMatMul(
            gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 3 backward: gated recurrence through time
        // h_t = f_t * h_{t-1} + i_t * x_t
        // Gradients flow backward through the recurrence chain
        var dh = new Tensor<T>(new[] { batchSize, _modelDimension });
        var dForgetGate = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dInputGate = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dProjected = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            // Accumulate gradient from output at this timestep
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T dRecVal = dRecurrence[new[] { bi, t, d }];
                    dh[new[] { bi, d }] = NumOps.Add(dh[new[] { bi, d }], dRecVal);
                }
            }

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T dhVal = dh[new[] { bi, d }];
                    T fVal = _lastForgetGate[new[] { bi, t, d }];
                    T iVal = _lastInputGate[new[] { bi, t, d }];
                    T xVal = _lastProjectedInput[new[] { bi, t, d }];
                    T hPrev = _lastHiddenStates[new[] { bi, t, d }]; // h_{t-1} at index t (offset by 1)

                    // dh/df = h_{t-1}
                    dForgetGate[new[] { bi, t, d }] = NumOps.Multiply(dhVal, hPrev);

                    // dh/di = x_t
                    dInputGate[new[] { bi, t, d }] = NumOps.Multiply(dhVal, xVal);

                    // dh/dx = i_t (gradient of recurrence input wrt projected input)
                    dProjected[new[] { bi, t, d }] = NumOps.Multiply(dhVal, iVal);

                    // dh/dh_{t-1} = f_t (propagate gradient to previous timestep)
                    dh[new[] { bi, d }] = NumOps.Multiply(dhVal, fVal);
                }
            }
        }

        // Step 2 backward: gate gradients through sigmoid
        // sigmoid derivative: sig'(x) = sig(x) * (1 - sig(x))
        // dForgetGateRaw = dForgetGate * f * (1 - f)
        // dInputGateRaw = dInputGate * i * (1 - i)
        var onesF = CreateOnesLike(_lastForgetGate);
        var fSigDeriv = Engine.TensorMultiply(_lastForgetGate,
            Engine.TensorSubtract(onesF, _lastForgetGate));
        var dForgetRaw = Engine.TensorMultiply(dForgetGate, fSigDeriv);

        var onesI = CreateOnesLike(_lastInputGate);
        var iSigDeriv = Engine.TensorMultiply(_lastInputGate,
            Engine.TensorSubtract(onesI, _lastInputGate));
        var dInputRaw = Engine.TensorMultiply(dInputGate, iSigDeriv);

        // Gate weight gradients: gate = sigmoid(p * W + b)
        var projFlat = _lastProjectedInput.Reshape(batchSize * seqLen, _modelDimension);
        var dForgetRawFlat = dForgetRaw.Reshape(batchSize * seqLen, _modelDimension);
        var dInputRawFlat = dInputRaw.Reshape(batchSize * seqLen, _modelDimension);

        _forgetGateWeightsGradient = Engine.TensorMatMul(
            projFlat.Transpose([1, 0]), dForgetRawFlat);
        _forgetGateBiasGradient = Engine.ReduceSum(dForgetRaw, new int[] { 0, 1 });

        _inputGateWeightsGradient = Engine.TensorMatMul(
            projFlat.Transpose([1, 0]), dInputRawFlat);
        _inputGateBiasGradient = Engine.ReduceSum(dInputRaw, new int[] { 0, 1 });

        // Accumulate projected input gradient from all gate paths plus recurrence
        var dProjFlat = dProjected.Reshape(batchSize * seqLen, _modelDimension);
        var dProjFromForget = Engine.TensorMatMul(
            dForgetRawFlat, _forgetGateWeights.Transpose([1, 0]));
        var dProjFromInput = Engine.TensorMatMul(
            dInputRawFlat, _inputGateWeights.Transpose([1, 0]));
        var dProjTotal = Engine.TensorAdd(dProjFlat,
            Engine.TensorAdd(dProjFromForget, dProjFromInput));

        // Step 1 backward: input projection projected = input * W_inp + b_inp
        var input2D = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        _inputProjectionWeightsGradient = Engine.TensorMatMul(
            input2D.Transpose([1, 0]), dProjTotal);
        _inputProjectionBiasGradient = Engine.ReduceSum(
            dProjTotal.Reshape(batchSize, seqLen, _modelDimension), new int[] { 0, 1 });

        var inputGradFlat = Engine.TensorMatMul(
            dProjTotal, _inputProjectionWeights.Transpose([1, 0]));
        var inputGrad3D = inputGradFlat.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return inputGrad3D.Reshape(_originalInputShape);

        return inputGrad3D;
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
        if (_inputProjectionWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights,
            Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias,
            Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _forgetGateWeights = Engine.TensorAdd(_forgetGateWeights,
            Engine.TensorMultiplyScalar(_forgetGateWeightsGradient!, negLR));
        _forgetGateBias = Engine.TensorAdd(_forgetGateBias,
            Engine.TensorMultiplyScalar(_forgetGateBiasGradient!, negLR));
        _inputGateWeights = Engine.TensorAdd(_inputGateWeights,
            Engine.TensorMultiplyScalar(_inputGateWeightsGradient!, negLR));
        _inputGateBias = Engine.TensorAdd(_inputGateBias,
            Engine.TensorMultiplyScalar(_inputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights,
            Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias,
            Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
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
        _inputProjectionWeights, _inputProjectionBias,
        _forgetGateWeights, _forgetGateBias,
        _inputGateWeights, _inputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedInput = null;
        _lastForgetGate = null;
        _lastInputGate = null;
        _lastHiddenStates = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
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
        metadata["ForgetBias"] = _forgetBias.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the forget gate weights for external inspection.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the learned weights that determine how much of the
    /// previous hidden state to retain at each timestep. The forget gate bias (set during construction)
    /// shifts these values to control the default memory retention rate for this layer.</para>
    /// </remarks>
    public Tensor<T> GetForgetGateWeights() => _forgetGateWeights;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;
}
