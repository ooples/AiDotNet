using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class RealGatedLinearRecurrenceLayer<T> : LayerBase<T>
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
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

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
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape.ToArray();

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
        var recGate3D = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _recurrenceDimension });
        var inpGate3D = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _recurrenceDimension });

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
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _recurrenceDimension });
        var h = TensorAllocator.Rent<T>(new[] { batchSize, _recurrenceDimension });
        var allHidden = new Tensor<T>(new[] { batchSize, seqLen + 1, _recurrenceDimension });
        var allDecay = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _recurrenceDimension });

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

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_inputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_recurrenceGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_recurrenceGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_inputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_valueProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

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

    public override Vector<T> GetParameterGradients()
    {
        if (_inputProjectionWeightsGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_inputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_recurrenceGateWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_recurrenceGateBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputGateWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputGateBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_decayParamGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _inputProjectionWeightsGradient = null; _inputProjectionBiasGradient = null;
        _recurrenceGateWeightsGradient = null; _recurrenceGateBiasGradient = null;
        _inputGateWeightsGradient = null; _inputGateBiasGradient = null;
        _valueProjectionWeightsGradient = null; _decayParamGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

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
