using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Extended LSTM (xLSTM) layer from Hochreiter et al., 2024.
/// </summary>
/// <remarks>
/// <para>
/// xLSTM modernizes the classic LSTM architecture with two key innovations:
/// 1. <b>sLSTM (scalar LSTM)</b>: Enhanced gating with exponential activation functions and
///    a new memory mixing mechanism. Uses scalar (diagonal) memory cells.
/// 2. <b>mLSTM (matrix LSTM)</b>: Replaces the scalar memory cell with a matrix-valued memory,
///    connecting LSTMs to modern linear attention/state space models.
/// </para>
/// <para>
/// This layer implements the mLSTM variant, which is the more impactful innovation:
/// <code>
///   // Gate computations
///   i_t = exp(W_i * x_t + b_i)    // Input gate (exponential, not sigmoid!)
///   f_t = sigmoid(W_f * x_t + b_f) OR exp(W_f * x_t + b_f)  // Forget gate
///   o_t = sigmoid(W_o * x_t + b_o)  // Output gate
///
///   // Key-Value projections (connecting to linear attention)
///   k_t = W_k * x_t / sqrt(d)
///   v_t = W_v * x_t
///   q_t = W_q * x_t
///
///   // Matrix memory cell update (covariance-based)
///   C_t = f_t * C_{t-1} + i_t * v_t * k_t^T    // Matrix cell = gated outer product
///   n_t = f_t * n_{t-1} + i_t * k_t              // Normalizer state
///
///   // Output
///   h_t = o_t * (C_t * q_t) / max(|n_t^T * q_t|, 1)
/// </code>
/// </para>
/// <para>
/// The connection to linear attention: if f_t = 1 and i_t = 1, the matrix cell C_t accumulates
/// k*v outer products exactly like the state matrix in linear attention. The gates allow
/// selective forgetting and input scaling, which is what makes xLSTM competitive.
/// </para>
/// <para><b>For Beginners:</b> xLSTM is a modernized version of the classic LSTM (1997).
///
/// The original LSTM was the dominant sequence model for years, but was overtaken by Transformers.
/// xLSTM brings it back by fixing key limitations:
///
/// 1. <b>Exponential gating</b>: Instead of sigmoid (0 to 1), gates use exp() which can amplify
///    important signals, not just dampen them.
///
/// 2. <b>Matrix memory</b>: Instead of a vector cell, mLSTM uses a matrix. This is like having
///    a lookup table that maps keys to values, similar to attention but stored as a running sum.
///
/// The result: an LSTM that matches Transformer performance at scale while maintaining the
/// efficient O(1) per-step inference of RNNs.
/// </para>
/// <para>
/// <b>Reference:</b> Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.
/// https://arxiv.org/abs/2405.04517
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving; relations DISCOVERED by probing, roles read from the forward. Like every layer in
// this folder it takes seqLen = Shape[rank-2] and modelDim = Shape[rank-1], so rank 2 is
// [Time, Features] with NO batch axis. OutputAxesFor is generated from these layouts.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ExtendedLSTMLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _headDimension;
    private readonly int _numHeads;

    // Input gate projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputGateBias;

    // Forget gate projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _forgetGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _forgetGateBias;

    // Output gate projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputGateBias;

    // Query, Key, Value projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached values
    private int[]? _originalInputShape;

    // Fixed (non-trainable) unit-gamma / zero-beta for the mLSTM output
    // normalization. Beck et al. 2024 ("xLSTM", §2.3) normalize the mLSTM cell
    // output before the up-projection so the covariance-cell signal — a product
    // of sub-unit q/k/v projections that the max(|n·q|, 1) normalizer never
    // up-scales — stays at unit scale. Without it, stacking N cells collapses the
    // activations by ~5 orders of magnitude per layer (≈1e-77 by 4 layers), and
    // the backward pass underflows to zero gradient everywhere so no parameter
    // ever updates. Standardization (mean 0, var 1) with fixed gamma/beta keeps
    // the layer output unit-scale without introducing extra trainable parameters
    // (which would change the serialized parameter layout).
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _outputNormGamma;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _outputNormBeta;

    // Gradients
    [Scratch]
    private Tensor<T>? _inputGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _inputGateBiasGradient;
    [Scratch]
    private Tensor<T>? _forgetGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _forgetGateBiasGradient;
    [Scratch]
    private Tensor<T>? _outputGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputGateBiasGradient;
    [Scratch]
    private Tensor<T>? _queryWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of heads for the matrix memory.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// Creates a new Extended LSTM (xLSTM) layer using the mLSTM (matrix memory) variant.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads for matrix memory. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own matrix memory cell.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public ExtendedLSTMLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        _sequenceLength = sequenceLength;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        _inputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _inputGateBias = new Tensor<T>([modelDimension]);
        _forgetGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _forgetGateBias = new Tensor<T>([modelDimension]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor(_inputGateWeights);
        _inputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_forgetGateWeights);
        // Forget gate bias initialized to positive values for long memory (LSTM best practice)
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(1.0);
        InitializeTensor(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_queryWeights);
        InitializeTensor(_keyWeights);
        InitializeTensor(_valueWeights);
        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, modelDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, modelDim });

        // Project the full sequence once. The matrix-memory scan accepts scalar gates per head;
        // preserve this layer's historical first-coordinate selection and log stabilization.
        var flat = Engine.Reshape(input3D, new[] { batchSize * seqLen, modelDim });
        Tensor<T> Project(Tensor<T> weights, Tensor<T>? bias = null)
        {
            var projected = Engine.TensorMatMul(flat, weights);
            if (bias is not null) projected = Engine.TensorAdd(projected, Engine.Reshape(bias, new[] { 1, modelDim }));
            return Engine.Reshape(projected, new[] { batchSize, seqLen, modelDim });
        }
        Tensor<T> HeadGate(Tensor<T> gate) => Engine.Reshape(Engine.TensorNarrow(
            Engine.Reshape(gate, new[] { batchSize, seqLen, _numHeads, _headDimension }), 3, 0, 1),
            new[] { batchSize, seqLen, _numHeads });
        var q = Project(_queryWeights); var k = Project(_keyWeights); var v = Project(_valueWeights);
        var inputLogits = Project(_inputGateWeights, _inputGateBias);
        var forget = Engine.Sigmoid(Project(_forgetGateWeights, _forgetGateBias));
        var outputGate = Engine.Sigmoid(Project(_outputGateWeights, _outputGateBias));
        var logInput = HeadGate(inputLogits);
        var logForget = Engine.TensorLog(Engine.TensorMax(HeadGate(forget),
            Tensor<T>.CreateDefault(new[] { batchSize, seqLen, _numHeads }, NumOps.FromDouble(1e-30))));
        var m = Tensor<T>.CreateDefault(new[] { batchSize, 1, _numHeads }, NumOps.FromDouble(-1e30));
        var inputScales = new Tensor<T>[seqLen]; var forgetScales = new Tensor<T>[seqLen];
        for (int t = 0; t < seqLen; t++)
        {
            var li = Engine.TensorNarrow(logInput, 1, t, 1);
            var carry = Engine.TensorAdd(Engine.TensorNarrow(logForget, 1, t, 1), m);
            var next = Engine.TensorMax(carry, li);
            inputScales[t] = Engine.TensorExp(Engine.TensorSubtract(li, next));
            forgetScales[t] = Engine.TensorExp(Engine.TensorSubtract(carry, next));
            m = next;
        }
        // Only the scalar stabilizer is sequential here; the matrix recurrence and its BPTT
        // are one shared engine operation. The scan applies the original 1/sqrt(headDim) key scale.
        var hidden = Engine.XLstmScanForward(q, k, v, Engine.TensorConcatenate(inputScales, 1),
            Engine.TensorConcatenate(forgetScales, 1), HeadGate(outputGate), _numHeads);
        var output = Engine.Reshape(Engine.TensorAdd(Engine.TensorMatMul(
            Engine.Reshape(hidden, new[] { batchSize * seqLen, modelDim }), _outputProjectionWeights),
            Engine.Reshape(_outputProjectionBias, new[] { 1, modelDim })), new[] { batchSize, seqLen, modelDim });

        // Preserve the normalized residual block and its serialized parameter layout.
        output = Engine.TensorAdd(output, input3D);
        if (_outputNormGamma is null || _outputNormBeta is null)
        {
            var gamma = new Tensor<T>(new[] { _modelDimension });
            var beta = new Tensor<T>(new[] { _modelDimension });
            for (int j = 0; j < _modelDimension; j++)
            {
                gamma[j] = NumOps.One;
                beta[j] = NumOps.Zero;
            }
            _outputNormGamma = gamma;
            _outputNormBeta = beta;
        }
        output = Engine.LayerNorm(output, _outputNormGamma, _outputNormBeta, 1e-5, out _, out _);

        var result = ApplyActivation(output);

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_inputGateWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        Engine.TensorAddInPlace(_inputGateWeights, Engine.TensorMultiplyScalar(_inputGateWeightsGradient, negLR));
        Engine.TensorAddInPlace(_inputGateBias, Engine.TensorMultiplyScalar((_inputGateBiasGradient ?? throw new InvalidOperationException("_inputGateBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_forgetGateWeights, Engine.TensorMultiplyScalar((_forgetGateWeightsGradient ?? throw new InvalidOperationException("_forgetGateWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_forgetGateBias, Engine.TensorMultiplyScalar((_forgetGateBiasGradient ?? throw new InvalidOperationException("_forgetGateBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputGateWeights, Engine.TensorMultiplyScalar((_outputGateWeightsGradient ?? throw new InvalidOperationException("_outputGateWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputGateBias, Engine.TensorMultiplyScalar((_outputGateBiasGradient ?? throw new InvalidOperationException("_outputGateBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_queryWeights, Engine.TensorMultiplyScalar((_queryWeightsGradient ?? throw new InvalidOperationException("_queryWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_keyWeights, Engine.TensorMultiplyScalar((_keyWeightsGradient ?? throw new InvalidOperationException("_keyWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_valueWeights, Engine.TensorMultiplyScalar((_valueWeightsGradient ?? throw new InvalidOperationException("_valueWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionWeights, Engine.TensorMultiplyScalar((_outputProjectionWeightsGradient ?? throw new InvalidOperationException("_outputProjectionWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionBias, Engine.TensorMultiplyScalar((_outputProjectionBiasGradient ?? throw new InvalidOperationException("_outputProjectionBiasGradient is null where it is required.")), negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_inputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_forgetGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_forgetGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _inputGateWeights, _inputGateBias,
        _forgetGateWeights, _forgetGateBias,
        _outputGateWeights, _outputGateBias,
        _queryWeights, _keyWeights, _valueWeights,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override void ClearGradients()
    {
        base.ClearGradients();
        _inputGateWeightsGradient = null; _inputGateBiasGradient = null;
        _forgetGateWeightsGradient = null; _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null;
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _originalInputShape = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

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
    /// Gets the forget gate weights for external inspection.
    /// </summary>
    public Tensor<T> GetForgetGateWeights() => _forgetGateWeights;
}
