using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[LayerCategory(LayerCategory.StateSpaceModel)]
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
public partial class DeltaNetLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _queryBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _keyBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _valueBias;

    // Beta (write strength) projection: [modelDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _betaWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _betaBias;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    [Scratch]
    private Tensor<T>? _lastInput;
    [Scratch]
    private Tensor<T>? _lastOutput;
    [Scratch]
    private Tensor<T>? _lastQuery;
    [Scratch]
    private Tensor<T>? _lastKey;
    [Scratch]
    private Tensor<T>? _lastValue;
    [Scratch]
    private Tensor<T>? _lastBeta;
    [Scratch]
    private Tensor<T>? _lastStates;
    [Scratch]
    private Tensor<T>? _lastDeltaRuleOutput;
    private int[]? _originalInputShape;

    // Gradients
    [Scratch]
    private Tensor<T>? _queryWeightsGradient;
    [Scratch]
    private Tensor<T>? _queryBiasGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyBiasGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueBiasGradient;
    [Scratch]
    private Tensor<T>? _betaWeightsGradient;
    [Scratch]
    private Tensor<T>? _betaBiasGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

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

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

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
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        _sequenceLength = sequenceLength;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

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

        _lastInput = input3D;

        // Step 1: Q, K, V projections
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });

        var qFlat = Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _queryWeights),
            Engine.Reshape(_queryBias, new[] { 1, _modelDimension }));
        var q = Engine.Reshape(qFlat, new[] { batchSize, seqLen, _modelDimension });

        var kFlat = Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _keyWeights),
            Engine.Reshape(_keyBias, new[] { 1, _modelDimension }));
        var k = Engine.Reshape(kFlat, new[] { batchSize, seqLen, _modelDimension });

        var vFlat = Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _valueWeights),
            Engine.Reshape(_valueBias, new[] { 1, _modelDimension }));
        var v = Engine.Reshape(vFlat, new[] { batchSize, seqLen, _modelDimension });

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Beta (write strength) via sigmoid
        var betaRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _betaWeights),
            Engine.Reshape(_betaBias, new[] { 1, _numHeads })), new[] { batchSize, seqLen, _numHeads });
        var beta = Engine.Sigmoid(betaRaw);
        _lastBeta = beta;

        // Step 3: Delta rule recurrence per head
        var output = DeltaRuleForward(q, k, v, beta, batchSize, seqLen);
        _lastDeltaRuleOutput = output;

        // Step 4: Output projection
        var outputFlat = Engine.TensorMatMul(
            Engine.Reshape(output, new[] { batchSize * seqLen, _modelDimension }),
            _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorAdd(outputFlat, outBias);
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
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
        int headBatch = batchSize * _numHeads;
        var qHeads = ToHeadMajor(q, batchSize, seqLen);
        var kHeads = Engine.TensorMultiplyScalar(
            ToHeadMajor(k, batchSize, seqLen),
            NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension)));
        var vHeads = ToHeadMajor(v, batchSize, seqLen);
        var betaHeads = Engine.Reshape(
            Engine.TensorPermute(beta, new[] { 0, 2, 1 }),
            new[] { headBatch, seqLen, 1 });

        // A recurrent state must itself stay on the tape. The old implementation wrote scalar
        // values into a rented tensor, so every projection before the recurrence received a zero
        // derivative even though the forward values were right.
        var state = Tensor<T>.CreateDefault(
            new[] { headBatch, _headDimension, _headDimension }, NumOps.Zero);
        var outputs = new List<Tensor<T>>(seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            var qCol = Engine.Reshape(Engine.TensorSliceAxis(qHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var kCol = Engine.Reshape(Engine.TensorSliceAxis(kHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var kRow = Engine.TensorPermute(kCol, new[] { 0, 2, 1 });
            var vCol = Engine.Reshape(Engine.TensorSliceAxis(vHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var betaT = Engine.Reshape(Engine.TensorSliceAxis(betaHeads, 1, t),
                new[] { headBatch, 1, 1 });

            var prediction = Engine.BatchMatMul(state, kCol);
            var delta = Engine.TensorSubtract(vCol, prediction);
            var update = Engine.TensorMultiply(
                Engine.BatchMatMul(delta, kRow), betaT);
            state = Engine.TensorAdd(state, update);

            var y = Engine.BatchMatMul(state, qCol);
            outputs.Add(Engine.Reshape(y, new[] { headBatch, 1, _headDimension }));
        }

        _lastStates = state;
        return FromHeadMajor(Engine.TensorConcatenate(outputs.ToArray(), 1), batchSize, seqLen);
    }

    private Tensor<T> ToHeadMajor(Tensor<T> value, int batchSize, int seqLen) =>
        Engine.Reshape(
            Engine.TensorPermute(
                Engine.Reshape(value, new[] { batchSize, seqLen, _numHeads, _headDimension }),
                new[] { 0, 2, 1, 3 }),
            new[] { batchSize * _numHeads, seqLen, _headDimension });

    private Tensor<T> FromHeadMajor(Tensor<T> value, int batchSize, int seqLen) =>
        Engine.Reshape(
            Engine.TensorPermute(
                Engine.Reshape(value, new[] { batchSize, _numHeads, seqLen, _headDimension }),
                new[] { 0, 2, 1, 3 }),
            new[] { batchSize, seqLen, _modelDimension });

    /// <summary>
    /// Creates a tensor of ones with the same shape as the template tensor.
    /// </summary>
    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var ones = new Tensor<T>(template._shape);
        ones.Fill(NumOps.One);
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

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_queryBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_betaWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_betaBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

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

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_queryBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_betaWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_betaBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _queryBiasGradient = null; _keyWeightsGradient = null; _keyBiasGradient = null; _valueWeightsGradient = null; _valueBiasGradient = null; _betaWeightsGradient = null; _betaBiasGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

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
