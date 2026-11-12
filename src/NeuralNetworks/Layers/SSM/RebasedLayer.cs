using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 8", TestConstructorArgs = "4, 8, 2")]
// Shape-preserving. Relations discovered by probing; roles read from the forward - this folder's
// convention is seqLen = Shape[rank-2], modelDim = Shape[rank-1], so rank 2 is [Time, Features].
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RebasedLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Output gate: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    [Scratch]
    private Tensor<T>? _lastQuery;
    [Scratch]
    private Tensor<T>? _lastKey;
    [Scratch]
    private Tensor<T>? _lastValue;
    [Scratch]
    private Tensor<T>? _lastPhiQ;
    [Scratch]
    private Tensor<T>? _lastPhiK;
    [Scratch]
    private Tensor<T>? _lastPhiQNorm;
    [Scratch]
    private Tensor<T>? _lastPhiKNorm;
    [Scratch]
    private Tensor<T>? _lastDenominators;
    private int[]? _originalInputShape;

    // Gradients
    [Scratch]
    private Tensor<T>? _queryWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputGateBiasGradient;
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
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

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
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
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
            T reluVal = NumOps.GreaterThan(val, NumOps.FromDouble(0.0)) ? val : NumOps.Zero;
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

        // Step 1: Q, K, V projections
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });
        var q = Engine.Reshape(Engine.TensorMatMul(inputFlat, _queryWeights), new[] { batchSize, seqLen, _modelDimension });
        var k = Engine.Reshape(Engine.TensorMatMul(inputFlat, _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var v = Engine.Reshape(Engine.TensorMatMul(inputFlat, _valueWeights), new[] { batchSize, seqLen, _modelDimension });
        SaveForBackward(ref _lastQuery, q);
        SaveForBackward(ref _lastKey, k);
        SaveForBackward(ref _lastValue, v);

        // Step 2: Output gate
        var gateRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var outputGate = Engine.Swish(gateRaw);

        // Step 3: Apply squared ReLU feature maps and run the shared, vectorized causal
        // recurrence. Prefix sums preserve the exact running-state semantics while keeping the
        // tape graph independent of sequence length.
        var phiQ = ApplySquaredReluFeatureMap(q, batchSize, seqLen);
        var phiK = ApplySquaredReluFeatureMap(k, batchSize, seqLen);
        var linearOutput = CausalLinearAttention.Normalized(
            Engine, phiQ, phiK, v, _numHeads, NumOps.FromDouble(1e-6));

        // Step 4: Gated output
        var gatedOutput = Engine.TensorMultiply(linearOutput, outputGate);

        // Step 5: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorAdd(outputFlat, outBias);
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }

    private Tensor<T> ApplySquaredReluFeatureMap(
        Tensor<T> input, int batchSize, int seqLen)
    {
        var heads = Engine.Reshape(
            input,
            new[] { batchSize, seqLen, _numHeads, _headDimension });
        var relu = Engine.ReLU(heads);
        var squared = Engine.TensorSquare(relu);
        var fourthPower = Engine.TensorSquare(squared);
        var norm = Engine.TensorSqrt(
            Engine.TensorAddScalar(
                Engine.ReduceSum(fourthPower, new[] { 3 }, keepDims: true),
                NumOps.FromDouble(1e-8)));
        return Engine.Reshape(
            Engine.TensorDivide(squared, norm),
            new[] { batchSize, seqLen, _modelDimension });
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        var sig = Engine.Sigmoid(x);
        var oneMinusSig = Engine.ScalarMinusTensor(NumOps.One, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAddScalar(xTimesOneMinusSig, NumOps.One);
        return Engine.TensorMultiply(sig, onePlusXSig);
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        Engine.TensorAddInPlace(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        Engine.TensorAddInPlace(_keyWeights, Engine.TensorMultiplyScalar((_keyWeightsGradient ?? throw new InvalidOperationException("_keyWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_valueWeights, Engine.TensorMultiplyScalar((_valueWeightsGradient ?? throw new InvalidOperationException("_valueWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputGateWeights, Engine.TensorMultiplyScalar((_outputGateWeightsGradient ?? throw new InvalidOperationException("_outputGateWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputGateBias, Engine.TensorMultiplyScalar((_outputGateBiasGradient ?? throw new InvalidOperationException("_outputGateBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionWeights, Engine.TensorMultiplyScalar((_outputProjectionWeightsGradient ?? throw new InvalidOperationException("_outputProjectionWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionBias, Engine.TensorMultiplyScalar((_outputProjectionBiasGradient ?? throw new InvalidOperationException("_outputProjectionBiasGradient is null where it is required.")), negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _keyWeights, _valueWeights,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastPhiQ = null;
        _lastPhiK = null;
        _lastPhiQNorm = null;
        _lastPhiKNorm = null;
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
