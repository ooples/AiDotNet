using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Gated DeltaProduct layer from "DeltaProduct: Increasing the Expressivity of DeltaNet
/// Through Products of Householders" (Siems et al., 2025).
/// </summary>
/// <remarks>
/// <para>
/// Gated DeltaProduct combines the Householder product state transitions of DeltaProduct with
/// output gating (similar to how GatedDeltaNet adds gating to DeltaNet). The forget gate (alpha)
/// provides an additional scalar decay before the Householder rotation, and the output gate
/// controls information flow to the final output.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute gates:
///      - alpha (forget gate via sigmoid): scalar decay per head
///      - beta (write strength via sigmoid): controls write magnitude
///      - gate (output gate via Swish): modulates the output
///   3. For each timestep, compute M Householder vectors {u_1, ..., u_M}
///   4. State update with gated Householder product:
///      H_t = (I - 2*u_M*u_M^T/||u_M||^2) * ... * (I - 2*u_1*u_1^T/||u_1||^2)
///      S_t = alpha_t * H_t * S_{t-1} + beta_t * v_t * k_t^T
///   5. Output: O_t = gate_t * (S_t * q_t)
///   6. Output projection
/// </code>
/// </para>
/// <para>
/// The combination of gating and Householder products provides maximum expressivity:
/// - Alpha gate: Controls how much old state to retain (like GatedDeltaNet)
/// - Householder product: ROTATES the retained state (like DeltaProduct)
/// - Beta gate: Controls how strongly to write new information
/// - Output gate: Controls what to expose from the state
///
/// This means the model can: fade old information (alpha), rearrange what remains (Householder),
/// write new information selectively (beta), and filter what to output (gate).
/// </para>
/// <para><b>For Beginners:</b> Gated DeltaProduct is the fully-featured version that combines
/// everything from both GatedDeltaNet and DeltaProduct.
///
/// Think of managing a library:
/// - Alpha (forget gate): "Remove 10% of books from each shelf" (uniform fading)
/// - Householder product: "Rearrange remaining books by topic instead of author" (rotation)
/// - Beta (write gate): "How many new books to add to the collection" (write strength)
/// - Output gate: "Which shelves to show to the current visitor" (output filtering)
///
/// Without gating (plain DeltaProduct): you can rearrange and add books, but can't thin out
/// the collection. Without Householder (plain GatedDeltaNet): you can thin out and add, but
/// can't rearrange. Gated DeltaProduct does all four operations at each step.
///
/// This makes it the most expressive variant in the DeltaNet family, at the cost of slightly
/// more compute per step.
/// </para>
/// <para>
/// <b>Reference:</b> Siems et al., "DeltaProduct: Increasing the Expressivity of DeltaNet Through
/// Products of Householders", 2025. https://arxiv.org/abs/2502.10297
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Gating)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving. Relations discovered by probing; roles read from the forward - this folder's
// convention is seqLen = Shape[rank-2], modelDim = Shape[rank-1], so rank 2 is [Time, Features] with
// NO batch axis. OutputAxesFor is generated from these layouts.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GatedDeltaProductLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _numHouseholders;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Beta (write strength) projection: [modelDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _betaWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _betaBias;

    // Alpha (forget gate) projection: [modelDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _alphaWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _alphaBias;

    // Householder vector projections: [numHouseholders, modelDim, headDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _householderWeights;

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
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastBeta;
    private Tensor<T>? _lastAlpha;
    private Tensor<T>? _lastHouseholderVecs;
    private Tensor<T>? _lastRecurrenceOutput;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _betaWeightsGradient;
    private Tensor<T>? _betaBiasGradient;
    private Tensor<T>? _alphaWeightsGradient;
    private Tensor<T>? _alphaBiasGradient;
    private Tensor<T>? _householderWeightsGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the number of Householder reflections per timestep.
    /// </summary>
    public int NumHouseholders => _numHouseholders;

    /// <summary>
    /// Creates a new Gated DeltaProduct layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own state matrix and Householder
    /// reflections. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="numHouseholders">
    /// Number of Householder reflections per timestep. Default: 4.
    /// <para><b>For Beginners:</b> More reflections give more expressive state rotations.
    /// 2-4 reflections capture most of the benefit in practice.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public GatedDeltaProductLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int numHouseholders = 4,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (sequenceLength <= 0)
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        if (numHouseholders <= 0)
            throw new ArgumentException($"Number of Householder reflections ({numHouseholders}) must be positive.", nameof(numHouseholders));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _numHouseholders = numHouseholders;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _betaWeights = new Tensor<T>([modelDimension, numHeads]);
        _betaBias = new Tensor<T>([numHeads]);
        _alphaWeights = new Tensor<T>([modelDimension, numHeads]);
        _alphaBias = new Tensor<T>([numHeads]);
        _householderWeights = new Tensor<T>([numHouseholders, modelDimension, _headDimension]);
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
        InitializeTensor2D(_betaWeights);
        _betaBias.Fill(NumOps.FromDouble(0.1));
        InitializeTensor2D(_alphaWeights);
        // Alpha bias ~ 2 so sigmoid(2) ~ 0.88 -> strong initial memory retention
        for (int i = 0; i < _alphaBias.Length; i++)
            _alphaBias[i] = NumOps.FromDouble(2.0);
        InitializeHouseholderWeights();
        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    private void InitializeHouseholderWeights()
    {
        InitializeLayerWeights(_householderWeights, _modelDimension, _headDimension);
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
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });

        // Step 1: Q, K, V projections
        var q = Engine.Reshape(Engine.TensorMatMul(inputFlat, _queryWeights), new[] { batchSize, seqLen, _modelDimension });
        var k = Engine.Reshape(Engine.TensorMatMul(inputFlat, _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var v = Engine.Reshape(Engine.TensorMatMul(inputFlat, _valueWeights), new[] { batchSize, seqLen, _modelDimension });
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Gates
        var betaProjection = Engine.TensorMatMul(inputFlat, _betaWeights);
        var betaBias = Engine.TensorBroadcastTo(
            Engine.Reshape(_betaBias, new[] { 1, _numHeads }),
            new[] { batchSize * seqLen, _numHeads });
        var betaRaw = Engine.Reshape(
            Engine.TensorAdd(betaProjection, betaBias), new[] { batchSize, seqLen, _numHeads });
        var beta = Engine.Sigmoid(betaRaw);
        _lastBeta = beta;

        var alphaProjection = Engine.TensorMatMul(inputFlat, _alphaWeights);
        var alphaBias = Engine.TensorBroadcastTo(
            Engine.Reshape(_alphaBias, new[] { 1, _numHeads }),
            new[] { batchSize * seqLen, _numHeads });
        var alphaRaw = Engine.Reshape(
            Engine.TensorAdd(alphaProjection, alphaBias), new[] { batchSize, seqLen, _numHeads });
        var alpha = Engine.Sigmoid(alphaRaw);
        _lastAlpha = alpha;

        var gateProjection = Engine.TensorMatMul(inputFlat, _outputGateWeights);
        var gateBias = Engine.TensorBroadcastTo(
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension }),
            new[] { batchSize * seqLen, _modelDimension });
        var gateRaw = Engine.Reshape(
            Engine.TensorAdd(gateProjection, gateBias), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Swish(gateRaw);
        _lastOutputGate = gate;
        _lastOutputGateRaw = gateRaw;

        // Step 3: Compute Householder vectors
        var hVecs = ComputeHouseholderVectors(inputFlat, batchSize, seqLen);
        _lastHouseholderVecs = hVecs;

        // Step 4: Gated DeltaProduct recurrence
        var recOutput = GatedDeltaProductRecurrence(q, k, v, alpha, beta, hVecs, batchSize, seqLen);
        _lastRecurrenceOutput = recOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(recOutput, gate);

        // Step 6: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias = Engine.TensorBroadcastTo(
            Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension }),
            new[] { batchSize * seqLen, _modelDimension });
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
    /// Computes Householder vectors from input for each timestep.
    /// Returns shape [batch * seqLen, numHouseholders, numHeads, headDim].
    /// </summary>
    /// <remarks>
    /// The projection does not depend on the head index at all - every head sees the same vector -
    /// so this is one matmul per Householder index, tiled across heads. Built from Engine ops so
    /// _householderWeights stays on the tape; the scalar loop this replaces severed it.
    /// </remarks>
    private Tensor<T> ComputeHouseholderVectors(Tensor<T> inputFlat, int batchSize, int seqLen)
    {
        int total = batchSize * seqLen;
        var reflections = new Tensor<T>[_numHouseholders];
        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            var weightSlice = Engine.TensorNarrow(_householderWeights, dim: 0, start: mi, length: 1);
            var weightMatrix = Engine.Reshape(weightSlice, new[] { _modelDimension, _headDimension });
            var projected = Engine.TensorMatMul(inputFlat, weightMatrix);
            var perHead = Engine.TensorBroadcastTo(
                Engine.Reshape(projected, new[] { total, 1, _headDimension }),
                new[] { total, _numHeads, _headDimension });
            reflections[mi] = Engine.Reshape(
                perHead, new[] { total, 1, _numHeads, _headDimension });
        }

        return Engine.TensorConcatenate(reflections, axis: 1);
    }

    /// <summary>
    /// Gated DeltaProduct recurrence: Householder-product transition, alpha gate, then delta update.
    /// <code>
    ///   S_t = alpha_t * (H_M ... H_1) * S_{t-1} + beta_t * v_t * k_t^T
    ///   O_t = S_t * q_t                       where H_m = I - 2 u u^T / (||u||^2 + eps)
    /// </code>
    /// </summary>
    /// <remarks>
    /// Written with Engine ops, mirroring the ungated <c>DeltaProductLayer</c> - which computes the
    /// same recurrence and is already fully differentiable. The scalar loop this replaces was
    /// detached from the tape, so q/k/v, both gates and the Householder weights (8 of 12 trainable
    /// tensors) received no gradient and never learned. This layer has no Backward override, so the
    /// tape was its only gradient path.
    /// </remarks>
    private Tensor<T> GatedDeltaProductRecurrence(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> alpha, Tensor<T> beta,
        Tensor<T> hVecs, int batchSize, int seqLen)
    {
        int batchHeads = batchSize * _numHeads;
        var qHeads = Engine.Reshape(q, new[] { batchSize, seqLen, _numHeads, _headDimension });
        var kHeads = Engine.Reshape(k, new[] { batchSize, seqLen, _numHeads, _headDimension });
        var vHeads = Engine.Reshape(v, new[] { batchSize, seqLen, _numHeads, _headDimension });
        var hByTime = Engine.Reshape(
            hVecs,
            new[] { batchSize, seqLen, _numHouseholders, _numHeads, _headDimension });
        var state = new Tensor<T>(new[] { batchHeads, _headDimension, _headDimension });
        var outputs = new Tensor<T>[seqLen];
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        T two = NumOps.FromDouble(2.0);
        T epsilon = NumOps.FromDouble(1e-8);
        int[] stateShape = { batchHeads, _headDimension, _headDimension };

        for (int t = 0; t < seqLen; t++)
        {
            for (int mi = 0; mi < _numHouseholders; mi++)
            {
                var timeSlice = Engine.TensorNarrow(hByTime, dim: 1, start: t, length: 1);
                var reflectionSlice = Engine.TensorNarrow(timeSlice, dim: 2, start: mi, length: 1);
                var uColumn = Engine.Reshape(
                    reflectionSlice, new[] { batchHeads, _headDimension, 1 });
                var uTranspose = Engine.TensorPermute(uColumn, new[] { 0, 2, 1 });
                var normSquared = Engine.ReduceSum(
                    Engine.TensorMultiply(uColumn, uColumn), new[] { 1 }, keepDims: true);
                normSquared = Engine.TensorAddScalar(normSquared, epsilon);

                var projectedState = Engine.TensorBatchMatMul(uTranspose, state);
                var correction = Engine.TensorBatchMatMul(uColumn, projectedState);
                correction = Engine.TensorMultiplyScalar(correction, two);
                var normBroadcast = Engine.TensorBroadcastTo(normSquared, stateShape);
                correction = Engine.TensorDivide(correction, normBroadcast);
                state = Engine.TensorSubtract(state, correction);
            }

            var alphaSlice = Engine.TensorNarrow(alpha, dim: 1, start: t, length: 1);
            var alphaScale = Engine.TensorBroadcastTo(
                Engine.Reshape(alphaSlice, new[] { batchHeads, 1, 1 }), stateShape);
            state = Engine.TensorMultiply(state, alphaScale);

            var keySlice = Engine.TensorNarrow(kHeads, dim: 1, start: t, length: 1);
            var valueSlice = Engine.TensorNarrow(vHeads, dim: 1, start: t, length: 1);
            var keyRow = Engine.Reshape(keySlice, new[] { batchHeads, 1, _headDimension });
            keyRow = Engine.TensorMultiplyScalar(keyRow, keyScale);
            var valueColumn = Engine.Reshape(valueSlice, new[] { batchHeads, _headDimension, 1 });
            var update = Engine.TensorBatchMatMul(valueColumn, keyRow);
            var betaSlice = Engine.TensorNarrow(beta, dim: 1, start: t, length: 1);
            var betaScale = Engine.TensorBroadcastTo(
                Engine.Reshape(betaSlice, new[] { batchHeads, 1, 1 }), stateShape);
            state = Engine.TensorAdd(state, Engine.TensorMultiply(update, betaScale));

            var querySlice = Engine.TensorNarrow(qHeads, dim: 1, start: t, length: 1);
            var queryColumn = Engine.Reshape(querySlice, new[] { batchHeads, _headDimension, 1 });
            var outputColumn = Engine.TensorBatchMatMul(state, queryColumn);
            outputs[t] = Engine.Reshape(
                outputColumn, new[] { batchSize, 1, _numHeads, _headDimension });
        }

        var outputHeads = Engine.TensorConcatenate(outputs, axis: 1);
        return Engine.Reshape(outputHeads, new[] { batchSize, seqLen, _modelDimension });
    }

    private Tensor<T> ToHeadMajor(Tensor<T> value, int batchSize, int seqLen) =>
        Engine.Reshape(Engine.TensorPermute(
            Engine.Reshape(value, new[] { batchSize, seqLen, _numHeads, _headDimension }),
            new[] { 0, 2, 1, 3 }),
            new[] { batchSize * _numHeads, seqLen, _headDimension });

    private Tensor<T> FromHeadMajor(Tensor<T> value, int batchSize, int seqLen) =>
        Engine.Reshape(Engine.TensorPermute(
            Engine.Reshape(value, new[] { batchSize, _numHeads, seqLen, _headDimension }),
            new[] { 0, 2, 1, 3 }),
            new[] { batchSize, seqLen, _modelDimension });

    /// <summary>
    /// Computes input gradient contribution from Householder vectors.
    /// </summary>
    private Tensor<T> ComputeHouseholderInputGradient(
        Tensor<T> dHVecs, int batchSize, int seqLen)
    {
        int total = batchSize * seqLen;
        var dInput = TensorAllocator.Rent<T>(new[] { total, _modelDimension });

        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            for (int pos = 0; pos < total; pos++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    for (int d = 0; d < _headDimension; d++)
                    {
                        T dH = dHVecs[new[] { pos, mi, hi, d }];
                        for (int j = 0; j < _modelDimension; j++)
                        {
                            dInput[new[] { pos, j }] = NumOps.Add(
                                dInput[new[] { pos, j }],
                                NumOps.Multiply(dH, _householderWeights[new[] { mi, j, d }]));
                        }
                    }
                }
            }
        }

        return dInput;
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        var sig = Engine.Sigmoid(x);
        var oneMinusSig = Engine.ScalarMinusTensor(NumOps.One, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAddScalar(xTimesOneMinusSig, NumOps.One);
        return Engine.TensorMultiply(sig, onePlusXSig);
    }

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
        var queryWeightsGrad = _queryWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var keyWeightsGrad = _keyWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var valueWeightsGrad = _valueWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var betaWeightsGrad = _betaWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var betaBiasGrad = _betaBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var alphaWeightsGrad = _alphaWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var alphaBiasGrad = _alphaBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var householderWeightsGrad = _householderWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputGateWeightsGrad = _outputGateWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputGateBiasGrad = _outputGateBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionWeightsGrad = _outputProjectionWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionBiasGrad = _outputProjectionBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(queryWeightsGrad, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(keyWeightsGrad, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(valueWeightsGrad, negLR));
        _betaWeights = Engine.TensorAdd(_betaWeights, Engine.TensorMultiplyScalar(betaWeightsGrad, negLR));
        _betaBias = Engine.TensorAdd(_betaBias, Engine.TensorMultiplyScalar(betaBiasGrad, negLR));
        _alphaWeights = Engine.TensorAdd(_alphaWeights, Engine.TensorMultiplyScalar(alphaWeightsGrad, negLR));
        _alphaBias = Engine.TensorAdd(_alphaBias, Engine.TensorMultiplyScalar(alphaBiasGrad, negLR));
        _householderWeights = Engine.TensorAdd(_householderWeights, Engine.TensorMultiplyScalar(householderWeightsGrad, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(outputGateWeightsGrad, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(outputGateBiasGrad, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(outputProjectionWeightsGrad, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(outputProjectionBiasGrad, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_betaWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_betaBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_alphaWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_alphaBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_householderWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _keyWeights, _valueWeights,
        _betaWeights, _betaBias,
        _alphaWeights, _alphaBias,
        _householderWeights,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_betaWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_betaBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_alphaWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_alphaBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_householderWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _betaWeightsGradient = null; _betaBiasGradient = null; _alphaWeightsGradient = null; _alphaBiasGradient = null; _householderWeightsGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
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
        _lastAlpha = null;
        _lastHouseholderVecs = null;
        _lastRecurrenceOutput = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _betaWeightsGradient = null;
        _betaBiasGradient = null;
        _alphaWeightsGradient = null;
        _alphaBiasGradient = null;
        _householderWeightsGradient = null;
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
        metadata["NumHouseholders"] = _numHouseholders.ToString();
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
    /// Gets the Householder projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetHouseholderWeights() => _householderWeights;
}
