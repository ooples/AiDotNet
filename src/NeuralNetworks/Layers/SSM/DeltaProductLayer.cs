using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the DeltaProduct layer from "DeltaProduct: Increasing the Expressivity of DeltaNet
/// Through Products of Householders" (Siems et al., 2025).
/// </summary>
/// <remarks>
/// <para>
/// DeltaProduct extends DeltaNet by replacing the scalar forget gate with a product of Householder
/// reflections for state transitions. A Householder reflection H = I - 2*v*v^T/||v||^2 is an
/// orthogonal transformation that reflects vectors across the hyperplane perpendicular to v.
/// By composing multiple Householder reflections, DeltaProduct can represent any orthogonal
/// transformation of the state, making the state transition far more expressive than a scalar decay.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute beta (write strength) via sigmoid
///   3. For each timestep, compute M Householder vectors {u_1, ..., u_M}
///   4. State update with product of Householder reflections:
///      H_t = (I - 2*u_M*u_M^T/||u_M||^2) * ... * (I - 2*u_1*u_1^T/||u_1||^2)
///      S_t = H_t * S_{t-1} + beta_t * v_t * k_t^T
///   5. Output: O_t = S_t * q_t
///   6. Output projection
/// </code>
/// </para>
/// <para>
/// The key insight: in standard DeltaNet, the state transition is S_t = alpha * S_{t-1} + ...,
/// where alpha is just a scalar decay. This limits how the state can evolve -- old information
/// can only fade uniformly. With Householder products, the state can be ROTATED and REFLECTED
/// before the new write, preserving information while restructuring it. Since any orthogonal
/// matrix can be decomposed into Householder reflections, M reflections can express any rotation
/// in the head dimension space when M >= headDim.
/// </para>
/// <para><b>For Beginners:</b> DeltaProduct improves DeltaNet by adding "rotations" to memory updates.
///
/// Think of the state matrix as a whiteboard of notes:
/// - DeltaNet: Before writing new notes, you can only FADE the old notes (scalar alpha)
/// - DeltaProduct: Before writing, you can REARRANGE the old notes (rotate/reflect them)
///
/// A Householder reflection is like flipping the whiteboard across a mirror:
/// - One reflection can flip everything across one axis
/// - Two reflections can rotate everything by any angle
/// - M reflections can do any rearrangement that preserves the "length" of your notes
///
/// This means DeltaProduct can:
/// - Move old information to make room for new information (rotation)
/// - Flip the organization of information (reflection)
/// - All while preserving the total amount of stored information (orthogonality)
///
/// The result is a more expressive model that better manages what it remembers and forgets.
/// </para>
/// <para>
/// <b>Reference:</b> Siems et al., "DeltaProduct: Increasing the Expressivity of DeltaNet Through
/// Products of Householders", 2025. https://arxiv.org/abs/2502.10297
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
public partial class DeltaProductLayer<T> : LayerBase<T>, IShapeContract
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

    // Householder vector projections: [numHouseholders, modelDim, headDim]
    // Each Householder gets its own projection from input to a per-head vector
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _householderWeights;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
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
    private Tensor<T>? _lastHouseholderVecs;
    [Scratch]
    private Tensor<T>? _lastStates;
    [Scratch]
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    [Scratch]
    private Tensor<T>? _queryWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _betaWeightsGradient;
    [Scratch]
    private Tensor<T>? _betaBiasGradient;
    [Scratch]
    private Tensor<T>? _householderWeightsGradient;
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

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// Creates a new DeltaProduct layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own state matrix and set of
    /// Householder reflections. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="numHouseholders">
    /// Number of Householder reflections per timestep. Default: 4.
    /// <para><b>For Beginners:</b> More reflections allow more complex state rotations.
    /// With M = headDim reflections, any orthogonal transformation is possible.
    /// In practice, 2-4 reflections capture most of the benefit.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public DeltaProductLayer(
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
        // Each Householder reflection needs a headDim vector per head, projected from modelDim input
        _householderWeights = new Tensor<T>([numHouseholders, modelDimension, _headDimension]);
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
        InitializeHouseholderWeights();
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

        // Step 2: Beta (write strength)
        var betaRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _betaWeights),
            Engine.Reshape(_betaBias, new[] { 1, _numHeads })), new[] { batchSize, seqLen, _numHeads });
        var beta = Engine.Sigmoid(betaRaw);
        _lastBeta = beta;

        // Step 3: Compute Householder vectors per timestep
        // householderVecs: [batch, seqLen, numHouseholders, numHeads, headDim]
        var hVecs = ComputeHouseholderVectors(inputFlat, batchSize, seqLen);
        _lastHouseholderVecs = hVecs;

        // Step 4: DeltaProduct recurrence
        var recOutput = DeltaProductRecurrence(q, k, v, beta, hVecs, batchSize, seqLen);
        _lastRecurrenceOutput = recOutput;

        // Step 5: Output projection
        var outputFlat = Engine.TensorMatMul(
            Engine.Reshape(recOutput, new[] { batchSize * seqLen, _modelDimension }), _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias);
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
    private Tensor<T> ComputeHouseholderVectors(Tensor<T> inputFlat, int batchSize, int seqLen)
    {
        int total = batchSize * seqLen;
        var projections = new List<Tensor<T>>(_numHouseholders);
        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            var weights = Engine.Reshape(
                Engine.TensorSlice(_householderWeights,
                    new[] { mi, 0, 0 }, new[] { 1, _modelDimension, _headDimension }),
                new[] { _modelDimension, _headDimension });
            var projected = Engine.TensorMatMul(inputFlat, weights); // [B*T,D]
            projections.Add(Engine.Reshape(
                Engine.TensorTile(Engine.Reshape(projected,
                    new[] { total, 1, 1, _headDimension }),
                    new[] { 1, 1, _numHeads, 1 }),
                new[] { total, 1, _numHeads, _headDimension }));
        }
        return Engine.TensorConcatenate(projections.ToArray(), axis: 1);
    }

    /// <summary>
    /// Applies the product of M Householder reflections to a matrix.
    /// H = H_M * ... * H_1, where H_m = I - 2*u*u^T/||u||^2.
    /// Computes H * S for state matrix S.
    /// </summary>
    private void ApplyHouseholderProduct(
        Tensor<T> state, Tensor<T> hVecs,
        int bi, int hi, int posFlat)
    {
        // Apply each Householder reflection sequentially: S <- (I - 2*u*u^T/||u||^2) * S
        for (int mi = 0; mi < _numHouseholders; mi++)
        {
            // Compute ||u||^2
            T normSq = NumOps.Zero;
            for (int d = 0; d < _headDimension; d++)
            {
                T u = hVecs[new[] { posFlat, mi, hi, d }];
                normSq = NumOps.Add(normSq, NumOps.Multiply(u, u));
            }
            T eps = NumOps.FromDouble(1e-8);
            normSq = NumOps.Add(normSq, eps);
            T twoOverNormSq = NumOps.Divide(NumOps.FromDouble(2.0), normSq);

            // For each column j of S: S[:,j] <- S[:,j] - (2/||u||^2) * u * (u^T * S[:,j])
            for (int j = 0; j < _headDimension; j++)
            {
                // dot = u^T * S[:,j]
                T dot = NumOps.Zero;
                for (int d = 0; d < _headDimension; d++)
                {
                    T u = hVecs[new[] { posFlat, mi, hi, d }];
                    dot = NumOps.Add(dot, NumOps.Multiply(u, state[new[] { bi, hi, d, j }]));
                }
                T factor = NumOps.Multiply(twoOverNormSq, dot);

                for (int d = 0; d < _headDimension; d++)
                {
                    T u = hVecs[new[] { posFlat, mi, hi, d }];
                    state[new[] { bi, hi, d, j }] = NumOps.Subtract(
                        state[new[] { bi, hi, d, j }],
                        NumOps.Multiply(factor, u));
                }
            }
        }
    }

    /// <summary>
    /// DeltaProduct recurrence: state update with Householder product transitions.
    /// S_t = H_t * S_{t-1} + beta_t * v_t * k_t^T
    /// O_t = S_t * q_t
    /// </summary>
    private Tensor<T> DeltaProductRecurrence(
        Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T> beta,
        Tensor<T> hVecs, int batchSize, int seqLen)
    {
        int headBatch = batchSize * _numHeads;
        var qHeads = ToHeadMajor(q, batchSize, seqLen);
        var kHeads = Engine.TensorMultiplyScalar(ToHeadMajor(k, batchSize, seqLen),
            NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension)));
        var vHeads = ToHeadMajor(v, batchSize, seqLen);
        var betaHeads = Engine.Reshape(Engine.TensorPermute(beta, new[] { 0, 2, 1 }),
            new[] { headBatch, seqLen, 1 });
        var state = Tensor<T>.CreateDefault(
            new[] { headBatch, _headDimension, _headDimension }, NumOps.Zero);
        var outputs = new List<Tensor<T>>(seqLen);
        var two = NumOps.FromDouble(2.0);

        for (int t = 0; t < seqLen; t++)
        {
            for (int mi = 0; mi < _numHouseholders; mi++)
            {
                var u = Engine.TensorSlice(hVecs,
                    new[] { t, mi, 0, 0 }, new[] { 1, 1, _numHeads, _headDimension });
                // hVecs is position-major; gather all batches for this time and flatten heads.
                if (batchSize > 1)
                {
                    var perBatch = new List<Tensor<T>>(batchSize);
                    for (int bi = 0; bi < batchSize; bi++)
                    {
                        int pos = bi * seqLen + t;
                        perBatch.Add(Engine.TensorSlice(hVecs,
                            new[] { pos, mi, 0, 0 }, new[] { 1, 1, _numHeads, _headDimension }));
                    }
                    u = Engine.TensorConcatenate(perBatch.ToArray(), 0);
                }
                var uCol = Engine.Reshape(u, new[] { headBatch, _headDimension, 1 });
                var uRow = Engine.TensorPermute(uCol, new[] { 0, 2, 1 });
                var normSq = Engine.BatchMatMul(uRow, uCol);
                var denominator = Engine.TensorAddScalar(normSq, NumOps.FromDouble(1e-8));
                var reflection = Engine.TensorBroadcastMultiply(
                    Engine.BatchMatMul(uCol, uRow),
                    Engine.TensorDivide(Tensor<T>.CreateDefault(
                        new[] { headBatch, 1, 1 }, two), denominator));
                state = Engine.TensorSubtract(state, Engine.BatchMatMul(reflection, state));
            }

            var qCol = Engine.Reshape(Engine.TensorSliceAxis(qHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var kCol = Engine.Reshape(Engine.TensorSliceAxis(kHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var vCol = Engine.Reshape(Engine.TensorSliceAxis(vHeads, 1, t),
                new[] { headBatch, _headDimension, 1 });
            var betaT = Engine.Reshape(Engine.TensorSliceAxis(betaHeads, 1, t),
                new[] { headBatch, 1, 1 });
            state = Engine.TensorAdd(state, Engine.TensorBroadcastMultiply(
                Engine.BatchMatMul(vCol, Engine.TensorPermute(kCol, new[] { 0, 2, 1 })), betaT));
            outputs.Add(Engine.Reshape(Engine.BatchMatMul(state, qCol),
                new[] { headBatch, 1, _headDimension }));
        }

        _lastStates = state;
        return FromHeadMajor(Engine.TensorConcatenate(outputs.ToArray(), 1), batchSize, seqLen);
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
    /// Accumulates Householder weight gradients from per-position gradients.
    /// </summary>
    private void AccumulateHouseholderWeightGradients(
        Tensor<T> dHVecs, Tensor<T> inputFlat, int batchSize, int seqLen)
    {
        int total = batchSize * seqLen;
        var gradient = _householderWeightsGradient ?? throw new InvalidOperationException("Gradients not initialized.");

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
                            gradient[new[] { mi, j, d }] = NumOps.Add(
                                gradient[new[] { mi, j, d }],
                                NumOps.Multiply(dH, inputFlat[new[] { pos, j }]));
                        }
                    }
                }
            }
        }
    }

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
        var householderWeightsGrad = _householderWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionWeightsGrad = _outputProjectionWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionBiasGrad = _outputProjectionBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(queryWeightsGrad, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(keyWeightsGrad, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(valueWeightsGrad, negLR));
        _betaWeights = Engine.TensorAdd(_betaWeights, Engine.TensorMultiplyScalar(betaWeightsGrad, negLR));
        _betaBias = Engine.TensorAdd(_betaBias, Engine.TensorMultiplyScalar(betaBiasGrad, negLR));
        _householderWeights = Engine.TensorAdd(_householderWeights, Engine.TensorMultiplyScalar(householderWeightsGrad, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(outputProjectionWeightsGrad, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(outputProjectionBiasGrad, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_betaWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_betaBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_householderWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _keyWeights, _valueWeights,
        _betaWeights, _betaBias,
        _householderWeights,
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
            new Vector<T>(_householderWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _betaWeightsGradient = null; _betaBiasGradient = null; _householderWeightsGradient = null;
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
        _lastHouseholderVecs = null;
        _lastStates = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _betaWeightsGradient = null;
        _betaBiasGradient = null;
        _householderWeightsGradient = null;
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
