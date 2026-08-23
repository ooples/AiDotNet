using AiDotNet.Autodiff;
using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Oblivious Decision Tree (ODT) for NODE architecture.
/// </summary>
/// <remarks>
/// <para>
/// An oblivious decision tree uses the same feature and threshold at each level,
/// making it more regularized and efficient than standard decision trees.
/// NODE uses differentiable ODTs with entmax splits for end-to-end learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> An oblivious tree is a special type of decision tree where:
/// - At level 1, ALL nodes use the same feature (e.g., "age > 30")
/// - At level 2, ALL nodes use the same feature (e.g., "income > 50k")
/// - And so on...
///
/// This is simpler than regular trees where each node can use different features.
/// The simplicity helps prevent overfitting and makes the tree faster.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
// Rank 2 EXACTLY, and this layer says so twice in its own words. OnFirstForward throws for any other
// rank - "requires rank-2 input [batch, features]" - and ForwardTraced re-checks the same thing on every
// call rather than only on the first, because "ODT Forward indexes the tensor as a flat [batch,
// _inputDim] matrix [...] rank-1 input would alias batch onto the feature axis and silently produce
// garbage". So BatchOptional is not merely unnecessary here, it would declare the precise form the layer
// was hardened to reject.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input,
    Note = "Unbatched or higher-rank data must be reshaped to [batch, features] upstream.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
// Declared so this layer is GENERATED A TEST AT ALL. TestScaffoldGenerator skips any layer with
// neither a parameterless constructor nor TestConstructorArgs, and the skip is a bare continue
// with no diagnostic, so an undeclared layer simply vanishes from generated coverage.
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2,
    TestInputShape = "1, 4", TestConstructorArgs = "4, 3, 2")]
public partial class ObliviousDecisionTreeLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written because the feature axis keeps its role but changes size, which a generated
    /// <c>Same(Features)</c> would misstate. The base constructors declare it directly -
    /// <c>base([inputDim], [outputDim])</c> and <c>base([-1], [outputDim])</c> - and
    /// <c>OnFirstForward</c> preserves it when it resolves the lazy form:
    /// <c>ResolveShapes(new[] { inputDim }, OutputShape)</c> re-uses the ALREADY-declared
    /// <c>OutputShape</c>, so resolving the input width never moves the output width.
    /// </para>
    /// <para>
    /// <c>Fixed(_outputDim)</c> rather than anything derived from the input, because an oblivious tree
    /// routes each sample to one of <c>2^depth</c> leaves and emits that leaf's value vector: the leaf
    /// table is <c>_leafValues</c>, allocated <c>[numLeaves, _outputDim]</c>. The number of input
    /// features decides which leaf is chosen, not how wide the answer is.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _outputDim <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputDim)),
        };
    }

    // Non-readonly: lazy ctor leaves _inputDim = -1 until OnFirstForward
    // resolves it from input.Shape[^1]. Eager ctor sets it at construction.
    private int _inputDim;
    private readonly int _depth;
    private readonly int _outputDim;
    private readonly double _initScale;
    private bool _isInitialized;

    // Each level has one feature selection and one threshold
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _featureSelectionWeights;  // [depth, inputDim] - softmax to select feature
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _thresholds;                // [depth]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _leafValues;                // [numLeaves, outputDim]

    // Gradients
    [Scratch]
    private Tensor<T> _featureSelectionGrad;
    [Scratch]
    private Tensor<T> _thresholdsGrad;
    [Scratch]
    private Tensor<T> _leafValuesGrad;

    // Cached values
    [Scratch]
    private Tensor<T>? _inputCache;
    [Scratch]
    private Tensor<T>? _featureSelectionsCache;
    [Scratch]
    private Tensor<T>? _splitDecisionsCache;
    [Scratch]
    private Tensor<T>? _leafProbabilitiesCache;

    private readonly int _numLeaves;

    /// <summary>
    /// Gets the number of leaf nodes (2^depth).
    /// </summary>
    public int NumLeaves => _numLeaves;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

                                // lazy: no params allocated yet

    /// <summary>
    /// Initializes an oblivious decision tree.
    /// </summary>
    /// <param name="inputDim">Input feature dimension.</param>
    /// <param name="depth">Tree depth (number of split levels).</param>
    /// <param name="outputDim">Output dimension per leaf.</param>
    /// <param name="initScale">Initialization scale.</param>
    /// <remarks>
    /// Not marked <c>[LayerState]</c>; the lazy overload below carries the annotation. The generator
    /// takes one constructor per type, first by source order, and this one takes <c>inputDim</c>,
    /// whose backing field is <c>-1</c> on a lazily-built layer that has not yet forwarded. The lazy
    /// overload re-resolves <c>inputDim</c> from the first input and so restores layers built
    /// either way.
    /// </remarks>
    public ObliviousDecisionTreeLayer(int inputDim, int depth = 6, int outputDim = 1, double initScale = 0.01)
        : base([inputDim], [outputDim])
    {
        if (inputDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputDim), "Input dimension must be positive.");
        if (depth <= 0 || depth > 30)
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be between 1 and 30.");
        if (outputDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputDim), "Output dimension must be positive.");

        _inputDim = inputDim;
        _depth = depth;
        _outputDim = outputDim;
        _initScale = initScale;
        _numLeaves = 1 << depth;  // 2^depth

        // Initialize parameters
        _featureSelectionWeights = new Tensor<T>([depth, inputDim]);
        _thresholds = new Tensor<T>([depth]);
        _leafValues = new Tensor<T>([_numLeaves, outputDim]);

        // Initialize gradients
        _featureSelectionGrad = new Tensor<T>([depth, inputDim]);
        _thresholdsGrad = new Tensor<T>([depth]);
        _leafValuesGrad = new Tensor<T>([_numLeaves, outputDim]);

        InitializeParameters(initScale);
        _isInitialized = true;
    }

    /// <summary>
    /// Lazy constructor: resolves <c>inputDim</c> from <c>input.Shape[^1]</c>
    /// on first <see cref="Forward"/>. <paramref name="depth"/> and
    /// <paramref name="outputDim"/> are architectural and stay required;
    /// only the input feature dimension is shape-dependent.
    /// </summary>
    /// <param name="depth">Tree depth (number of split levels).</param>
    /// <param name="outputDim">Output dimension per leaf.</param>
    /// <param name="initScale">Initialization scale.</param>
    public ObliviousDecisionTreeLayer([LayerState] int depth = 6, [LayerState] int outputDim = 1, [LayerState] double initScale = 0.01)
        : base([-1], [outputDim])
    {
        if (depth <= 0 || depth > 30)
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be between 1 and 30.");
        if (outputDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputDim), "Output dimension must be positive.");

        _inputDim = -1;
        _depth = depth;
        _outputDim = outputDim;
        _initScale = initScale;
        _numLeaves = 1 << depth;

        // Empty placeholders — EnsureInitialized will re-allocate against
        // the resolved inputDim once OnFirstForward fires. Keeping the
        // not-null reference contract intact for code paths that walk
        // these fields unconditionally (GetParameters, ClearGradients).
        _featureSelectionWeights = new Tensor<T>([0, 0]);
        _thresholds = new Tensor<T>([0]);
        _leafValues = new Tensor<T>([0, 0]);
        _featureSelectionGrad = new Tensor<T>([0, 0]);
        _thresholdsGrad = new Tensor<T>([0]);
        _leafValuesGrad = new Tensor<T>([0, 0]);
        _isInitialized = false;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Reads the input feature count from <c>input.Shape[^1]</c> and
    /// resolves the lazy shape so the rest of the forward pass + parameter
    /// access can index against a real <c>InputShape[0]</c>.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        // ODT Forward indexes the tensor as a flat [batch, _inputDim]
        // matrix (see ComputeSplitDecisions); rank-1 input would alias
        // batch onto the feature axis and silently produce garbage. Lock
        // the contract here so lazy first forward fails fast instead of
        // resolving _inputDim from the wrong axis.
        if (rank != 2)
            throw new ArgumentException(
                $"ObliviousDecisionTreeLayer requires rank-2 input [batch, features]; " +
                $"got rank {rank} with shape [{string.Join(", ", input.Shape)}]. If your " +
                $"data is unbatched, add a leading batch axis (e.g. tensor.Reshape([1, " +
                $"features])); higher-rank inputs must be flattened to [batch, features] " +
                $"upstream.", nameof(input));

        int inputDim = input.Shape[rank - 1];
        if (inputDim <= 0)
            throw new ArgumentException(
                $"ObliviousDecisionTreeLayer's input feature dimension must be positive; got {inputDim} from input shape.",
                nameof(input));

        _inputDim = inputDim;
        ResolveShapes(new[] { inputDim }, OutputShape);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Lazy initialization: allocate parameter and gradient tensors against
    /// the resolved <c>_inputDim</c> and run the standard ODT initialization.
    /// Eager-ctor instances bypass this path because <see cref="_isInitialized"/>
    /// is set to true at construction.
    /// </remarks>
    protected override void EnsureInitialized()
    {
        if (_isInitialized) return;
        if (_inputDim <= 0)
            throw new InvalidOperationException(
                "ObliviousDecisionTreeLayer cannot initialize until OnFirstForward has resolved the input dimension from input shape.");

        // Idempotent allocation: correctly-shaped parameters are already present when a deserialize
        // installed them or a copy-on-write clone shared them, and re-initializing would replace
        // trained weights with fresh noise (#1221 Clone_AfterTraining). See ConvolutionalLayer. A
        // freshly-constructed lazy layer holds the constructor's [0, 0] placeholders, which cannot
        // match a positive _depth, so it still initializes.
        bool weightsAlreadyValid =
            _featureSelectionWeights is { Rank: 2 } fs && fs.Shape[0] == _depth && fs.Shape[1] == _inputDim
            && _thresholds is { Rank: 1 } th && th.Shape[0] == _depth
            && _leafValues is { Rank: 2 } lv && lv.Shape[0] == _numLeaves && lv.Shape[1] == _outputDim;

        if (!weightsAlreadyValid)
        {
            _featureSelectionWeights = AllocateLazyWeight([_depth, _inputDim]);
            _thresholds = AllocateLazyWeight([_depth]);
            _leafValues = AllocateLazyWeight([_numLeaves, _outputDim]);
        }

        // Gradient buffers don't go through the streaming pool — they
        // mirror the weight shapes but are owned by the autograd tape,
        // not registered with the pool. Plain new Tensor here. Allocated
        // unconditionally, since a restore installs weights but never
        // gradients.
        _featureSelectionGrad = new Tensor<T>([_depth, _inputDim]);
        _thresholdsGrad = new Tensor<T>([_depth]);
        _leafValuesGrad = new Tensor<T>([_numLeaves, _outputDim]);

        if (!weightsAlreadyValid)
        {
            InitializeParameters(_initScale);
        }

        _isInitialized = true;
    }

    private void InitializeParameters(double scale)
    {
        // Initialize feature selection weights (uniform, will be softmaxed)
        for (int i = 0; i < _featureSelectionWeights.Length; i++)
        {
            _featureSelectionWeights[i] = NumOps.FromDouble(Random.NextGaussian() * scale);
        }

        // Initialize thresholds to small random values
        for (int i = 0; i < _thresholds.Length; i++)
        {
            _thresholds[i] = NumOps.FromDouble(Random.NextGaussian() * scale * 0.1);
        }

        // Initialize leaf values
        for (int i = 0; i < _leafValues.Length; i++)
        {
            _leafValues[i] = NumOps.FromDouble(Random.NextGaussian() * scale);
        }
    }

    /// <summary>
    /// Forward pass through the oblivious decision tree.
    /// </summary>
    /// <param name="input">Input features [batchSize, inputDim].</param>
    /// <returns>Tree output [batchSize, outputDim].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Lazy-ctor instances start with _inputDim = -1; resolve from
        // input.Shape on first call, then materialize parameter tensors.
        // Eager-ctor instances are already initialized (IsShapeResolved=true,
        // _isInitialized=true) so both calls are no-ops.
        if (!IsShapeResolved) OnFirstForward(input);
        EnsureInitialized();

        // Re-validate the shape contract on every call, not just the
        // lazy-first one. OnFirstForward only fires once; subsequent
        // inputs with a different rank or feature count would otherwise
        // index past _featureSelectionWeights[level*_inputDim+f].
        if (input.Shape.Length != 2)
            throw new ArgumentException(
                $"ObliviousDecisionTreeLayer requires rank-2 input [batch, features]; " +
                $"got rank {input.Shape.Length}.", nameof(input));
        if (input.Shape[1] != _inputDim)
            throw new ArgumentException(
                $"ObliviousDecisionTreeLayer's input feature dimension mismatch: layer " +
                $"was resolved with _inputDim={_inputDim}, but input has " +
                $"{input.Shape[1]} features.", nameof(input));

        _inputCache = input;
        int batchSize = input.Shape[0];

        // Compute soft feature selections for each level (entmax/softmax)
        var featureSelections = ComputeFeatureSelections();
        _featureSelectionsCache = featureSelections;

        // Compute split decisions for each level
        var splitDecisions = ComputeSplitDecisions(input, featureSelections, batchSize);
        _splitDecisionsCache = splitDecisions;

        // Compute leaf probabilities
        var leafProbs = ComputeLeafProbabilities(splitDecisions, batchSize);
        _leafProbabilitiesCache = leafProbs;

        // Weighted sum of leaf values
        var output = ComputeOutput(leafProbs, batchSize);

        return output;
    }

    private Tensor<T> ComputeFeatureSelections()
    {
        return Engine.TensorSoftmax(_featureSelectionWeights, axis: 1);
    }

    private Tensor<T> ComputeSplitDecisions(Tensor<T> input, Tensor<T> featureSelections, int batchSize)
    {
        // [batch, input] @ [input, depth] => [batch, depth]. Keep both the
        // feature-selection weights and thresholds on the active tape.
        var weightedFeatures = Engine.TensorMatMul(
            input,
            Engine.TensorTranspose(featureSelections));
        var negativeThresholds = Engine.Reshape(
            Engine.TensorNegate(_thresholds),
            [1, _depth]);
        return Engine.Sigmoid(Engine.TensorAdd(weightedFeatures, negativeThresholds));
    }

    private Tensor<T> ComputeLeafProbabilities(Tensor<T> splitDecisions, int batchSize)
    {
        var paths = new List<Tensor<T>>
        {
            Tensor<T>.CreateDefault([batchSize, 1], NumOps.One)
        };

        for (int level = 0; level < _depth; level++)
        {
            var right = Engine.TensorNarrow(splitDecisions, dim: 1, start: level, length: 1);
            var left = Engine.TensorNegate(Engine.TensorSubtractScalar(right, NumOps.One));
            var next = new List<Tensor<T>>(paths.Count * 2);
            foreach (var parent in paths)
            {
                next.Add(Engine.TensorMultiply(parent, left));
                next.Add(Engine.TensorMultiply(parent, right));
            }
            paths = next;
        }

        return paths.Count == 1
            ? paths[0]
            : Engine.TensorConcatenate(paths.ToArray(), axis: 1);
    }

    private Tensor<T> ComputeOutput(Tensor<T> leafProbs, int batchSize)
    {
        return Engine.TensorMatMul(leafProbs, _leafValues);
    }

    /// <summary>
    /// Gets feature importance based on selection weights.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the layer was constructed with the lazy ctor and has not yet
    /// seen a Forward call — feature importance can't be computed without a
    /// resolved <c>_inputDim</c> and allocated <c>_featureSelectionWeights</c>.
    /// </exception>
    public Vector<T> GetFeatureImportance()
    {
        if (_inputDim <= 0)
        {
            throw new InvalidOperationException(
                "ObliviousDecisionTreeLayer.GetFeatureImportance(): the layer was " +
                "constructed via the lazy ctor (no inputDim arg) and has not yet seen " +
                "a Forward call, so the input dimension and parameter tensors are not " +
                "yet resolved. Run at least one Forward(input) before querying feature " +
                "importance, or construct via the eager ctor with an explicit inputDim.");
        }
        var importance = new Vector<T>(_inputDim);

        if (_featureSelectionsCache != null)
        {
            for (int f = 0; f < _inputDim; f++)
            {
                var sum = NumOps.Zero;
                for (int level = 0; level < _depth; level++)
                {
                    sum = NumOps.Add(sum, _featureSelectionsCache[level * _inputDim + f]);
                }
                importance[f] = NumOps.Divide(sum, NumOps.FromDouble(_depth));
            }
        }
        else
        {
            // Use raw weights if forward hasn't been called
            var selections = ComputeFeatureSelections();
            for (int f = 0; f < _inputDim; f++)
            {
                var sum = NumOps.Zero;
                for (int level = 0; level < _depth; level++)
                {
                    sum = NumOps.Add(sum, selections[level * _inputDim + f]);
                }
                importance[f] = NumOps.Divide(sum, NumOps.FromDouble(_depth));
            }
        }

        return importance;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _featureSelectionWeights = Engine.TensorSubtract(_featureSelectionWeights,
            Engine.TensorMultiplyScalar(_featureSelectionGrad, learningRate));
        _thresholds = Engine.TensorSubtract(_thresholds,
            Engine.TensorMultiplyScalar(_thresholdsGrad, learningRate));
        _leafValues = Engine.TensorSubtract(_leafValues,
            Engine.TensorMultiplyScalar(_leafValuesGrad, learningRate));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_featureSelectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_thresholds, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_leafValues, PersistentTensorRole.Weights);

    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _inputCache = null;
        _featureSelectionsCache = null;
        _splitDecisionsCache = null;
        _leafProbabilitiesCache = null;

        Engine.TensorFill(_featureSelectionGrad, NumOps.Zero);
        Engine.TensorFill(_thresholdsGrad, NumOps.Zero);
        Engine.TensorFill(_leafValuesGrad, NumOps.Zero);
    }

}
