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
public partial class ObliviousDecisionTreeLayer<T> : LayerBase<T>
{
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
    private Tensor<T> _featureSelectionGrad;
    private Tensor<T> _thresholdsGrad;
    private Tensor<T> _leafValuesGrad;

    // Cached values
    private Tensor<T>? _inputCache;
    private Tensor<T>? _featureSelectionsCache;
    private Tensor<T>? _splitDecisionsCache;
    private Tensor<T>? _leafProbabilitiesCache;

    private readonly int _numLeaves;

    /// <summary>
    /// Gets the number of leaf nodes (2^depth).
    /// </summary>
    public int NumLeaves => _numLeaves;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override int ParameterCount =>
        _inputDim > 0
            ? _depth * _inputDim +      // feature selection weights
              _depth +                   // thresholds
              _numLeaves * _outputDim    // leaf values
            : 0;                         // lazy: no params allocated yet

    /// <summary>
    /// Initializes an oblivious decision tree.
    /// </summary>
    /// <param name="inputDim">Input feature dimension.</param>
    /// <param name="depth">Tree depth (number of split levels).</param>
    /// <param name="outputDim">Output dimension per leaf.</param>
    /// <param name="initScale">Initialization scale.</param>
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
    public ObliviousDecisionTreeLayer(int depth = 6, int outputDim = 1, double initScale = 0.01)
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

        _featureSelectionWeights = new Tensor<T>([_depth, _inputDim]);
        _thresholds = new Tensor<T>([_depth]);
        _leafValues = new Tensor<T>([_numLeaves, _outputDim]);
        _featureSelectionGrad = new Tensor<T>([_depth, _inputDim]);
        _thresholdsGrad = new Tensor<T>([_depth]);
        _leafValuesGrad = new Tensor<T>([_numLeaves, _outputDim]);
        InitializeParameters(_initScale);
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
    public override Tensor<T> Forward(Tensor<T> input)
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
        var selections = new Tensor<T>([_depth, _inputDim]);

        for (int level = 0; level < _depth; level++)
        {
            // Softmax over features for this level
            var maxVal = _featureSelectionWeights[level * _inputDim];
            for (int f = 1; f < _inputDim; f++)
            {
                var val = _featureSelectionWeights[level * _inputDim + f];
                if (NumOps.Compare(val, maxVal) > 0)
                    maxVal = val;
            }

            var sumExp = NumOps.Zero;
            for (int f = 0; f < _inputDim; f++)
            {
                var exp = NumOps.Exp(NumOps.Subtract(
                    _featureSelectionWeights[level * _inputDim + f], maxVal));
                selections[level * _inputDim + f] = exp;
                sumExp = NumOps.Add(sumExp, exp);
            }

            for (int f = 0; f < _inputDim; f++)
            {
                selections[level * _inputDim + f] = NumOps.Divide(
                    selections[level * _inputDim + f], sumExp);
            }
        }

        return selections;
    }

    private Tensor<T> ComputeSplitDecisions(Tensor<T> input, Tensor<T> featureSelections, int batchSize)
    {
        // [batchSize, depth] - probability of going right at each level
        var decisions = TensorAllocator.Rent<T>([batchSize, _depth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int level = 0; level < _depth; level++)
            {
                // Compute weighted feature value
                var weightedFeature = NumOps.Zero;
                for (int f = 0; f < _inputDim; f++)
                {
                    weightedFeature = NumOps.Add(weightedFeature,
                        NumOps.Multiply(
                            featureSelections[level * _inputDim + f],
                            input[b * _inputDim + f]));
                }

                // Soft split decision using sigmoid
                var logit = NumOps.Subtract(weightedFeature, _thresholds[level]);
                var rightProb = Sigmoid(logit);
                decisions[b * _depth + level] = rightProb;
            }
        }

        return decisions;
    }

    private Tensor<T> ComputeLeafProbabilities(Tensor<T> splitDecisions, int batchSize)
    {
        var leafProbs = TensorAllocator.Rent<T>([batchSize, _numLeaves]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int leaf = 0; leaf < _numLeaves; leaf++)
            {
                // Compute probability of reaching this leaf
                var prob = NumOps.One;
                for (int level = 0; level < _depth; level++)
                {
                    var rightProb = splitDecisions[b * _depth + level];
                    var leftProb = NumOps.Subtract(NumOps.One, rightProb);

                    // Check if this leaf goes right or left at this level
                    bool goRight = ((leaf >> (_depth - 1 - level)) & 1) == 1;
                    prob = NumOps.Multiply(prob, goRight ? rightProb : leftProb);
                }
                leafProbs[b * _numLeaves + leaf] = prob;
            }
        }

        return leafProbs;
    }

    private Tensor<T> ComputeOutput(Tensor<T> leafProbs, int batchSize)
    {
        var output = TensorAllocator.Rent<T>([batchSize, _outputDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < _outputDim; o++)
            {
                var sum = NumOps.Zero;
                for (int leaf = 0; leaf < _numLeaves; leaf++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(
                        leafProbs[b * _numLeaves + leaf],
                        _leafValues[leaf * _outputDim + o]));
                }
                output[b * _outputDim + o] = sum;
            }
        }

        return output;
    }

    private T Sigmoid(T x)
    {
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
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

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int total = _featureSelectionWeights.Length + _thresholds.Length + _leafValues.Length;
        var result = new Vector<T>(total);
        int offset = 0;
        for (int i = 0; i < _featureSelectionWeights.Length; i++)
            result[offset++] = _featureSelectionWeights[i];
        for (int i = 0; i < _thresholds.Length; i++)
            result[offset++] = _thresholds[i];
        for (int i = 0; i < _leafValues.Length; i++)
            result[offset++] = _leafValues[i];
        return result;
    }
}
