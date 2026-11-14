using System;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using System.Collections.Generic;
using AiDotNet.Models.Parameters;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Engines;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for NODE (Neural Oblivious Decision Ensembles).
/// </summary>
/// <remarks>
/// <para>
/// NODE uses differentiable oblivious decision trees that can be trained end-to-end:
/// - Each tree uses the same feature at each depth level (oblivious)
/// - Split decisions are soft (differentiable) using sigmoid
/// - Trees are combined additively for the final output
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of NODE as making decision trees trainable like neural networks:
///
/// - **Oblivious trees**: At each depth, all nodes split on the same feature.
///   This makes trees faster and more regularized.
/// - **Soft splits**: Instead of "if feature &gt; threshold then go right",
///   we use a smooth function that gradually transitions.
/// - **Ensemble**: Multiple trees vote together for the final answer.
///
/// The result is a model that combines tree interpretability with deep learning power.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class NODEBase<T> : IParameterSource<T>
{
    protected readonly NODEOptions<T> Options;
    protected readonly int NumFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected IEngine Engine => AiDotNetEngine.Current;
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature preprocessing (optional)
    private readonly FullyConnectedLayer<T>? _featurePreprocessing;

    // Per-tree parameters
    private readonly Tensor<T>[] _featureSelectionWeights;  // [depth, features] per tree
    private readonly Tensor<T>[] _splitThresholds;          // [depth] per tree
    private readonly Tensor<T>[] _leafValues;               // [2^depth, output_dim] per tree

    /// <summary>
    /// Gets the tree output dimension.
    /// </summary>
    public int TreeOutputDimension => Options.TreeOutputDimension;

    /// <summary>Built once on first parameter access, then reused.</summary>
    private ParameterComponentRegistry<T>? _parameterRegistry;

    /// <summary>
    /// Extra trainable layers a subclass contributes, folded after the shared backbone.
    /// </summary>
    /// <remarks>
    /// The regression and classification variants share this whole backbone and differ only by a
    /// final projection. Each used to override <see cref="ParameterCount"/> purely to append that
    /// one layer -- and because this base had no GetParameters or SetParameters at all, the head was
    /// COUNTED and never read, never restored and never checkpointed. The count grew; the model that
    /// could be saved did not. Declaring the head here means the subclass states WHAT it adds and
    /// the registry decides where it goes, so count, vector and restore cannot disagree about it.
    /// </remarks>
    protected virtual IEnumerable<IParameterSource<T>> GetExtraTrainableLayers()
        => GeneratedParameterDiscovery.EnumerateDerivedSources<T>(this, typeof(NODEBase<T>));

    /// <summary>
    /// The single ordered traversal of this model's parameter-bearing components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Count, read and restore all derive from THIS, rather than each restating the component list.
    /// Three parallel walks are how a count and a vector come to describe different models: they
    /// agree until someone adds a component to two of them, and nothing reports the disagreement
    /// because the lengths still look plausible. One enumeration makes that unrepresentable.
    /// </para>
    /// <para>
    /// The stable IDs carry a numeric prefix because the registry orders by identity rather than by
    /// the order Register happened to be called -- so the prefix, not the call order, is what pins
    /// serialization order, and it survives a component being added in the middle later.
    /// </para>
    /// </remarks>
    private ParameterComponentRegistry<T> ParameterRegistry
    {
        get
        {
            if (_parameterRegistry is not null) return _parameterRegistry;

            var registry = new ParameterComponentRegistry<T>();
            registry.Register("00000000/preprocessing", _featurePreprocessing);

            registry.Register("00000001/featureSelection",
                new TensorCollectionParameterSource<T>(() => _featureSelectionWeights));
            registry.Register("00000002/thresholds",
                new TensorCollectionParameterSource<T>(() => _splitThresholds));
            registry.Register("00000003/leaves",
                new TensorCollectionParameterSource<T>(() => _leafValues));
            int extraIndex = 0;
            foreach (var extra in GetExtraTrainableLayers())
            {
                if (extra is not null) registry.Register($"00009000/{extraIndex++:D8}", extra);
            }

            _parameterRegistry = registry;
            return registry;
        }
    }

    /// <inheritdoc cref="GetParameters"/>
    public virtual long ParameterCount => ParameterRegistry.ParameterCount;

    /// <summary>
    /// Reads every parameter in traversal order. <see cref="SetParameters"/> reads it back in the
    /// same order, and <see cref="ParameterCount"/> is the length of what this returns.
    /// </summary>
    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    /// <summary>Restores every parameter, in the order <see cref="GetParameters"/> emitted them.</summary>
    public virtual void SetParameters(Vector<T> parameters) => ParameterRegistry.SetParameters(parameters);

    /// <summary>
    /// Initializes a new instance of the NODEBase class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected NODEBase(int numFeatures, NODEOptions<T>? options = null)
    {
        Options = options ?? new NODEOptions<T>();
        NumFeatures = numFeatures;

        if (numFeatures < 1)
        {
            throw new ArgumentException("Number of features must be at least 1", nameof(numFeatures));
        }

        // The choice and bin functions are the closed-form alpha-entmax cases; other alphas would need the
        // general bisection solver, which has no engine op
        if (Options.EntmaxAlpha != 1.0 && Options.EntmaxAlpha != 1.5 && Options.EntmaxAlpha != 2.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), "NODE supports EntmaxAlpha 1.0 (softmax), 1.5 (entmax) or 2.0 (sparsemax).");
        }

        // Optional feature preprocessing
        if (Options.UseFeaturePreprocessing)
        {
            _featurePreprocessing = Options.HiddenVectorActivation is null
                ? new FullyConnectedLayer<T>(
                    numFeatures,
                    numFeatures,
                    Options.HiddenActivation ?? new ReLUActivation<T>())
                : FullyConnectedLayer<T>.WithVectorActivation(
                    numFeatures,
                    numFeatures,
                    Options.HiddenVectorActivation);
        }

        // Initialize tree parameters
        _featureSelectionWeights = new Tensor<T>[Options.NumTrees];
        _splitThresholds = new Tensor<T>[Options.NumTrees];
        _leafValues = new Tensor<T>[Options.NumTrees];

        for (int t = 0; t < Options.NumTrees; t++)
        {
            // Feature selection weights for each depth
            _featureSelectionWeights[t] = new Tensor<T>(new[] { Options.TreeDepth, NumFeatures });
            InitializeWeights(_featureSelectionWeights[t]);

            // Split thresholds for each depth
            _splitThresholds[t] = new Tensor<T>(new[] { Options.TreeDepth });  // zero-initialised

            // Leaf values
            _leafValues[t] = new Tensor<T>(new[] { Options.NumLeaves, Options.TreeOutputDimension });
            InitializeWeights(_leafValues[t]);
        }
    }

    private void InitializeWeights(Tensor<T> tensor)
    {
        var scale = NumOps.FromDouble(Options.InitScale);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(
                NumOps.FromDouble(_random.NextGaussian()),
                scale);
        }
    }

    /// <summary>
    /// Performs the forward pass through the NODE backbone.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Aggregated tree outputs [batch_size, tree_output_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features)
    {
        int batchSize = features.Shape[0];

        // Optional feature preprocessing
        var processed = features;
        if (_featurePreprocessing != null)
        {
            processed = _featurePreprocessing.Forward(features);
        }

        // Aggregate outputs from all trees
        var output = new Tensor<T>(new[] { batchSize, Options.TreeOutputDimension });

        if (Options.NumTrees <= 0)
        {
            throw new InvalidOperationException("NumTrees must be greater than 0.");
        }

        for (int t = 0; t < Options.NumTrees; t++)
        {
            var treeOutput = ForwardTree(t, processed, batchSize);

            // Aggregate tree outputs (additive ensemble)
            output = Engine.TensorAdd(output, treeOutput);
        }

        // Average over trees
        output = Engine.TensorMultiplyScalar(output, NumOps.FromDouble(1.0 / Options.NumTrees));

        return output;
    }

    private Tensor<T> ForwardTree(int treeIdx, Tensor<T> features, int batchSize)
    {
        int depth = Options.TreeDepth;
        int numLeaves = Options.NumLeaves;

        // Soft feature selection: selected[b, d] = sum_f entmax(W[d, :])[f] * x[b, f]
        var x = TabularMetrics.Flat(features, batchSize * NumFeatures).Reshape(batchSize, NumFeatures);
        var selection = ComputeChoiceWeights(_featureSelectionWeights[treeIdx]);
        var selectedFeature = Engine.TensorMatMul(x, Engine.TensorTranspose(selection));

        // Soft split decision (probability of going right) at every depth: the two-class entmax of the
        // scaled threshold logit (Popov et al. 2020, eq. 2), entmoid for the default alpha = 1.5
        var thresholds = _splitThresholds[treeIdx].Reshape(1, depth);
        var temp = NumOps.FromDouble(Options.Temperature);
        var scaledDiff = Engine.TensorDivideScalar(Engine.TensorSubtract(selectedFeature, thresholds), temp);
        var splitProbs = ComputeBinProbabilities(scaledDiff);
        var stayLeftProbs = Engine.ScalarMinusTensor(NumOps.One, splitProbs);

        // goesRight[d, leaf] = (leaf >> (depth - 1 - d)) & 1, computed as floor(leaf / 2^k) mod 2
        var leafIds = Engine.TensorArange<T>(0, numLeaves).Reshape(1, numLeaves);
        var pathScale = Engine.TensorExp2(Engine.TensorArange<T>(depth - 1, -1, -1)).Reshape(depth, 1);
        var quotient = Engine.TensorFloor(Engine.TensorDivide(leafIds, pathScale));
        var goesRight = Engine.TensorSubtract(
            quotient, Engine.TensorMultiplyScalar(Engine.TensorFloor(Engine.TensorDivideScalar(quotient, NumOps.FromDouble(2.0))), NumOps.FromDouble(2.0)));
        var goesLeft = Engine.ScalarMinusTensor(NumOps.One, goesRight);

        // Leaf weight = product over depths of p_d (right) or 1 - p_d (left), multiplied in depth order
        var leafWeights = Engine.TensorFullLike(new Tensor<T>(new[] { batchSize, numLeaves }), NumOps.One);
        for (int d = 0; d < depth; d++)
        {
            var right = Engine.TensorNarrow(goesRight, 0, d, 1);
            var left = Engine.TensorNarrow(goesLeft, 0, d, 1);
            var factor = Engine.TensorAdd(
                Engine.TensorMultiply(right, Engine.TensorNarrow(splitProbs, 1, d, 1)),
                Engine.TensorMultiply(left, Engine.TensorNarrow(stayLeftProbs, 1, d, 1)));
            leafWeights = Engine.TensorMultiply(leafWeights, factor);
        }

        // Compute tree output as weighted sum of leaf values
        return Engine.TensorMatMul(
            leafWeights, _leafValues[treeIdx].Reshape(numLeaves, Options.TreeOutputDimension));
    }

    /// <summary>
    /// The choice function over each depth's feature-selection logits: [tree_depth, num_features]. It is the
    /// alpha-entmax of <see cref="NODEOptions{T}.EntmaxAlpha"/>: softmax at 1, entmax-1.5 at the paper's
    /// default 1.5 and sparsemax at 2.
    /// </summary>
    private Tensor<T> ComputeChoiceWeights(Tensor<T> weights)
    {
        var logits = weights.Reshape(Options.TreeDepth, NumFeatures);
        if (Options.EntmaxAlpha == 1.0) return Engine.Softmax(logits, -1);
        if (Options.EntmaxAlpha == 2.0) return Engine.Sparsemax(logits, -1);
        return Entmax15(logits);
    }

    /// <summary>
    /// The bin function of the split logits: the two-class alpha-entmax, i.e. sigmoid at alpha 1, entmoid-1.5
    /// at 1.5 and sparsemoid <c>clamp((x + 1) / 2, 0, 1)</c> at 2.
    /// </summary>
    private Tensor<T> ComputeBinProbabilities(Tensor<T> logits)
    {
        if (Options.EntmaxAlpha == 1.0) return Engine.Sigmoid(logits);
        if (Options.EntmaxAlpha == 2.0)
        {
            var half = NumOps.FromDouble(0.5);
            return Engine.TensorClamp(
                TensorTapeOps.TapeAddScalar(Engine, TensorTapeOps.TapeMultiplyScalar(Engine, logits, half), half),
                NumOps.Zero,
                NumOps.One);
        }

        return Entmoid15(logits);
    }

    /// <summary>
    /// 1.5-entmax along the last axis of a [rows, n] tensor (Peters, Niculae and Martins 2019):
    /// <c>y = ReLU(z / 2 - tau)^2</c> with tau the unique threshold making each row sum to one.
    /// </summary>
    /// <remarks>
    /// The threshold is found exactly by sorting: with X = z / 2 sorted descending, the candidate for a support
    /// of size r is <c>tau_r = mean_r - sqrt((1 - r (meansq_r - mean_r^2)) / r)</c>, and the support is
    /// <c>{ r : tau_r &lt;= X_(r) }</c>. The returned value recomputes tau from that support with differentiable
    /// engine ops, so on the tape the gradient is the exact entmax Jacobian.
    /// </remarks>
    internal static Tensor<T> Entmax15(Tensor<T> logits)
    {
        var engine = AiDotNetEngine.Current;
        int rows = logits.Shape[0];
        int n = logits.Shape[1];
        var one = NumOps.One;

        // X = z / 2, shifted by the row max (entmax is shift invariant)
        var halved = TensorTapeOps.TapeMultiplyScalar(engine, logits.Reshape(rows, n), NumOps.FromDouble(0.5));
        var x = engine.TensorSubtract(halved, engine.ReduceMax(halved, new[] { 1 }, true));

        // Threshold candidates over the descending order, and the support size
        var sorted = engine.TensorSort(x, -1, descending: true).Values;
        var rank = engine.TensorArange<T>(1, n + 1).Reshape(1, n);
        var mean = engine.TensorDivide(engine.TensorCumSum(sorted, 1), rank);
        var meanSquare = engine.TensorDivide(engine.TensorCumSum(engine.TensorMultiply(sorted, sorted), 1), rank);
        var spread = engine.TensorMultiply(rank, engine.TensorSubtract(meanSquare, engine.TensorMultiply(mean, mean)));
        var delta = engine.TensorDivide(engine.ScalarMinusTensor(one, spread), rank);
        var candidates = engine.TensorSubtract(mean, engine.TensorSqrt(engine.ReLU(delta)));
        // Counted in int: a 0/1 sum in T stops counting at 2048 in Half and would pick the wrong threshold
        var outside = engine.TensorLessThan(engine.TensorSubtract(sorted, candidates), NumOps.Zero);
        var supportCount = engine.ScalarMinusTensor(n, engine.ReduceSum(outside.Cast<int>(), new[] { 1 }, true));
        var supportSize = supportCount.Cast<T>();

        // Membership in original order: x >= the smallest supported value
        var lastIndex = engine.TensorAddScalar(supportCount, -1);
        var smallestSupported = engine.TensorTakeAlongDim(sorted, lastIndex, 1);
        var support = engine.ScalarMinusTensor(
            one, engine.TensorLessThan(engine.TensorSubtract(x, smallestSupported), NumOps.Zero));

        // tau over the support: mean_S - sqrt((1 - ss_S) / k), ss_S = sum_S x^2 - k mean_S^2
        var sum = engine.ReduceSum(engine.TensorMultiply(support, x), new[] { 1 }, true);
        var sumSquares = engine.ReduceSum(engine.TensorMultiply(support, engine.TensorMultiply(x, x)), new[] { 1 }, true);
        var supportMean = engine.TensorDivide(sum, supportSize);
        var supportSpread = engine.TensorSubtract(sumSquares, engine.TensorMultiply(sum, supportMean));
        var tau = engine.TensorSubtract(
            supportMean,
            engine.TensorSqrt(engine.TensorDivide(engine.ScalarMinusTensor(one, supportSpread), supportSize)));

        var shifted = engine.ReLU(engine.TensorSubtract(x, tau));
        return engine.TensorMultiply(shifted, shifted).Reshape(logits._shape);
    }

    /// <summary>
    /// entmoid-1.5, the two-class 1.5-entmax <c>[entmax15([x, 0])]_0</c>, element-wise (Popov et al. 2020):
    /// with a = |x|, <c>t = (a + sqrt(8 - a^2)) / 2</c> (2 once t &lt;= a), <c>y- = ReLU(t - a)^2 / 4</c> and
    /// <c>y = 1 - y-</c> for x &gt;= 0, <c>y-</c> otherwise.
    /// </summary>
    internal static Tensor<T> Entmoid15(Tensor<T> logits)
    {
        var engine = AiDotNetEngine.Current;
        var two = NumOps.FromDouble(2.0);
        var a = engine.TensorAbs(logits);

        // a is clamped to 2 inside the root: the result is unchanged (t is replaced by 2 beyond a = 2) and the
        // root's argument stays >= 4, so its gradient is finite everywhere
        var clamped = engine.TensorClamp(a, NumOps.Zero, two);
        var root = engine.TensorSqrt(engine.ScalarMinusTensor(NumOps.FromDouble(8.0), engine.TensorMultiply(clamped, clamped)));
        var t = TensorTapeOps.TapeMultiplyScalar(engine, engine.TensorAdd(a, root), NumOps.FromDouble(0.5));
        t = engine.TensorWhere(engine.TensorGreaterThan(t, a), t, engine.TensorFullLike(t, two));

        var gap = engine.ReLU(engine.TensorSubtract(t, a));
        var negative = TensorTapeOps.TapeMultiplyScalar(engine, engine.TensorMultiply(gap, gap), NumOps.FromDouble(0.25));
        return engine.TensorWhere(
            engine.TensorLessThan(logits, NumOps.Zero), negative, engine.ScalarMinusTensor(NumOps.One, negative));
    }

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    /// <returns>Feature importance scores [num_features].</returns>
    public Vector<T> GetFeatureImportance()
    {
        var importance = new Tensor<T>(new[] { NumFeatures });

        for (int t = 0; t < Options.NumTrees; t++)
        {
            var softmaxWeights = ComputeChoiceWeights(_featureSelectionWeights[t]);
            importance = Engine.TensorAdd(importance, Engine.ReduceSum(softmaxWeights, new[] { 0 }, false));
        }

        // Normalize
        var scale = NumOps.FromDouble(1.0 / (Options.NumTrees * Options.TreeDepth));
        return Engine.TensorMultiplyScalar(importance, scale).ToVector();
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        _featurePreprocessing?.UpdateParameters(learningRate);

        // Update tree parameters (simplified gradient descent)
        // In practice, this would use proper gradient computation
    }

    /// <summary>
    /// The live tensors a tape training step updates: the feature-selection logits, split thresholds and leaf responses of every tree, the optional preprocessing layer and the task head.
    /// </summary>
    protected IEnumerable<Tensor<T>> GetTrainableTensors()
    {
        var tensors = new List<Tensor<T>>();
        tensors.AddRange(_featureSelectionWeights);
        tensors.AddRange(_splitThresholds);
        tensors.AddRange(_leafValues);
        var layers = new List<ILayer<T>>();
        if (_featurePreprocessing is not null) layers.Add(_featurePreprocessing);
        layers.AddRange(GetExtraTrainableLayers().OfType<ILayer<T>>());
        tensors.AddRange(TabularTraining.LayerTensors(layers));
        return tensors;
    }

    /// <summary>
    /// One tape training step: gradients of <paramref name="objective"/> over every trainable tensor, then SGD.
    /// </summary>
    /// <returns>The loss before the update.</returns>
    protected T TrainWithTape(Func<Tensor<T>> objective, T learningRate)
    {
        try
        {
            return TabularTraining.SgdStep(GetTrainableTensors, objective, learningRate);
        }
        finally
        {
            ResetState();
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _featurePreprocessing?.ResetState();
    }
}
