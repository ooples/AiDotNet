using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Interpretability.Helpers;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Layer-wise Relevance Propagation (LRP) explainer for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LRP explains neural network predictions by propagating
/// "relevance" scores backward from the output to the input.
///
/// Key idea:
/// - Start with the prediction score as the total "relevance"
/// - At each layer, distribute relevance to neurons in the previous layer
/// - Continue until you reach the input features
/// - Each input feature gets a relevance score
///
/// Conservation principle:
/// - The total relevance is conserved at each layer
/// - Relevance in = Relevance out (like energy conservation)
/// - This means attributions sum to the prediction value
///
/// LRP rules (how to distribute relevance):
/// - <b>LRP-0 (Basic)</b>: Distribute proportional to contribution
/// - <b>LRP-ε (Epsilon)</b>: Adds stability with small epsilon
/// - <b>LRP-γ (Gamma)</b>: Emphasizes positive contributions
/// - <b>LRP-αβ</b>: Separately handles positive and negative contributions
///
/// When to use LRP:
/// - You want to understand which inputs were "responsible" for the output
/// - You need attributions that sum exactly to the prediction
/// - You want a principled way to handle different layer types
///
/// Example: For image classification:
/// - LRP shows which pixels contributed positively or negatively
/// - Red = positive relevance (supported the prediction)
/// - Blue = negative relevance (contradicted the prediction)
/// </para>
/// </remarks>
public class LayerwiseRelevancePropagationExplainer<T> : ILocalExplainer<T, LRPExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>[]>? _getLayerActivations;
    private readonly Func<int, Matrix<T>>? _getLayerWeights;
    private readonly int _numFeatures;
    private readonly int[]? _layerSizes;
    private readonly string[]? _featureNames;
    private readonly LRPRule _rule;
    private readonly double _epsilon;
    private readonly double _gamma;

    /// <inheritdoc/>
    public string MethodName => "LRP";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new LRP explainer.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions.</param>
    /// <param name="getLayerActivations">Function that returns activations for all layers.
    /// Returns array where [0] is input, [L] is output.</param>
    /// <param name="getLayerWeights">Function that returns weight matrix for a layer index.
    /// Returns matrix of shape [input_neurons, output_neurons].</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="layerSizes">Sizes of each layer including input and output.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="rule">LRP rule to use (default: Epsilon).</param>
    /// <param name="epsilon">Epsilon for LRP-ε rule (default: 1e-4).</param>
    /// <param name="gamma">Gamma for LRP-γ rule (default: 0.25).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>rule</b>: Epsilon rule is most commonly used. Use Gamma rule to suppress noise.
    /// - <b>epsilon</b>: Small value that prevents division by zero. Don't set too high.
    /// - <b>gamma</b>: Higher values emphasize positive contributions more.
    /// </para>
    /// </remarks>
    public LayerwiseRelevancePropagationExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>[]>? getLayerActivations = null,
        Func<int, Matrix<T>>? getLayerWeights = null,
        int numFeatures = 0,
        int[]? layerSizes = null,
        string[]? featureNames = null,
        LRPRule rule = LRPRule.Epsilon,
        double epsilon = 1e-4,
        double gamma = 0.25)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        _getLayerActivations = getLayerActivations;
        _getLayerWeights = getLayerWeights;
        _numFeatures = numFeatures;
        _layerSizes = layerSizes;
        _featureNames = featureNames;
        _rule = rule;
        _epsilon = epsilon;
        _gamma = gamma;
    }

    /// <summary>
    /// Computes LRP relevance scores for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>LRP explanation with relevance scores.</returns>
    public LRPExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, outputIndex: -1);
    }

    /// <summary>
    /// Computes LRP relevance scores for a specific output.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="outputIndex">Index of the output to explain (-1 for highest scoring class).</param>
    /// <returns>LRP explanation with relevance scores.</returns>
    public LRPExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        int numFeatures = instance.Length;

        // Get prediction
        var prediction = _predictFunction(instance);

        // Determine target output
        if (outputIndex < 0)
        {
            outputIndex = prediction.Length == 0 ? 0 : ExplainerTensorOps<T>.ArgMax(prediction);
        }

        T outputValue = outputIndex < prediction.Length
            ? prediction[outputIndex]
            : NumOps.Zero;

        Vector<T> relevanceScores;

        if (_getLayerActivations != null && _getLayerWeights != null && _layerSizes != null)
        {
            // Full LRP with layer access
            relevanceScores = ComputeFullLRP(instance, outputIndex);
        }
        else
        {
            // Approximate LRP using gradient-weighted input
            relevanceScores = ComputeApproximateLRP(instance, outputIndex);
        }

        // Compute positive and negative relevance
        var engine = AiDotNetEngine.Current;
        var scores = relevanceScores.GetSubVector(0, numFeatures);
        var positivePart = engine.ReLU(scores);
        T posRelevance = engine.Sum(positivePart);
        T negRelevance = engine.Sum(engine.Subtract(scores, positivePart));

        return new LRPExplanation<T>
        {
            RelevanceScores = relevanceScores,
            Input = instance,
            Prediction = prediction,
            OutputIndex = outputIndex,
            OutputValue = outputValue,
            TotalPositiveRelevance = posRelevance,
            TotalNegativeRelevance = negRelevance,
            FeatureNames = _featureNames ?? Enumerable.Range(0, numFeatures).Select(i => $"Feature {i}").ToArray(),
            Rule = _rule
        };
    }

    /// <inheritdoc/>
    public LRPExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new LRPExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Computes full LRP using layer activations and weights.
    /// </summary>
    private Vector<T> ComputeFullLRP(Vector<T> input, int outputIndex)
    {
        // Get activations
        var activations = (_getLayerActivations ?? throw new InvalidOperationException("_getLayerActivations is null where it is required."))(input);
        int numLayers = activations.Length;

        // Initialize relevance at output layer
        var relevance = new Vector<T>(activations[numLayers - 1].Length);
        relevance[outputIndex] = activations[numLayers - 1][outputIndex];

        // Propagate backward through layers
        for (int layer = numLayers - 2; layer >= 0; layer--)
        {
            var weights = (_getLayerWeights ?? throw new InvalidOperationException("_getLayerWeights is null where it is required."))(layer);
            var lowerActivations = activations[layer];

            relevance = _rule switch
            {
                LRPRule.Basic => PropagateBasicRule(lowerActivations, weights, relevance),
                LRPRule.Epsilon => PropagateEpsilonRule(lowerActivations, weights, relevance),
                LRPRule.Gamma => PropagateGammaRule(lowerActivations, weights, relevance),
                LRPRule.AlphaBeta => PropagateAlphaBetaRule(lowerActivations, weights, relevance),
                _ => new Vector<T>(lowerActivations.Length)
            };
        }

        return relevance;
    }

    /// <summary>
    /// LRP-0 Basic rule.
    /// </summary>
    private Vector<T> PropagateBasicRule(Vector<T> lowerAct, Matrix<T> weights, Vector<T> upperRel)
    {
        var engine = AiDotNetEngine.Current;
        var w = weights.Rows == lowerAct.Length && weights.Columns == upperRel.Length
            ? weights
            : weights.GetSubMatrix(0, 0, lowerAct.Length, upperRel.Length);

        // z_k = sum_i a_i w_ik; R_j = sum_k (a_j w_jk / z_k) R_k, skipping neurons with |z_k| <= 1e-10
        var z = engine.MatrixVectorMultiply(engine.MatrixTranspose(w), lowerAct);
        return Redistribute(lowerAct, w, SafeRatio(upperRel, z));
    }

    /// <summary>
    /// LRP-ε Epsilon rule for numerical stability.
    /// </summary>
    private Vector<T> PropagateEpsilonRule(Vector<T> lowerAct, Matrix<T> weights, Vector<T> upperRel)
    {
        var engine = AiDotNetEngine.Current;
        var w = weights.Rows == lowerAct.Length && weights.Columns == upperRel.Length
            ? weights
            : weights.GetSubMatrix(0, 0, lowerAct.Length, upperRel.Length);

        // z_k = sum_i a_i w_ik, then add epsilon with the sign of z_k (zero counts as positive)
        var z = engine.MatrixVectorMultiply(engine.MatrixTranspose(w), lowerAct).AsTensor();
        var nonNegative = engine.TensorGreaterEqual(z, engine.TensorFullLike(z, NumOps.Zero));
        T epsilon = NumOps.FromDouble(_epsilon);
        var signedEpsilon = engine.TensorWhere(nonNegative, engine.TensorFullLike(z, epsilon),
            engine.TensorFullLike(z, NumOps.Negate(epsilon)));
        var stabilized = engine.TensorAdd(z, signedEpsilon);
        var share = engine.TensorDivide(upperRel.AsTensor(), stabilized).ToVector();
        return Redistribute(lowerAct, w, share);
    }

    /// <summary>
    /// LRP-γ Gamma rule emphasizing positive contributions.
    /// </summary>
    private Vector<T> PropagateGammaRule(Vector<T> lowerAct, Matrix<T> weights, Vector<T> upperRel)
    {
        var engine = AiDotNetEngine.Current;
        var w = weights.Rows == lowerAct.Length && weights.Columns == upperRel.Length
            ? weights
            : weights.GetSubMatrix(0, 0, lowerAct.Length, upperRel.Length);

        // Modify weights: w + gamma * max(0, w), increasing the positive weights
        var positiveWeights = engine.ReLU(Tensor<T>.FromMatrix(w)).ToMatrix();
        var modified = engine.MatrixAdd(w, engine.MatrixMultiplyScalar(positiveWeights, NumOps.FromDouble(_gamma)));
        var z = engine.MatrixVectorMultiply(engine.MatrixTranspose(modified), lowerAct);
        return Redistribute(lowerAct, modified, SafeRatio(upperRel, z));
    }

    /// <summary>
    /// LRP-αβ rule separating positive and negative contributions.
    /// </summary>
    private Vector<T> PropagateAlphaBetaRule(Vector<T> lowerAct, Matrix<T> weights, Vector<T> upperRel, double alpha = 2, double beta = 1)
    {
        var engine = AiDotNetEngine.Current;
        var w = weights.Rows == lowerAct.Length && weights.Columns == upperRel.Length
            ? weights
            : weights.GetSubMatrix(0, 0, lowerAct.Length, upperRel.Length);

        // Positive and negative weight parts and their per-neuron totals z_k+ / z_k-
        var positiveWeights = engine.ReLU(Tensor<T>.FromMatrix(w)).ToMatrix();
        var negativeWeights = engine.MatrixSubtract(w, positiveWeights);
        var zPlus = engine.MatrixVectorMultiply(engine.MatrixTranspose(positiveWeights), lowerAct);
        var zMinus = engine.MatrixVectorMultiply(engine.MatrixTranspose(negativeWeights), lowerAct);

        // R_j = a_j * sum_k (alpha w+_jk / z_k+ - beta w-_jk / z_k-) R_k, a ratio counting 0 where |z| <= 1e-10
        var positiveShare = engine.Multiply(SafeRatio(upperRel, zPlus), NumOps.FromDouble(alpha));
        var negativeShare = engine.Multiply(SafeRatio(upperRel, zMinus), NumOps.FromDouble(beta));
        var spread = engine.Subtract(
            engine.MatrixVectorMultiply(positiveWeights, positiveShare),
            engine.MatrixVectorMultiply(negativeWeights, negativeShare));
        return engine.Multiply(lowerAct, spread);
    }

    /// <summary>
    /// relevance / z, with 0 where |z| &lt;= 1e-10.
    /// </summary>
    private static Vector<T> SafeRatio(Vector<T> relevance, Vector<T> z)
    {
        var engine = AiDotNetEngine.Current;
        var zTensor = z.AsTensor();
        var usable = engine.TensorGreaterThan(engine.TensorAbs(zTensor), NumOps.FromDouble(1e-10));
        var safeZ = engine.TensorWhere(usable, zTensor, engine.TensorFullLike(zTensor, NumOps.One));
        var ratio = engine.TensorDivide(relevance.AsTensor(), safeZ);
        return engine.TensorWhere(usable, ratio, engine.TensorFullLike(zTensor, NumOps.Zero)).ToVector();
    }

    /// <summary>
    /// R_j = a_j * sum_k w_jk * share_k: the lower-layer relevance from per-neuron shares.
    /// </summary>
    private static Vector<T> Redistribute(Vector<T> lowerAct, Matrix<T> weights, Vector<T> share)
    {
        var engine = AiDotNetEngine.Current;
        return engine.Multiply(lowerAct, engine.MatrixVectorMultiply(weights, share));
    }

    /// <summary>
    /// Computes approximate LRP using gradient × input (a simple approximation).
    /// </summary>
    private Vector<T> ComputeApproximateLRP(Vector<T> input, int outputIndex)
    {
        // Compute gradients
        var gradients = ComputeNumericalGradient(input, outputIndex);

        // Relevance ≈ gradient × input (a simple approximation to LRP)
        return AiDotNetEngine.Current.Multiply(gradients, input);
    }

    /// <summary>
    /// Computes numerical gradient.
    /// </summary>
    private Vector<T> ComputeNumericalGradient(Vector<T> input, int outputIndex)
    {
        return ExplainerTensorOps<T>.CentralDifferenceGradient(
            _predictFunction, input, outputIndex, 1e-4, input.Length, outOfRangeIsZero: true);
    }
}

/// <summary>
/// LRP propagation rules.
/// </summary>
public enum LRPRule
{
    /// <summary>
    /// Basic LRP-0 rule.
    /// </summary>
    Basic,

    /// <summary>
    /// LRP-ε rule with stabilizing epsilon.
    /// </summary>
    Epsilon,

    /// <summary>
    /// LRP-γ rule emphasizing positive contributions.
    /// </summary>
    Gamma,

    /// <summary>
    /// LRP-αβ rule separating positive and negative.
    /// </summary>
    AlphaBeta
}

/// <summary>
/// Represents the result of an LRP analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LRPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the relevance scores for each input feature.
    /// </summary>
    public Vector<T> RelevanceScores { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Input { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the model prediction.
    /// </summary>
    public Vector<T> Prediction { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the output index that was explained.
    /// </summary>
    public int OutputIndex { get; set; }

    /// <summary>
    /// Gets or sets the output value being explained.
    /// </summary>
    public required T OutputValue { get; set; }

    /// <summary>
    /// Gets or sets the sum of positive relevance scores.
    /// </summary>
    public required T TotalPositiveRelevance { get; set; }

    /// <summary>
    /// Gets or sets the sum of negative relevance scores.
    /// </summary>
    public required T TotalNegativeRelevance { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the LRP rule used.
    /// </summary>
    public LRPRule Rule { get; set; }

    /// <summary>
    /// Gets relevance scores sorted by absolute value (most relevant first).
    /// </summary>
    public List<(string name, T relevance)> GetSortedRelevance()
    {
        var result = new List<(string, T)>();
        for (int i = 0; i < RelevanceScores.Length; i++)
        {
            result.Add((FeatureNames[i], RelevanceScores[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item2))).ToList();
    }

    /// <summary>
    /// Gets features with positive relevance (supporting the prediction).
    /// </summary>
    public List<(string name, T relevance)> GetPositiveRelevance()
    {
        return GetSortedRelevance()
            .Where(x => NumOps.GreaterThan(x.relevance, NumOps.Zero))
            .ToList();
    }

    /// <summary>
    /// Gets features with negative relevance (contradicting the prediction).
    /// </summary>
    public List<(string name, T relevance)> GetNegativeRelevance()
    {
        return GetSortedRelevance()
            .Where(x => NumOps.LessThan(x.relevance, NumOps.Zero))
            .OrderBy(x => NumOps.ToDouble(x.relevance))
            .ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        T totalRel = NumOps.Add(TotalPositiveRelevance, TotalNegativeRelevance);

        var lines = new List<string>
        {
            $"Layer-wise Relevance Propagation (Rule: {Rule}):",
            $"  Output value: {OutputValue:F4}",
            $"  Total relevance: {totalRel:F4}",
            $"  Positive relevance: {TotalPositiveRelevance:F4}",
            $"  Negative relevance: {TotalNegativeRelevance:F4}",
            "",
            "Top Positive (supporting prediction):"
        };

        var positive = GetPositiveRelevance().Take(5);
        foreach (var (name, rel) in positive)
        {
            lines.Add($"  {name}: +{rel:F4}");
        }

        lines.Add("");
        lines.Add("Top Negative (contradicting prediction):");

        var negative = GetNegativeRelevance().Take(5);
        foreach (var (name, rel) in negative)
        {
            lines.Add($"  {name}: {rel:F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
