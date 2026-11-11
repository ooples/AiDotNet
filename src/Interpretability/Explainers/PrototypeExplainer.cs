using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Interpretability.Helpers;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Prototype-based explainer that explains predictions using similar examples.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sometimes the best explanation is an example!
/// This explainer answers "Why this prediction?" with "Because it's similar to these examples."
///
/// How it works:
/// 1. Maintain a set of "prototypes" (representative examples from training data)
/// 2. For a new prediction, find the most similar prototypes
/// 3. Explain the prediction based on these similar examples
///
/// Types of prototype explanations:
/// - <b>Nearest neighbors</b>: Show the K closest training examples
/// - <b>Same-class prototypes</b>: Show similar examples with the same prediction
/// - <b>Contrast prototypes</b>: Show similar examples with different predictions
///
/// Why prototypes are useful:
/// - Intuitive: "This loan was approved because it's similar to John's approved loan"
/// - Concrete: Shows actual examples, not abstract feature weights
/// - Trustworthy: Users can verify the similarity themselves
///
/// Example use cases:
/// - Medical diagnosis: "This case is similar to these past cases that were diagnosed as..."
/// - Credit decisions: "Your application is similar to these approved/denied applications"
/// - Image classification: "This image looks like these training images of cats"
/// </para>
/// </remarks>
public partial class PrototypeExplainer<T> : ILocalExplainer<T, PrototypeExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Matrix<T> _prototypes;
    private readonly Vector<T>? _prototypeLabels;
    private readonly int _numNeighbors;
    private readonly DistanceMetric _distanceMetric;
    private readonly string[]? _featureNames;
    private readonly string[]? _prototypeNames;

    /// <inheritdoc/>
    public string MethodName => "Prototype";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new Prototype explainer.
    /// </summary>
    /// <param name="predictFunction">Function that takes batch input and returns predictions.</param>
    /// <param name="prototypes">Set of prototype examples (representative training data).</param>
    /// <param name="prototypeLabels">Optional labels for each prototype.</param>
    /// <param name="numNeighbors">Number of nearest prototypes to return (default: 5).</param>
    /// <param name="distanceMetric">Distance metric to use (default: Euclidean).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="prototypeNames">Optional names/IDs for prototypes.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>prototypes</b>: These should be representative examples from your data.
    ///   You can use all training data or select key examples using clustering.
    /// - <b>distanceMetric</b>: Euclidean works well for normalized numeric features.
    ///   Use Cosine for text embeddings or high-dimensional sparse data.
    /// </para>
    /// </remarks>
    public PrototypeExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Matrix<T> prototypes,
        Vector<T>? prototypeLabels = null,
        int numNeighbors = 5,
        DistanceMetric distanceMetric = DistanceMetric.Euclidean,
        string[]? featureNames = null,
        string[]? prototypeNames = null)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        Guard.NotNull(prototypes);
        _prototypes = prototypes;

        if (prototypes.Rows == 0)
            throw new ArgumentException("Prototypes must have at least one example.", nameof(prototypes));
        if (numNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(numNeighbors));
        if (prototypeLabels != null && prototypeLabels.Length != prototypes.Rows)
            throw new ArgumentException($"prototypeLabels length ({prototypeLabels.Length}) must match prototypes.Rows ({prototypes.Rows}).", nameof(prototypeLabels));
        if (featureNames != null && featureNames.Length != prototypes.Columns)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match prototypes.Columns ({prototypes.Columns}).", nameof(featureNames));
        if (prototypeNames != null && prototypeNames.Length != prototypes.Rows)
            throw new ArgumentException($"prototypeNames length ({prototypeNames.Length}) must match prototypes.Rows ({prototypes.Rows}).", nameof(prototypeNames));

        _prototypeLabels = prototypeLabels;
        _numNeighbors = numNeighbors;
        _distanceMetric = distanceMetric;
        _featureNames = featureNames;
        _prototypeNames = prototypeNames;
    }

    /// <summary>
    /// Explains a prediction using similar prototypes.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>Prototype-based explanation.</returns>
    public PrototypeExplanation<T> Explain(Vector<T> instance)
    {
        if (instance.Length != _prototypes.Columns)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_prototypes.Columns}.", nameof(instance));

        int numPrototypes = _prototypes.Rows;
        int numFeatures = instance.Length;
        var engine = AiDotNetEngine.Current;

        // Get prediction for the instance
        var prediction = _predictFunction(ExplainerTensorOps<T>.RowMatrix(instance))[0];
        int predictedClass = NumOps.ToInt32(NumOps.Round(prediction));

        // Compute distances to all prototypes, and their similarities 1 / (1 + distance)
        var distances = ComputeDistances(instance);
        var similarities = engine.TensorDivide(
            engine.TensorFullLike(distances, NumOps.One), engine.TensorAddScalar(distances, NumOps.One));
        var labels = _prototypeLabels != null
            ? ExplainerTensorOps<T>.PadOrTruncate(_prototypeLabels, numPrototypes)
            : new Vector<T>(numPrototypes);
        var labelClasses = engine.TensorRound(labels.AsTensor()).Cast<int>().ToArray();

        // Sort by distance (stable: equal distances keep prototype order)
        var distanceRanks = numPrototypes > 0 ? ExplainerTensorOps<T>.DenseRanks(distances) : Array.Empty<int>();
        var order = Enumerable.Range(0, numPrototypes).OrderBy(i => distanceRanks[i]).ToArray();
        var distanceValues = distances.ToArray();
        var similarityValues = similarities.ToArray();
        var labelValues = labels.ToArray();
        var sorted = order
            .Select(i => (index: i, distance: distanceValues[i], similarity: similarityValues[i], label: labelValues[i], labelClass: labelClasses[i]))
            .ToList();

        PrototypeMatch<T> ToMatch((int index, T distance, T similarity, T label, int labelClass) d, bool isSameClass) => new PrototypeMatch<T>
        {
            PrototypeIndex = d.index,
            PrototypeName = _prototypeNames != null && d.index < _prototypeNames.Length
                ? _prototypeNames[d.index]
                : $"Prototype {d.index}",
            Distance = d.distance,
            Similarity = d.similarity,
            Label = d.label,
            PrototypeFeatures = _prototypes.GetRow(d.index),
            IsSameClass = isSameClass
        };

        // Get K nearest neighbors
        var nearestPrototypes = sorted
            .Take(_numNeighbors)
            .Select(d => ToMatch(d, d.labelClass == predictedClass))
            .ToList();

        // Get same-class prototypes (supporting the prediction)
        var sameClassPrototypes = sorted
            .Where(d => d.labelClass == predictedClass)
            .Take(_numNeighbors)
            .Select(d => ToMatch(d, true))
            .ToList();

        // Get different-class prototypes (contrasting)
        var contrastPrototypes = sorted
            .Where(d => d.labelClass != predictedClass)
            .Take(_numNeighbors)
            .Select(d => ToMatch(d, false))
            .ToList();

        // Compute feature differences with nearest same-class prototype
        Dictionary<int, T>? featureDifferences = sameClassPrototypes.Count > 0
            ? SignificantDifferences(instance, sameClassPrototypes[0].PrototypeFeatures)
            : null;

        // Compute distinguishing features (what makes this different from contrast prototypes)
        Dictionary<int, T>? distinguishingFeatures = contrastPrototypes.Count > 0
            ? SignificantDifferences(instance, contrastPrototypes[0].PrototypeFeatures)
            : null;

        return new PrototypeExplanation<T>
        {
            Input = instance,
            Prediction = prediction,
            PredictedClass = predictedClass,
            NearestPrototypes = nearestPrototypes,
            SameClassPrototypes = sameClassPrototypes,
            ContrastPrototypes = contrastPrototypes,
            FeatureDifferences = featureDifferences,
            DistinguishingFeatures = distinguishingFeatures,
            FeatureNames = _featureNames ?? Enumerable.Range(0, numFeatures).Select(i => $"Feature {i}").ToArray(),
            DistanceMetric = _distanceMetric
        };
    }

    /// <summary>
    /// instance - prototype at the features where it exceeds 1e-6 in magnitude, keyed by feature index.
    /// </summary>
    private static Dictionary<int, T> SignificantDifferences(Vector<T> instance, Vector<T> prototype)
    {
        var engine = AiDotNetEngine.Current;
        var diff = engine.Subtract(instance, prototype.GetSubVector(0, instance.Length));
        var significant = engine.TensorGreaterThan(engine.Abs(diff).AsTensor(), NumOps.FromDouble(1e-6)).ToVector();
        var indices = significant.NonZeroIndices().ToList();
        return indices.Zip(diff.GetElements(indices), (index, value) => (index, value))
            .ToDictionary(entry => entry.index, entry => entry.value);
    }

    /// <inheritdoc/>
    public PrototypeExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new PrototypeExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Computes distance between two vectors.
    /// </summary>
    private Tensor<T> ComputeDistances(Vector<T> instance)
    {
        if (_prototypes.Rows == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        var engine = AiDotNetEngine.Current;
        var prototypes = Tensor<T>.FromMatrix(_prototypes);
        var instanceRows = engine.TensorBroadcastTo(ExplainerTensorOps<T>.RowTensor(instance), prototypes.Shape);
        var diff = engine.TensorSubtract(prototypes, instanceRows);

        switch (_distanceMetric)
        {
            case DistanceMetric.Manhattan:
                return engine.ReduceSum(engine.TensorAbs(diff), new[] { 1 }, false);
            case DistanceMetric.Cosine:
            {
                // 1 - clamp(p . x / sqrt(|p|^2 |x|^2), -1, 1), with similarity 0 when the denominator is <= 1e-10
                var dot = engine.MatrixVectorMultiply(_prototypes, instance).AsTensor();
                var prototypeNorms = engine.ReduceSum(engine.TensorSquare(prototypes), new[] { 1 }, false);
                var denominator = engine.TensorSqrt(engine.TensorMultiplyScalar(prototypeNorms, engine.DotProduct(instance, instance)));
                var usable = engine.TensorGreaterThan(denominator, NumOps.FromDouble(1e-10));
                var safeDenominator = engine.TensorWhere(usable, denominator, engine.TensorFullLike(denominator, NumOps.One));
                var cosine = engine.TensorMax(engine.TensorMin(engine.TensorDivide(dot, safeDenominator), NumOps.One), NumOps.Negate(NumOps.One));
                var similarity = engine.TensorWhere(usable, cosine, engine.TensorFullLike(cosine, NumOps.Zero));
                return engine.TensorSubtract(engine.TensorFullLike(similarity, NumOps.One), similarity);
            }
            default:
                return engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(diff), new[] { 1 }, false));
        }
    }

}

/// <summary>
/// Distance metrics for prototype matching.
/// </summary>
public enum DistanceMetric
{
    /// <summary>
    /// Euclidean (L2) distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Manhattan (L1) distance.
    /// </summary>
    Manhattan,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    Cosine
}

/// <summary>
/// Represents a matched prototype.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PrototypeMatch<T>
{
    /// <summary>
    /// Gets or sets the index of the prototype in the prototype set.
    /// </summary>
    public int PrototypeIndex { get; set; }

    /// <summary>
    /// Gets or sets the name/ID of the prototype.
    /// </summary>
    public string PrototypeName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the distance from the input to this prototype.
    /// </summary>
    public T Distance { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the similarity (1 / (1 + distance)).
    /// </summary>
    public T Similarity { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the label of the prototype.
    /// </summary>
    public T Label { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the feature values of the prototype.
    /// </summary>
    public Vector<T> PrototypeFeatures { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets whether this prototype has the same class as the input.
    /// </summary>
    public bool IsSameClass { get; set; }
}

/// <summary>
/// Represents the result of a Prototype-based explanation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PrototypeExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Input { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the prediction value.
    /// </summary>
    public T Prediction { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the predicted class.
    /// </summary>
    public int PredictedClass { get; set; }

    /// <summary>
    /// Gets or sets the nearest prototypes (regardless of class).
    /// </summary>
    public List<PrototypeMatch<T>> NearestPrototypes { get; set; } = new();

    /// <summary>
    /// Gets or sets prototypes with the same class (supporting evidence).
    /// </summary>
    public List<PrototypeMatch<T>> SameClassPrototypes { get; set; } = new();

    /// <summary>
    /// Gets or sets prototypes with different class (contrast examples).
    /// </summary>
    public List<PrototypeMatch<T>> ContrastPrototypes { get; set; } = new();

    /// <summary>
    /// Gets or sets feature differences from the nearest same-class prototype.
    /// </summary>
    public Dictionary<int, T>? FeatureDifferences { get; set; }

    /// <summary>
    /// Gets or sets features that distinguish from the nearest contrast prototype.
    /// </summary>
    public Dictionary<int, T>? DistinguishingFeatures { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the distance metric used.
    /// </summary>
    public DistanceMetric DistanceMetric { get; set; }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            "Prototype-based Explanation:",
            $"  Predicted class: {PredictedClass}",
            $"  Distance metric: {DistanceMetric}",
            ""
        };

        lines.Add("Nearest Similar Examples (same class):");
        foreach (var proto in SameClassPrototypes.Take(3))
        {
            lines.Add($"  {proto.PrototypeName}: distance={proto.Distance:F4}, " +
                      $"similarity={proto.Similarity:F4}");
        }

        if (ContrastPrototypes.Count > 0)
        {
            lines.Add("");
            lines.Add("Nearest Contrasting Examples (different class):");
            var contrasts = ContrastPrototypes.Take(3).ToList();
            var engine = AiDotNetEngine.Current;
            var labelClasses = engine.TensorRound(new Vector<T>(contrasts.Select(proto => proto.Label)).AsTensor()).Cast<int>().ToArray();
            foreach (var (proto, labelClass) in contrasts.Zip(labelClasses, (match, label) => (match, label)))
            {
                lines.Add($"  {proto.PrototypeName}: class={labelClass}, " +
                          $"distance={proto.Distance:F4}");
            }
        }

        if (DistinguishingFeatures != null && DistinguishingFeatures.Count > 0)
        {
            lines.Add("");
            lines.Add("Key Distinguishing Features (vs nearest different class):");
            var engine = AiDotNetEngine.Current;
            var entries = DistinguishingFeatures.ToList();
            var magnitudeRanks = ExplainerTensorOps<T>.DenseRanks(engine.TensorAbs(new Vector<T>(entries.Select(kvp => kvp.Value)).AsTensor()));
            var top = Enumerable.Range(0, entries.Count).OrderByDescending(k => magnitudeRanks[k]).Take(5).Select(k => entries[k]).ToList();
            var diffs = new Vector<T>(top.Select(kvp => kvp.Value)).AsTensor();
            var magnitudes = engine.TensorAbs(diffs).ToArray();
            var higher = engine.TensorGreaterThan(diffs, NumOps.Zero).Cast<int>().ToArray();
            for (int k = 0; k < top.Count; k++)
            {
                int featureIndex = top[k].Key;
                string featureName = featureIndex < FeatureNames.Length ? FeatureNames[featureIndex] : $"Feature {featureIndex}";
                string direction = higher[k] != 0 ? "higher" : "lower";
                lines.Add($"  {featureName}: {magnitudes[k]:F4} {direction}");
            }
        }

        return string.Join(Environment.NewLine, lines);
    }
}
