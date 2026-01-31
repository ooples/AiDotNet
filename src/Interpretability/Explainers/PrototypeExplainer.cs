using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

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
public class PrototypeExplainer<T> : ILocalExplainer<T, PrototypeExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
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
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _prototypes = prototypes ?? throw new ArgumentNullException(nameof(prototypes));

        if (prototypes.Rows == 0)
            throw new ArgumentException("Prototypes must have at least one example.", nameof(prototypes));
        if (numNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(numNeighbors));

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
        int numPrototypes = _prototypes.Rows;
        int numFeatures = instance.Length;

        // Get prediction for the instance
        var instanceMatrix = CreateSingleRowMatrix(instance);
        var prediction = _predictFunction(instanceMatrix)[0];
        int predictedClass = (int)Math.Round(NumOps.ToDouble(prediction));

        // Compute distances to all prototypes
        var distances = new (int index, double distance, T label, int labelClass)[numPrototypes];

        for (int i = 0; i < numPrototypes; i++)
        {
            var prototype = _prototypes.GetRow(i);
            double dist = ComputeDistance(instance, prototype);
            T label = _prototypeLabels != null && i < _prototypeLabels.Length
                ? _prototypeLabels[i]
                : NumOps.Zero;
            int labelClass = (int)Math.Round(NumOps.ToDouble(label));
            distances[i] = (i, dist, label, labelClass);
        }

        // Sort by distance
        Array.Sort(distances, (a, b) => a.distance.CompareTo(b.distance));

        // Get K nearest neighbors
        var nearestPrototypes = distances
            .Take(_numNeighbors)
            .Select(d => new PrototypeMatch<T>
            {
                PrototypeIndex = d.index,
                PrototypeName = _prototypeNames != null && d.index < _prototypeNames.Length
                    ? _prototypeNames[d.index]
                    : $"Prototype {d.index}",
                Distance = NumOps.FromDouble(d.distance),
                Similarity = NumOps.FromDouble(1.0 / (1.0 + d.distance)),
                Label = d.label,
                PrototypeFeatures = _prototypes.GetRow(d.index),
                IsSameClass = d.labelClass == predictedClass
            })
            .ToList();

        // Get same-class prototypes (supporting the prediction)
        var sameClassPrototypes = distances
            .Where(d => d.labelClass == predictedClass)
            .Take(_numNeighbors)
            .Select(d => new PrototypeMatch<T>
            {
                PrototypeIndex = d.index,
                PrototypeName = _prototypeNames != null && d.index < _prototypeNames.Length
                    ? _prototypeNames[d.index]
                    : $"Prototype {d.index}",
                Distance = NumOps.FromDouble(d.distance),
                Similarity = NumOps.FromDouble(1.0 / (1.0 + d.distance)),
                Label = d.label,
                PrototypeFeatures = _prototypes.GetRow(d.index),
                IsSameClass = true
            })
            .ToList();

        // Get different-class prototypes (contrasting)
        var contrastPrototypes = distances
            .Where(d => d.labelClass != predictedClass)
            .Take(_numNeighbors)
            .Select(d => new PrototypeMatch<T>
            {
                PrototypeIndex = d.index,
                PrototypeName = _prototypeNames != null && d.index < _prototypeNames.Length
                    ? _prototypeNames[d.index]
                    : $"Prototype {d.index}",
                Distance = NumOps.FromDouble(d.distance),
                Similarity = NumOps.FromDouble(1.0 / (1.0 + d.distance)),
                Label = d.label,
                PrototypeFeatures = _prototypes.GetRow(d.index),
                IsSameClass = false
            })
            .ToList();

        // Compute feature differences with nearest same-class prototype
        Dictionary<int, T>? featureDifferences = null;
        if (sameClassPrototypes.Count > 0)
        {
            featureDifferences = new Dictionary<int, T>();
            var nearestSame = sameClassPrototypes[0];
            for (int j = 0; j < numFeatures; j++)
            {
                double diff = NumOps.ToDouble(instance[j]) - NumOps.ToDouble(nearestSame.PrototypeFeatures[j]);
                if (Math.Abs(diff) > 1e-6)
                {
                    featureDifferences[j] = NumOps.FromDouble(diff);
                }
            }
        }

        // Compute distinguishing features (what makes this different from contrast prototypes)
        Dictionary<int, T>? distinguishingFeatures = null;
        if (contrastPrototypes.Count > 0)
        {
            distinguishingFeatures = new Dictionary<int, T>();
            var nearestContrast = contrastPrototypes[0];
            for (int j = 0; j < numFeatures; j++)
            {
                double instanceVal = NumOps.ToDouble(instance[j]);
                double contrastVal = NumOps.ToDouble(nearestContrast.PrototypeFeatures[j]);
                double diff = instanceVal - contrastVal;
                if (Math.Abs(diff) > 1e-6)
                {
                    distinguishingFeatures[j] = NumOps.FromDouble(diff);
                }
            }
        }

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
    private double ComputeDistance(Vector<T> a, Vector<T> b)
    {
        int n = Math.Min(a.Length, b.Length);

        switch (_distanceMetric)
        {
            case DistanceMetric.Euclidean:
                double sumSq = 0;
                for (int i = 0; i < n; i++)
                {
                    double diff = NumOps.ToDouble(a[i]) - NumOps.ToDouble(b[i]);
                    sumSq += diff * diff;
                }
                return Math.Sqrt(sumSq);

            case DistanceMetric.Manhattan:
                double sumAbs = 0;
                for (int i = 0; i < n; i++)
                {
                    sumAbs += Math.Abs(NumOps.ToDouble(a[i]) - NumOps.ToDouble(b[i]));
                }
                return sumAbs;

            case DistanceMetric.Cosine:
                double dotProduct = 0;
                double normA = 0;
                double normB = 0;
                for (int i = 0; i < n; i++)
                {
                    double aVal = NumOps.ToDouble(a[i]);
                    double bVal = NumOps.ToDouble(b[i]);
                    dotProduct += aVal * bVal;
                    normA += aVal * aVal;
                    normB += bVal * bVal;
                }
                double cosineSim = dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-10);
                return 1.0 - cosineSim; // Convert to distance

            default:
                goto case DistanceMetric.Euclidean;
        }
    }

    private Matrix<T> CreateSingleRowMatrix(Vector<T> row)
    {
        var matrix = new Matrix<T>(1, row.Length);
        for (int j = 0; j < row.Length; j++)
            matrix[0, j] = row[j];
        return matrix;
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
    public T Distance { get; set; } = default!;

    /// <summary>
    /// Gets or sets the similarity (1 / (1 + distance)).
    /// </summary>
    public T Similarity { get; set; } = default!;

    /// <summary>
    /// Gets or sets the label of the prototype.
    /// </summary>
    public T Label { get; set; } = default!;

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
    public T Prediction { get; set; } = default!;

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
            lines.Add($"  {proto.PrototypeName}: distance={NumOps.ToDouble(proto.Distance):F4}, " +
                      $"similarity={NumOps.ToDouble(proto.Similarity):F4}");
        }

        if (ContrastPrototypes.Count > 0)
        {
            lines.Add("");
            lines.Add("Nearest Contrasting Examples (different class):");
            foreach (var proto in ContrastPrototypes.Take(3))
            {
                int labelClass = (int)Math.Round(NumOps.ToDouble(proto.Label));
                lines.Add($"  {proto.PrototypeName}: class={labelClass}, " +
                          $"distance={NumOps.ToDouble(proto.Distance):F4}");
            }
        }

        if (DistinguishingFeatures != null && DistinguishingFeatures.Count > 0)
        {
            lines.Add("");
            lines.Add("Key Distinguishing Features (vs nearest different class):");
            var sorted = DistinguishingFeatures
                .OrderByDescending(kvp => Math.Abs(NumOps.ToDouble(kvp.Value)))
                .Take(5);
            foreach (var kvp in sorted)
            {
                string featureName = kvp.Key < FeatureNames.Length ? FeatureNames[kvp.Key] : $"Feature {kvp.Key}";
                double diff = NumOps.ToDouble(kvp.Value);
                string direction = diff > 0 ? "higher" : "lower";
                lines.Add($"  {featureName}: {Math.Abs(diff):F4} {direction}");
            }
        }

        return string.Join(Environment.NewLine, lines);
    }
}
