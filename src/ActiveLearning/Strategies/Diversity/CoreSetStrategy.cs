using AiDotNet.ActiveLearning.Config;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Strategies.Diversity;

/// <summary>
/// CoreSet strategy for active learning using geometric diversity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CoreSet is a diversity-based strategy that aims to select
/// samples that best represent the entire dataset. It's like choosing a small set of
/// "core" points that cover the data space well.</para>
///
/// <para><b>How CoreSet Works:</b></para>
/// <list type="number">
/// <item><description>Extract feature representations of all samples</description></item>
/// <item><description>Start with labeled samples as initial "covered" points</description></item>
/// <item><description>Iteratively select unlabeled points that maximize minimum distance to covered set</description></item>
/// <item><description>This ensures selected points spread across the feature space</description></item>
/// </list>
///
/// <para><b>Distance Metrics:</b></para>
/// <list type="bullet">
/// <item><description><b>Euclidean:</b> Standard L2 distance in feature space</description></item>
/// <item><description><b>Cosine:</b> Angle-based distance (1 - cosine similarity)</description></item>
/// <item><description><b>Manhattan:</b> L1 distance for sparse features</description></item>
/// </list>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Want diversity rather than just uncertainty</description></item>
/// <item><description>Data has meaningful geometric structure</description></item>
/// <item><description>Concerned about sample redundancy</description></item>
/// </list>
///
/// <para><b>Reference:</b> Sener and Savarese "Active Learning for Convolutional Neural Networks: A Core-Set Approach" (ICLR 2018)</para>
/// </remarks>
public class CoreSetStrategy<T, TInput, TOutput> : IDiversityStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly DistanceMetric _distanceMetric;
    private readonly ActiveLearnerConfig<T>? _config;
    private List<Vector<T>>? _labeledFeatures;
    private List<Vector<T>>? _unlabeledFeatures;

    /// <inheritdoc/>
    public string Name => $"CoreSet ({_distanceMetric})";

    /// <inheritdoc/>
    public string Description =>
        "Selects samples that maximize geometric coverage of the feature space using greedy furthest-first traversal";

    /// <summary>
    /// Initializes a new CoreSet strategy with Euclidean distance.
    /// </summary>
    public CoreSetStrategy()
        : this(DistanceMetric.Euclidean, null)
    {
    }

    /// <summary>
    /// Initializes a new CoreSet strategy with specified distance metric.
    /// </summary>
    /// <param name="distanceMetric">The distance metric to use.</param>
    /// <param name="config">Optional configuration.</param>
    public CoreSetStrategy(DistanceMetric distanceMetric, ActiveLearnerConfig<T>? config = null)
    {
        _distanceMetric = distanceMetric;
        _config = config;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeScores(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool)
    {
        // For CoreSet, we compute distances to the labeled set
        // Score = minimum distance to any labeled sample
        var scores = new T[unlabeledPool.Count];

        // Extract features for unlabeled samples
        var unlabeledFeatures = ExtractFeatures(model, unlabeledPool);

        if (_labeledFeatures == null || _labeledFeatures.Count == 0)
        {
            // No labeled samples yet - all samples get equal high scores
            for (int i = 0; i < unlabeledPool.Count; i++)
            {
                scores[i] = NumOps.One;
            }
        }
        else
        {
            for (int i = 0; i < unlabeledPool.Count; i++)
            {
                scores[i] = ComputeMinDistanceToLabeled(unlabeledFeatures[i]);
            }
        }

        return new Vector<T>(scores);
    }

    /// <inheritdoc/>
    public int[] SelectSamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        var batchSizeToUse = Math.Min(batchSize, unlabeledPool.Count);

        // Extract features
        var unlabeledFeatures = ExtractFeatures(model, unlabeledPool);
        _unlabeledFeatures = unlabeledFeatures;

        // Use greedy furthest-first traversal (k-center greedy)
        var selectedIndices = new List<int>();
        var coveredPoints = new List<Vector<T>>();

        // Initialize with labeled features if available
        if (_labeledFeatures != null)
        {
            coveredPoints.AddRange(_labeledFeatures);
        }

        // Greedy selection
        for (int b = 0; b < batchSizeToUse; b++)
        {
            T maxMinDistance = NumOps.MinValue;
            int bestIndex = -1;

            for (int i = 0; i < unlabeledFeatures.Count; i++)
            {
                if (selectedIndices.Contains(i))
                {
                    continue;
                }

                // Compute minimum distance to all covered points
                T minDistance = ComputeMinDistance(unlabeledFeatures[i], coveredPoints);

                if (NumOps.Compare(minDistance, maxMinDistance) > 0)
                {
                    maxMinDistance = minDistance;
                    bestIndex = i;
                }
            }

            if (bestIndex >= 0)
            {
                selectedIndices.Add(bestIndex);
                coveredPoints.Add(unlabeledFeatures[bestIndex]);
            }
        }

        return selectedIndices.ToArray();
    }

    /// <inheritdoc/>
    public T ComputeDiversity(TInput sample1, TInput sample2)
    {
        // Convert samples to vectors for distance computation
        if (sample1 is Vector<T> vec1 && sample2 is Vector<T> vec2)
        {
            return ComputeDistance(vec1, vec2);
        }

        // Default: return zero (samples assumed similar)
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeDensityWeights(IDataset<T, TInput, TOutput> pool)
    {
        // Density weighting: prefer points in dense regions to represent them
        // Weight = average inverse distance to k nearest neighbors
        // Higher density (smaller avg distance to neighbors) = higher weight
        if (pool.Count == 0)
        {
            return new Vector<T>(0);
        }

        var weights = new T[pool.Count];
        int k = Math.Min(5, pool.Count - 1);

        if (k == 0)
        {
            // Only one sample, give it weight 1
            weights[0] = NumOps.One;
            return new Vector<T>(weights);
        }

        // Extract features for all samples
        var features = new List<Vector<T>>();
        for (int i = 0; i < pool.Count; i++)
        {
            features.Add(GetFeatureRepresentation(pool.GetInput(i)));
        }

        // Compute density for each sample based on k-nearest neighbors
        for (int i = 0; i < pool.Count; i++)
        {
            // Compute distances to all other points
            var distances = new List<(int Index, T Distance)>();
            for (int j = 0; j < pool.Count; j++)
            {
                if (i != j)
                {
                    var dist = ComputeDistance(features[i], features[j]);
                    distances.Add((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.Sort((a, b) => NumOps.Compare(a.Distance, b.Distance));
            var kNearest = distances.Take(k).ToList();

            // Compute average distance to k nearest neighbors
            T avgDistance = NumOps.Zero;
            foreach (var (_, distance) in kNearest)
            {
                avgDistance = NumOps.Add(avgDistance, distance);
            }
            avgDistance = NumOps.Divide(avgDistance, NumOps.FromDouble(k));

            // Density weight = 1 / (avgDistance + epsilon) for stability
            // Higher density (smaller distance) = higher weight
            var epsilon = NumOps.FromDouble(1e-6);
            weights[i] = NumOps.Divide(NumOps.One, NumOps.Add(avgDistance, epsilon));
        }

        // Normalize weights to sum to 1
        T totalWeight = NumOps.Zero;
        foreach (var w in weights)
        {
            totalWeight = NumOps.Add(totalWeight, w);
        }

        if (NumOps.Compare(totalWeight, NumOps.Zero) > 0)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NumOps.Divide(weights[i], totalWeight);
            }
        }

        return new Vector<T>(weights);
    }

    /// <inheritdoc/>
    public void UpdateState(int[] newlyLabeledIndices, TOutput[] labels)
    {
        if (_unlabeledFeatures == null) return;

        // Move selected features to labeled set
        _labeledFeatures ??= new List<Vector<T>>();

        foreach (var index in newlyLabeledIndices)
        {
            if (index >= 0 && index < _unlabeledFeatures.Count)
            {
                _labeledFeatures.Add(_unlabeledFeatures[index]);
            }
        }
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _labeledFeatures = null;
        _unlabeledFeatures = null;
    }

    /// <inheritdoc/>
    public int[] SelectDiverseSamples(
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData,
        int numSamples)
    {
        var batchSizeToUse = Math.Min(numSamples, unlabeledData.Count);

        // Extract features from unlabeled data
        var unlabeledFeatures = new List<Vector<T>>();
        for (int i = 0; i < unlabeledData.Count; i++)
        {
            unlabeledFeatures.Add(GetFeatureRepresentation(unlabeledData.GetInput(i)));
        }

        // Extract features from labeled data if available
        var coveredPoints = new List<Vector<T>>();
        if (labeledData != null)
        {
            for (int i = 0; i < labeledData.Count; i++)
            {
                coveredPoints.Add(GetFeatureRepresentation(labeledData.GetInput(i)));
            }
        }

        // Use greedy furthest-first traversal (k-center greedy)
        var selectedIndices = new List<int>();

        for (int b = 0; b < batchSizeToUse; b++)
        {
            T maxMinDistance = NumOps.MinValue;
            int bestIndex = -1;

            for (int i = 0; i < unlabeledFeatures.Count; i++)
            {
                if (selectedIndices.Contains(i))
                {
                    continue;
                }

                // Compute minimum distance to all covered points
                T minDistance = coveredPoints.Count > 0
                    ? ComputeMinDistance(unlabeledFeatures[i], coveredPoints)
                    : NumOps.MaxValue;

                if (NumOps.Compare(minDistance, maxMinDistance) > 0)
                {
                    maxMinDistance = minDistance;
                    bestIndex = i;
                }
            }

            if (bestIndex >= 0)
            {
                selectedIndices.Add(bestIndex);
                coveredPoints.Add(unlabeledFeatures[bestIndex]);
            }
        }

        return selectedIndices.ToArray();
    }

    /// <inheritdoc/>
    public Vector<T> ComputeDiversityScores(
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData)
    {
        var scores = new T[unlabeledData.Count];

        // Extract features from labeled data
        var labeledFeatures = new List<Vector<T>>();
        if (labeledData != null)
        {
            for (int i = 0; i < labeledData.Count; i++)
            {
                labeledFeatures.Add(GetFeatureRepresentation(labeledData.GetInput(i)));
            }
        }

        if (labeledFeatures.Count == 0)
        {
            // No labeled samples yet - all samples get equal high scores
            for (int i = 0; i < unlabeledData.Count; i++)
            {
                scores[i] = NumOps.One;
            }
        }
        else
        {
            for (int i = 0; i < unlabeledData.Count; i++)
            {
                var feature = GetFeatureRepresentation(unlabeledData.GetInput(i));
                scores[i] = ComputeMinDistance(feature, labeledFeatures);
            }
        }

        return new Vector<T>(scores);
    }

    /// <inheritdoc/>
    public Matrix<T> ComputeDistanceMatrix(IDataset<T, TInput, TOutput> dataset)
    {
        int n = dataset.Count;
        var distances = new Matrix<T>(n, n);

        // Extract all features first
        var features = new List<Vector<T>>();
        for (int i = 0; i < n; i++)
        {
            features.Add(GetFeatureRepresentation(dataset.GetInput(i)));
        }

        // Compute pairwise distances
        for (int i = 0; i < n; i++)
        {
            distances[i, i] = NumOps.Zero;
            for (int j = i + 1; j < n; j++)
            {
                var dist = ComputeDistance(features[i], features[j]);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    /// <inheritdoc/>
    public Vector<T> GetFeatureRepresentation(TInput input)
    {
        // If input is already a vector, use it directly
        if (input is Vector<T> vectorInput)
        {
            return vectorInput;
        }

        // If input is an array, convert to vector
        if (input is T[] arrayInput)
        {
            return new Vector<T>(arrayInput);
        }

        // If input is a single numeric value, create single-element vector
        if (input is T singleValue)
        {
            return new Vector<T>(new[] { singleValue });
        }

        // If input is double/float/int, convert and create vector
        if (input is double doubleValue)
        {
            return new Vector<T>(new[] { NumOps.FromDouble(doubleValue) });
        }
        if (input is float floatValue)
        {
            return new Vector<T>(new[] { NumOps.FromDouble(floatValue) });
        }
        if (input is int intValue)
        {
            return new Vector<T>(new[] { NumOps.FromDouble(intValue) });
        }

        // Fallback: return zero vector (should be overridden for complex types)
        return new Vector<T>(1);
    }

    /// <summary>
    /// Sets the initial labeled features for distance computation.
    /// </summary>
    /// <param name="labeledData">The labeled dataset.</param>
    /// <param name="model">The model for feature extraction.</param>
    public void InitializeLabeledFeatures(
        IDataset<T, TInput, TOutput> labeledData,
        IFullModel<T, TInput, TOutput> model)
    {
        _labeledFeatures = ExtractFeatures(model, labeledData);
    }

    #region Private Methods

    private List<Vector<T>> ExtractFeatures(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> dataset)
    {
        var features = new List<Vector<T>>();

        for (int i = 0; i < dataset.Count; i++)
        {
            var input = dataset.GetInput(i);
            var feature = ExtractFeature(model, input);
            features.Add(feature);
        }

        return features;
    }

    private Vector<T> ExtractFeature(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        // If model supports feature extraction, use it
        if (model is IFeatureExtractor<T, TInput> featureExtractor)
        {
            return featureExtractor.ExtractFeatures(input);
        }

        // Fallback: use input directly if it's a vector
        if (input is Vector<T> vectorInput)
        {
            return vectorInput;
        }

        // Otherwise, use prediction as features
        var prediction = model.Predict(input);
        if (prediction is Vector<T> vectorPred)
        {
            return vectorPred;
        }

        // Last resort: single value feature
        return new Vector<T>(new[] { ConvertToNumeric(prediction) });
    }

    private T ComputeMinDistanceToLabeled(Vector<T> feature)
    {
        if (_labeledFeatures == null || _labeledFeatures.Count == 0)
        {
            return NumOps.MaxValue;
        }

        return ComputeMinDistance(feature, _labeledFeatures);
    }

    private T ComputeMinDistance(Vector<T> point, List<Vector<T>> coveredPoints)
    {
        if (coveredPoints.Count == 0)
        {
            return NumOps.MaxValue;
        }

        T minDist = NumOps.MaxValue;

        foreach (var covered in coveredPoints)
        {
            var dist = ComputeDistance(point, covered);
            if (NumOps.Compare(dist, minDist) < 0)
            {
                minDist = dist;
            }
        }

        return minDist;
    }

    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        return _distanceMetric switch
        {
            DistanceMetric.Euclidean => EuclideanDistance(a, b),
            DistanceMetric.Cosine => CosineDistance(a, b),
            DistanceMetric.Manhattan => ManhattanDistance(a, b),
            _ => EuclideanDistance(a, b)
        };
    }

    private T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        int length = Math.Min(a.Length, b.Length);
        T sumSquared = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sumSquared);
    }

    private T CosineDistance(Vector<T> a, Vector<T> b)
    {
        // Cosine distance = 1 - cosine similarity
        int length = Math.Min(a.Length, b.Length);
        T dotProduct = NumOps.Zero;
        T normA = NumOps.Zero;
        T normB = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
            normA = NumOps.Add(normA, NumOps.Multiply(a[i], a[i]));
            normB = NumOps.Add(normB, NumOps.Multiply(b[i], b[i]));
        }

        var denominator = NumOps.Multiply(NumOps.Sqrt(normA), NumOps.Sqrt(normB));
        if (NumOps.Compare(denominator, NumOps.Zero) <= 0)
        {
            return NumOps.One;
        }

        var similarity = NumOps.Divide(dotProduct, denominator);
        return NumOps.Subtract(NumOps.One, similarity);
    }

    private T ManhattanDistance(Vector<T> a, Vector<T> b)
    {
        int length = Math.Min(a.Length, b.Length);
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Abs(diff));
        }

        return sum;
    }

    private T ConvertToNumeric(TOutput output)
    {
        if (output is T typedValue) return typedValue;
        if (output is double doubleValue) return NumOps.FromDouble(doubleValue);
        if (output is float floatValue) return NumOps.FromDouble(floatValue);
        if (output is int intValue) return NumOps.FromDouble(intValue);
        return NumOps.Zero;
    }

    #endregion
}

/// <summary>
/// Distance metrics for diversity-based strategies.
/// </summary>
public enum DistanceMetric
{
    /// <summary>
    /// Euclidean (L2) distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    Cosine,

    /// <summary>
    /// Manhattan (L1) distance.
    /// </summary>
    Manhattan
}

/// <summary>
/// Interface for models that can extract feature representations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
public interface IFeatureExtractor<T, TInput>
{
    /// <summary>
    /// Extracts feature representation from an input.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <returns>Feature vector representation.</returns>
    Vector<T> ExtractFeatures(TInput input);
}
