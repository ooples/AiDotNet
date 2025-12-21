using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ActiveLearning.QueryStrategies;

/// <summary>
/// Core-set selection strategy for active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Core-set selection chooses examples that are representative
/// of the unlabeled data distribution. Instead of selecting uncertain examples, it selects
/// examples that best "cover" the feature space.</para>
///
/// <para><b>Key Idea:</b>
/// The goal is to minimize the maximum distance from any unlabeled point to its nearest
/// labeled point. This ensures good coverage of the data distribution.
/// </para>
///
/// <para><b>How it works:</b>
/// 1. Represent each example by its features (often using embeddings from a neural network)
/// 2. Select examples that maximize coverage of the feature space
/// 3. Use greedy k-center algorithm:
///    - Start with current labeled set
///    - Repeatedly add the unlabeled point that is farthest from all labeled points
///    - Continue until budget is exhausted
/// </para>
///
/// <para><b>Advantages:</b>
/// - Ensures diverse selection (no redundant examples)
/// - Works well when labeled data should represent full distribution
/// - Effective for imbalanced datasets
/// - Can be combined with uncertainty sampling
/// </para>
///
/// <para><b>Disadvantages:</b>
/// - Requires computing pairwise distances (can be expensive)
/// - May select outliers or noisy examples
/// - Doesn't consider model uncertainty
/// </para>
///
/// <para><b>Variants:</b>
/// - <b>k-Center:</b> Greedy algorithm for maximum coverage
/// - <b>k-Means++:</b> Probabilistic selection based on distance
/// - <b>Facility Location:</b> Optimize total distance to nearest labeled point
/// </para>
///
/// <para><b>Reference:</b> Sener and Savarese "Active Learning for Convolutional Neural Networks: A Core-Set Approach" (2018)</para>
/// </remarks>
public class CoreSetSelection<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Algorithm for core-set selection.
    /// </summary>
    public enum CoreSetAlgorithm
    {
        /// <summary>
        /// Greedy k-center algorithm
        /// </summary>
        KCenter,

        /// <summary>
        /// k-means++ style probabilistic selection
        /// </summary>
        KMeansPlusPlus
    }

    private readonly CoreSetAlgorithm _algorithm;
    private readonly Func<TInput, Vector<T>>? _featureExtractor;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new core-set selection strategy.
    /// </summary>
    /// <param name="algorithm">The core-set algorithm to use.</param>
    /// <param name="featureExtractor">Function to extract features from inputs (uses model embeddings if null).</param>
    /// <param name="seed">Random seed for reproducibility (null for non-deterministic).</param>
    public CoreSetSelection(
        CoreSetAlgorithm algorithm = CoreSetAlgorithm.KCenter,
        Func<TInput, Vector<T>>? featureExtractor = null,
        int? seed = null)
    {
        _algorithm = algorithm;
        _featureExtractor = featureExtractor;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc/>
    public string Name => $"CoreSet-{_algorithm}";

    /// <inheritdoc/>
    public Vector<T> ScoreExamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData = null)
    {
        if (unlabeledData == null)
            throw new ArgumentNullException(nameof(unlabeledData));

        // For core-set, scoring is based on distance to labeled set
        // Without labeled data context, return uniform scores
        // SelectBatch will perform the actual distance-based selection

        int numExamples = unlabeledData.Count;
        var scores = Enumerable.Repeat(NumOps.FromDouble(1.0), numExamples).ToArray();
        return new Vector<T>(scores);
    }

    /// <inheritdoc/>
    public Vector<int> SelectBatch(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        int k,
        IDataset<T, TInput, TOutput>? labeledData = null)
    {
        return _algorithm switch
        {
            CoreSetAlgorithm.KCenter => SelectKCenter(model, unlabeledData, k, labeledData),
            CoreSetAlgorithm.KMeansPlusPlus => SelectKMeansPlusPlus(model, unlabeledData, k, labeledData),
            _ => throw new NotSupportedException($"Algorithm {_algorithm} not supported")
        };
    }

    /// <summary>
    /// Selects examples using greedy k-center algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// 1. Initialize: all labeled examples are "centers"
    /// 2. For each iteration:
    ///    - For each unlabeled example, compute distance to nearest center
    ///    - Select the unlabeled example with maximum distance to nearest center
    ///    - Add this example to the set of centers
    /// 3. Return k selected examples
    /// </para>
    /// <para><b>Complexity:</b> O(k * n * d) where n = unlabeled set size, d = feature dimension</para>
    /// </remarks>
    private Vector<int> SelectKCenter(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        int k,
        IDataset<T, TInput, TOutput>? labeledData)
    {
        if (unlabeledData == null)
            throw new ArgumentNullException(nameof(unlabeledData));

        int numExamples = unlabeledData.Count;

        // Extract features for all unlabeled examples
        var unlabeledFeatures = new List<Vector<T>>();
        for (int i = 0; i < numExamples; i++)
        {
            var input = unlabeledData.GetInput(i);
            var feat = ExtractFeatures(model, input);
            unlabeledFeatures.Add(feat);
        }

        // Initialize centers with existing labeled examples
        // This ensures new selections are diverse relative to what's already labeled
        var centers = new List<Vector<T>>();
        if (labeledData != null)
        {
            for (int i = 0; i < labeledData.Count; i++)
            {
                var input = labeledData.GetInput(i);
                var feat = ExtractFeatures(model, input);
                centers.Add(feat);
            }
        }

        var selected = new List<int>();

        // Greedy k-center algorithm: iteratively select the unlabeled example
        // that is farthest from all current centers (labeled + newly selected)
        for (int iteration = 0; iteration < Math.Min(k, numExamples); iteration++)
        {
            int farthestIdx = -1;
            T maxDist = NumOps.FromDouble(double.MinValue);

            // Find unlabeled example with maximum distance to nearest center
            for (int idx = 0; idx < numExamples; idx++)
            {
                // Skip already selected examples
                if (selected.Contains(idx))
                    continue;

                T dist = MinDistanceToCenters(unlabeledFeatures[idx], centers);
                if (Convert.ToDouble(dist) > Convert.ToDouble(maxDist))
                {
                    maxDist = dist;
                    farthestIdx = idx;
                }
            }

            // Add the farthest example to selected set and centers
            if (farthestIdx >= 0)
            {
                selected.Add(farthestIdx);
                centers.Add(unlabeledFeatures[farthestIdx]);
            }
            else
            {
                // No more unlabeled examples available
                break;
            }
        }

        return new Vector<int>(selected.ToArray());
    }

    /// <summary>
    /// Selects examples using k-means++ style probabilistic selection.
    /// </summary>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// 1. Initialize: all labeled examples are "centers"
    /// 2. For each iteration:
    ///    - For each unlabeled example, compute D(x) = distance to nearest center
    ///    - Sample next example with probability proportional to D(x)^2
    ///    - Add sampled example to centers
    /// 3. Return k selected examples
    /// </para>
    /// <para><b>Difference from k-center:</b>
    /// - k-means++ uses probabilistic sampling instead of greedy max
    /// - Can avoid selecting outliers
    /// - More diverse selection in practice
    /// </para>
    /// </remarks>
    private Vector<int> SelectKMeansPlusPlus(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        int k,
        IDataset<T, TInput, TOutput>? labeledData)
    {
        // Extract features for all unlabeled examples
        var unlabeledFeatures = new List<Vector<T>>();
        for (int i = 0; i < unlabeledData.Count; i++)
        {
            var input = unlabeledData.GetInput(i);
            unlabeledFeatures.Add(ExtractFeatures(model, input));
        }

        // Initialize centers with labeled data features
        var centers = new List<Vector<T>>();
        if (labeledData != null)
        {
            for (int i = 0; i < labeledData.Count; i++)
            {
                var input = labeledData.GetInput(i);
                centers.Add(ExtractFeatures(model, input));
            }
        }

        var selected = new List<int>();
        var availableIndices = new HashSet<int>(Enumerable.Range(0, unlabeledData.Count));

        // Select k examples using probabilistic sampling
        for (int iteration = 0; iteration < k; iteration++)
        {
            if (availableIndices.Count == 0)
                break;

            // Compute D(x)^2 for each available unlabeled point
            var distances = new List<double>();
            var indices = new List<int>();

            foreach (var idx in availableIndices)
            {
                var dist = MinDistanceToCenters(unlabeledFeatures[idx], centers);
                var distSquared = Convert.ToDouble(dist) * Convert.ToDouble(dist);
                distances.Add(distSquared);
                indices.Add(idx);
            }

            // Sample proportional to distance squared
            int selectedIdx = WeightedRandomSample(indices, distances);
            selected.Add(selectedIdx);
            centers.Add(unlabeledFeatures[selectedIdx]);
            availableIndices.Remove(selectedIdx);
        }

        return new Vector<int>(selected.ToArray());
    }

    /// <summary>
    /// Samples an index with probability proportional to weights.
    /// </summary>
    private int WeightedRandomSample(List<int> indices, List<double> weights)
    {
        double totalWeight = weights.Sum();
        if (totalWeight <= double.Epsilon)
        {
            // All weights are zero, sample uniformly
            return indices[_random.Next(indices.Count)];
        }

        double sample = _random.NextDouble() * totalWeight;
        double cumulative = 0;

        for (int i = 0; i < indices.Count; i++)
        {
            cumulative += weights[i];
            if (sample <= cumulative)
                return indices[i];
        }

        // Fallback (shouldn't reach here due to floating point precision)
        return indices[indices.Count - 1];
    }

    /// <summary>
    /// Computes Euclidean distance between two feature vectors.
    /// </summary>
    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have same length");

        T sumSquaredDiff = NumOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }

        // Return sqrt(sum)
        double distance = Math.Sqrt(Convert.ToDouble(sumSquaredDiff));
        return NumOps.FromDouble(distance);
    }

    /// <summary>
    /// Computes minimum distance from a point to a set of centers.
    /// </summary>
    private T MinDistanceToCenters(Vector<T> point, List<Vector<T>> centers)
    {
        if (centers.Count == 0)
            return NumOps.FromDouble(double.MaxValue);

        T minDist = ComputeDistance(point, centers[0]);

        for (int i = 1; i < centers.Count; i++)
        {
            T dist = ComputeDistance(point, centers[i]);
            if (Convert.ToDouble(dist) < Convert.ToDouble(minDist))
                minDist = dist;
        }

        return minDist;
    }

    /// <summary>
    /// Extracts features from an input.
    /// </summary>
    private Vector<T> ExtractFeatures(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        if (_featureExtractor != null)
            return _featureExtractor(input);

        // Default: use model prediction as feature representation
        // This gives each input a unique feature vector based on model output
        var prediction = model.Predict(input);
        return ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
    }
}
