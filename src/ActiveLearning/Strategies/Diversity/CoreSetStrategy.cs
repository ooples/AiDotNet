using AiDotNet.ActiveLearning.Config;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.Clustering)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Active Learning for Convolutional Neural Networks: A Core-Set Approach", "https://arxiv.org/abs/1708.00489", Year = 2018, Authors = "Ozan Sener, Silvio Savarese")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class CoreSetStrategy<T, TInput, TOutput> : IDiversityStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly T CosineEpsilon = NumOps.FromDouble(1e-10);

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

        // Extract features for unlabeled samples
        var unlabeledFeatures = ExtractFeatures(model, unlabeledPool);

        if (_labeledFeatures == null || _labeledFeatures.Count == 0)
        {
            // No labeled samples yet - all samples get equal high scores
            return AiDotNetEngine.Current.Fill(unlabeledPool.Count, NumOps.One);
        }

        return MinDistances(Stack(unlabeledFeatures), Stack(_labeledFeatures)).ToVector();
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

        // Use greedy furthest-first traversal (k-center greedy), initialized with the labeled features
        return KCenterGreedy(unlabeledFeatures, _labeledFeatures, batchSizeToUse);
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

        int k = Math.Min(5, pool.Count - 1);

        if (k == 0)
        {
            // Only one sample, give it weight 1
            return new Vector<T>(new[] { NumOps.One });
        }

        // Extract features for all samples
        var features = new List<Vector<T>>();
        for (int i = 0; i < pool.Count; i++)
        {
            features.Add(GetFeatureRepresentation(pool.GetInput(i)));
        }

        // Compute density for each sample from the sum of its k nearest-neighbor distances (self excluded)
        var engine = AiDotNetEngine.Current;
        var rows = Stack(features);
        int n = rows.Shape[0];
        int blockRows = ActiveLearningEngineOps.PairwiseBlockRows(n, rows.Shape[1]);
        var nearestSums = new List<Tensor<T>>();
        for (int start = 0; start < n; start += blockRows)
        {
            int count = Math.Min(blockRows, n - start);
            var block = engine.TensorSlice(rows, new[] { start, 0 }, new[] { count, rows.Shape[1] });
            var others = engine.TensorMaskedFill(CrossDistances(block, rows), ActiveLearningEngineOps.SelfMask(start, count, n), NumOps.MaxValue);
            var (nearest, _) = engine.TopK(others, k, 1, false);
            nearestSums.Add(engine.ReduceSum(nearest, new[] { 1 }, keepDims: false));
        }

        var sums = nearestSums.Count == 1 ? nearestSums[0] : engine.TensorConcatenate(nearestSums.ToArray(), 0);

        // Density weight = 1 / (avgDistance + epsilon) for stability
        // Higher density (smaller distance) = higher weight
        var avgDistance = engine.TensorDivideScalar(sums, NumOps.FromDouble(k));
        var weights = engine.TensorReciprocal(engine.TensorAddScalar(avgDistance, NumOps.FromDouble(1e-6)));

        // Normalize weights to sum to 1
        T totalWeight = engine.TensorSum(weights);

        if (NumOps.Compare(totalWeight, NumOps.Zero) > 0)
        {
            weights = engine.TensorDivideScalar(weights, totalWeight);
        }

        return weights.ToVector();
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
        return KCenterGreedy(unlabeledFeatures, coveredPoints, batchSizeToUse);
    }

    /// <inheritdoc/>
    public Vector<T> ComputeDiversityScores(
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData)
    {
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
            return AiDotNetEngine.Current.Fill(unlabeledData.Count, NumOps.One);
        }

        var unlabeledFeatures = new List<Vector<T>>();
        for (int i = 0; i < unlabeledData.Count; i++)
        {
            unlabeledFeatures.Add(GetFeatureRepresentation(unlabeledData.GetInput(i)));
        }

        return MinDistances(Stack(unlabeledFeatures), Stack(labeledFeatures)).ToVector();
    }

    /// <inheritdoc/>
    public Matrix<T> ComputeDistanceMatrix(IDataset<T, TInput, TOutput> dataset)
    {
        int n = dataset.Count;

        // Extract all features first
        var features = new List<Vector<T>>();
        for (int i = 0; i < n; i++)
        {
            features.Add(GetFeatureRepresentation(dataset.GetInput(i)));
        }

        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // Compute pairwise distances; the diagonal is exactly zero
        var engine = AiDotNetEngine.Current;
        var rows = Stack(features);
        var all = engine.TensorMaskedFill(CrossDistances(rows, rows), ActiveLearningEngineOps.SelfMask(0, n, n), NumOps.Zero);
        return Matrix<T>.FromMemory(all.ToArray(), n, n);
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

    /// <summary>
    /// Greedy furthest-first traversal (k-center greedy): repeatedly selects the sample whose minimum
    /// distance to the covered points (initial points plus selections) is largest, first index on ties.
    /// </summary>
    private int[] KCenterGreedy(List<Vector<T>> unlabeledFeatures, List<Vector<T>>? initialCovered, int batchSize)
    {
        int n = unlabeledFeatures.Count;
        if (n == 0 || batchSize <= 0)
        {
            return Array.Empty<int>();
        }

        var engine = AiDotNetEngine.Current;
        var rows = Stack(unlabeledFeatures);
        int dim = rows.Shape[1];
        var minDistance = initialCovered is { Count: > 0 }
            ? MinDistances(rows, Stack(initialCovered))
            : Filled(n, NumOps.MaxValue);
        var selectedIndices = new List<int>();
        var isSelected = new bool[n];

        for (int b = 0; b < batchSize; b++)
        {
            int bestIndex = ActiveLearningEngineOps.ArgMaxAbove(minDistance.Reshape(new[] { 1, n }), isSelected, NumOps.MinValue);
            if (bestIndex < 0)
            {
                break;
            }

            selectedIndices.Add(bestIndex);
            isSelected[bestIndex] = true;

            var newest = engine.TensorSlice(rows, new[] { bestIndex, 0 }, new[] { 1, dim });
            minDistance = engine.TensorMin(minDistance, CrossDistances(rows, newest).Reshape(new[] { n }));
        }

        return selectedIndices.ToArray();
    }

    /// <summary>
    /// Minimum distance from every row of <paramref name="points"/> to the rows of
    /// <paramref name="covered"/>, capped at MaxValue (the value used when nothing is covered).
    /// </summary>
    /// <returns>A [m] tensor.</returns>
    private Tensor<T> MinDistances(Tensor<T> points, Tensor<T> covered)
    {
        var engine = AiDotNetEngine.Current;
        int m = points.Shape[0];
        int k = covered.Shape[0];
        if (k == 0)
        {
            return Filled(m, NumOps.MaxValue);
        }

        int blockRows = ActiveLearningEngineOps.BlockRows(m, (long)k * Math.Max(1, points.Shape[1]));
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < m; start += blockRows)
        {
            int count = Math.Min(blockRows, m - start);
            var block = engine.TensorSlice(points, new[] { start, 0 }, new[] { count, points.Shape[1] });
            blocks.Add(engine.TensorAmin(CrossDistances(block, covered), new[] { 1 }, keepDims: false));
        }

        var minimum = blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
        return engine.TensorMin(minimum, NumOps.MaxValue);
    }

    /// <summary>
    /// Distances under the configured metric between every row of a [m, d] and a [k, d] tensor.
    /// </summary>
    /// <returns>A [m, k] tensor.</returns>
    private Tensor<T> CrossDistances(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        int m = a.Shape[0];
        int k = b.Shape[0];
        int d = a.Shape[1];
        if (d != b.Shape[1])
        {
            throw new ArgumentException($"Vectors must have the same length. Got {d} and {b.Shape[1]}.", nameof(b));
        }

        switch (_distanceMetric)
        {
            case DistanceMetric.Cosine:
            {
                // 1 - clamp(a.b / sqrt(|a|^2 |b|^2), -1, 1); similarity 0 when the denominator is ~0
                var shape = new[] { m, k };
                var dot = d == 0 ? new Tensor<T>(shape) : engine.TensorMatMul(a, engine.TensorTranspose(b));
                var normA = engine.TensorBroadcastTo(SquaredNorms(a).Reshape(new[] { m, 1 }), shape);
                var normB = engine.TensorBroadcastTo(SquaredNorms(b).Reshape(new[] { 1, k }), shape);
                var denominator = engine.TensorSqrt(engine.TensorMultiply(normA, normB));
                var similarity = engine.TensorMax(engine.TensorMin(engine.TensorDivide(dot, denominator), NumOps.One), NumOps.Negate(NumOps.One));
                var valid = engine.TensorGreaterThan(denominator, CosineEpsilon);
                similarity = engine.TensorWhere(valid, similarity, new Tensor<T>(shape));
                return engine.ScalarMinusTensor(NumOps.One, similarity);
            }

            case DistanceMetric.Manhattan:
            {
                if (d == 0 || m == 0 || k == 0)
                {
                    return new Tensor<T>(new[] { m, k });
                }

                var shape = new[] { m, k, d };
                var left = engine.TensorBroadcastTo(a.Reshape(new[] { m, 1, d }), shape);
                var right = engine.TensorBroadcastTo(b.Reshape(new[] { 1, k, d }), shape);
                return engine.ReduceSum(engine.TensorAbs(engine.TensorSubtract(left, right)), new[] { 2 }, keepDims: false);
            }

            default:
                return engine.TensorSqrt(ActiveLearningEngineOps.SquaredDistances(a, b));
        }
    }

    private static Tensor<T> SquaredNorms(Tensor<T> rows)
    {
        var engine = AiDotNetEngine.Current;
        return rows.Shape[1] == 0
            ? new Tensor<T>(new[] { rows.Shape[0] })
            : engine.ReduceSum(engine.TensorMultiply(rows, rows), new[] { 1 }, keepDims: false);
    }

    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vectors must have the same length. Got {a.Length} and {b.Length}.", nameof(b));
        }

        var left = Tensor<T>.FromVector(a).Reshape(new[] { 1, a.Length });
        var right = Tensor<T>.FromVector(b).Reshape(new[] { 1, b.Length });
        return CrossDistances(left, right).GetFlat(0);
    }

    private static Tensor<T> Stack(List<Vector<T>> features)
    {
        return features.Count == 0
            ? new Tensor<T>(new[] { 0, 0 })
            : ActiveLearningEngineOps.StackVectors(features);
    }

    private static Tensor<T> Filled(int n, T value)
    {
        var filled = new Tensor<T>(new[] { n });
        filled.Fill(value);
        return filled;
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
[AiDotNet.Configuration.YamlConfigurable("ALFeatureExtractor")]
public interface IFeatureExtractor<T, TInput>
{
    /// <summary>
    /// Extracts feature representation from an input.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <returns>Feature vector representation.</returns>
    Vector<T> ExtractFeatures(TInput input);
}
