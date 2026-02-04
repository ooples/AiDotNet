using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.SurvivalAnalysis;

/// <summary>
/// Implements Random Survival Forest for survival analysis using ensemble of survival trees.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Random Survival Forest extends Random Forest to handle survival data.
/// Instead of predicting classes or values, it predicts survival curves. Each tree uses log-rank
/// statistics to find splits that maximize survival difference between groups.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Build many survival trees using bootstrap samples</item>
/// <item>At each node, select random features and find the split that maximizes log-rank statistic</item>
/// <item>Store survival estimates at each terminal node</item>
/// <item>Average survival curves from all trees for prediction</item>
/// </list>
/// </para>
///
/// <para><b>Key advantages:</b>
/// <list type="bullet">
/// <item>Handles non-linear relationships and interactions automatically</item>
/// <item>Provides variable importance through permutation</item>
/// <item>Robust to outliers and doesn't require proportional hazards assumption</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Ishwaran et al., "Random Survival Forests" (2008)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomSurvivalForest<T> : SurvivalModelBase<T>
{
    /// <summary>
    /// Gets the number of trees in the forest.
    /// </summary>
    public int NumTrees { get; }

    /// <summary>
    /// Gets the maximum depth of each tree.
    /// </summary>
    public int MaxDepth { get; }

    /// <summary>
    /// Gets the minimum samples per leaf.
    /// </summary>
    public int MinSamplesLeaf { get; }

    /// <summary>
    /// Gets the number of features to consider at each split.
    /// </summary>
    public int MaxFeatures { get; private set; }

    /// <summary>
    /// Random generator for reproducibility.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// The survival trees.
    /// </summary>
    private List<SurvivalTree>? _trees;

    /// <summary>
    /// Creates a new Random Survival Forest.
    /// </summary>
    /// <param name="numTrees">Number of trees (default: 100).</param>
    /// <param name="maxDepth">Maximum tree depth (default: 10).</param>
    /// <param name="minSamplesLeaf">Minimum samples per leaf (default: 6).</param>
    /// <param name="maxFeatures">Features to consider at each split (default: 0 = sqrt(p)).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public RandomSurvivalForest(
        int numTrees = 100,
        int maxDepth = 10,
        int minSamplesLeaf = 6,
        int maxFeatures = 0,
        int? seed = null) : base()
    {
        if (numTrees < 1)
            throw new ArgumentOutOfRangeException(nameof(numTrees), "Must have at least 1 tree.");
        if (maxDepth < 1)
            throw new ArgumentOutOfRangeException(nameof(maxDepth), "Max depth must be at least 1.");
        if (minSamplesLeaf < 1)
            throw new ArgumentOutOfRangeException(nameof(minSamplesLeaf), "Min samples per leaf must be at least 1.");

        NumTrees = numTrees;
        MaxDepth = maxDepth;
        MinSamplesLeaf = minSamplesLeaf;
        MaxFeatures = maxFeatures;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Fits the Random Survival Forest.
    /// </summary>
    protected override void FitSurvivalCore(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Set max features if not specified
        if (MaxFeatures <= 0)
            MaxFeatures = (int)Math.Ceiling(Math.Sqrt(p));
        MaxFeatures = Math.Min(MaxFeatures, p);

        // Store event times
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        // Build trees
        _trees = new List<SurvivalTree>();

        for (int t = 0; t < NumTrees; t++)
        {
            // Bootstrap sample
            var bootstrapIndices = new int[n];
            for (int i = 0; i < n; i++)
                bootstrapIndices[i] = _random.Next(n);

            // Build tree
            var tree = BuildTree(x, times, events, bootstrapIndices, 0);
            _trees.Add(tree);
        }

        // Compute baseline survival (average across all trees)
        if (TrainedEventTimes.Length > 0)
        {
            BaselineSurvivalFunction = new Vector<T>(TrainedEventTimes.Length);
            var allIndices = Enumerable.Range(0, n).ToArray();

            // Get average survival at each time point
            var survivalProbs = PredictSurvivalProbability(x, TrainedEventTimes);
            for (int t = 0; t < TrainedEventTimes.Length; t++)
            {
                double avgSurvival = 0;
                for (int i = 0; i < n; i++)
                    avgSurvival += NumOps.ToDouble(survivalProbs[i, t]);
                avgSurvival /= n;
                BaselineSurvivalFunction[t] = NumOps.FromDouble(avgSurvival);
            }
        }
    }

    /// <summary>
    /// Recursively builds a survival tree.
    /// </summary>
    private SurvivalTree BuildTree(Matrix<T> x, Vector<T> times, Vector<int> events, int[] indices, int depth)
    {
        int n = indices.Length;

        // Check stopping conditions
        if (depth >= MaxDepth || n < 2 * MinSamplesLeaf)
            return CreateLeafNode(times, events, indices);

        // Check if all events are the same
        int numEvents = indices.Count(i => events[i] == 1);
        if (numEvents == 0 || numEvents == n)
            return CreateLeafNode(times, events, indices);

        // Select random features
        var featureIndices = Enumerable.Range(0, x.Columns).OrderBy(_ => _random.Next()).Take(MaxFeatures).ToArray();

        // Find best split using log-rank statistic
        double bestLogRank = double.NegativeInfinity;
        int bestFeature = -1;
        double bestThreshold = 0;
        int[]? bestLeftIndices = null;
        int[]? bestRightIndices = null;

        foreach (int feature in featureIndices)
        {
            // Get unique values for this feature
            var values = indices.Select(i => NumOps.ToDouble(x[i, feature])).Distinct().OrderBy(v => v).ToArray();

            if (values.Length < 2)
                continue;

            // Try midpoints as thresholds
            for (int i = 0; i < values.Length - 1; i++)
            {
                double threshold = (values[i] + values[i + 1]) / 2;

                var leftIndices = indices.Where(idx => NumOps.ToDouble(x[idx, feature]) <= threshold).ToArray();
                var rightIndices = indices.Where(idx => NumOps.ToDouble(x[idx, feature]) > threshold).ToArray();

                if (leftIndices.Length < MinSamplesLeaf || rightIndices.Length < MinSamplesLeaf)
                    continue;

                // Compute log-rank statistic
                double logRank = ComputeLogRankStatistic(times, events, leftIndices, rightIndices);

                if (logRank > bestLogRank)
                {
                    bestLogRank = logRank;
                    bestFeature = feature;
                    bestThreshold = threshold;
                    bestLeftIndices = leftIndices;
                    bestRightIndices = rightIndices;
                }
            }
        }

        // If no valid split found, create leaf
        if (bestFeature < 0 || bestLeftIndices is null || bestRightIndices is null)
            return CreateLeafNode(times, events, indices);

        // Recursively build children
        var leftChild = BuildTree(x, times, events, bestLeftIndices, depth + 1);
        var rightChild = BuildTree(x, times, events, bestRightIndices, depth + 1);

        return new SurvivalTree
        {
            IsLeaf = false,
            SplitFeature = bestFeature,
            SplitThreshold = bestThreshold,
            Left = leftChild,
            Right = rightChild
        };
    }

    /// <summary>
    /// Creates a leaf node with Kaplan-Meier survival estimate.
    /// </summary>
    private SurvivalTree CreateLeafNode(Vector<T> times, Vector<int> events, int[] indices)
    {
        // Compute Kaplan-Meier estimate for this leaf
        var uniqueTimes = indices
            .Where(i => events[i] == 1)
            .Select(i => NumOps.ToDouble(times[i]))
            .Distinct()
            .OrderBy(t => t)
            .ToArray();

        var survivalTimes = new List<double>();
        var survivalProbs = new List<double>();

        double survival = 1.0;

        foreach (double t in uniqueTimes)
        {
            int atRisk = indices.Count(i => NumOps.ToDouble(times[i]) >= t);
            int eventsAtT = indices.Count(i => Math.Abs(NumOps.ToDouble(times[i]) - t) < 1e-10 && events[i] == 1);

            if (atRisk > 0)
                survival *= (double)(atRisk - eventsAtT) / atRisk;

            survivalTimes.Add(t);
            survivalProbs.Add(survival);
        }

        return new SurvivalTree
        {
            IsLeaf = true,
            SurvivalTimes = survivalTimes.ToArray(),
            SurvivalProbs = survivalProbs.ToArray()
        };
    }

    /// <summary>
    /// Computes the log-rank statistic for a split.
    /// </summary>
    private double ComputeLogRankStatistic(Vector<T> times, Vector<int> events, int[] leftIndices, int[] rightIndices)
    {
        // Get all event times
        var allEventTimes = leftIndices.Concat(rightIndices)
            .Where(i => events[i] == 1)
            .Select(i => NumOps.ToDouble(times[i]))
            .Distinct()
            .OrderBy(t => t)
            .ToArray();

        if (allEventTimes.Length == 0)
            return 0;

        double observed = 0;
        double expected = 0;
        double variance = 0;

        var leftSet = new HashSet<int>(leftIndices);

        foreach (double t in allEventTimes)
        {
            int leftAtRisk = leftIndices.Count(i => NumOps.ToDouble(times[i]) >= t);
            int rightAtRisk = rightIndices.Count(i => NumOps.ToDouble(times[i]) >= t);
            int totalAtRisk = leftAtRisk + rightAtRisk;

            if (totalAtRisk <= 1)
                continue;

            int leftEvents = leftIndices.Count(i => Math.Abs(NumOps.ToDouble(times[i]) - t) < 1e-10 && events[i] == 1);
            int rightEvents = rightIndices.Count(i => Math.Abs(NumOps.ToDouble(times[i]) - t) < 1e-10 && events[i] == 1);
            int totalEvents = leftEvents + rightEvents;

            // Expected events in left group under null
            double expectedLeft = (double)(leftAtRisk * totalEvents) / totalAtRisk;

            observed += leftEvents;
            expected += expectedLeft;

            // Variance contribution
            if (totalAtRisk > 1)
            {
                variance += (double)(leftAtRisk * rightAtRisk * totalEvents * (totalAtRisk - totalEvents))
                           / (totalAtRisk * totalAtRisk * (totalAtRisk - 1));
            }
        }

        if (variance < 1e-10)
            return 0;

        // Log-rank statistic
        return Math.Abs((observed - expected) / Math.Sqrt(variance));
    }

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        int numSubjects = x.Rows;
        var result = new Matrix<T>(numSubjects, times.Length);

        for (int i = 0; i < numSubjects; i++)
        {
            // Get survival from each tree and average
            var survivalCurve = new double[times.Length];

            foreach (var tree in _trees!)
            {
                var leafSurvival = GetLeafSurvival(tree, x, i);

                for (int t = 0; t < times.Length; t++)
                {
                    double queryTime = NumOps.ToDouble(times[t]);
                    survivalCurve[t] += InterpolateLeafSurvival(leafSurvival, queryTime);
                }
            }

            // Average across trees
            for (int t = 0; t < times.Length; t++)
                result[i, t] = NumOps.FromDouble(survivalCurve[t] / NumTrees);
        }

        return result;
    }

    /// <summary>
    /// Traverses tree to get leaf survival estimate.
    /// </summary>
    private SurvivalTree GetLeafSurvival(SurvivalTree node, Matrix<T> x, int sampleIdx)
    {
        if (node.IsLeaf)
            return node;

        double value = NumOps.ToDouble(x[sampleIdx, node.SplitFeature]);
        return value <= node.SplitThreshold
            ? GetLeafSurvival(node.Left!, x, sampleIdx)
            : GetLeafSurvival(node.Right!, x, sampleIdx);
    }

    /// <summary>
    /// Interpolates survival probability at a specific time from leaf node.
    /// </summary>
    private static double InterpolateLeafSurvival(SurvivalTree leaf, double queryTime)
    {
        if (leaf.SurvivalTimes is null || leaf.SurvivalProbs is null || leaf.SurvivalTimes.Length == 0)
            return 1.0;

        // Find the appropriate time point
        for (int i = leaf.SurvivalTimes.Length - 1; i >= 0; i--)
        {
            if (leaf.SurvivalTimes[i] <= queryTime)
                return leaf.SurvivalProbs[i];
        }

        return 1.0; // Before first event
    }

    /// <summary>
    /// Predicts mortality risk scores (higher = higher risk).
    /// </summary>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        // Use ensemble risk: 1 / median survival time
        var medianSurvival = Predict(x);
        var risk = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            double median = NumOps.ToDouble(medianSurvival[i]);
            risk[i] = NumOps.FromDouble(1.0 / Math.Max(1e-10, median));
        }

        return risk;
    }

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        if (BaselineSurvivalFunction is null || TrainedEventTimes is null)
            return new Vector<T>(times.Length);

        var result = new Vector<T>(times.Length);
        for (int t = 0; t < times.Length; t++)
        {
            double queryTime = NumOps.ToDouble(times[t]);

            // Find closest event time
            int idx = -1;
            for (int i = TrainedEventTimes.Length - 1; i >= 0; i--)
            {
                if (NumOps.ToDouble(TrainedEventTimes[i]) <= queryTime)
                {
                    idx = i;
                    break;
                }
            }

            result[t] = idx >= 0 ? BaselineSurvivalFunction[idx] : NumOps.One;
        }

        return result;
    }

    /// <summary>
    /// Predicts median survival time.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        EnsureFitted();

        if (TrainedEventTimes is null || TrainedEventTimes.Length == 0)
            return new Vector<T>(input.Rows);

        var survivalProbs = PredictSurvivalProbability(input, TrainedEventTimes);
        var result = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Find time when survival crosses 0.5
            double medianTime = NumOps.ToDouble(TrainedEventTimes[TrainedEventTimes.Length - 1]);

            for (int t = 0; t < TrainedEventTimes.Length - 1; t++)
            {
                double prob = NumOps.ToDouble(survivalProbs[i, t]);
                double nextProb = NumOps.ToDouble(survivalProbs[i, t + 1]);

                if (prob >= 0.5 && nextProb < 0.5)
                {
                    // Linear interpolation
                    double t1 = NumOps.ToDouble(TrainedEventTimes[t]);
                    double t2 = NumOps.ToDouble(TrainedEventTimes[t + 1]);
                    double fraction = (prob - 0.5) / (prob - nextProb);
                    medianTime = t1 + fraction * (t2 - t1);
                    break;
                }
            }

            result[i] = NumOps.FromDouble(medianTime);
        }

        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Random Forest parameters are complex - just return tree count
        return new Vector<T>(new[] { NumOps.FromDouble(NumTrees) });
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Parameters are read-only for Random Forest
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return new RandomSurvivalForest<T>(NumTrees, MaxDepth, MinSamplesLeaf, MaxFeatures);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new RandomSurvivalForest<T>(NumTrees, MaxDepth, MinSamplesLeaf, MaxFeatures);
    }

    /// <inheritdoc />
    public override ModelType GetModelType() => ModelType.RandomSurvivalForest;

    /// <summary>
    /// Internal survival tree node.
    /// </summary>
    private class SurvivalTree
    {
        public bool IsLeaf { get; set; }
        public int SplitFeature { get; set; }
        public double SplitThreshold { get; set; }
        public SurvivalTree? Left { get; set; }
        public SurvivalTree? Right { get; set; }
        public double[]? SurvivalTimes { get; set; }
        public double[]? SurvivalProbs { get; set; }
    }
}
