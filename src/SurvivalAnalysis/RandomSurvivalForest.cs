using System.Text;
using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// // age and treatment arm per patient
/// var features = new Matrix&lt;double&gt;(new double[,]
/// {
///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
/// });
/// // months each patient was observed
/// var times = new Vector&lt;double&gt;(new double[] { 5.0, 12.0, 3.0, 18.0, 9.0, 21.0 });
/// // 1 = the event happened; 0 = censored, still fine when the study ended or lost to follow-up
/// var events = new Vector&lt;int&gt;(new int[] { 1, 0, 1, 0, 1, 1 });
///
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new RandomSurvivalForest&lt;double&gt;(numTrees: 100, maxDepth: 10))
///     .Build(features, times, events);
///
/// var risk = result.Predict(features);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.SurvivalModel)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Random Survival Forests", "https://doi.org/10.1214/08-AOAS169", Year = 2008, Authors = "Hemant Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, Michael S. Lauer")]
public partial class RandomSurvivalForest<T> : SurvivalModelBase<T>
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
    /// Gets whether each tree is grown on a bootstrap sample (true, as in Ishwaran et al.) or on the full data.
    /// </summary>
    public bool Bootstrap { get; }

    /// <summary>
    /// Random generator for reproducibility.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// The survival trees.
    /// </summary>
    [FittedParameter]
    private List<SurvivalTree>? _trees;

    /// <summary>
    /// Nelson-Aalen cumulative hazard of every terminal node at the training event times, one row per leaf.
    /// </summary>
    [FittedParameter]
    private Matrix<T>? _leafCumulativeHazards;

    /// <summary>
    /// Kaplan-Meier survival of every terminal node at the training event times, one row per leaf.
    /// </summary>
    [FittedParameter]
    private Matrix<T>? _leafSurvival;

    /// <summary>
    /// Creates a new Random Survival Forest.
    /// </summary>
    /// <param name="numTrees">Number of trees (default: 100).</param>
    /// <param name="maxDepth">Maximum tree depth (default: 10).</param>
    /// <param name="minSamplesLeaf">Minimum samples per leaf (default: 6).</param>
    /// <param name="maxFeatures">Features to consider at each split (default: 0 = sqrt(p)).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="bootstrap">Grow each tree on a bootstrap sample (default: true, as in Ishwaran et al.).</param>
    public RandomSurvivalForest(
        int numTrees = 100,
        int maxDepth = 10,
        int minSamplesLeaf = 6,
        int maxFeatures = 0,
        int? seed = null,
        bool bootstrap = true) : base()
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
        Bootstrap = bootstrap;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Fits the Random Survival Forest.
    /// </summary>
    protected override void FitSurvivalCore(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Rows;
        int p = x.Columns;

        // Set max features if not specified
        if (MaxFeatures <= 0)
            MaxFeatures = (int)Math.Ceiling(Math.Sqrt(p));
        MaxFeatures = Math.Min(MaxFeatures, p);

        var features = Tensor<T>.FromMatrix(x);
        var timeTensor = times.AsTensor();
        var eventTensor = events.AsTensor().Cast<T>();

        // Store event times (sorted, unique, computed in T so they compare exactly with the data)
        TrainedEventTimes = UniqueEventTimes(engine, timeTensor, eventTensor).ToVector();

        // Build trees
        _trees = new List<SurvivalTree>();
        var leafHazards = new List<Tensor<T>>();
        var leafSurvival = new List<Tensor<T>>();

        for (int t = 0; t < NumTrees; t++)
        {
            var sampleX = features;
            var sampleTimes = timeTensor;
            var sampleEvents = eventTensor;

            if (Bootstrap)
            {
                // Bootstrap sample
                var bootstrapIndices = new int[n];
                for (int i = 0; i < n; i++)
                    bootstrapIndices[i] = _random.Next(n);

                var rows = new Tensor<int>(bootstrapIndices, new[] { n });
                sampleX = engine.TensorIndexSelect(features, rows, 0);
                sampleTimes = engine.TensorIndexSelect(timeTensor, rows, 0);
                sampleEvents = engine.TensorIndexSelect(eventTensor, rows, 0);
            }

            // Build tree
            var tree = BuildTree(engine, sampleX, sampleTimes, sampleEvents, 0, leafHazards, leafSurvival);
            _trees.Add(tree);
        }

        if (TrainedEventTimes.Length > 0)
        {
            _leafCumulativeHazards = engine.TensorStack(leafHazards.ToArray(), 0).ToMatrix();
            _leafSurvival = engine.TensorStack(leafSurvival.ToArray(), 0).ToMatrix();

            // Baseline survival: the ensemble survival averaged over the training subjects
            // Use internal method to avoid EnsureFitted() check during fitting
            var survivalProbs = Tensor<T>.FromMatrix(PredictSurvivalProbabilityInternal(x, TrainedEventTimes));
            BaselineSurvivalFunction = engine.ReduceMean(survivalProbs, new[] { 0 }, keepDims: false).ToVector();
        }
        else
        {
            _leafCumulativeHazards = null;
            _leafSurvival = null;
        }
    }

    /// <summary>
    /// Recursively builds a survival tree on the node's samples.
    /// </summary>
    /// <remarks>
    /// A node is split on the feature and threshold that maximise the two-sample log-rank statistic
    /// (Ishwaran et al. 2008, Section 3; the statistic scikit-survival's LogrankCriterion uses). A node
    /// whose candidate splits all have zero log-rank variance (for example, all censored) is a leaf.
    /// </remarks>
    private SurvivalTree BuildTree(
        IEngine engine, Tensor<T> x, Tensor<T> times, Tensor<T> events, int depth,
        List<Tensor<T>> leafHazards, List<Tensor<T>> leafSurvival)
    {
        int n = x.Shape[0];

        // Check stopping conditions
        if (depth >= MaxDepth || n < 2 * MinSamplesLeaf)
            return CreateLeafNode(engine, times, events, leafHazards, leafSurvival);

        // A node without events cannot be split by the log-rank statistic
        if (!NumOps.GreaterThan(engine.TensorSum(events), NumOps.Zero))
            return CreateLeafNode(engine, times, events, leafHazards, leafSurvival);

        // Select random features
        var featureIndices = Enumerable.Range(0, x.Shape[1]).OrderBy(_ => _random.Next()).Take(MaxFeatures).ToArray();

        // Find best split using log-rank statistic
        var split = FindBestSplit(engine, x, times, events, featureIndices);

        // If no valid split found, create leaf
        if (split is null)
            return CreateLeafNode(engine, times, events, leafHazards, leafSurvival);

        var (feature, threshold) = split.Value;
        var column = engine.TensorSelect(x, 1, feature);
        var goesLeft = engine.TensorLessEqual(column, Tensor<T>.CreateDefault(new[] { n }, threshold));
        var leftRows = Positions(engine, goesLeft);
        var rightRows = Positions(engine, engine.TensorSubtract(Tensor<T>.CreateDefault(new[] { n }, NumOps.One), goesLeft));

        // Recursively build children
        var leftChild = BuildTree(
            engine,
            engine.TensorIndexSelect(x, leftRows, 0),
            engine.TensorIndexSelect(times, leftRows, 0),
            engine.TensorIndexSelect(events, leftRows, 0),
            depth + 1,
            leafHazards,
            leafSurvival);
        var rightChild = BuildTree(
            engine,
            engine.TensorIndexSelect(x, rightRows, 0),
            engine.TensorIndexSelect(times, rightRows, 0),
            engine.TensorIndexSelect(events, rightRows, 0),
            depth + 1,
            leafHazards,
            leafSurvival);

        return new SurvivalTree
        {
            IsLeaf = false,
            SplitFeature = feature,
            SplitThreshold = threshold,
            Left = leftChild,
            Right = rightChild
        };
    }

    /// <summary>
    /// Finds the split with the largest log-rank statistic over the candidate features.
    /// </summary>
    /// <returns>The feature and threshold, or null when no split is admissible.</returns>
    /// <remarks>
    /// For every feature the node is sorted by the feature value, and the risk-set and death counts of
    /// each candidate left child come from cumulative sums over the sorted samples, so all thresholds
    /// (midpoints between distinct values) are scored at once. Candidates are compared in feature order
    /// and then threshold order, and the first maximum wins.
    /// </remarks>
    private (int Feature, T Threshold)? FindBestSplit(
        IEngine engine, Tensor<T> x, Tensor<T> times, Tensor<T> events, int[] featureIndices)
    {
        int n = x.Shape[0];
        int candidates = n - 1;

        // Event times of this node; other times contribute nothing to the statistic.
        var nodeEventTimes = UniqueEventTimes(engine, times, events);
        int k = nodeEventTimes.Length;
        int[] grid = { n, k };
        var eventGrid = engine.TensorBroadcastTo(engine.Reshape(nodeEventTimes, new[] { 1, k }), grid);

        // Left-child sizes 1..n-1 that keep both children at MinSamplesLeaf or more.
        var leftSizes = engine.TensorArange<T>(1, n, 1);
        var sizeAllowed = engine.TensorMultiply(
            engine.TensorGreaterThan(leftSizes, NumOps.FromDouble(MinSamplesLeaf - 0.5)),
            engine.TensorLessThan(leftSizes, NumOps.FromDouble(n - MinSamplesLeaf + 0.5)));

        var scores = new List<Tensor<T>>();
        var sortedColumns = new List<Tensor<T>>();
        foreach (int feature in featureIndices)
        {
            var (sortedValues, order) = engine.TensorSort(engine.TensorSelect(x, 1, feature), 0, false);
            sortedColumns.Add(sortedValues);

            var sortedTimes = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorIndexSelect(times, order, 0), new[] { n, 1 }), grid);
            var sortedEvents = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorIndexSelect(events, order, 0), new[] { n, 1 }), grid);

            // Risk set and deaths of the first i + 1 sorted samples at every node event time.
            var cumulativeRisk = engine.TensorCumSum(engine.TensorGreaterEqual(sortedTimes, eventGrid), 0);
            var cumulativeDeaths = engine.TensorCumSum(
                engine.TensorMultiply(engine.TensorEquals(sortedTimes, eventGrid), sortedEvents), 0);

            int[] candidateGrid = { candidates, k };
            var atRisk = engine.TensorBroadcastTo(engine.TensorNarrow(cumulativeRisk, 0, n - 1, 1), candidateGrid);
            var deaths = engine.TensorBroadcastTo(engine.TensorNarrow(cumulativeDeaths, 0, n - 1, 1), candidateGrid);
            var leftAtRisk = engine.TensorNarrow(cumulativeRisk, 0, 0, candidates);
            var leftDeaths = engine.TensorNarrow(cumulativeDeaths, 0, 0, candidates);

            // |Z| where the variance is positive, the threshold separates distinct values and both
            // children are large enough; -1 marks an inadmissible candidate.
            var (statistic, hasVariance) = LogRankStatistic(engine, leftAtRisk, leftDeaths, atRisk, deaths);
            var distinct = engine.TensorGreaterThan(engine.TensorDiff(sortedValues, 1, 0), NumOps.Zero);
            var admissible = engine.TensorMultiply(engine.TensorMultiply(hasVariance, distinct), sizeAllowed);
            scores.Add(engine.TensorWhere(
                admissible, statistic, Tensor<T>.CreateDefault(new[] { candidates }, NumOps.FromDouble(-1.0))));
        }

        var allScores = engine.Concat(scores, 0);
        int best = engine.TensorArgMax(allScores, 0).ToArray()[0];
        if (NumOps.LessThan(allScores.ToArray()[best], NumOps.Zero))
            return null;

        int featurePosition = best / candidates;
        int leftSize = best % candidates + 1;
        T[] bestValues = sortedColumns[featurePosition].ToArray();
        T threshold = NumOps.Divide(
            NumOps.Add(bestValues[leftSize - 1], bestValues[leftSize]), NumOps.FromDouble(2.0));

        return (featureIndices[featurePosition], threshold);
    }

    /// <summary>
    /// Two-sample log-rank statistic |Z| of candidate left children from their risk-set counts.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="leftAtRisk">Y_L, the number at risk in the left child at each event time, [c, k].</param>
    /// <param name="leftDeaths">d_L, the deaths in the left child at each event time, [c, k].</param>
    /// <param name="atRisk">Y, the number at risk in the node, [c, k].</param>
    /// <param name="deaths">d, the deaths in the node, [c, k].</param>
    /// <returns>|Z| = |sum(d_L - d Y_L / Y)| / sqrt(sum(Y_L/Y (1 - Y_L/Y) (Y - d)/(Y - 1) d)) per candidate
    /// (0 where that variance is zero), and 1 where the variance is positive.</returns>
    private (Tensor<T> Statistic, Tensor<T> HasVariance) LogRankStatistic(
        IEngine engine, Tensor<T> leftAtRisk, Tensor<T> leftDeaths, Tensor<T> atRisk, Tensor<T> deaths)
    {
        T one = NumOps.One;
        int candidates = atRisk.Shape[0];
        int[] candidateGrid = { candidates, atRisk.Shape[1] };

        var ratio = engine.TensorDivide(leftAtRisk, engine.TensorClampMin(atRisk, one));
        var numerator = engine.ReduceSum(
            engine.TensorSubtract(leftDeaths, engine.TensorMultiply(deaths, ratio)), new[] { 1 }, keepDims: false);
        var tieFactor = engine.TensorMultiply(
            engine.TensorDivide(
                engine.TensorMultiply(engine.TensorSubtract(atRisk, deaths), deaths),
                engine.TensorClampMin(engine.TensorSubtractScalar(atRisk, one), one)),
            engine.TensorGreaterThan(atRisk, one));
        var spread = engine.TensorMultiply(
            ratio, engine.TensorSubtract(Tensor<T>.CreateDefault(candidateGrid, one), ratio));
        var variance = engine.ReduceSum(engine.TensorMultiply(spread, tieFactor), new[] { 1 }, keepDims: false);

        var hasVariance = engine.TensorGreaterThan(variance, NumOps.Zero);
        var safeVariance = engine.TensorAdd(
            engine.TensorMultiply(variance, hasVariance),
            engine.TensorSubtract(Tensor<T>.CreateDefault(new[] { candidates }, one), hasVariance));
        var statistic = engine.TensorMultiply(
            engine.TensorDivide(engine.TensorAbs(numerator), engine.TensorSqrt(safeVariance)), hasVariance);
        return (statistic, hasVariance);
    }

    /// <summary>
    /// Computes the log-rank statistic |Z| for a split of the given subjects into two groups
    /// (0 when the split carries no information, for example when every subject is censored).
    /// </summary>
    private T ComputeLogRankStatistic(Vector<T> times, Vector<int> events, int[] leftIndices, int[] rightIndices)
    {
        var engine = AiDotNetEngine.Current;
        int left = leftIndices.Length;
        int total = left + rightIndices.Length;
        var members = new Tensor<int>(leftIndices.Concat(rightIndices).ToArray(), new[] { total });
        var nodeTimes = engine.TensorIndexSelect(times.AsTensor(), members, 0);
        var nodeEvents = engine.TensorIndexSelect(events.AsTensor().Cast<T>(), members, 0);
        var eventTimes = UniqueEventTimes(engine, nodeTimes, nodeEvents);
        if (eventTimes.Length == 0)
            return NumOps.Zero;

        var inLeft = engine.Concat(
            new[] { Tensor<T>.CreateDefault(new[] { left }, NumOps.One), new Tensor<T>(new[] { total - left }) }, 0);
        var (atRisk, deaths) = RiskSetCounts(engine, nodeTimes, nodeEvents, eventTimes);
        var (leftAtRisk, leftDeaths) = RiskSetCounts(
            engine, nodeTimes, engine.TensorMultiply(nodeEvents, inLeft), eventTimes, inLeft);
        int k = eventTimes.Length;
        var (statistic, _) = LogRankStatistic(
            engine,
            engine.Reshape(leftAtRisk, new[] { 1, k }),
            engine.Reshape(leftDeaths, new[] { 1, k }),
            engine.Reshape(atRisk, new[] { 1, k }),
            engine.Reshape(deaths, new[] { 1, k }));
        return statistic.ToArray()[0];
    }

    /// <summary>
    /// Creates a leaf node with its Nelson-Aalen cumulative hazard and Kaplan-Meier survival
    /// at the training event times (Ishwaran et al. 2008, equation 2).
    /// </summary>
    private SurvivalTree CreateLeafNode(
        IEngine engine, Tensor<T> times, Tensor<T> events,
        List<Tensor<T>> leafHazards, List<Tensor<T>> leafSurvival)
    {
        var eventTimes = TrainedEventTimes;
        if (eventTimes is null || eventTimes.Length == 0)
            return new SurvivalTree { IsLeaf = true, LeafIndex = -1, SplitThreshold = NumOps.Zero };

        var (atRisk, deaths) = RiskSetCounts(engine, times, events, eventTimes.AsTensor());
        var hazardIncrements = engine.TensorDivide(deaths, engine.TensorClampMin(atRisk, NumOps.One));
        leafHazards.Add(engine.TensorCumSum(hazardIncrements, 0));
        leafSurvival.Add(engine.TensorCumProd(
            engine.TensorSubtract(Tensor<T>.CreateDefault(new[] { eventTimes.Length }, NumOps.One), hazardIncrements), 0));

        return new SurvivalTree { IsLeaf = true, LeafIndex = leafHazards.Count - 1, SplitThreshold = NumOps.Zero };
    }

    /// <summary>
    /// Number at risk (time &gt;= t) and number of deaths (time == t with an event) at each time t,
    /// optionally counting only the subjects whose <paramref name="members"/> entry is 1.
    /// </summary>
    private (Tensor<T> AtRisk, Tensor<T> Deaths) RiskSetCounts(
        IEngine engine, Tensor<T> times, Tensor<T> events, Tensor<T> at, Tensor<T>? members = null)
    {
        int n = times.Length;
        int k = at.Length;
        int[] grid = { n, k };
        var timeGrid = engine.TensorBroadcastTo(engine.Reshape(times, new[] { n, 1 }), grid);
        var atGrid = engine.TensorBroadcastTo(engine.Reshape(at, new[] { 1, k }), grid);
        var eventGrid = engine.TensorBroadcastTo(engine.Reshape(events, new[] { n, 1 }), grid);

        var atRiskIndicator = engine.TensorGreaterEqual(timeGrid, atGrid);
        if (members is not null)
            atRiskIndicator = engine.TensorMultiply(
                atRiskIndicator, engine.TensorBroadcastTo(engine.Reshape(members, new[] { n, 1 }), grid));

        var atRisk = engine.ReduceSum(atRiskIndicator, new[] { 0 }, keepDims: false);
        var deaths = engine.ReduceSum(
            engine.TensorMultiply(engine.TensorEquals(timeGrid, atGrid), eventGrid), new[] { 0 }, keepDims: false);
        return (atRisk, deaths);
    }

    /// <summary>
    /// Sorted unique times of the observed events.
    /// </summary>
    private static Tensor<T> UniqueEventTimes(IEngine engine, Tensor<T> times, Tensor<T> events)
    {
        var observed = engine.TensorIndexSelect(times, Positions(engine, events), 0);
        if (observed.Length == 0)
            return observed;

        var sorted = engine.TensorSort(observed, 0, false).Values;
        var isNew = engine.Concat(
            new[]
            {
                Tensor<T>.CreateDefault(new[] { 1 }, MathHelper.GetNumericOperations<T>().One),
                engine.TensorNotEquals(engine.TensorDiff(sorted, 1, 0), MathHelper.GetNumericOperations<T>().Zero),
            },
            0);
        return engine.TensorIndexSelect(sorted, Positions(engine, isNew), 0);
    }

    /// <summary>
    /// Positions of the non-zero entries of a 1-D tensor.
    /// </summary>
    private static Tensor<int> Positions(IEngine engine, Tensor<T> mask)
    {
        var coordinates = engine.TensorNonzero(mask);
        return engine.Reshape(coordinates, new[] { coordinates.Shape[0] });
    }

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        // Accepts either the covariates alone or the [event | covariates] design matrix Train takes.
        // The trees split on training-covariate indices, so an extra leading column shifts every
        // split onto the wrong feature. Predict and PredictHazardRatio both route through here.
        return PredictSurvivalProbabilityInternal(ExtractCovariates(x), times);
    }

    /// <summary>
    /// Predicts the ensemble cumulative hazard H_e(t | x) at the specified times.
    /// </summary>
    /// <remarks>
    /// <para>The ensemble cumulative hazard is the average over the trees of the Nelson-Aalen estimate in
    /// the terminal node the subject falls into (Ishwaran et al. 2008, equation 3).</para>
    /// </remarks>
    public override Matrix<T> PredictCumulativeHazard(Vector<T> times, Matrix<T>? features = null)
    {
        EnsureFitted();

        var covariates = features is null
            ? new Matrix<T>(1, NumFeatures > 0 ? NumFeatures : 1)
            : ExtractCovariates(features);
        return StepLookup(EnsembleCurves(covariates).Hazard, times, NumOps.Zero);
    }

    private Matrix<T> PredictSurvivalProbabilityInternal(Matrix<T> x, Vector<T> times)
    {
        // Average of the terminal-node Kaplan-Meier curves, as a step function of time.
        return StepLookup(EnsembleCurves(x).Survival, times, NumOps.One);
    }

    /// <summary>
    /// Ensemble cumulative hazard and survival of each subject at the training event times, [n, K] each.
    /// </summary>
    private (Tensor<T> Hazard, Tensor<T> Survival) EnsembleCurves(Matrix<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var trees = _trees ?? throw new InvalidOperationException("_trees is null where it is required.");
        int n = x.Rows;
        int k = TrainedEventTimes?.Length ?? 0;
        var hazardSum = new Tensor<T>(new[] { n, k });
        var survivalSum = new Tensor<T>(new[] { n, k });

        if (k == 0 || n == 0 || _leafCumulativeHazards is null || _leafSurvival is null)
            return (hazardSum, Tensor<T>.CreateDefault(new[] { n, k }, NumOps.One));

        var features = Tensor<T>.FromMatrix(x);
        var leafHazards = Tensor<T>.FromMatrix(_leafCumulativeHazards);
        var leafSurvival = Tensor<T>.FromMatrix(_leafSurvival);
        var allRows = engine.TensorArange<T>(0, n, 1).Cast<int>();

        foreach (var tree in trees)
        {
            var parts = new List<(Tensor<int> Rows, int Leaf)>();
            Route(engine, tree, features, allRows, parts);

            var rows = engine.Concat(parts.Select(part => part.Rows).ToList(), 0);
            var leaves = engine.Concat(
                parts.Select(part => Tensor<int>.CreateDefault(new[] { part.Rows.Length }, part.Leaf)).ToList(), 0);
            hazardSum = engine.TensorIndexAdd(hazardSum, 0, rows, engine.TensorIndexSelect(leafHazards, leaves, 0));
            survivalSum = engine.TensorIndexAdd(survivalSum, 0, rows, engine.TensorIndexSelect(leafSurvival, leaves, 0));
        }

        T treeCount = NumOps.FromDouble(trees.Count);
        return (engine.TensorDivideScalar(hazardSum, treeCount), engine.TensorDivideScalar(survivalSum, treeCount));
    }

    /// <summary>
    /// Sends the given rows down a tree and records which leaf each group of rows reaches.
    /// </summary>
    private void Route(IEngine engine, SurvivalTree node, Tensor<T> x, Tensor<int> rows, List<(Tensor<int> Rows, int Leaf)> parts)
    {
        if (rows.Length == 0)
            return;

        if (node.IsLeaf)
        {
            parts.Add((rows, node.LeafIndex));
            return;
        }

        var values = engine.TensorIndexSelect(engine.TensorSelect(x, 1, node.SplitFeature), rows, 0);
        var goesLeft = engine.TensorLessEqual(values, Tensor<T>.CreateDefault(new[] { rows.Length }, node.SplitThreshold));
        var goesRight = engine.TensorSubtract(Tensor<T>.CreateDefault(new[] { rows.Length }, NumOps.One), goesLeft);

        Route(engine, node.Left ?? throw new InvalidOperationException("node.Left is null where it is required."),
            x, engine.TensorIndexSelect(rows, Positions(engine, goesLeft), 0), parts);
        Route(engine, node.Right ?? throw new InvalidOperationException("node.Right is null where it is required."),
            x, engine.TensorIndexSelect(rows, Positions(engine, goesRight), 0), parts);
    }

    /// <summary>
    /// Evaluates curves stored at the training event times as right-continuous step functions:
    /// the value at the last event time at or before each query time, or <paramref name="before"/>
    /// before the first event time.
    /// </summary>
    private Matrix<T> StepLookup(Tensor<T> curves, Vector<T> times, T before)
    {
        var engine = AiDotNetEngine.Current;
        int n = curves.Shape[0];
        var eventTimes = TrainedEventTimes;
        if (eventTimes is null || eventTimes.Length == 0)
            return Tensor<T>.CreateDefault(new[] { n, times.Length }, before).ToMatrix();

        var padded = engine.Concat(new[] { Tensor<T>.CreateDefault(new[] { n, 1 }, before), curves }, 1);
        var columns = engine.TensorSearchSorted(eventTimes.AsTensor(), times.AsTensor(), right: true);
        return engine.TensorIndexSelect(padded, columns, 1).ToMatrix();
    }

    /// <summary>
    /// Predicts the ensemble mortality (higher = higher risk).
    /// </summary>
    /// <remarks>
    /// <para>Ensemble mortality is the ensemble cumulative hazard summed over the training event times,
    /// M_e(x) = sum_j H_e(t_j | x) (Ishwaran et al. 2008, Section 4.1); it is the expected number of deaths
    /// for a subject like x and is what scikit-survival's RandomSurvivalForest.predict returns.</para>
    /// </remarks>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        var covariates = ExtractCovariates(x);
        if (TrainedEventTimes is null || TrainedEventTimes.Length == 0)
            return new Vector<T>(covariates.Rows);

        var hazard = EnsembleCurves(covariates).Hazard;
        return AiDotNetEngine.Current.ReduceSum(hazard, new[] { 1 }, keepDims: false).ToVector();
    }

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        if (BaselineSurvivalFunction is null || TrainedEventTimes is null)
            return new Vector<T>(times.Length);

        // Value at the last event time at or before each query time, 1 before the first event.
        var engine = AiDotNetEngine.Current;
        var padded = engine.Concat(
            new[] { Tensor<T>.CreateDefault(new[] { 1 }, NumOps.One), BaselineSurvivalFunction.AsTensor() }, 0);
        var positions = engine.TensorSearchSorted(TrainedEventTimes.AsTensor(), times.AsTensor(), right: true);
        return engine.TensorIndexSelect(padded, positions, 0).ToVector();
    }

    /// <summary>
    /// Predicts median survival time.
    /// </summary>
    /// <remarks>
    /// <para>The median is where the ensemble survival curve crosses 0.5, linearly interpolated between the
    /// two event times around the crossing. A curve that never falls below 0.5 gets the last event time;
    /// one that is already below 0.5 at the first event time gets that first event time.</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        EnsureFitted();

        if (TrainedEventTimes is null || TrainedEventTimes.Length == 0)
            return new Vector<T>(input.Rows);

        var engine = AiDotNetEngine.Current;
        var eventTimes = TrainedEventTimes.AsTensor();
        int n = input.Rows;
        int k = TrainedEventTimes.Length;
        T half = NumOps.FromDouble(0.5);
        T lastTime = TrainedEventTimes[k - 1];
        if (k == 1)
            return Tensor<T>.CreateDefault(new[] { n }, lastTime).ToVector();

        var survival = Tensor<T>.FromMatrix(PredictSurvivalProbability(input, TrainedEventTimes));

        // The averaged survival curve is non-increasing, so the number of event times with S >= 0.5
        // locates the crossing: S(t_c) >= 0.5 > S(t_c+1) with c = count - 1.
        var aboveHalf = engine.ReduceSum(engine.TensorGreaterEqual(survival, Tensor<T>.CreateDefault(new[] { n, k }, half)), new[] { 1 }, keepDims: false);
        var crossing = engine.TensorClamp(engine.TensorSubtractScalar(aboveHalf, NumOps.One), NumOps.Zero, NumOps.FromDouble(k - 2));
        var crossingIndex = crossing.Cast<int>();
        var rowOffsets = engine.TensorMultiplyScalar(engine.TensorArange<T>(0, n, 1), NumOps.FromDouble(k));
        var flatSurvival = engine.Reshape(survival, new[] { n * k });
        var flatIndex = engine.TensorAdd(rowOffsets, crossing).Cast<int>();
        var nextIndex = engine.TensorAddScalar(engine.TensorAdd(rowOffsets, crossing), NumOps.One).Cast<int>();

        var probability = engine.TensorIndexSelect(flatSurvival, flatIndex, 0);
        var nextProbability = engine.TensorIndexSelect(flatSurvival, nextIndex, 0);
        var time = engine.TensorIndexSelect(eventTimes, crossingIndex, 0);
        var nextTime = engine.TensorIndexSelect(eventTimes, engine.TensorAddScalar(crossing, NumOps.One).Cast<int>(), 0);

        // Linear interpolation
        var drop = engine.TensorSubtract(probability, nextProbability);
        var safeDrop = engine.TensorAdd(drop, engine.TensorEquals(drop, NumOps.Zero));
        var fraction = engine.TensorDivide(engine.TensorSubtractScalar(probability, half), safeDrop);
        var interpolated = engine.TensorAdd(time, engine.TensorMultiply(fraction, engine.TensorSubtract(nextTime, time)));

        var neverBelow = engine.TensorEquals(aboveHalf, NumOps.FromDouble(k));
        var belowAtFirst = engine.TensorEquals(aboveHalf, NumOps.Zero);
        var median = engine.TensorWhere(neverBelow, Tensor<T>.CreateDefault(new[] { n }, lastTime), interpolated);
        median = engine.TensorWhere(belowAtFirst, Tensor<T>.CreateDefault(new[] { n }, TrainedEventTimes[0]), median);
        return median.ToVector();
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return new RandomSurvivalForest<T>(NumTrees, MaxDepth, MinSamplesLeaf, MaxFeatures, bootstrap: Bootstrap);
    }

    /// <summary>
    /// Internal survival tree node. Terminal nodes point at their row of the forest's leaf curves.
    /// </summary>
    private class SurvivalTree
    {
        [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
        public SurvivalTree() { }

        public bool IsLeaf { get; set; }
        public int SplitFeature { get; set; }
        public required T SplitThreshold { get; set; }
        public SurvivalTree? Left { get; set; }
        public SurvivalTree? Right { get; set; }
        public int LeafIndex { get; set; }
    }
}
