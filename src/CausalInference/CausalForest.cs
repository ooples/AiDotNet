using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

using AiDotNet.Models.Parameters;

namespace AiDotNet.CausalInference;

#pragma warning disable CS8618 // Generic T properties use default(T) - always used with value types
/// <summary>
/// Causal Forest for heterogeneous treatment effect estimation using random forests.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Causal Forests extend the random forest framework to estimate Conditional Average
/// Treatment Effects (CATE) - how treatment effects vary across individuals.
/// </para>
/// <para>
/// <b>For Beginners:</b> Causal Forests are like regular random forests, but instead
/// of predicting outcomes, they predict how much the treatment CHANGES the outcome
/// for each individual.
///
/// Key concepts:
/// 1. CATE (Conditional Average Treatment Effect): The expected treatment effect
///    for individuals with specific characteristics.
/// 2. Honest estimation: Using separate data for building trees vs estimating effects.
/// 3. Heterogeneity: Treatment effects can vary across the population.
///
/// How it works:
/// 1. Build many decision trees on bootstrap samples
/// 2. At each node, split to maximize treatment effect heterogeneity
/// 3. For prediction, average treatment effect estimates across trees
///
/// Example interpretation:
/// - CATE = +5 for young patients: Treatment increases outcome by 5 for young patients
/// - CATE = -2 for elderly: Treatment decreases outcome by 2 for elderly patients
/// - This helps target treatments to those who benefit most
///
/// References:
/// - Athey &amp; Imbens (2016). "Recursive Partitioning for Heterogeneous Causal Effects"
/// - Wager &amp; Athey (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var covariates = new Matrix&lt;double&gt;(new double[,]
/// {
///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
/// });
/// var treatment = new Vector&lt;int&gt;(new int[] { 0, 1, 0, 1, 0, 1 });   // who was treated
/// var outcome = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 2.8, 8.0, 3.4, 7.6 });
///
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new CausalForest&lt;double&gt;(numTrees: 100, maxDepth: 10))
///     .Build(covariates, treatment, outcome);
///
/// // One effect per person, over the covariates alone.
/// Vector&lt;double&gt; cate = result.EstimateTreatmentEffect(covariates);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Estimation and Inference of Heterogeneous Treatment Effects using Random Forests", "https://doi.org/10.1080/01621459.2017.1319839", Year = 2018, Authors = "Stefan Wager, Susan Athey")]
public partial class CausalForest<T> : CausalModelBase<T>
{

    /// <inheritdoc />
    /// <remarks>The propensity-score coefficients. This also FIXES a measured mismatch: ParameterCount came from CausalModelBase as NumFeatures, which SetParameters set to length - 1 for the intercept, so the count reported 5 against a 6-element vector after any restore and a saved vector could not be reloaded. Count and vector now both fold this one source; NumFeatures keeps its old meaning for everything else that reads it.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VectorFieldParameterSource<T>(
            () => _propensityCoefficients,
            value =>
            {
                _propensityCoefficients = value;
                NumFeatures = value.Length - 1;   // -1 for the intercept, as before
            }));
    }
    /// <summary>
    /// Number of trees in the forest.
    /// </summary>
    private readonly int _numTrees;

    /// <summary>
    /// Maximum depth of each tree.
    /// </summary>
    private readonly int _maxDepth;

    /// <summary>
    /// Minimum samples required in a leaf node.
    /// </summary>
    private readonly int _minSamplesLeaf;

    /// <summary>
    /// Number of features to consider at each split.
    /// </summary>
    private readonly int? _maxFeatures;

    /// <summary>
    /// Whether to use honest estimation (separate data for structure vs estimation).
    /// </summary>
    private readonly bool _honest;

    /// <summary>
    /// Fraction of data to use for tree building when honest=true.
    /// </summary>
    private readonly double _honestFraction;

    /// <summary>
    /// Fraction of the rows each tree subsamples without replacement.
    /// </summary>
    private readonly double _sampleFraction;

    /// <summary>
    /// Random number generator for reproducibility.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// The trained causal trees.
    /// </summary>
    private List<CausalTree>? _trees;

    /// <summary>
    /// Propensity score coefficients for overlap adjustment.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _propensityCoefficients;

    /// <summary>
    /// Cached treatment vector from fitting.
    /// </summary>
    [Scratch]
    private Vector<int>? _cachedTreatment;

    /// <summary>
    /// Cached outcome vector from fitting.
    /// </summary>
    [Scratch]
    private Vector<T>? _cachedOutcome;

    /// <summary>
    /// Cached feature matrix from fitting.
    /// </summary>
    [Scratch]
    private Matrix<T>? _cachedFeatures;

    /// <summary>
    /// Gets the model type.
    /// </summary>

    /// <summary>
    /// Initializes a new instance of the CausalForest class.
    /// </summary>
    /// <param name="numTrees">Number of trees in the forest. Default is 100.</param>
    /// <param name="maxDepth">Maximum depth of each tree. Default is 10.</param>
    /// <param name="minSamplesLeaf">Minimum samples in leaf. Default is 5.</param>
    /// <param name="maxFeatures">Max features per split. Default is sqrt(n_features).</param>
    /// <param name="honest">Whether to use honest estimation. Default is true.</param>
    /// <param name="honestFraction">Fraction for tree building when honest. Default is 0.5.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="sampleFraction">Fraction of the rows each tree subsamples without replacement. Default is 0.5 (grf's default).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters control the forest structure:
    ///
    /// - numTrees: More trees = more stable estimates but slower
    /// - maxDepth: Deeper trees = more complex patterns but risk overfitting
    /// - minSamplesLeaf: Larger values = more stable leaf estimates
    /// - honest: Recommended! Uses separate data for building vs estimating
    ///
    /// Honest estimation splits data:
    /// - Half for deciding where to split (structure)
    /// - Half for estimating treatment effects (estimation)
    /// This reduces overfitting and provides valid confidence intervals.
    /// </para>
    /// </remarks>
    public CausalForest(
        int numTrees = 100,
        int maxDepth = 10,
        int minSamplesLeaf = 5,
        int? maxFeatures = null,
        bool honest = true,
        double honestFraction = 0.5,
        int? seed = null,
        double sampleFraction = 0.5)
        : base(seed)
    {
        if (!(sampleFraction > 0 && sampleFraction <= 1))
            throw new ArgumentOutOfRangeException(nameof(sampleFraction), "sampleFraction must be in (0, 1].");

        _numTrees = numTrees;
        _maxDepth = maxDepth;
        _minSamplesLeaf = minSamplesLeaf;
        _maxFeatures = maxFeatures;
        _honest = honest;
        _honestFraction = honestFraction;
        _sampleFraction = sampleFraction;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Fits the causal forest to the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fitting builds the forest by:
    /// 1. For each tree, draw a subsample without replacement
    /// 2. If honest, split it into disjoint structure and estimation samples
    /// 3. Build tree by recursively splitting to maximize treatment effect heterogeneity
    /// 4. Store leaf treatment effects for prediction
    /// </para>
    /// <para>
    /// The subsample is drawn without replacement, as in Wager &amp; Athey's causal forest (and grf, whose default
    /// sample fraction is 0.5). A bootstrap sample with replacement, as used before, put copies of the same subject in
    /// both the structure and the estimation halves, which defeats honesty.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        ValidateCausalData(x, treatment, outcome);

        NumFeatures = x.Columns;
        int n = x.Rows;

        // Cache data for predictions
        _cachedFeatures = x;
        _cachedTreatment = treatment;
        _cachedOutcome = outcome;

        // Fit propensity model for overlap adjustment
        _propensityCoefficients = FitLogisticRegression(x, treatment);

        // Initialize trees
        _trees = new List<CausalTree>();

        int featuresPerSplit = _maxFeatures ?? (int)Math.Sqrt(NumFeatures);
        var data = new TreeData(
            Tensor<T>.FromMatrix(x),
            treatment.AsTensor(),
            Engine.TensorEquals(treatment.AsTensor(), 1).Cast<T>(),
            outcome.AsTensor());

        for (int t = 0; t < _numTrees; t++)
        {
            var (structureIndices, estimationIndices) = DrawTreeSamples(n);

            CausalTree tree;
            if (_honest)
            {
                tree = BuildHonestTree(data, structureIndices, estimationIndices, featuresPerSplit);
            }
            else
            {
                tree = BuildTree(data, structureIndices, featuresPerSplit, 0);
            }

            _trees.Add(tree);
        }

        IsFitted = true;
    }

    /// <summary>
    /// Draws one tree's subsample without replacement (a partial Fisher–Yates shuffle) and, when honest, splits it
    /// into the structure sample (the first <c>honestFraction</c>) and the estimation sample (the rest). The two are
    /// disjoint; without honesty the whole subsample is the structure sample and the estimation sample is empty.
    /// </summary>
    internal (Tensor<int> Structure, Tensor<int> Estimation) DrawTreeSamples(int n)
    {
        int size = Math.Max(1, Math.Min(n, (int)(n * _sampleFraction)));
        var permutation = Enumerable.Range(0, n).ToArray();
        for (int i = 0; i < size; i++)
        {
            int j = i + _random.Next(n - i);
            (permutation[i], permutation[j]) = (permutation[j], permutation[i]);
        }

        int splitPoint = _honest ? (int)(size * _honestFraction) : size;
        var structure = new Tensor<int>(permutation.Take(splitPoint).ToArray(), new[] { splitPoint });
        var estimation = new Tensor<int>(permutation.Skip(splitPoint).Take(size - splitPoint).ToArray(), new[] { size - splitPoint });
        return (structure, estimation);
    }

    /// <summary>
    /// Fits the causal model using the ICausalModel interface signature.
    /// </summary>
    /// <param name="features">The feature matrix (covariates).</param>
    /// <param name="treatment">Treatment indicators as generic type (0 or 1).</param>
    /// <param name="outcome">The outcome variable.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts the generic treatment vector to integer format
    /// and then fits the causal forest. Causal forests are particularly good at finding
    /// heterogeneous treatment effects - where treatment helps some people more than others.
    /// </para>
    /// </remarks>
    public override void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome)
    {
        // Convert treatment vector to int
        var treatmentInt = Engine.TensorRound(treatment.AsTensor()).Cast<int>().ToVector();

        // Call the original fit method
        Fit(features, treatmentInt, outcome);
    }

    /// <summary>
    /// Estimates treatment effects for individuals using the causal forest.
    /// </summary>
    /// <param name="features">The feature matrix for which to estimate effects.</param>
    /// <returns>A vector of estimated treatment effects (CATE for each individual).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Causal forests excel at estimating Conditional Average Treatment
    /// Effects (CATE) - how much the treatment would change the outcome for each person.
    /// Unlike simpler methods that give everyone the same effect estimate, causal forests
    /// provide personalized effect estimates based on each person's features.
    /// </para>
    /// </remarks>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        return PredictTreatmentEffect(features);
    }

    /// <summary>
    /// Predicts outcomes under treatment for the given features.
    /// </summary>
    /// <param name="features">The feature matrix.</param>
    /// <returns>Predicted outcomes if treated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This estimates what the outcome would be if each individual received treatment.
    /// It uses the control outcome prediction plus the estimated treatment effect:
    /// Y(1) = Y(0) + CATE
    /// </para>
    /// </remarks>
    public override Vector<T> PredictTreated(Matrix<T> features)
    {
        EnsureFitted();

        var controlPredictions = PredictControl(features);
        var treatmentEffects = EstimateTreatmentEffect(features);
        return Engine.Add(controlPredictions, treatmentEffects);
    }

    /// <summary>
    /// Predicts outcomes under control for the given features.
    /// </summary>
    /// <param name="features">The feature matrix.</param>
    /// <returns>Predicted outcomes if not treated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This estimates what the outcome would be if each individual did NOT receive treatment.
    /// We use k-nearest neighbors weighted by propensity scores to estimate control outcomes
    /// from the training data control group.
    /// </para>
    /// </remarks>
    public override Vector<T> PredictControl(Matrix<T> features)
    {
        EnsureFitted();

        if (_cachedFeatures is null || _cachedTreatment is null || _cachedOutcome is null)
        {
            throw new InvalidOperationException("Model must be fitted with outcome data for control predictions.");
        }

        // Kernel-weighted outcomes of the control individuals with similar propensity scores (0 where none weigh in)
        var control = Engine.TensorEquals(_cachedTreatment.AsTensor(), 0).Cast<T>();
        var (mean, _) = PropensityKernelMean(
            EstimatePropensityScores(features).AsTensor(), EstimatePropensityScores(_cachedFeatures).AsTensor(),
            control, _cachedOutcome);
        return mean.ToVector();
    }

    /// <summary>The training data a tree is grown on, as tensors.</summary>
    private sealed class TreeData
    {
        public TreeData(Tensor<T> features, Tensor<int> treatment, Tensor<T> treated, Tensor<T> outcome)
        {
            Features = features;
            Treatment = treatment;
            Treated = treated;
            Outcome = outcome;
        }

        /// <summary>[n, p] covariates.</summary>
        public Tensor<T> Features { get; }

        /// <summary>[n] 0/1 treatment as ints (for exact counts).</summary>
        public Tensor<int> Treatment { get; }

        /// <summary>[n] 1 for treated, 0 otherwise.</summary>
        public Tensor<T> Treated { get; }

        /// <summary>[n] outcomes.</summary>
        public Tensor<T> Outcome { get; }
    }

    /// <summary>
    /// Builds a causal tree using honest estimation.
    /// </summary>
    private CausalTree BuildHonestTree(
        TreeData data,
        Tensor<int> structureIndices,
        Tensor<int> estimationIndices,
        int featuresPerSplit)
    {
        // Build tree structure using structure data
        var tree = BuildTree(data, structureIndices, featuresPerSplit, 0);

        // Re-estimate treatment effects using estimation data
        ReEstimateLeafEffects(tree, data, estimationIndices);

        return tree;
    }

    /// <summary>
    /// Builds a causal tree recursively.
    /// </summary>
    private CausalTree BuildTree(
        TreeData data,
        Tensor<int> indices,
        int featuresPerSplit,
        int depth)
    {
        var tree = new CausalTree();
        int count = indices.Length;

        // Check stopping conditions
        if (depth >= _maxDepth || count < 2 * _minSamplesLeaf)
        {
            return MakeLeaf(tree, data, indices);
        }

        // Count treated and control in this node
        int numTreated = Engine.TensorCountNonzero(Engine.TensorIndexSelect(data.Treatment, indices, 0));
        if (numTreated < _minSamplesLeaf || count - numTreated < _minSamplesLeaf)
        {
            return MakeLeaf(tree, data, indices);
        }

        // Find best split
        var (bestFeature, bestThreshold, found) = FindBestSplit(data, indices, featuresPerSplit);
        if (!found)
        {
            return MakeLeaf(tree, data, indices);
        }

        // Split the data
        var (leftIndices, rightIndices) = Partition(data.Features, indices, bestFeature, bestThreshold);
        if (leftIndices.Length < _minSamplesLeaf || rightIndices.Length < _minSamplesLeaf)
        {
            return MakeLeaf(tree, data, indices);
        }

        // Build child nodes
        tree.IsLeaf = false;
        tree.FeatureIndex = bestFeature;
        tree.Threshold = bestThreshold;
        tree.NumSamples = count;
        tree.TreatmentEffect = EstimateLeafEffect(data, indices);
        tree.Left = BuildTree(data, leftIndices, featuresPerSplit, depth + 1);
        tree.Right = BuildTree(data, rightIndices, featuresPerSplit, depth + 1);

        return tree;
    }

    /// <summary>Turns <paramref name="tree"/> into a leaf holding the effect of <paramref name="indices"/>.</summary>
    private CausalTree MakeLeaf(CausalTree tree, TreeData data, Tensor<int> indices)
    {
        tree.IsLeaf = true;
        tree.TreatmentEffect = EstimateLeafEffect(data, indices);
        tree.NumSamples = indices.Length;
        return tree;
    }

    /// <summary>The rows of <paramref name="indices"/> with feature ≤ threshold, and the rest, in order.</summary>
    private (Tensor<int> Left, Tensor<int> Right) Partition(Tensor<T> features, Tensor<int> indices, int feature, T threshold)
    {
        var column = Engine.Reshape(
            Engine.TensorIndexSelect(Engine.TensorIndexSelect(features, indices, 0), new Tensor<int>(new[] { feature }, new[] { 1 }), 1),
            new[] { indices.Length });
        var right = Engine.TensorGreaterThan(column, threshold);
        var left = Engine.TensorEquals(right, NumOps.Zero);
        return (RowsWhere(indices, left), RowsWhere(indices, right));
    }

    /// <summary>The entries of <paramref name="indices"/> where <paramref name="mask"/> is nonzero.</summary>
    private Tensor<int> RowsWhere(Tensor<int> indices, Tensor<T> mask)
    {
        var coordinates = Engine.TensorNonzero(mask);
        return Engine.TensorIndexSelect(indices, Engine.Reshape(coordinates, new[] { coordinates.Length }), 0);
    }

    /// <summary>
    /// Finds the best split that maximizes treatment effect heterogeneity.
    /// </summary>
    /// <remarks>
    /// Every threshold of every candidate feature is scored at once: each feature's values are sorted, the treated
    /// and control counts and outcome sums are prefix sums along the sort, and a threshold sits midway between
    /// consecutive distinct values. The gain is the size-weighted squared deviation of the child effects from the
    /// parent's; the first best (in feature order, then ascending threshold) wins, as before.
    /// </remarks>
    private (int Feature, T Threshold, bool Found) FindBestSplit(
        TreeData data,
        Tensor<int> indices,
        int featuresPerSplit)
    {
        int m = indices.Length;

        // Randomly select features to consider
        var featureIndices = Enumerable.Range(0, NumFeatures).ToList();
        Shuffle(featureIndices);
        var selectedFeatures = featureIndices.Take(Math.Min(featuresPerSplit, NumFeatures)).ToArray();
        int f = selectedFeatures.Length;
        if (m < 2 || f == 0)
            return (-1, NumOps.Zero, false);

        // Node data, outcomes scaled by their peak (the gain only has to be compared, and nothing squared overflows)
        var nodeRows = Engine.TensorIndexSelect(data.Features, indices, 0);
        var values = Engine.TensorTranspose(Engine.TensorIndexSelect(nodeRows, new Tensor<int>(selectedFeatures, new[] { f }), 1));
        var nodeOutcome = Engine.TensorIndexSelect(data.Outcome, indices, 0);
        var y = Engine.TensorDivideScalar(nodeOutcome, PeakMagnitude(nodeOutcome));
        var treated = Engine.TensorIndexSelect(data.Treated, indices, 0);
        var treatedInt = Engine.TensorIndexSelect(data.Treatment, indices, 0);

        // Parent node treatment effect
        T parentEffect = DifferenceInMeans(y, treated);

        // Sort each feature's values, carrying the treatment and outcome along
        var order = Engine.TensorArgsort(values, 1, false);
        var sorted = Engine.TensorGather(values, order, 1);
        var tSorted = Engine.TensorGather(Engine.TensorBroadcastTo(Engine.Reshape(treated, new[] { 1, m }), new[] { f, m }), order, 1);
        var tIntSorted = Engine.TensorGather(Engine.TensorBroadcastTo(Engine.Reshape(treatedInt, new[] { 1, m }), new[] { f, m }), order, 1);
        var ySorted = Engine.TensorGather(Engine.TensorBroadcastTo(Engine.Reshape(y, new[] { 1, m }), new[] { f, m }), order, 1);

        // Left child after position i holds sorted rows 0..i: exact integer counts, T outcome sums
        int k = m - 1;
        var leftCount = Engine.TensorBroadcastTo(
            new Tensor<int>(Enumerable.Range(1, k).ToArray(), new[] { 1, k }), new[] { f, k });
        var leftTreatedCount = Engine.TensorNarrow(Engine.TensorCumSum(tIntSorted, 1), 1, 0, k);
        var leftControlCount = Engine.TensorSubtract(leftCount, leftTreatedCount);
        int totalTreated = Engine.TensorCountNonzero(treatedInt);
        var rightTreatedCount = Engine.TensorSubtract(Tensor<int>.CreateDefault(new[] { f, k }, totalTreated), leftTreatedCount);
        var rightControlCount = Engine.TensorSubtract(
            Engine.TensorSubtract(Tensor<int>.CreateDefault(new[] { f, k }, m), leftCount), rightTreatedCount);

        var treatedY = Engine.TensorMultiply(tSorted, ySorted);
        var cumTreatedY = Engine.TensorCumSum(treatedY, 1);
        var cumY = Engine.TensorCumSum(ySorted, 1);
        var totalTreatedY = Engine.TensorBroadcastTo(Engine.TensorNarrow(cumTreatedY, 1, k, 1), new[] { f, k });
        var totalY = Engine.TensorBroadcastTo(Engine.TensorNarrow(cumY, 1, k, 1), new[] { f, k });
        var leftTreatedY = Engine.TensorNarrow(cumTreatedY, 1, 0, k);
        var leftControlY = Engine.TensorSubtract(Engine.TensorNarrow(cumY, 1, 0, k), leftTreatedY);
        var rightTreatedY = Engine.TensorSubtract(totalTreatedY, leftTreatedY);
        var rightControlY = Engine.TensorSubtract(Engine.TensorSubtract(totalY, leftTreatedY), Engine.TensorAdd(leftControlY, rightTreatedY));

        var leftEffect = GroupDifference(leftTreatedY, leftTreatedCount, leftControlY, leftControlCount);
        var rightEffect = GroupDifference(rightTreatedY, rightTreatedCount, rightControlY, rightControlCount);

        // A threshold exists only between distinct values, with enough rows (and both groups) on each side
        var distinct = Engine.TensorNotEquals(Engine.TensorNarrow(sorted, 1, 0, k), Engine.TensorNarrow(sorted, 1, 1, k));
        var rightCount = Engine.TensorSubtract(Tensor<int>.CreateDefault(new[] { f, k }, m), leftCount);
        var valid = Engine.TensorMultiply(distinct, Engine.TensorMultiply(
            AtLeast(leftCount, _minSamplesLeaf), AtLeast(rightCount, _minSamplesLeaf)).Cast<T>());
        valid = Engine.TensorMultiply(valid, Engine.TensorMultiply(
            Engine.TensorMultiply(AtLeast(leftTreatedCount, 1), AtLeast(leftControlCount, 1)),
            Engine.TensorMultiply(AtLeast(rightTreatedCount, 1), AtLeast(rightControlCount, 1))).Cast<T>());

        // Gain is weighted variance of treatment effects
        T invM = NumOps.FromDouble(1.0 / m);
        var leftShare = Engine.TensorMultiplyScalar(leftCount.Cast<T>(), invM);
        var rightShare = Engine.TensorMultiplyScalar(rightCount.Cast<T>(), invM);
        var gain = Engine.TensorAdd(
            Engine.TensorMultiply(leftShare, Engine.TensorSquare(Engine.TensorSubtractScalar(leftEffect, parentEffect))),
            Engine.TensorMultiply(rightShare, Engine.TensorSquare(Engine.TensorSubtractScalar(rightEffect, parentEffect))));
        var scored = Engine.Reshape(
            Engine.TensorWhere(valid, gain, Tensor<T>.CreateDefault(new[] { f, k }, NumOps.MinValue)), new[] { f * k });

        T bestGain = Engine.TensorMaxValue(scored);
        if (!NumOps.GreaterThan(bestGain, NumOps.Zero))
            return (-1, NumOps.Zero, false);

        var best = Engine.TensorNonzero(Engine.TensorEquals(scored, bestGain));
        int flat = best[0, 0];
        int row = flat / k;
        int position = flat % k;
        T threshold = NumOps.Divide(NumOps.Add(sorted[row, position], sorted[row, position + 1]), NumOps.FromDouble(2.0));
        return (selectedFeatures[row], threshold, true);
    }

    /// <summary>1 where <paramref name="counts"/> ≥ <paramref name="minimum"/>, as ints.</summary>
    private Tensor<int> AtLeast(Tensor<int> counts, int minimum)
        => Engine.TensorGreaterThan(counts, minimum - 1);

    /// <summary>Treated mean minus control mean from sums and counts; 0 where either group is empty.</summary>
    private Tensor<T> GroupDifference(Tensor<T> treatedSum, Tensor<int> treatedCount, Tensor<T> controlSum, Tensor<int> controlCount)
    {
        var tCount = treatedCount.Cast<T>();
        var cCount = controlCount.Cast<T>();
        var tHas = Engine.TensorGreaterThan(tCount, NumOps.Zero);
        var cHas = Engine.TensorGreaterThan(cCount, NumOps.Zero);
        var ones = Engine.TensorOnesLike(tCount);
        var difference = Engine.TensorSubtract(
            Engine.TensorDivide(treatedSum, Engine.TensorWhere(tHas, tCount, ones)),
            Engine.TensorDivide(controlSum, Engine.TensorWhere(cHas, cCount, ones)));
        return Engine.TensorMultiply(Engine.TensorMultiply(tHas, cHas), difference);
    }

    /// <summary>Treated mean minus control mean of <paramref name="y"/>; 0 when either group is empty.</summary>
    private T DifferenceInMeans(Tensor<T> y, Tensor<T> treated)
    {
        var control = Engine.TensorSubtract(Engine.TensorOnesLike(treated), treated);
        T treatedMean = WeightedMean(y, treated, out bool hasTreated);
        T controlMean = WeightedMean(y, control, out bool hasControl);
        return hasTreated && hasControl ? NumOps.Subtract(treatedMean, controlMean) : NumOps.Zero;
    }

    /// <summary>
    /// Estimates the treatment effect in a leaf node.
    /// </summary>
    private T EstimateLeafEffect(TreeData data, Tensor<int> indices)
    {
        return DifferenceInMeans(
            Engine.TensorIndexSelect(data.Outcome, indices, 0), Engine.TensorIndexSelect(data.Treated, indices, 0));
    }

    /// <summary>
    /// Re-estimates leaf effects using estimation sample (for honest estimation).
    /// </summary>
    private void ReEstimateLeafEffects(
        CausalTree tree,
        TreeData data,
        Tensor<int> estimationIndices)
    {
        // For each leaf, find which estimation samples fall into it and re-estimate
        var leafAssignments = new Dictionary<CausalTree, Tensor<int>>();
        AssignToLeaves(tree, data.Features, estimationIndices, leafAssignments);

        foreach (var kvp in leafAssignments)
        {
            if (kvp.Value.Length > 0)
            {
                kvp.Key.TreatmentEffect = EstimateLeafEffect(data, kvp.Value);
                kvp.Key.NumSamples = kvp.Value.Length;
            }
        }
    }

    /// <summary>
    /// Assigns samples to leaf nodes.
    /// </summary>
    private void AssignToLeaves(
        CausalTree tree,
        Tensor<T> features,
        Tensor<int> indices,
        Dictionary<CausalTree, Tensor<int>> assignments)
    {
        if (tree.IsLeaf)
        {
            assignments[tree] = indices;
            return;
        }

        var (leftIndices, rightIndices) = Partition(features, indices, tree.FeatureIndex, tree.Threshold);

        if (tree.Left is not null)
        {
            AssignToLeaves(tree.Left, features, leftIndices, assignments);
        }

        if (tree.Right is not null)
        {
            AssignToLeaves(tree.Right, features, rightIndices, assignments);
        }
    }

    /// <summary>
    /// Shuffles a list in place.
    /// </summary>
    private void Shuffle<TItem>(List<TItem> list)
    {
        int n = list.Count;
        for (int i = n - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    /// <summary>
    /// Adds each row's prediction through a tree into <paramref name="sums"/> (rows routed left when the split
    /// feature is ≤ the threshold; a missing child keeps the node's own effect).
    /// </summary>
    private Tensor<T> AccumulateTree(CausalTree tree, Tensor<T> features, Tensor<int> rows, Tensor<T> sums)
    {
        if (rows.Length == 0)
        {
            return sums;
        }

        if (tree.IsLeaf)
        {
            return Engine.TensorScatterAdd(sums, rows, Tensor<T>.CreateDefault(new[] { rows.Length }, tree.TreatmentEffect), 0);
        }

        var (leftRows, rightRows) = Partition(features, rows, tree.FeatureIndex, tree.Threshold);
        sums = tree.Left is not null
            ? AccumulateTree(tree.Left, features, leftRows, sums)
            : Engine.TensorScatterAdd(sums, leftRows, Tensor<T>.CreateDefault(new[] { leftRows.Length }, tree.TreatmentEffect), 0);
        return tree.Right is not null
            ? AccumulateTree(tree.Right, features, rightRows, sums)
            : Engine.TensorScatterAdd(sums, rightRows, Tensor<T>.CreateDefault(new[] { rightRows.Length }, tree.TreatmentEffect), 0);
    }

    #region ICausalModel Implementation

    /// <summary>
    /// Estimates the Average Treatment Effect (ATE).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ATE is the average of CATE across all individuals.
    /// It represents the overall average treatment effect in the population.
    /// </para>
    /// </remarks>
    public override (T estimate, T standardError) EstimateATE(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment, outcome);
        }

        // ATE is average of CATE
        T ate = Engine.Mean(EstimateCATEPerIndividual(x, treatment, outcome));

        // Bootstrap standard error
        T se = CalculateBootstrapStandardError(
            (xBoot, treatBoot, outBoot) => Engine.Mean(EstimateCATEPerIndividual(xBoot, treatBoot, outBoot)),
            x, treatment, outcome, 50);

        return (ate, se);
    }

    /// <summary>
    /// Estimates the Average Treatment Effect on the Treated (ATT).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ATT is the average treatment effect among those who
    /// actually received treatment. This is relevant when treatment effects differ
    /// between treated and control groups.
    /// </para>
    /// </remarks>
    public override (T estimate, T standardError) EstimateATT(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment, outcome);
        }

        // ATT is average CATE among treated
        T att = MeanOverTreated(EstimateCATEPerIndividual(x, treatment, outcome), treatment);

        // Bootstrap standard error
        T se = CalculateBootstrapStandardError(
            (xBoot, treatBoot, outBoot) => MeanOverTreated(EstimateCATEPerIndividual(xBoot, treatBoot, outBoot), treatBoot),
            x, treatment, outcome, 50);

        return (att, se);
    }

    /// <summary>Mean of <paramref name="values"/> over the treated (0 when there are none).</summary>
    private T MeanOverTreated(Vector<T> values, Vector<int> treatment)
    {
        T mean = WeightedMean(values.AsTensor(), Engine.TensorEquals(treatment.AsTensor(), 1).Cast<T>(), out bool any);
        return any ? mean : NumOps.Zero;
    }

    /// <summary>
    /// Estimates CATE for each individual.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CATE (Conditional Average Treatment Effect) is the
    /// expected treatment effect for individuals with specific characteristics.
    /// Causal Forest's main strength is providing individual-level CATE estimates.
    ///
    /// Each individual gets their own treatment effect estimate based on their
    /// features, enabling personalized treatment decisions.
    /// </para>
    /// </remarks>
    public override Vector<T> EstimateCATEPerIndividual(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment, outcome);
        }

        return PredictTreatmentEffect(x);
    }

    /// <summary>
    /// Predicts treatment effects for new individuals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For each new individual, averages the treatment effect
    /// predictions across all trees in the forest. This ensemble approach provides
    /// robust, stable estimates.
    /// </para>
    /// </remarks>
    public override Vector<T> PredictTreatmentEffect(Matrix<T> x)
    {
        EnsureFitted();

        if (_trees is null || _trees.Count == 0)
        {
            throw new InvalidOperationException("Causal forest has no trees.");
        }

        int n = x.Rows;
        var features = Tensor<T>.FromMatrix(x);
        var rows = new Tensor<int>(Enumerable.Range(0, n).ToArray(), new[] { n });
        var sums = new Tensor<T>(new[] { n });
        foreach (var tree in _trees)
        {
            sums = AccumulateTree(tree, features, rows, sums);
        }

        return Engine.TensorDivideScalar(sums, NumOps.FromDouble(_trees.Count)).ToVector();
    }

    /// <summary>
    /// Estimates propensity scores.
    /// </summary>
    protected override Vector<T> EstimatePropensityScoresCore(Matrix<T> x)
    {
        if (_propensityCoefficients is null)
        {
            throw new InvalidOperationException("Propensity model not fitted.");
        }

        return PredictPropensityWithCoefficients(x, _propensityCoefficients);
    }

    /// <summary>
    /// Standard prediction - returns treatment effect predictions.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        return PredictTreatmentEffect(input);
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Creates a new instance with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new CausalForest<T>(_numTrees, _maxDepth, _minSamplesLeaf, _maxFeatures, _honest, _honestFraction, sampleFraction: _sampleFraction);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Gets feature importance based on split frequency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Features that appear more often in tree splits are
    /// considered more important for predicting treatment effect heterogeneity.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();

        if (_trees is null)
        {
            return base.GetFeatureImportance();
        }

        // Count feature splits across all trees
        var splitCounts = new int[NumFeatures];
        foreach (var tree in _trees)
        {
            CountFeatureSplits(tree, splitCounts);
        }

        int totalSplits = splitCounts.Sum();
        if (totalSplits == 0) totalSplits = 1;

        var shares = Engine.TensorDivideScalar(
            new Tensor<int>(splitCounts, new[] { NumFeatures }).Cast<T>(), NumOps.FromDouble(totalSplits)).ToArray();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            importance[name] = shares[i];
        }

        return importance;
    }

    /// <summary>
    /// Counts feature splits in a tree.
    /// </summary>
    private void CountFeatureSplits(CausalTree tree, int[] counts)
    {
        if (tree.IsLeaf) return;

        if (tree.FeatureIndex >= 0 && tree.FeatureIndex < counts.Length)
        {
            counts[tree.FeatureIndex]++;
        }

        if (tree.Left is not null)
        {
            CountFeatureSplits(tree.Left, counts);
        }

        if (tree.Right is not null)
        {
            CountFeatureSplits(tree.Right, counts);
        }
    }

    #endregion

    /// <summary>
    /// Gets the number of trees in the forest.
    /// </summary>
    public int NumTrees => _trees?.Count ?? 0;

    /// <summary>
    /// Internal class representing a causal tree node.
    /// </summary>
    private class CausalTree
    {
        [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
        public CausalTree() { }

        public bool IsLeaf { get; set; }
        public int FeatureIndex { get; set; }
        public required T Threshold { get; set; }
        public required T TreatmentEffect { get; set; }
        public int NumSamples { get; set; }
        public CausalTree? Left { get; set; }
        public CausalTree? Right { get; set; }
    }
}
