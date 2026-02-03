using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.CausalInference;

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
public class CausalForest<T> : CausalModelBase<T>
{
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
    private Vector<T>? _propensityCoefficients;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.CausalForest;

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
        int? seed = null)
    {
        _numTrees = numTrees;
        _maxDepth = maxDepth;
        _minSamplesLeaf = minSamplesLeaf;
        _maxFeatures = maxFeatures;
        _honest = honest;
        _honestFraction = honestFraction;
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
    /// 1. For each tree, draw a bootstrap sample
    /// 2. If honest, split into structure and estimation samples
    /// 3. Build tree by recursively splitting to maximize treatment effect heterogeneity
    /// 4. Store leaf treatment effects for prediction
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        ValidateCausalData(x, treatment, outcome);

        NumFeatures = x.Columns;
        int n = x.Rows;

        // Fit propensity model for overlap adjustment
        _propensityCoefficients = FitLogisticRegression(x, treatment);

        // Initialize trees
        _trees = new List<CausalTree>();

        int featuresPerSplit = _maxFeatures ?? (int)Math.Sqrt(NumFeatures);

        for (int t = 0; t < _numTrees; t++)
        {
            // Bootstrap sample
            var bootstrapIndices = new List<int>();
            for (int i = 0; i < n; i++)
            {
                bootstrapIndices.Add(_random.Next(n));
            }

            CausalTree tree;
            if (_honest)
            {
                // Split bootstrap into structure and estimation samples
                int splitPoint = (int)(bootstrapIndices.Count * _honestFraction);
                var structureIndices = bootstrapIndices.Take(splitPoint).ToList();
                var estimationIndices = bootstrapIndices.Skip(splitPoint).ToList();

                tree = BuildHonestTree(x, treatment, outcome, structureIndices, estimationIndices, featuresPerSplit);
            }
            else
            {
                tree = BuildTree(x, treatment, outcome, bootstrapIndices, featuresPerSplit, 0);
            }

            _trees.Add(tree);
        }

        IsFitted = true;
    }

    /// <summary>
    /// Builds a causal tree using honest estimation.
    /// </summary>
    private CausalTree BuildHonestTree(
        Matrix<T> x,
        Vector<int> treatment,
        Vector<T> outcome,
        List<int> structureIndices,
        List<int> estimationIndices,
        int featuresPerSplit)
    {
        // Build tree structure using structure data
        var tree = BuildTree(x, treatment, outcome, structureIndices, featuresPerSplit, 0);

        // Re-estimate treatment effects using estimation data
        ReEstimateLeafEffects(tree, x, treatment, outcome, estimationIndices);

        return tree;
    }

    /// <summary>
    /// Builds a causal tree recursively.
    /// </summary>
    private CausalTree BuildTree(
        Matrix<T> x,
        Vector<int> treatment,
        Vector<T> outcome,
        List<int> indices,
        int featuresPerSplit,
        int depth)
    {
        var tree = new CausalTree();

        // Check stopping conditions
        if (depth >= _maxDepth || indices.Count < 2 * _minSamplesLeaf)
        {
            tree.IsLeaf = true;
            tree.TreatmentEffect = EstimateLeafEffect(treatment, outcome, indices);
            tree.NumSamples = indices.Count;
            return tree;
        }

        // Count treated and control in this node
        int numTreated = 0;
        foreach (int i in indices)
        {
            numTreated += treatment[i];
        }

        if (numTreated < _minSamplesLeaf || indices.Count - numTreated < _minSamplesLeaf)
        {
            tree.IsLeaf = true;
            tree.TreatmentEffect = EstimateLeafEffect(treatment, outcome, indices);
            tree.NumSamples = indices.Count;
            return tree;
        }

        // Find best split
        var (bestFeature, bestThreshold, bestGain) = FindBestSplit(x, treatment, outcome, indices, featuresPerSplit);

        if (bestGain <= 0)
        {
            tree.IsLeaf = true;
            tree.TreatmentEffect = EstimateLeafEffect(treatment, outcome, indices);
            tree.NumSamples = indices.Count;
            return tree;
        }

        // Split the data
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        foreach (int i in indices)
        {
            double xVal = NumOps.ToDouble(x[i, bestFeature]);
            if (xVal <= bestThreshold)
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        if (leftIndices.Count < _minSamplesLeaf || rightIndices.Count < _minSamplesLeaf)
        {
            tree.IsLeaf = true;
            tree.TreatmentEffect = EstimateLeafEffect(treatment, outcome, indices);
            tree.NumSamples = indices.Count;
            return tree;
        }

        // Build child nodes
        tree.IsLeaf = false;
        tree.FeatureIndex = bestFeature;
        tree.Threshold = bestThreshold;
        tree.NumSamples = indices.Count;
        tree.TreatmentEffect = EstimateLeafEffect(treatment, outcome, indices);
        tree.Left = BuildTree(x, treatment, outcome, leftIndices, featuresPerSplit, depth + 1);
        tree.Right = BuildTree(x, treatment, outcome, rightIndices, featuresPerSplit, depth + 1);

        return tree;
    }

    /// <summary>
    /// Finds the best split that maximizes treatment effect heterogeneity.
    /// </summary>
    private (int feature, double threshold, double gain) FindBestSplit(
        Matrix<T> x,
        Vector<int> treatment,
        Vector<T> outcome,
        List<int> indices,
        int featuresPerSplit)
    {
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = double.NegativeInfinity;

        // Randomly select features to consider
        var featureIndices = Enumerable.Range(0, NumFeatures).ToList();
        Shuffle(featureIndices);
        var selectedFeatures = featureIndices.Take(Math.Min(featuresPerSplit, NumFeatures)).ToList();

        // Parent node treatment effect
        double parentEffect = EstimateLeafEffectDouble(treatment, outcome, indices);

        foreach (int f in selectedFeatures)
        {
            // Get unique values for this feature
            var values = indices.Select(i => NumOps.ToDouble(x[i, f])).Distinct().OrderBy(v => v).ToList();

            if (values.Count < 2) continue;

            // Try splitting at midpoints between consecutive values
            for (int v = 0; v < values.Count - 1; v++)
            {
                double threshold = (values[v] + values[v + 1]) / 2.0;

                var leftIndices = new List<int>();
                var rightIndices = new List<int>();

                foreach (int i in indices)
                {
                    if (NumOps.ToDouble(x[i, f]) <= threshold)
                    {
                        leftIndices.Add(i);
                    }
                    else
                    {
                        rightIndices.Add(i);
                    }
                }

                if (leftIndices.Count < _minSamplesLeaf || rightIndices.Count < _minSamplesLeaf)
                    continue;

                // Count treatment/control in each split
                int leftTreated = leftIndices.Count(i => treatment[i] == 1);
                int rightTreated = rightIndices.Count(i => treatment[i] == 1);

                if (leftTreated < 1 || leftIndices.Count - leftTreated < 1)
                    continue;
                if (rightTreated < 1 || rightIndices.Count - rightTreated < 1)
                    continue;

                // Calculate treatment effect heterogeneity gain
                double leftEffect = EstimateLeafEffectDouble(treatment, outcome, leftIndices);
                double rightEffect = EstimateLeafEffectDouble(treatment, outcome, rightIndices);

                // Gain is weighted variance of treatment effects
                double nLeft = leftIndices.Count;
                double nRight = rightIndices.Count;
                double n = indices.Count;

                double gain = (nLeft / n) * (leftEffect - parentEffect) * (leftEffect - parentEffect)
                            + (nRight / n) * (rightEffect - parentEffect) * (rightEffect - parentEffect);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }

        return (bestFeature, bestThreshold, bestGain);
    }

    /// <summary>
    /// Estimates the treatment effect in a leaf node.
    /// </summary>
    private T EstimateLeafEffect(Vector<int> treatment, Vector<T> outcome, List<int> indices)
    {
        return NumOps.FromDouble(EstimateLeafEffectDouble(treatment, outcome, indices));
    }

    /// <summary>
    /// Estimates the treatment effect in a leaf node (double version).
    /// </summary>
    private double EstimateLeafEffectDouble(Vector<int> treatment, Vector<T> outcome, List<int> indices)
    {
        double treatedSum = 0;
        int treatedCount = 0;
        double controlSum = 0;
        int controlCount = 0;

        foreach (int i in indices)
        {
            double y = NumOps.ToDouble(outcome[i]);
            if (treatment[i] == 1)
            {
                treatedSum += y;
                treatedCount++;
            }
            else
            {
                controlSum += y;
                controlCount++;
            }
        }

        if (treatedCount == 0 || controlCount == 0)
        {
            return 0;
        }

        double treatedMean = treatedSum / treatedCount;
        double controlMean = controlSum / controlCount;

        return treatedMean - controlMean;
    }

    /// <summary>
    /// Re-estimates leaf effects using estimation sample (for honest estimation).
    /// </summary>
    private void ReEstimateLeafEffects(
        CausalTree tree,
        Matrix<T> x,
        Vector<int> treatment,
        Vector<T> outcome,
        List<int> estimationIndices)
    {
        // For each leaf, find which estimation samples fall into it and re-estimate
        var leafAssignments = new Dictionary<CausalTree, List<int>>();
        AssignToLeaves(tree, x, estimationIndices, leafAssignments);

        foreach (var kvp in leafAssignments)
        {
            if (kvp.Value.Count > 0)
            {
                kvp.Key.TreatmentEffect = EstimateLeafEffect(treatment, outcome, kvp.Value);
                kvp.Key.NumSamples = kvp.Value.Count;
            }
        }
    }

    /// <summary>
    /// Assigns samples to leaf nodes.
    /// </summary>
    private void AssignToLeaves(
        CausalTree tree,
        Matrix<T> x,
        List<int> indices,
        Dictionary<CausalTree, List<int>> assignments)
    {
        if (tree.IsLeaf)
        {
            assignments[tree] = indices;
            return;
        }

        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        foreach (int i in indices)
        {
            double xVal = NumOps.ToDouble(x[i, tree.FeatureIndex]);
            if (xVal <= tree.Threshold)
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        if (tree.Left is not null)
        {
            AssignToLeaves(tree.Left, x, leftIndices, assignments);
        }

        if (tree.Right is not null)
        {
            AssignToLeaves(tree.Right, x, rightIndices, assignments);
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
    /// Predicts a single sample through a tree.
    /// </summary>
    private T PredictTree(CausalTree tree, Matrix<T> x, int rowIndex)
    {
        if (tree.IsLeaf)
        {
            return tree.TreatmentEffect;
        }

        double xVal = NumOps.ToDouble(x[rowIndex, tree.FeatureIndex]);
        if (xVal <= tree.Threshold)
        {
            return tree.Left is not null ? PredictTree(tree.Left, x, rowIndex) : tree.TreatmentEffect;
        }
        else
        {
            return tree.Right is not null ? PredictTree(tree.Right, x, rowIndex) : tree.TreatmentEffect;
        }
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

        var cate = EstimateCATEPerIndividual(x, treatment, outcome);

        // ATE is average of CATE
        double sum = 0;
        for (int i = 0; i < cate.Length; i++)
        {
            sum += NumOps.ToDouble(cate[i]);
        }
        double ate = sum / cate.Length;

        // Bootstrap standard error
        T se = CalculateBootstrapStandardError(
            (xBoot, treatBoot, outBoot) =>
            {
                var cateBoot = EstimateCATEPerIndividual(xBoot, treatBoot, outBoot);
                double s = 0;
                for (int i = 0; i < cateBoot.Length; i++)
                {
                    s += NumOps.ToDouble(cateBoot[i]);
                }
                return NumOps.FromDouble(s / cateBoot.Length);
            },
            x, treatment, outcome, 50);

        return (NumOps.FromDouble(ate), se);
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

        var cate = EstimateCATEPerIndividual(x, treatment, outcome);

        // ATT is average CATE among treated
        double sum = 0;
        int count = 0;
        for (int i = 0; i < cate.Length; i++)
        {
            if (treatment[i] == 1)
            {
                sum += NumOps.ToDouble(cate[i]);
                count++;
            }
        }
        double att = count > 0 ? sum / count : 0;

        // Bootstrap standard error
        T se = CalculateBootstrapStandardError(
            (xBoot, treatBoot, outBoot) =>
            {
                var cateBoot = EstimateCATEPerIndividual(xBoot, treatBoot, outBoot);
                double s = 0;
                int c = 0;
                for (int i = 0; i < cateBoot.Length; i++)
                {
                    if (treatBoot[i] == 1)
                    {
                        s += NumOps.ToDouble(cateBoot[i]);
                        c++;
                    }
                }
                return NumOps.FromDouble(c > 0 ? s / c : 0);
            },
            x, treatment, outcome, 50);

        return (NumOps.FromDouble(att), se);
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
        var effects = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            foreach (var tree in _trees)
            {
                sum += NumOps.ToDouble(PredictTree(tree, x, i));
            }
            effects[i] = NumOps.FromDouble(sum / _trees.Count);
        }

        return effects;
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
    /// Gets all model parameters.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        // For tree models, we return propensity coefficients
        return _propensityCoefficients ?? new Vector<T>(0);
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length > 0)
        {
            _propensityCoefficients = parameters;
            NumFeatures = parameters.Length - 1;
        }
    }

    /// <summary>
    /// Creates a new instance with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new CausalForest<T>(_numTrees, _maxDepth, _minSamplesLeaf, _maxFeatures, _honest, _honestFraction);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of the same type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new CausalForest<T>(_numTrees, _maxDepth, _minSamplesLeaf, _maxFeatures, _honest, _honestFraction);
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

        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            importance[name] = NumOps.FromDouble((double)splitCounts[i] / totalSplits);
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
        public bool IsLeaf { get; set; }
        public int FeatureIndex { get; set; }
        public double Threshold { get; set; }
        public T TreatmentEffect { get; set; } = default!;
        public int NumSamples { get; set; }
        public CausalTree? Left { get; set; }
        public CausalTree? Right { get; set; }
    }
}
