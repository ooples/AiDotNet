using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Filter-Wrapper Hybrid Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines filter and wrapper methods: first uses a fast filter to reduce
/// the feature space, then applies a more accurate wrapper method on the
/// reduced set.
/// </para>
/// <para><b>For Beginners:</b> This method gets the best of both worlds.
/// First, it quickly eliminates obviously irrelevant features using simple
/// statistics (filter). Then, it carefully evaluates the remaining features
/// using model performance (wrapper). This is faster than pure wrapper methods
/// but more accurate than pure filter methods.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FilterWrapperHybridSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _filterRatio;
    private readonly int _wrapperIterations;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FilterWrapperHybridSelector(
        int nFeaturesToSelect = 10,
        int filterRatio = 3,
        int wrapperIterations = 50,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _filterRatio = filterRatio;
        _wrapperIterations = wrapperIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FilterWrapperHybridSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var rand = RandomHelper.CreateSecureRandom();
        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Phase 1: Filter - compute correlation-based scores
        var filterScores = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Select top features from filter
        int filterCount = Math.Min(_nFeaturesToSelect * _filterRatio, p);
        var remaining = PreprocessingTensorOps.StableOrder(filterScores, descending: true).Take(filterCount).ToList();

        // Phase 2: Wrapper - forward selection on filtered features. One shuffle defines the cross-validation
        // folds of every candidate subset, so all candidates are compared on the same splits.
        var foldOrder = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).ToArray();
        var selected = new List<int>();
        var bestOverallScore = PreprocessingTensorOps.Filled(new[] { 1 }, NumOps.MinValue);

        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            int bestFeature = -1;
            Tensor<T>? bestScore = null;

            foreach (int j in remaining)
            {
                var testSet = new List<int>(selected) { j };
                var score = CrossValidatedRSquared(x, y, testSet, foldOrder);
                if (bestScore is null || PopulationSearchOps.Greater(score, bestScore))
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0 && bestScore is not null
                && PopulationSearchOps.Greater(bestScore, PopulationSearchOps.Affine(bestOverallScore, 1, -0.001)))
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                bestOverallScore = bestScore;
            }
            else
            {
                break;
            }
        }

        // Compute final scores (combination of filter and selection order): filter score + (count - rank)
        var rankBonus = new int[p];
        for (int i = 0; i < selected.Count; i++)
            rankBonus[selected[i]] = selected.Count - i;
        var bonus = new Tensor<int>(rankBonus, new[] { p }).Cast<T>();
        var featureScores = engine.TensorMultiply(
            engine.TensorAdd(filterScores, bonus), PopulationSearchOps.IndexMask<T>(selected.ToArray(), p));
        _featureScores = featureScores.ToVector();

        _selectedIndices = selected.OrderBy(i => i).ToArray();
        IsFitted = true;
    }

    /// <summary>
    /// Mean over min(5, n) contiguous folds of <paramref name="foldOrder"/> of the test R^2 of a least-squares fit
    /// with intercept (ridge 1e-6) on the other rows. A fold with fewer training rows than coefficients, no test
    /// rows or a constant test target contributes 0. Returns [1].
    /// </summary>
    internal static Tensor<T> CrossValidatedRSquared(Tensor<T> x, Tensor<T> y, List<int> features, int[] foldOrder)
    {
        // Half and BFloat16 cross-validate in double: the fold Gram matrix passes 65504 (eight rows of 300 give
        // 720000) before RidgeCoefficients sees it, and widening inside the solve cannot recover it.
        if (PreprocessingTensorOps.IsNarrowFloat<T>())
            return FilterWrapperHybridSelector<double>.CrossValidatedRSquared(
                x.Cast<double>(), y.Cast<double>(), features, foldOrder).Cast<T>();

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = foldOrder.Length;
        int folds = Math.Min(5, n);
        var total = new Tensor<T>(new[] { 1 });
        if (features.Count == 0)
            return total;

        var design = SubsetRegressionOps.WithIntercept(SubsetRegressionOps.Columns(x, features));
        for (int fold = 0; fold < folds; fold++)
        {
            int testStart = fold * n / folds;
            int testEnd = (fold + 1) * n / folds;
            var trainIdx = foldOrder.Where((_, i) => i < testStart || i >= testEnd).ToArray();
            var testIdx = foldOrder.Skip(testStart).Take(testEnd - testStart).ToArray();
            if (trainIdx.Length < features.Count + 1 || testIdx.Length == 0)
                continue;

            var coefficients = SubsetRegressionOps.RidgeCoefficients(
                SubsetRegressionOps.Rows(design, trainIdx), SubsetRegressionOps.Rows(y, trainIdx), 1e-6);
            var testY = SubsetRegressionOps.Rows(y, testIdx);
            var prediction = SubsetRegressionOps.Predict(SubsetRegressionOps.Rows(design, testIdx), coefficients);
            var rSquared = SubsetRegressionOps.RSquared(
                testY, prediction, engine.ReduceMean(testY, new[] { 0 }, true), out _);
            total = engine.TensorAdd(total, rSquared);
        }

        return engine.TensorDivideScalar(total, ops.FromDouble(folds));
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FilterWrapperHybridSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FilterWrapperHybridSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FilterWrapperHybridSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
