using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Mode options for GenericUnivariateSelect.
/// </summary>
public enum SelectionMode
{
    /// <summary>Select top k features by score.</summary>
    KBest,
    /// <summary>Select top percentile of features by score.</summary>
    Percentile,
    /// <summary>Select features with p-value below alpha threshold.</summary>
    FPR,
    /// <summary>Select features with false discovery rate below alpha.</summary>
    FDR,
    /// <summary>Select features controlling family-wise error rate.</summary>
    FWE
}

/// <summary>
/// Generic univariate feature selector with configurable selection mode.
/// </summary>
/// <remarks>
/// <para>
/// GenericUnivariateSelect is a flexible feature selector that can operate in different
/// modes: k-best, percentile, FPR (false positive rate), FDR (false discovery rate),
/// or FWE (family-wise error rate). This provides a unified interface for various
/// univariate selection strategies.
/// </para>
/// <para><b>For Beginners:</b> This is like a Swiss Army knife for feature selection.
/// Instead of using different classes for different strategies, you pick a mode:
/// - KBest: "Give me the top 10 features"
/// - Percentile: "Give me the top 10% of features"
/// - FPR/FDR/FWE: "Give me features with statistical significance below 0.05"
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GenericUnivariateSelect<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly SelectionMode _mode;
    private readonly double _param;
    private readonly Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T> PValues)>? _scoreFunc;
    private readonly UnivariateScoreFunction _defaultScoreFunc;

    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public SelectionMode Mode => _mode;
    public double Param => _param;
    public Vector<T>? Scores => _scores;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GenericUnivariateSelect(
        SelectionMode mode = SelectionMode.KBest,
        double param = 10.0,
        Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T> PValues)>? scoreFunc = null,
        UnivariateScoreFunction defaultScoreFunc = UnivariateScoreFunction.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _mode = mode;
        _param = param;
        _scoreFunc = scoreFunc;
        _defaultScoreFunc = defaultScoreFunc;

        ValidateParameters();
    }

    private void ValidateParameters()
    {
        if (!Enum.IsDefined(typeof(UnivariateScoreFunction), _defaultScoreFunc))
            throw new ArgumentException(
                $"Unknown univariate score function {_defaultScoreFunc}.", "defaultScoreFunc");

        // FPR, FDR and FWE threshold p-values; mutual information is a score without a test distribution
        // (scikit-learn's mutual_info_classif returns no p-values), so it cannot drive them.
        bool needsPValues = _mode == SelectionMode.FPR || _mode == SelectionMode.FDR || _mode == SelectionMode.FWE;
        if (needsPValues && _scoreFunc is null && !ProducesPValues(_defaultScoreFunc))
            throw new ArgumentException(
                $"{_defaultScoreFunc} produces no p-values, so it cannot be used with {_mode} mode. " +
                "Use FClassif, FRegression or Chi2, or pass a custom scoreFunc that returns p-values.",
                "defaultScoreFunc");

        switch (_mode)
        {
            case SelectionMode.KBest:
                if (_param < 1)
                    throw new ArgumentException("k must be at least 1 for KBest mode.");
                break;
            case SelectionMode.Percentile:
                if (_param <= 0 || _param > 100)
                    throw new ArgumentException("Percentile must be in (0, 100].");
                break;
            case SelectionMode.FPR:
            case SelectionMode.FDR:
            case SelectionMode.FWE:
                if (_param <= 0 || _param > 1)
                    throw new ArgumentException("Alpha must be in (0, 1] for statistical tests.");
                break;
        }
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GenericUnivariateSelect requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute scores and p-values
        if (_scoreFunc != null)
        {
            var result = _scoreFunc(data, target);
            _scores = result.Scores;
            _pValues = result.PValues;
        }
        else
        {
            (_scores, _pValues) = _defaultScoreFunc switch
            {
                UnivariateScoreFunction.FClassif => ComputeFClassif(data, target, n),
                UnivariateScoreFunction.FRegression => ComputeFRegression(data, target, n),
                UnivariateScoreFunction.Chi2 => ComputeChi2(data, target),
                UnivariateScoreFunction.MutualInfoClassif => ComputeMutualInfoClassif(data, target),
                _ => throw new ArgumentException($"Unknown univariate score function {_defaultScoreFunc}.")
            };
        }

        // Select features based on mode
        _selectedIndices = _mode switch
        {
            SelectionMode.KBest => SelectKBest(p),
            SelectionMode.Percentile => SelectPercentile(p),
            SelectionMode.FPR => SelectFPR(),
            SelectionMode.FDR => SelectFDR(p),
            SelectionMode.FWE => SelectFWE(p),
            _ => SelectKBest(p)
        };

        IsFitted = true;
    }

    private int[] SelectKBest(int p)
    {
        var scores = _scores ?? throw new InvalidOperationException("_scores has not been initialized.");
        int k = Math.Min((int)_param, p);
        return PreprocessingTensorOps.TopKIndicesSorted(FilterStatisticTensorOps.AsTensor(scores), k);
    }

    private int[] SelectPercentile(int p)
    {
        var scores = _scores ?? throw new InvalidOperationException("_scores has not been initialized.");
        int numToSelect = Math.Max(1, (int)Math.Ceiling(p * _param / 100.0));
        return PreprocessingTensorOps.TopKIndicesSorted(FilterStatisticTensorOps.AsTensor(scores), numToSelect);
    }

    private int[] SelectFPR()
    {
        if (_pValues is null || _pValues.Length == 0)
            return [0];

        // Select features with p-value below alpha
        return BelowOrBest(FilterStatisticTensorOps.AsTensor(_pValues), _param);
    }

    private int[] SelectFDR(int p)
    {
        if (_pValues is null || _pValues.Length == 0)
            return [0];

        // Benjamini-Hochberg step-up procedure (scikit-learn SelectFdr): with the p-values sorted ascending,
        // find the largest rank i with p_(i) <= i * alpha / p and keep every feature ranked at or before it.
        // A NaN p-value sorts last and never passes, so it is never kept (scikit-learn's np.sort does the same).
        var engine = Engine;
        var pValues = FilterStatisticTensorOps.AsTensor(_pValues);
        int count = pValues.Length;
        var order = PreprocessingTensorOps.StableOrder(pValues, descending: false);
        var sorted = engine.TensorIndexSelect(pValues, new Tensor<int>((int[])order.Clone(), new[] { count }), 0);
        var thresholds = engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(1, count + 1, 1), NumOps.FromDouble(_param)),
            NumOps.FromDouble(p));
        var passing = FilterStatisticTensorOps.MaskPositions(engine.TensorLessEqual(sorted, thresholds));

        if (passing.Length == 0)
            return BestUnlessNaN(pValues, order[0]);

        var selected = order.Take(passing[passing.Length - 1] + 1).ToArray();
        Array.Sort(selected);
        return selected;
    }

    private int[] SelectFWE(int p)
    {
        if (_pValues is null || _pValues.Length == 0)
            return [0];

        // Bonferroni correction for FWE control
        return BelowOrBest(FilterStatisticTensorOps.AsTensor(_pValues), _param / p);
    }

    // The features whose p-value is below the threshold, or the one with the lowest p-value when none is.
    // NaN compares false against the threshold and orders last, so a NaN p-value is never selected.
    private int[] BelowOrBest(Tensor<T> pValues, double threshold)
    {
        var selected = FilterStatisticTensorOps.MaskPositions(Engine.TensorLessThan(pValues, NumOps.FromDouble(threshold)));
        return selected.Length > 0 ? selected : BestUnlessNaN(pValues, FilterStatisticTensorOps.FirstArgMin(pValues));
    }

    // The fallback "best feature" when nothing passes the test; empty when even the best p-value is NaN
    // (every p-value is NaN), since a NaN p-value carries no evidence for selecting the feature.
    private int[] BestUnlessNaN(Tensor<T> pValues, int best) =>
        NumOps.IsNaN(pValues[best]) ? Array.Empty<int>() : new[] { best };

    // Whether the built-in scorer comes with a p-value per feature.
    private static bool ProducesPValues(UnivariateScoreFunction scoreFunction) =>
        scoreFunction != UnivariateScoreFunction.MutualInfoClassif;

    private static (Vector<T> Scores, Vector<T>? PValues) ComputeChi2(Matrix<T> data, Vector<T> target)
    {
        var scores = SelectorTensorOps.Chi2(Tensor<T>.FromMatrix(data), target, out var pValues);
        return (scores.ToVector(), pValues.ToVector());
    }

    private static (Vector<T> Scores, Vector<T>? PValues) ComputeMutualInfoClassif(Matrix<T> data, Vector<T> target) =>
        (SelectorTensorOps.MutualInfoClassif(Tensor<T>.FromMatrix(data), target).ToVector(), null);

    private (Vector<T> Scores, Vector<T>? PValues) ComputeFClassif(Matrix<T> data, Vector<T> target, int n)
    {
        var engine = Engine;
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int k);

        // One-way ANOVA sums of squares of every feature across the classes
        var (ssb, ssw) = FilterStatisticTensorOps.AnovaSums(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.ClassOneHot<T>(classIndex, k));
        var msb = engine.TensorDivideScalar(ssb, NumOps.FromDouble(Math.Max(1, k - 1)));
        var msw = engine.TensorDivideScalar(ssw, NumOps.FromDouble(Math.Max(1, n - k)));

        // F = MSB / MSW; a feature with no within-class scatter whose class means differ separates the
        // classes perfectly and scores +inf (scikit-learn f_classif), a constant feature scores 0
        var scores = FilterStatisticTensorOps.SeparationAwareRatio(msb, msw, 1e-10);
        return (scores.ToVector(), FPValues(scores, k - 1, n - k));
    }

    private (Vector<T> Scores, Vector<T>? PValues) ComputeFRegression(Matrix<T> data, Vector<T> target, int n)
    {
        var engine = Engine;

        // Centred sums of every feature with the target
        var (sxx, sxy, ssTotal) = FilterStatisticTensorOps.CenteredCrossSums(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));
        var beta = FilterStatisticTensorOps.GuardedDivide(sxy, sxx, 1e-10, NumOps.Zero);

        // SS_residual = sum (y - ybar - beta (x - xbar))^2 = Syy - 2 beta Sxy + beta^2 Sxx
        var ssTotalTensor = FilterStatisticTensorOps.FilledLike(sxx, ssTotal);
        var ssResidual = engine.TensorMax(
            engine.TensorAdd(
                engine.TensorSubtract(ssTotalTensor, engine.TensorMultiplyScalar(engine.TensorMultiply(beta, sxy), NumOps.FromDouble(2))),
                engine.TensorMultiply(engine.TensorSquare(beta), sxx)),
            NumOps.Zero);
        var ssRegression = engine.TensorSubtract(ssTotalTensor, ssResidual);
        var msResidual = engine.TensorDivideScalar(ssResidual, NumOps.FromDouble(Math.Max(1, n - 2)));

        // F = SS_regression / MS_residual; a perfect linear fit leaves no residual and scores +inf
        // (scikit-learn f_regression), a constant feature scores 0
        var scores = FilterStatisticTensorOps.SeparationAwareRatio(ssRegression, msResidual, 1e-10);
        return (scores.ToVector(), FPValues(scores, 1, n - 2));
    }

    // P(F(df1, df2) > F), the exact F-distribution survival function (scipy.stats.f.sf); 1 when a
    // degrees-of-freedom count is not positive.
    private Vector<T> FPValues(Tensor<T> fStatistics, int df1, int df2) =>
        df1 > 0 && df2 > 0
            ? FilterStatisticTensorOps.FSurvival(fStatistics, df1, df2).ToVector()
            : FilterStatisticTensorOps.FilledLike(fStatistics, NumOps.One).ToVector();

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
            throw new InvalidOperationException("GenericUnivariateSelect has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GenericUnivariateSelect does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GenericUnivariateSelect has not been fitted.");

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
