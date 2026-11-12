using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Boruta algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Boruta creates "shadow" features by shuffling each original feature, then trains
/// a model to compare original features against their shadows. Features that consistently
/// beat the best shadow feature are confirmed as important; those that don't are rejected.
/// </para>
/// <para><b>For Beginners:</b> Boruta creates fake versions of your features by randomly
/// shuffling their values. It then asks: "Is this real feature more useful than the best
/// fake feature?" Features that consistently outperform their fake counterparts are kept.
/// This is more rigorous than just picking the top N features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BorutaFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _nIterations;
    private readonly double _percentile;
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _getImportances;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NIterations => _nIterations;
    public double Percentile => _percentile;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BorutaFS(
        int nIterations = 100,
        double percentile = 99.0,
        Func<Matrix<T>, Vector<T>, Vector<T>>? getImportances = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));
        if (percentile <= 0 || percentile > 100)
            throw new ArgumentException("Percentile must be between 0 and 100.", nameof(percentile));

        _nIterations = nIterations;
        _percentile = percentile;
        _getImportances = getImportances;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BorutaFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var getImportances = _getImportances ?? DefaultGetImportances;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            columns[j] = PreprocessingTensorOps.ColumnVector(X, j);
        T zero = NumOps.Zero;

        // Track hit counts (feature beats max shadow)
        var hits = new int[p];
        var importanceSums = new Tensor<T>(new[] { p });

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Create shadow features by shuffling original, one permutation per feature in order
            var shadows = new Tensor<T>[p];
            for (int j = 0; j < p; j++)
            {
                var shuffledIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                shadows[j] = engine.Reshape(
                    engine.TensorIndexSelect(columns[j], new Tensor<int>(shuffledIndices, new[] { n }), 0), new[] { n, 1 });
            }

            var extendedMatrix = engine.TensorConcatenate(new[] { X, engine.TensorConcatenate(shadows, 1) }, 1).ToMatrix();

            // Get importances for all features
            var importances = Tensor<T>.FromVector(getImportances(extendedMatrix, target));
            var original = engine.TensorSlice(importances, new[] { 0 }, new[] { p });

            // Find max shadow importance (at least 0)
            T maxShadow = PreprocessingTensorOps.ScalarMax(zero,
                engine.TensorMaxValue(engine.TensorSlice(importances, new[] { p }, new[] { p })));

            // Count hits (original feature > max shadow)
            foreach (int j in PreprocessingTensorOps.MaskPositions(engine.TensorGreaterThan(original, maxShadow)))
                hits[j]++;
            importanceSums = engine.TensorAdd(importanceSums, original);
        }

        // Average importances
        var featureImportances = engine.TensorDivideScalar(importanceSums, NumOps.FromDouble(_nIterations));
        _featureImportances = featureImportances.ToVector();

        // Compute threshold using binomial test approximation
        // Feature is confirmed if hits significantly exceed 50%
        double expectedHits = _nIterations * 0.5;
        double stdHits = Math.Sqrt(_nIterations * 0.25);

        // Threshold: at least (expected + z*std) hits
        // Using percentile to determine z-score
        double zScore = NormalInverse(_percentile / 100.0);
        double threshold = expectedHits + zScore * stdHits;

        // Select confirmed features
        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => hits[j] >= threshold)
            .ToArray();

        // If no features confirmed, take top by importance
        if (_selectedIndices.Length == 0)
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(featureImportances, Math.Min(10, p));
        }

        IsFitted = true;
    }

    private Vector<T> DefaultGetImportances(Matrix<T> data, Vector<T> target)
    {
        // Use absolute Pearson correlation as simple importance measure (0 for a constant feature or target)
        return Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target))).ToVector();
    }

    /// <summary>
    /// Inverse standard normal CDF by Acklam's rational approximation (relative error below 1.2e-9):
    /// a central rational function in <c>(p - 0.5)^2</c> for <c>0.02425 &lt;= p &lt;= 0.97575</c> and a tail rational
    /// function in <c>sqrt(-2 ln(min(p, 1 - p)))</c> beyond.
    /// </summary>
    private static double NormalInverse(double p)
    {
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        double[] a = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00
        ];
        double[] b = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01
        ];
        double[] c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00
        ];
        double[] d = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00
        ];
        const double pLow = 0.02425;

        if (p < pLow || p > 1 - pLow)
        {
            double tail = Math.Sqrt(-2 * Math.Log(p < pLow ? p : 1 - p));
            double result = (((((c[0] * tail + c[1]) * tail + c[2]) * tail + c[3]) * tail + c[4]) * tail + c[5]) /
                            ((((d[0] * tail + d[1]) * tail + d[2]) * tail + d[3]) * tail + 1);
            return p < pLow ? result : -result;
        }

        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
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
            throw new InvalidOperationException("BorutaFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BorutaFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BorutaFS has not been fitted.");

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
