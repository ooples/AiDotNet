using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class BorutaFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nIterations;
    private readonly double _percentile;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _getImportances;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NIterations => _nIterations;
    public double Percentile => _percentile;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BorutaFS(
        int nIterations = 100,
        double percentile = 99.0,
        Func<Matrix<T>, Vector<T>, double[]>? getImportances = null,
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BorutaFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

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

        // Track hit counts (feature beats max shadow)
        var hits = new int[p];
        _featureImportances = new double[p];

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Create shadow features by shuffling original
            var extendedData = new T[n, p * 2];
            for (int j = 0; j < p; j++)
            {
                // Original feature
                for (int i = 0; i < n; i++)
                    extendedData[i, j] = data[i, j];

                // Shadow feature (shuffled)
                var shuffledIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                for (int i = 0; i < n; i++)
                    extendedData[i, p + j] = data[shuffledIndices[i], j];
            }

            var extendedMatrix = new Matrix<T>(extendedData);

            // Get importances for all features
            var importances = getImportances(extendedMatrix, target);

            // Find max shadow importance
            double maxShadow = 0;
            for (int j = p; j < 2 * p; j++)
            {
                if (importances[j] > maxShadow)
                    maxShadow = importances[j];
            }

            // Count hits (original feature > max shadow)
            for (int j = 0; j < p; j++)
            {
                if (importances[j] > maxShadow)
                    hits[j]++;
                _featureImportances[j] += importances[j];
            }
        }

        // Average importances
        for (int j = 0; j < p; j++)
            _featureImportances[j] /= _nIterations;

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
            .OrderByDescending(j => _featureImportances[j])
            .OrderBy(x => x)
            .ToArray();

        // If no features confirmed, take top by importance
        if (_selectedIndices.Length == 0)
        {
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _featureImportances[j])
                .Take(Math.Min(10, p))
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double[] DefaultGetImportances(Matrix<T> data, Vector<T> target)
    {
        // Use correlation as simple importance measure
        int n = data.Rows;
        int p = data.Columns;
        var importances = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            double corr = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
            importances[j] = Math.Abs(corr);
        }

        return importances;
    }

    private double NormalInverse(double p)
    {
        // Approximation of inverse normal CDF
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

        double q = p - 0.5;
        double r;

        if (Math.Abs(q) <= 0.425)
        {
            r = 0.180625 - q * q;
            return q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) /
                       (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }

        r = q < 0 ? p : 1 - p;
        r = Math.Sqrt(-Math.Log(r));

        double result = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) /
                        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);

        return q < 0 ? -result : result;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BorutaFS has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BorutaFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BorutaFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
