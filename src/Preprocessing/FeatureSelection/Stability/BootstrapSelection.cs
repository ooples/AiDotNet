using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Bootstrap-based Feature Selection for robust feature importance estimation.
/// </summary>
/// <remarks>
/// <para>
/// Uses bootstrap resampling (sampling with replacement) to estimate the
/// variability of feature importance scores. Features with consistently
/// high importance across bootstrap samples are selected.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap is like asking the same question many
/// times to slightly different versions of your data (by random sampling with
/// replacement). This helps you understand which features are reliably important
/// and which ones might just be important by chance in one particular sample.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BootstrapSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _confidenceLevel;
    private readonly int? _randomState;

    private double[]? _meanImportances;
    private double[]? _stdImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBootstraps => _nBootstraps;
    public double ConfidenceLevel => _confidenceLevel;
    public double[]? MeanImportances => _meanImportances;
    public double[]? StdImportances => _stdImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BootstrapSelection(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double confidenceLevel = 0.95,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBootstraps < 1)
            throw new ArgumentException("Number of bootstraps must be at least 1.", nameof(nBootstraps));
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentException("Confidence level must be between 0 and 1.", nameof(confidenceLevel));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _confidenceLevel = confidenceLevel;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var allImportances = new double[_nBootstraps, p];

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Bootstrap sample (with replacement)
            var bootstrapIndices = new int[n];
            for (int i = 0; i < n; i++)
                bootstrapIndices[i] = random.Next(n);

            // Compute importance scores on bootstrap sample
            double yMean = 0;
            foreach (int i in bootstrapIndices)
                yMean += NumOps.ToDouble(target[i]);
            yMean /= n;

            for (int j = 0; j < p; j++)
            {
                double xMean = 0;
                foreach (int i in bootstrapIndices)
                    xMean += NumOps.ToDouble(data[i, j]);
                xMean /= n;

                double sxy = 0, sxx = 0, syy = 0;
                foreach (int i in bootstrapIndices)
                {
                    double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                    double yDiff = NumOps.ToDouble(target[i]) - yMean;
                    sxy += xDiff * yDiff;
                    sxx += xDiff * xDiff;
                    syy += yDiff * yDiff;
                }

                allImportances[b, j] = (sxx > 1e-10 && syy > 1e-10)
                    ? Math.Abs(sxy / Math.Sqrt(sxx * syy))
                    : 0;
            }
        }

        // Compute mean and std of importances
        _meanImportances = new double[p];
        _stdImportances = new double[p];

        for (int j = 0; j < p; j++)
        {
            var values = new double[_nBootstraps];
            for (int b = 0; b < _nBootstraps; b++)
                values[b] = allImportances[b, j];

            _meanImportances[j] = values.Average();
            double variance = values.Sum(v => Math.Pow(v - _meanImportances[j], 2)) / _nBootstraps;
            _stdImportances[j] = Math.Sqrt(variance);
        }

        // Compute lower confidence bound (mean - z*std)
        double zScore = NormalQuantile(_confidenceLevel);
        var confidenceScores = new double[p];
        for (int j = 0; j < p; j++)
            confidenceScores[j] = _meanImportances[j] - zScore * _stdImportances[j] / Math.Sqrt(_nBootstraps);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => confidenceScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double NormalQuantile(double p)
    {
        // Approximation of inverse normal CDF
        double[] a = { -3.969683028665376e+01, 2.209460984245205e+02,
                       -2.759285104469687e+02, 1.383577518672690e+02,
                       -3.066479806614716e+01, 2.506628277459239e+00 };
        double[] b = { -5.447609879822406e+01, 1.615858368580409e+02,
                       -1.556989798598866e+02, 6.680131188771972e+01,
                       -1.328068155288572e+01 };
        double[] c = { -7.784894002430293e-03, -3.223964580411365e-01,
                       -2.400758277161838e+00, -2.549732539343734e+00,
                        4.374664141464968e+00, 2.938163982698783e+00 };
        double[] d = { 7.784695709041462e-03, 3.224671290700398e-01,
                       2.445134137142996e+00, 3.754408661907416e+00 };

        double pLow = 0.02425;
        double pHigh = 1 - pLow;
        double q, r;

        if (p < pLow)
        {
            q = Math.Sqrt(-2 * Math.Log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        else if (p <= pHigh)
        {
            q = p - 0.5;
            r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }
        else
        {
            q = Math.Sqrt(-2 * Math.Log(1 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapSelection has not been fitted.");

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
        throw new NotSupportedException("BootstrapSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapSelection has not been fitted.");

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
