using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Regression;

/// <summary>
/// F-statistic based feature selection for regression.
/// </summary>
/// <remarks>
/// <para>
/// F-Regression computes the F-statistic between each feature and the target,
/// which measures the linear dependency between them. Features with higher
/// F-scores have stronger linear relationships with the target.
/// </para>
/// <para><b>For Beginners:</b> The F-test checks how well each feature can predict
/// the target using a simple line (linear relationship). Features that follow
/// a straighter line with the target get higher scores. Good for finding features
/// that have clear linear patterns with your outcome.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FRegression<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fScores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FScores => _fScores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FRegression(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FRegression requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _fScores = new double[p];
        _pValues = new double[p];

        // Compute target mean and SST
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double sst = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(target[i]) - yMean;
            sst += diff * diff;
        }

        for (int j = 0; j < p; j++)
        {
            // Compute feature statistics
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            // Compute covariance and variance
            double sxy = 0, sxx = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
            }

            if (sxx < 1e-10 || sst < 1e-10)
            {
                _fScores[j] = 0;
                _pValues[j] = 1.0;
                continue;
            }

            // Regression coefficient and R-squared
            double beta = sxy / sxx;
            double ssr = beta * beta * sxx; // Regression sum of squares
            double sse = sst - ssr; // Error sum of squares

            // F-statistic: (SSR/1) / (SSE/(n-2))
            double msr = ssr;
            double mse = sse / (n - 2);

            _fScores[j] = mse > 1e-10 ? msr / mse : 0;

            // Approximate p-value using F-distribution approximation
            // For large n, F follows chi-square approximately
            _pValues[j] = ComputePValue(_fScores[j], 1, n - 2);
        }

        // Select top features by F-score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _fScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputePValue(double fStatistic, int df1, int df2)
    {
        if (fStatistic <= 0 || df2 <= 0) return 1.0;

        // Use incomplete beta function approximation
        double x = df2 / (df2 + df1 * fStatistic);

        // Simple approximation using normal distribution for large df
        if (df1 >= 1 && df2 >= 30)
        {
            double z = Math.Pow(fStatistic, 1.0 / 3.0) * (1 - 2.0 / (9 * df2)) - (1 - 2.0 / (9 * df1));
            double se = Math.Sqrt(2.0 / (9 * df1) + 2.0 / (9 * df2));
            double standardZ = z / se;

            // One-sided p-value from standard normal
            return 0.5 * (1 - Erf(standardZ / Math.Sqrt(2)));
        }

        // For smaller df, use a rough approximation
        return Math.Exp(-0.5 * fStatistic / Math.Max(1, df1));
    }

    private double Erf(double x)
    {
        // Approximation of the error function
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FRegression has not been fitted.");

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
        throw new NotSupportedException("FRegression does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FRegression has not been fitted.");

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
