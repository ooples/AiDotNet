using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// F-Regression for feature selection based on linear relationship with continuous target.
/// </summary>
/// <remarks>
/// <para>
/// Computes the F-statistic for a univariate linear regression between each feature and
/// the continuous target. Features with higher F-statistics have stronger linear
/// relationships with the target.
/// </para>
/// <para><b>For Beginners:</b> For regression problems (predicting continuous values),
/// we need to know which features have a linear relationship with what we're predicting.
/// F-regression fits a simple line between each feature and the target, then measures
/// how well that line explains the variation in the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FRegression<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private double[]? _fStatistics;
    private double[]? _pValues;
    private double[]? _rSquaredValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FStatistics => _fStatistics;
    public double[]? PValues => _pValues;
    public double[]? RSquaredValues => _rSquaredValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FRegression(
        int nFeaturesToSelect = 10,
        double pValueThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _pValueThreshold = pValueThreshold;
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

        // Compute target mean
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        // Total sum of squares for target
        double ssTotal = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(target[i]) - yMean;
            ssTotal += diff * diff;
        }

        _fStatistics = new double[p];
        _pValues = new double[p];
        _rSquaredValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute feature mean and variance
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            // Compute regression coefficients
            double sxx = 0, sxy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxx += xDiff * xDiff;
                sxy += xDiff * yDiff;
            }

            // Slope (beta1) and intercept (beta0)
            double beta1 = sxx > 1e-10 ? sxy / sxx : 0;

            // Compute residual sum of squares
            double ssResidual = 0;
            for (int i = 0; i < n; i++)
            {
                double predicted = yMean + beta1 * (NumOps.ToDouble(data[i, j]) - xMean);
                double residual = NumOps.ToDouble(target[i]) - predicted;
                ssResidual += residual * residual;
            }

            // Regression sum of squares
            double ssRegression = ssTotal - ssResidual;

            // R-squared
            _rSquaredValues[j] = ssTotal > 1e-10 ? Math.Max(0, ssRegression / ssTotal) : 0;

            // F-statistic: F = (SSR/1) / (SSE/(n-2))
            double msRegression = ssRegression;
            double msResidual = ssResidual / Math.Max(1, n - 2);

            _fStatistics[j] = msResidual > 1e-10 ? msRegression / msResidual : 0;

            // P-value from F-distribution (1, n-2 degrees of freedom)
            _pValues[j] = FDistributionPValue(_fStatistics[j], 1, Math.Max(1, n - 2));
        }

        // Select significant features or top by F-statistic
        var significant = new List<int>();
        for (int j = 0; j < p; j++)
            if (_pValues[j] < _pValueThreshold)
                significant.Add(j);

        if (significant.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = significant
                .OrderBy(j => _pValues[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _fStatistics
                .Select((f, idx) => (F: f, Index: idx))
                .OrderByDescending(x => x.F)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double FDistributionPValue(double f, int df1, int df2)
    {
        if (f <= 0 || df1 <= 0 || df2 <= 0) return 1;

        // Use incomplete beta function relationship
        double x = df2 / (df2 + df1 * f);
        return IncompleteBeta(df2 / 2.0, df1 / 2.0, x);
    }

    private double IncompleteBeta(double a, double b, double x)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

        // Use continued fraction representation
        double bt = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) +
                            a * Math.Log(x) + b * Math.Log(1 - x));

        if (x < (a + 1) / (a + b + 2))
            return bt * BetaCF(a, b, x) / a;
        else
            return 1 - bt * BetaCF(b, a, 1 - x) / b;
    }

    private double BetaCF(double a, double b, double x)
    {
        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        if (Math.Abs(d) < 1e-30) d = 1e-30;
        d = 1 / d;
        double h = d;

        for (int m = 1; m <= 100; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1) < 1e-10) break;
        }

        return h;
    }

    private double LogGamma(double x)
    {
        double[] cof = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
                        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
            ser += cof[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
