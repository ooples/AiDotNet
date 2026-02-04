using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Welch's t-test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Welch's t-test is a variant of the Student's t-test that does not assume equal
/// variances between the two groups. It uses the Welch-Satterthwaite approximation
/// for degrees of freedom.
/// </para>
/// <para><b>For Beginners:</b> When comparing two groups that might have different
/// amounts of spread (variance), Welch's test is more reliable than the standard
/// t-test. It's generally recommended as the default choice for comparing two
/// group means.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class WelchTTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private double[]? _tStatistics;
    private double[]? _pValues;
    private double[]? _degreesOfFreedom;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double[]? TStatistics => _tStatistics;
    public double[]? PValues => _pValues;
    public double[]? DegreesOfFreedom => _degreesOfFreedom;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public WelchTTest(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WelchTTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Split samples by class
        var group0 = new List<int>();
        var group1 = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                group0.Add(i);
            else
                group1.Add(i);
        }

        int n0 = group0.Count;
        int n1 = group1.Count;

        if (n0 < 2 || n1 < 2)
        {
            _tStatistics = new double[p];
            _pValues = Enumerable.Repeat(1.0, p).ToArray();
            _degreesOfFreedom = Enumerable.Repeat((double)(n - 2), p).ToArray();
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        _tStatistics = new double[p];
        _pValues = new double[p];
        _degreesOfFreedom = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute means
            double mean0 = group0.Sum(i => NumOps.ToDouble(data[i, j])) / n0;
            double mean1 = group1.Sum(i => NumOps.ToDouble(data[i, j])) / n1;

            // Compute sample variances
            double var0 = group0.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2)) / (n0 - 1);
            double var1 = group1.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2)) / (n1 - 1);

            // Welch's t-statistic
            double se = Math.Sqrt(var0 / n0 + var1 / n1);
            double t = se > 1e-10 ? (mean0 - mean1) / se : 0;

            // Welch-Satterthwaite degrees of freedom
            double num = Math.Pow(var0 / n0 + var1 / n1, 2);
            double den = Math.Pow(var0 / n0, 2) / (n0 - 1) + Math.Pow(var1 / n1, 2) / (n1 - 1);
            double df = den > 0 ? num / den : n0 + n1 - 2;

            _tStatistics[j] = t;
            _degreesOfFreedom[j] = df;

            // Use t-distribution CDF approximation
            _pValues[j] = 2 * (1 - TDistributionCDF(Math.Abs(t), df));
        }

        // Select features with smallest p-values
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _pValues
            .Select((pval, idx) => (PValue: pval, Index: idx))
            .OrderBy(x => x.PValue)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double TDistributionCDF(double t, double df)
    {
        // Approximation using incomplete beta function relationship
        // For large df, use normal approximation
        if (df > 100)
        {
            return NormalCDF(t);
        }

        // Use approximation for smaller df
        double x = df / (df + t * t);
        double a = df / 2;
        double b = 0.5;

        // Incomplete beta function approximation
        double beta = IncompleteBeta(x, a, b);

        if (t > 0)
            return 1 - 0.5 * beta;
        else
            return 0.5 * beta;
    }

    private double IncompleteBeta(double x, double a, double b)
    {
        // Continued fraction approximation for incomplete beta
        if (x < 0 || x > 1) return 0;
        if (x == 0) return 0;
        if (x == 1) return 1;

        // Use symmetry if needed
        if (x > (a + 1) / (a + b + 2))
            return 1 - IncompleteBeta(1 - x, b, a);

        double bt = Math.Exp(
            LogGamma(a + b) - LogGamma(a) - LogGamma(b) +
            a * Math.Log(x) + b * Math.Log(1 - x));

        // Continued fraction approximation
        double cf = ContinuedFraction(x, a, b);

        return bt * cf / a;
    }

    private double ContinuedFraction(double x, double a, double b)
    {
        const int maxIter = 100;
        const double eps = 1e-10;

        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;

        if (Math.Abs(d) < eps) d = eps;
        d = 1 / d;
        double h = d;

        for (int m = 1; m <= maxIter; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < eps) d = eps;
            c = 1 + aa / c;
            if (Math.Abs(c) < eps) c = eps;
            d = 1 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < eps) d = eps;
            c = 1 + aa / c;
            if (Math.Abs(c) < eps) c = eps;
            d = 1 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1) < eps) break;
        }

        return h;
    }

    private double LogGamma(double x)
    {
        // Stirling's approximation
        if (x <= 0) return 0;

        double[] coef = { 76.18009172947146, -86.50532032941677,
            24.01409824083091, -1.231739572450155,
            0.1208650973866179e-2, -0.5395239384953e-5 };

        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;

        for (int j = 0; j < 6; j++)
            ser += coef[j] / ++y;

        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    private double NormalCDF(double x)
    {
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WelchTTest has not been fitted.");

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
        throw new NotSupportedException("WelchTTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WelchTTest has not been fitted.");

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
