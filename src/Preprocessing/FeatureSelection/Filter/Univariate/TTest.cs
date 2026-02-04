using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Student's t-test for binary classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Performs independent two-sample t-tests between feature values for each class.
/// Features with significant differences between class means are selected as they
/// help distinguish between classes.
/// </para>
/// <para><b>For Beginners:</b> The t-test asks: are the average values of this feature
/// really different between class 0 and class 1, or could the difference be due to chance?
/// A small p-value (high t-statistic) means the feature genuinely helps separate classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private double[]? _tStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? TStatistics => _tStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TTest(
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
            "TTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Separate samples by class (binary)
        var class0 = new List<int>();
        var class1 = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        if (class0.Count < 2 || class1.Count < 2)
            throw new ArgumentException("Both classes must have at least 2 samples for t-test.");

        int n0 = class0.Count;
        int n1 = class1.Count;
        int df = n0 + n1 - 2;

        _tStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute means and variances for each class
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0)
                mean0 += NumOps.ToDouble(data[i, j]);
            mean0 /= n0;

            foreach (int i in class1)
                mean1 += NumOps.ToDouble(data[i, j]);
            mean1 /= n1;

            double var0 = 0, var1 = 0;
            foreach (int i in class0)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean0;
                var0 += diff * diff;
            }
            var0 /= (n0 - 1);

            foreach (int i in class1)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean1;
                var1 += diff * diff;
            }
            var1 /= (n1 - 1);

            // Pooled variance
            double pooledVar = ((n0 - 1) * var0 + (n1 - 1) * var1) / df;
            double pooledStdErr = Math.Sqrt(pooledVar * (1.0 / n0 + 1.0 / n1));

            // t-statistic
            _tStatistics[j] = pooledStdErr > 1e-10 ? (mean0 - mean1) / pooledStdErr : 0;

            // Two-tailed p-value using t-distribution approximation
            _pValues[j] = TDistributionPValue(Math.Abs(_tStatistics[j]), df);
        }

        // Select significant features or top by t-statistic
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
            _selectedIndices = _tStatistics
                .Select((t, idx) => (T: Math.Abs(t), Index: idx))
                .OrderByDescending(x => x.T)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double TDistributionPValue(double t, int df)
    {
        // Two-tailed p-value using approximation
        double x = df / (df + t * t);
        return IncompleteBeta(df / 2.0, 0.5, x);
    }

    private double IncompleteBeta(double a, double b, double x)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

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
            throw new InvalidOperationException("TTest has not been fitted.");

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
        throw new NotSupportedException("TTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TTest has not been fitted.");

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
