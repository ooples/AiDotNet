using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Select features based on False Positive Rate threshold.
/// </summary>
/// <remarks>
/// <para>
/// Selects features with p-values below a specified alpha threshold. Does not control
/// for multiple testing, so at alpha=0.05, approximately 5% of null (non-informative)
/// features will be incorrectly selected.
/// </para>
/// <para><b>For Beginners:</b> This method uses statistical testing to find features
/// that are unlikely to be uninformative by chance. If you set alpha=0.05, it keeps
/// features where there's less than 5% chance the feature-target relationship is random.
/// However, when testing many features, some false positives slip through.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SelectFPR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _alpha;
    private readonly Func<Matrix<T>, Vector<T>, int, double>? _scoringFunction;

    private double[]? _scores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Alpha => _alpha;
    public double[]? Scores => _scores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SelectFPR(
        double alpha = 0.05,
        Func<Matrix<T>, Vector<T>, int, double>? scoringFunction = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (alpha <= 0 || alpha >= 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));

        _alpha = alpha;
        _scoringFunction = scoringFunction;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectFPR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _scores = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            if (_scoringFunction is not null)
            {
                _scores[j] = _scoringFunction(data, target, j);
                _pValues[j] = 1.0 / (1.0 + _scores[j]);
            }
            else
            {
                var (fStat, pVal) = ComputeFStatistic(data, target, j, n);
                _scores[j] = fStat;
                _pValues[j] = pVal;
            }
        }

        // Select features with p-value < alpha
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_pValues[j] < _alpha)
                selected.Add(j);
        }

        // If none selected, select the best one
        if (selected.Count == 0)
        {
            int best = 0;
            for (int j = 1; j < p; j++)
                if (_pValues[j] < _pValues[best])
                    best = j;
            selected.Add(best);
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private (double FStat, double PValue) ComputeFStatistic(Matrix<T> data, Vector<T> target, int featureIdx, int n)
    {
        // Compute correlation F-statistic
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, featureIdx]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double ssXY = 0, ssXX = 0, ssYY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = NumOps.ToDouble(data[i, featureIdx]) - xMean;
            double dy = NumOps.ToDouble(target[i]) - yMean;
            ssXY += dx * dy;
            ssXX += dx * dx;
            ssYY += dy * dy;
        }

        if (ssXX < 1e-10 || ssYY < 1e-10)
            return (0, 1);

        double r2 = (ssXY * ssXY) / (ssXX * ssYY);
        double fStat = r2 * (n - 2) / (1 - r2 + 1e-10);

        // Approximate p-value using F-distribution
        double pValue = ApproximateFPValue(fStat, 1, n - 2);
        return (fStat, pValue);
    }

    private double ApproximateFPValue(double f, int df1, int df2)
    {
        if (f <= 0 || df1 <= 0 || df2 <= 0)
            return 1;

        // Using incomplete beta function approximation
        double x = df2 / (df2 + df1 * f);
        double a = df2 / 2.0;
        double b = df1 / 2.0;

        return IncompleteBeta(x, a, b);
    }

    private double IncompleteBeta(double x, double a, double b)
    {
        if (x < 0 || x > 1) return 1;
        if (x == 0) return 0;
        if (x == 1) return 1;

        double bt = Math.Exp(a * Math.Log(x) + b * Math.Log(1 - x));
        if (x < (a + 1) / (a + b + 2))
            return bt * BetaCF(x, a, b) / a;
        else
            return 1 - bt * BetaCF(1 - x, b, a) / b;
    }

    private double BetaCF(double x, double a, double b)
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

            if (Math.Abs(del - 1) < 1e-7) break;
        }

        return h;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectFPR has not been fitted.");

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
        throw new NotSupportedException("SelectFPR does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectFPR has not been fitted.");

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
