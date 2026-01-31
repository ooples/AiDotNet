using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Kruskal-Wallis H test for non-parametric feature selection.
/// </summary>
/// <remarks>
/// <para>
/// A non-parametric alternative to one-way ANOVA that doesn't assume normal
/// distributions. Tests whether samples from different groups originate from
/// the same distribution based on ranks.
/// </para>
/// <para><b>For Beginners:</b> Unlike ANOVA which assumes data follows a normal
/// bell curve, Kruskal-Wallis works with any data distribution. It converts
/// values to ranks and tests whether the ranks differ between groups. This is
/// more robust when your data has outliers or non-normal distributions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KruskalWallisTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private double[]? _hStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? HStatistics => _hStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KruskalWallisTest(
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
            "KruskalWallisTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var classGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classGroups.ContainsKey(label))
                classGroups[label] = new List<int>();
            classGroups[label].Add(i);
        }

        int k = classGroups.Count;
        if (k < 2)
            throw new ArgumentException("At least 2 classes are required.");

        _hStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get all values and compute ranks
            var valuesWithIndices = new List<(double Value, int Index)>();
            for (int i = 0; i < n; i++)
                valuesWithIndices.Add((NumOps.ToDouble(data[i, j]), i));

            // Sort and assign ranks (handling ties with average rank)
            var sorted = valuesWithIndices.OrderBy(x => x.Value).ToList();
            var ranks = new double[n];

            int i2 = 0;
            while (i2 < n)
            {
                int j2 = i2;
                while (j2 < n && Math.Abs(sorted[j2].Value - sorted[i2].Value) < 1e-10)
                    j2++;

                // Average rank for ties
                double avgRank = (i2 + j2 + 1) / 2.0;
                for (int m = i2; m < j2; m++)
                    ranks[sorted[m].Index] = avgRank;

                i2 = j2;
            }

            // Compute H statistic
            double sumRankSq = 0;
            foreach (var group in classGroups.Values)
            {
                double groupRankSum = group.Sum(idx => ranks[idx]);
                sumRankSq += (groupRankSum * groupRankSum) / group.Count;
            }

            double H = (12.0 / (n * (n + 1))) * sumRankSq - 3 * (n + 1);

            // Tie correction
            var tieGroups = new Dictionary<double, int>();
            for (int m = 0; m < n; m++)
            {
                double val = NumOps.ToDouble(data[m, j]);
                if (!tieGroups.ContainsKey(val))
                    tieGroups[val] = 0;
                tieGroups[val]++;
            }

            double tieCorrection = 0;
            foreach (var count in tieGroups.Values.Where(c => c > 1))
                tieCorrection += count * count * count - count;

            if (tieCorrection > 0)
                H /= (1 - tieCorrection / (n * n * n - n));

            _hStatistics[j] = H;

            // P-value using chi-square approximation with k-1 degrees of freedom
            int df = k - 1;
            _pValues[j] = ChiSquarePValue(H, df);
        }

        // Select significant features
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
            _selectedIndices = _hStatistics
                .Select((h, idx) => (H: h, Index: idx))
                .OrderByDescending(x => x.H)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double ChiSquarePValue(double chi2, int df)
    {
        if (chi2 <= 0 || df <= 0) return 1;
        return 1 - IncompleteGamma(df / 2.0, chi2 / 2.0);
    }

    private double IncompleteGamma(double a, double x)
    {
        if (x < 0 || a <= 0) return 0;
        if (x == 0) return 0;

        if (x < a + 1)
            return GammaSeries(a, x);
        else
            return 1 - GammaCF(a, x);
    }

    private double GammaSeries(double a, double x)
    {
        double sum = 1 / a;
        double del = 1 / a;
        for (int i = 1; i <= 100; i++)
        {
            del *= x / (a + i);
            sum += del;
            if (Math.Abs(del) < Math.Abs(sum) * 1e-10) break;
        }
        return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
    }

    private double GammaCF(double a, double x)
    {
        double b = x + 1 - a;
        double c = 1e30;
        double d = 1 / b;
        double h = d;

        for (int i = 1; i <= 100; i++)
        {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (Math.Abs(del - 1) < 1e-10) break;
        }

        return h * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
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
            throw new InvalidOperationException("KruskalWallisTest has not been fitted.");

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
        throw new NotSupportedException("KruskalWallisTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallisTest has not been fitted.");

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
