using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Chi-square test for feature selection in classification with categorical features.
/// </summary>
/// <remarks>
/// <para>
/// Uses the chi-square test of independence to measure the association between
/// features and target classes. Features with high chi-square values show strong
/// dependence on the target and are selected.
/// </para>
/// <para><b>For Beginners:</b> Chi-square tests whether a feature's values are
/// distributed differently across classes. If feature values are independent of
/// the class (evenly distributed), they can't help predict the class. High
/// chi-square scores indicate strong association with the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChiSquareTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;
    private readonly int _nBins;

    private double[]? _chiSquareStats;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ChiSquareStats => _chiSquareStats;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ChiSquareTest(
        int nFeaturesToSelect = 10,
        double pValueThreshold = 0.05,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _pValueThreshold = pValueThreshold;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ChiSquareTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get unique classes
        var classes = new HashSet<int>();
        for (int i = 0; i < n; i++)
            classes.Add((int)Math.Round(NumOps.ToDouble(target[i])));

        int nClasses = classes.Count;
        var classLabels = classes.OrderBy(x => x).ToArray();
        var classIndex = classLabels.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

        _chiSquareStats = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get feature range for binning
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }

            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            // Create contingency table
            var contingency = new int[_nBins, nClasses];
            var rowTotals = new int[_nBins];
            var colTotals = new int[nClasses];

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                int bin = Math.Min((int)(((val - minVal) / range) * _nBins), _nBins - 1);
                int classIdx = classIndex[(int)Math.Round(NumOps.ToDouble(target[i]))];

                contingency[bin, classIdx]++;
                rowTotals[bin]++;
                colTotals[classIdx]++;
            }

            // Compute chi-square statistic
            double chi2 = 0;
            for (int b = 0; b < _nBins; b++)
            {
                for (int c = 0; c < nClasses; c++)
                {
                    double expected = (double)rowTotals[b] * colTotals[c] / n;
                    if (expected > 0)
                    {
                        double observed = contingency[b, c];
                        chi2 += Math.Pow(observed - expected, 2) / expected;
                    }
                }
            }

            _chiSquareStats[j] = chi2;

            // Degrees of freedom
            int df = (_nBins - 1) * (nClasses - 1);
            _pValues[j] = ChiSquarePValue(chi2, df);
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
            _selectedIndices = _chiSquareStats
                .Select((chi2, idx) => (Chi2: chi2, Index: idx))
                .OrderByDescending(x => x.Chi2)
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
            throw new InvalidOperationException("ChiSquareTest has not been fitted.");

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
        throw new NotSupportedException("ChiSquareTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquareTest has not been fitted.");

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
