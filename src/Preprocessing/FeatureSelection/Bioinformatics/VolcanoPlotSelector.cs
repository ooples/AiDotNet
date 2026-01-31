using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Volcano plot-based feature selection combining fold change and statistical significance.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that show both large fold changes AND statistical significance.
/// Named after the volcano-shaped plot with log fold change on x-axis and
/// -log(p-value) on y-axis.
/// </para>
/// <para><b>For Beginners:</b> A feature might change a lot but by random chance,
/// or change little but reliably. Volcano selection requires BOTH: the feature must
/// show a large difference between groups AND that difference must be statistically
/// significant (unlikely to be just noise).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VolcanoPlotSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minFoldChange;
    private readonly double _pValueThreshold;

    private double[]? _log2FoldChanges;
    private double[]? _pValues;
    private double[]? _volcanoScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Log2FoldChanges => _log2FoldChanges;
    public double[]? PValues => _pValues;
    public double[]? VolcanoScores => _volcanoScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public VolcanoPlotSelector(
        int nFeaturesToSelect = 100,
        double minFoldChange = 1.5,
        double pValueThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minFoldChange = minFoldChange;
        _pValueThreshold = pValueThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "VolcanoPlotSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Separate samples by class
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

        _log2FoldChanges = new double[p];
        _pValues = new double[p];
        _volcanoScores = new double[p];

        int n0 = class0.Count;
        int n1 = class1.Count;

        for (int j = 0; j < p; j++)
        {
            // Compute means and variances for each class
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0)
                mean0 += NumOps.ToDouble(data[i, j]);
            foreach (int i in class1)
                mean1 += NumOps.ToDouble(data[i, j]);

            mean0 /= n0;
            mean1 /= n1;

            double var0 = 0, var1 = 0;
            foreach (int i in class0)
                var0 += Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2);
            foreach (int i in class1)
                var1 += Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2);

            var0 /= (n0 - 1);
            var1 /= (n1 - 1);

            // Log2 fold change
            double eps = 1e-10;
            double ratio = (mean1 + eps) / (mean0 + eps);
            _log2FoldChanges[j] = Math.Log(ratio) / Math.Log(2);

            // Welch's t-test
            double se = Math.Sqrt(var0 / n0 + var1 / n1 + eps);
            double tStat = (mean1 - mean0) / se;

            // Approximate p-value using t-distribution
            double df = ApproxWelchDF(var0, var1, n0, n1);
            _pValues[j] = TwoTailedTTest(Math.Abs(tStat), df);

            // Volcano score: combines significance and effect size
            _volcanoScores[j] = Math.Abs(_log2FoldChanges[j]) * (-Math.Log10(_pValues[j] + eps));
        }

        // Select significant features with high fold change
        var candidates = new List<(int Index, double Score)>();
        double log2Threshold = Math.Log(_minFoldChange) / Math.Log(2);

        for (int j = 0; j < p; j++)
        {
            if (Math.Abs(_log2FoldChanges[j]) >= log2Threshold && _pValues[j] <= _pValueThreshold)
                candidates.Add((j, _volcanoScores[j]));
        }

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(x => x.Score)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Fall back to top by volcano score
            _selectedIndices = _volcanoScores
                .Select((s, idx) => (Score: s, Index: idx))
                .OrderByDescending(x => x.Score)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double ApproxWelchDF(double var0, double var1, int n0, int n1)
    {
        double v0 = var0 / n0;
        double v1 = var1 / n1;
        double num = Math.Pow(v0 + v1, 2);
        double denom = (v0 * v0) / (n0 - 1) + (v1 * v1) / (n1 - 1);
        return denom > 0 ? num / denom : n0 + n1 - 2;
    }

    private double TwoTailedTTest(double t, double df)
    {
        // Approximate using incomplete beta function
        if (df <= 0) return 1;
        double x = df / (df + t * t);
        double a = df / 2.0;
        double b = 0.5;
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
            throw new InvalidOperationException("VolcanoPlotSelector has not been fitted.");

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
        throw new NotSupportedException("VolcanoPlotSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VolcanoPlotSelector has not been fitted.");

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
