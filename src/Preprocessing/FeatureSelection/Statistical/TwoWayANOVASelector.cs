using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Two-Way ANOVA based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on two-way ANOVA, which analyzes the effect of
/// two categorical factors and their interaction on a continuous response.
/// </para>
/// <para><b>For Beginners:</b> Two-way ANOVA extends one-way ANOVA by considering
/// two grouping variables simultaneously. It can detect main effects of each factor
/// and interaction effects between them. Features with significant F-statistics
/// for either main effects or interactions are selected.
/// </para>
/// </remarks>
public class TwoWayANOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FStatistics => _fStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TwoWayANOVASelector(
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
            "TwoWayANOVASelector requires two factor vectors. Use Fit(Matrix<T> data, Vector<T> factor1, Vector<T> factor2) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> factor1, Vector<T> factor2)
    {
        if (data.Rows != factor1.Length || data.Rows != factor2.Length)
            throw new ArgumentException("Factor lengths must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var f1 = new int[n];
        var f2 = new int[n];

        for (int i = 0; i < n; i++)
        {
            f1[i] = (int)Math.Round(NumOps.ToDouble(factor1[i]));
            f2[i] = (int)Math.Round(NumOps.ToDouble(factor2[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        var levels1 = f1.Distinct().OrderBy(x => x).ToList();
        var levels2 = f2.Distinct().OrderBy(x => x).ToList();

        _fStatistics = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double grandMean = col.Average();
            double ssTotal = col.Sum(v => (v - grandMean) * (v - grandMean));

            // Calculate cell means and marginal means
            var cellMeans = new Dictionary<(int, int), double>();
            var cellCounts = new Dictionary<(int, int), int>();

            for (int i = 0; i < n; i++)
            {
                var key = (f1[i], f2[i]);
                if (!cellMeans.ContainsKey(key))
                {
                    cellMeans[key] = 0;
                    cellCounts[key] = 0;
                }
                cellMeans[key] += col[i];
                cellCounts[key]++;
            }

            foreach (var key in cellMeans.Keys.ToList())
                cellMeans[key] /= cellCounts[key];

            // Marginal means for factor 1
            var means1 = new Dictionary<int, double>();
            var counts1 = new Dictionary<int, int>();
            for (int i = 0; i < n; i++)
            {
                if (!means1.ContainsKey(f1[i]))
                {
                    means1[f1[i]] = 0;
                    counts1[f1[i]] = 0;
                }
                means1[f1[i]] += col[i];
                counts1[f1[i]]++;
            }
            foreach (var k in means1.Keys.ToList())
                means1[k] /= counts1[k];

            // Marginal means for factor 2
            var means2 = new Dictionary<int, double>();
            var counts2 = new Dictionary<int, int>();
            for (int i = 0; i < n; i++)
            {
                if (!means2.ContainsKey(f2[i]))
                {
                    means2[f2[i]] = 0;
                    counts2[f2[i]] = 0;
                }
                means2[f2[i]] += col[i];
                counts2[f2[i]]++;
            }
            foreach (var k in means2.Keys.ToList())
                means2[k] /= counts2[k];

            // SS for factor 1
            double ss1 = 0;
            foreach (var lv in levels1)
                ss1 += counts1[lv] * (means1[lv] - grandMean) * (means1[lv] - grandMean);

            // SS for factor 2
            double ss2 = 0;
            foreach (var lv in levels2)
                ss2 += counts2[lv] * (means2[lv] - grandMean) * (means2[lv] - grandMean);

            // SS within (error)
            double ssWithin = 0;
            for (int i = 0; i < n; i++)
            {
                var key = (f1[i], f2[i]);
                double diff = col[i] - cellMeans[key];
                ssWithin += diff * diff;
            }

            // Degrees of freedom
            int df1 = levels1.Count - 1;
            int df2 = levels2.Count - 1;
            int dfWithin = n - levels1.Count * levels2.Count;

            if (dfWithin <= 0 || df1 <= 0 || df2 <= 0)
            {
                _fStatistics[j] = 0;
                continue;
            }

            // F-statistics (combine main effects)
            double msWithin = ssWithin / dfWithin;
            double f1Stat = msWithin > 1e-10 ? (ss1 / df1) / msWithin : 0;
            double f2Stat = msWithin > 1e-10 ? (ss2 / df2) / msWithin : 0;

            // Use maximum F-statistic
            _fStatistics[j] = Math.Max(f1Stat, f2Stat);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _fStatistics[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TwoWayANOVASelector has not been fitted.");

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
        throw new NotSupportedException("TwoWayANOVASelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TwoWayANOVASelector has not been fitted.");

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
