using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Information;

/// <summary>
/// Maximal Information Coefficient (MIC) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Maximal Information Coefficient to detect both linear and non-linear
/// dependencies between features and the target.
/// </para>
/// <para><b>For Beginners:</b> MIC is like correlation but can detect any type
/// of relationship, not just linear ones. It scores relationships on a scale
/// of 0 to 1, where 1 means a perfect (possibly complex) relationship exists.
/// </para>
/// </remarks>
public class MaximalInformationCoefficientSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _c;

    private double[]? _micScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MICScores => _micScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MaximalInformationCoefficientSelector(
        int nFeaturesToSelect = 10,
        double alpha = 0.6,
        double c = 15,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _c = c;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MaximalInformationCoefficientSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        _micScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            var x = new double[n];
            for (int i = 0; i < n; i++) x[i] = X[i, j];
            _micScores[j] = ComputeMIC(x, y, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _micScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeMIC(double[] x, double[] y, int n)
    {
        // Compute B(n) - maximum number of bins
        int maxBins = (int)Math.Pow(n, _alpha);
        maxBins = Math.Max(2, Math.Min(maxBins, (int)_c));

        double maxMI = 0;

        // Search over different grid resolutions
        for (int xBins = 2; xBins <= maxBins; xBins++)
        {
            for (int yBins = 2; yBins <= maxBins; yBins++)
            {
                if (xBins * yBins > maxBins * maxBins) continue;

                double mi = ComputeGridMI(x, y, n, xBins, yBins);
                double normalizedMI = mi / Math.Log(Math.Min(xBins, yBins)) * Math.Log(2);
                maxMI = Math.Max(maxMI, normalizedMI);
            }
        }

        return Math.Min(1.0, maxMI);
    }

    private double ComputeGridMI(double[] x, double[] y, int n, int xBins, int yBins)
    {
        // Discretize with equipartition
        var xRanked = x.Select((v, i) => (v, i)).OrderBy(t => t.v).Select((t, r) => (t.i, r)).OrderBy(t => t.i).Select(t => t.r).ToArray();
        var yRanked = y.Select((v, i) => (v, i)).OrderBy(t => t.v).Select((t, r) => (t.i, r)).OrderBy(t => t.i).Select(t => t.r).ToArray();

        var xDiscrete = new int[n];
        var yDiscrete = new int[n];
        for (int i = 0; i < n; i++)
        {
            xDiscrete[i] = Math.Min(xRanked[i] * xBins / n, xBins - 1);
            yDiscrete[i] = Math.Min(yRanked[i] * yBins / n, yBins - 1);
        }

        // Compute mutual information
        var jointCounts = new int[xBins, yBins];
        var xCounts = new int[xBins];
        var yCounts = new int[yBins];

        for (int i = 0; i < n; i++)
        {
            jointCounts[xDiscrete[i], yDiscrete[i]]++;
            xCounts[xDiscrete[i]]++;
            yCounts[yDiscrete[i]]++;
        }

        double mi = 0;
        for (int xi = 0; xi < xBins; xi++)
        {
            for (int yi = 0; yi < yBins; yi++)
            {
                if (jointCounts[xi, yi] == 0) continue;
                double pxy = (double)jointCounts[xi, yi] / n;
                double px = (double)xCounts[xi] / n;
                double py = (double)yCounts[yi] / n;
                mi += pxy * Math.Log(pxy / (px * py + 1e-10) + 1e-10);
            }
        }

        return Math.Max(0, mi);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformationCoefficientSelector has not been fitted.");

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
        throw new NotSupportedException("MaximalInformationCoefficientSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformationCoefficientSelector has not been fitted.");

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
