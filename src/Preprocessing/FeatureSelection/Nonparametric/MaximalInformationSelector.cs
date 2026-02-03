using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonparametric;

/// <summary>
/// Maximal Information Coefficient (MIC) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their maximal information coefficient with the target,
/// capturing many different types of relationships including nonlinear ones.
/// </para>
/// <para><b>For Beginners:</b> MIC tries different ways of binning data to find
/// the binning that maximizes mutual information. This makes it very good at
/// detecting all kinds of relationships - linear, nonlinear, periodic, etc.
/// </para>
/// </remarks>
public class MaximalInformationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxBins;

    private double[]? _micScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxBins => _maxBins;
    public double[]? MICScores => _micScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MaximalInformationSelector(
        int nFeaturesToSelect = 10,
        int maxBins = 15,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxBins = maxBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MaximalInformationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _micScores[j] = ComputeMIC(col, y, n);
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
        double maxMI = 0;
        int maxBinsX = Math.Min(_maxBins, (int)Math.Sqrt(n));
        int maxBinsY = Math.Min(_maxBins, (int)Math.Sqrt(n));

        double xMin = x.Min(), xMax = x.Max();
        double yMin = y.Min(), yMax = y.Max();

        // Try different grid sizes
        for (int bX = 2; bX <= maxBinsX; bX++)
        {
            for (int bY = 2; bY <= maxBinsY; bY++)
            {
                if (bX * bY > n) continue; // Not enough samples

                double mi = ComputeMIForGrid(x, y, n, bX, bY, xMin, xMax, yMin, yMax);
                double normalizedMI = mi / Math.Log(Math.Min(bX, bY)) * Math.Log(2);

                maxMI = Math.Max(maxMI, normalizedMI);
            }
        }

        return maxMI;
    }

    private double ComputeMIForGrid(double[] x, double[] y, int n, int bX, int bY,
        double xMin, double xMax, double yMin, double yMax)
    {
        double xBinWidth = (xMax - xMin) / bX + 1e-10;
        double yBinWidth = (yMax - yMin) / bY + 1e-10;

        var jointCounts = new int[bX, bY];
        var xCounts = new int[bX];
        var yCounts = new int[bY];

        for (int i = 0; i < n; i++)
        {
            int xBin = Math.Min((int)((x[i] - xMin) / xBinWidth), bX - 1);
            int yBin = Math.Min((int)((y[i] - yMin) / yBinWidth), bY - 1);

            jointCounts[xBin, yBin]++;
            xCounts[xBin]++;
            yCounts[yBin]++;
        }

        double mi = 0;
        for (int xb = 0; xb < bX; xb++)
        {
            for (int yb = 0; yb < bY; yb++)
            {
                if (jointCounts[xb, yb] > 0 && xCounts[xb] > 0 && yCounts[yb] > 0)
                {
                    double pxy = (double)jointCounts[xb, yb] / n;
                    double px = (double)xCounts[xb] / n;
                    double py = (double)yCounts[yb] / n;
                    mi += pxy * Math.Log(pxy / (px * py) + 1e-10);
                }
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
            throw new InvalidOperationException("MaximalInformationSelector has not been fitted.");

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
        throw new NotSupportedException("MaximalInformationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformationSelector has not been fitted.");

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
