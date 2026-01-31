using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonlinear;

/// <summary>
/// Maximal Information Coefficient (MIC) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on MIC, which finds the maximum mutual information
/// over different bin configurations, providing equitability across relationship types.
/// </para>
/// <para><b>For Beginners:</b> MIC is a powerful measure that tries different ways
/// of binning the data to find the maximum mutual information. It scores relationships
/// from 0 (independent) to 1 (perfect relationship) regardless of whether the
/// relationship is linear, exponential, periodic, or complex.
/// </para>
/// </remarks>
public class MaximalInformationCoefficientSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxBins;

    private double[]? _micValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxBins => _maxBins;
    public double[]? MICValues => _micValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MaximalInformationCoefficientSelector(
        int nFeaturesToSelect = 10,
        int maxBins = 15,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxBins < 2)
            throw new ArgumentException("Max bins must be at least 2.", nameof(maxBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxBins = maxBins;
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

        _micValues = new double[p];

        // Compute MIC bound: B(n) = n^0.6
        int bMax = (int)Math.Ceiling(Math.Pow(n, 0.6));
        bMax = Math.Min(bMax, _maxBins);
        bMax = Math.Max(bMax, 2);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _micValues[j] = ComputeMIC(col, y, bMax);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _micValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeMIC(double[] x, double[] y, int bMax)
    {
        int n = x.Length;
        double maxMI = 0;

        // Try different grid sizes
        for (int xBins = 2; xBins <= bMax; xBins++)
        {
            for (int yBins = 2; yBins <= bMax; yBins++)
            {
                if (xBins * yBins > bMax) continue;

                double mi = ComputeNormalizedMI(x, y, xBins, yBins);
                double normalizedMI = mi / Math.Log(Math.Min(xBins, yBins)) * Math.Log(2);

                maxMI = Math.Max(maxMI, normalizedMI);
            }
        }

        return Math.Min(1, maxMI); // MIC is bounded by 1
    }

    private double ComputeNormalizedMI(double[] x, double[] y, int xBins, int yBins)
    {
        int n = x.Length;

        // Get bin assignments
        double xMin = x.Min(), xMax = x.Max();
        double yMin = y.Min(), yMax = y.Max();
        double xRange = xMax - xMin;
        double yRange = yMax - yMin;

        if (xRange < 1e-10 || yRange < 1e-10)
            return 0;

        // Joint counts
        var jointCounts = new int[xBins, yBins];
        var xCounts = new int[xBins];
        var yCounts = new int[yBins];

        for (int i = 0; i < n; i++)
        {
            int xb = (int)((x[i] - xMin) / xRange * (xBins - 1));
            int yb = (int)((y[i] - yMin) / yRange * (yBins - 1));
            xb = Math.Min(xb, xBins - 1);
            yb = Math.Min(yb, yBins - 1);

            jointCounts[xb, yb]++;
            xCounts[xb]++;
            yCounts[yb]++;
        }

        // Compute MI
        double mi = 0;
        for (int xi = 0; xi < xBins; xi++)
        {
            for (int yi = 0; yi < yBins; yi++)
            {
                if (jointCounts[xi, yi] > 0 && xCounts[xi] > 0 && yCounts[yi] > 0)
                {
                    double pxy = (double)jointCounts[xi, yi] / n;
                    double px = (double)xCounts[xi] / n;
                    double py = (double)yCounts[yi] / n;
                    mi += pxy * Math.Log(pxy / (px * py));
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
