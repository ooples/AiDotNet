using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonlinear;

/// <summary>
/// Mutual Information based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their mutual information with the target,
/// capturing both linear and nonlinear dependencies.
/// </para>
/// <para><b>For Beginners:</b> Mutual information measures how much knowing
/// one variable tells you about another. Unlike correlation, it detects
/// any type of relationship - linear, nonlinear, or complex. High MI means
/// the feature is strongly related to the target in some way.
/// </para>
/// </remarks>
public class MutualInformationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _miValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? MIValues => _miValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MutualInformationSelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MutualInformationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _miValues = new double[p];

        // Discretize target
        var yBins = Discretize(y, _nBins);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Discretize feature
            var xBins = Discretize(col, _nBins);

            // Compute mutual information
            _miValues[j] = ComputeMI(xBins, yBins, _nBins);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _miValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] Discretize(double[] data, int nBins)
    {
        int n = data.Length;
        var result = new int[n];

        double min = data.Min();
        double max = data.Max();
        double range = max - min;

        if (range < 1e-10)
        {
            // Constant - all same bin
            for (int i = 0; i < n; i++)
                result[i] = 0;
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                int bin = (int)((data[i] - min) / range * (nBins - 1));
                result[i] = Math.Min(bin, nBins - 1);
            }
        }

        return result;
    }

    private double ComputeMI(int[] x, int[] y, int nBins)
    {
        int n = x.Length;

        // Joint and marginal counts
        var jointCounts = new int[nBins, nBins];
        var xCounts = new int[nBins];
        var yCounts = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            jointCounts[x[i], y[i]]++;
            xCounts[x[i]]++;
            yCounts[y[i]]++;
        }

        // Compute MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
        double mi = 0;
        for (int xi = 0; xi < nBins; xi++)
        {
            for (int yi = 0; yi < nBins; yi++)
            {
                if (jointCounts[xi, yi] > 0 && xCounts[xi] > 0 && yCounts[yi] > 0)
                {
                    double pxy = (double)jointCounts[xi, yi] / n;
                    double px = (double)xCounts[xi] / n;
                    double py = (double)yCounts[yi] / n;
                    mi += pxy * Math.Log(pxy / (px * py)) / Math.Log(2);
                }
            }
        }

        return Math.Max(0, mi); // MI is always non-negative
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInformationSelector has not been fitted.");

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
        throw new NotSupportedException("MutualInformationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInformationSelector has not been fitted.");

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
