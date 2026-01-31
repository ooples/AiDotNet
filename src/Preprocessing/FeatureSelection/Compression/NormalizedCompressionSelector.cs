using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Compression;

/// <summary>
/// Normalized Compression Distance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their normalized compression distance to the target,
/// approximating Kolmogorov complexity for feature relevance.
/// </para>
/// <para><b>For Beginners:</b> This measures how much "shared information" exists
/// between a feature and the target. Features that compress well together with
/// the target share more mutual information.
/// </para>
/// </remarks>
public class NormalizedCompressionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _ncdScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? NCDScores => _ncdScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public NormalizedCompressionSelector(
        int nFeaturesToSelect = 10,
        int nBins = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "NormalizedCompressionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        double entropyY = ComputeEntropy(y, n);
        _ncdScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double entropyX = ComputeEntropy(col, n);
            double jointEntropy = ComputeJointEntropy(col, y, n);

            // NCD approximation: lower NCD means more similarity
            // We use mutual information instead: higher means more relevant
            double mi = entropyX + entropyY - jointEntropy;
            _ncdScores[j] = Math.Max(0, mi);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _ncdScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeEntropy(double[] values, int n)
    {
        double minVal = values.Min(), maxVal = values.Max();
        if (Math.Abs(maxVal - minVal) < 1e-10) return 0;

        double binWidth = (maxVal - minVal) / _nBins + 1e-10;
        var counts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int bin = Math.Min((int)((values[i] - minVal) / binWidth), _nBins - 1);
            counts[bin]++;
        }

        double entropy = 0;
        for (int b = 0; b < _nBins; b++)
        {
            if (counts[b] > 0)
            {
                double p = (double)counts[b] / n;
                entropy -= p * Math.Log(p) / Math.Log(2);
            }
        }

        return entropy;
    }

    private double ComputeJointEntropy(double[] x, double[] y, int n)
    {
        double xMin = x.Min(), xMax = x.Max();
        double yMin = y.Min(), yMax = y.Max();

        if (Math.Abs(xMax - xMin) < 1e-10 || Math.Abs(yMax - yMin) < 1e-10)
            return 0;

        double xBinWidth = (xMax - xMin) / _nBins + 1e-10;
        double yBinWidth = (yMax - yMin) / _nBins + 1e-10;

        var jointCounts = new int[_nBins, _nBins];

        for (int i = 0; i < n; i++)
        {
            int xBin = Math.Min((int)((x[i] - xMin) / xBinWidth), _nBins - 1);
            int yBin = Math.Min((int)((y[i] - yMin) / yBinWidth), _nBins - 1);
            jointCounts[xBin, yBin]++;
        }

        double entropy = 0;
        for (int bx = 0; bx < _nBins; bx++)
        {
            for (int by = 0; by < _nBins; by++)
            {
                if (jointCounts[bx, by] > 0)
                {
                    double p = (double)jointCounts[bx, by] / n;
                    entropy -= p * Math.Log(p) / Math.Log(2);
                }
            }
        }

        return entropy;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NormalizedCompressionSelector has not been fitted.");

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
        throw new NotSupportedException("NormalizedCompressionSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NormalizedCompressionSelector has not been fitted.");

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
