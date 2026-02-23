using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Information;

/// <summary>
/// Conditional Entropy-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that minimize the conditional entropy of the target
/// given the feature, maximizing the information about the target.
/// </para>
/// <para><b>For Beginners:</b> Conditional entropy measures the uncertainty
/// remaining about the target after knowing a feature's value. Features that
/// reduce uncertainty the most (lower conditional entropy) are more useful.
/// </para>
/// </remarks>
public class ConditionalEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _conditionalEntropies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ConditionalEntropies => _conditionalEntropies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ConditionalEntropySelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
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
            "ConditionalEntropySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize target
        var yBins = Discretize(y, n, _nBins);

        _conditionalEntropies = new double[p];
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++) col[i] = X[i, j];
            var xBins = Discretize(col, n, _nBins);

            _conditionalEntropies[j] = ComputeConditionalEntropy(xBins, yBins, n, _nBins);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        // Select features with LOWEST conditional entropy (most information)
        _selectedIndices = Enumerable.Range(0, p)
            .OrderBy(j => _conditionalEntropies[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] Discretize(double[] values, int n, int nBins)
    {
        double min = values.Min();
        double max = values.Max();
        double range = max - min + 1e-10;

        var bins = new int[n];
        for (int i = 0; i < n; i++)
            bins[i] = Math.Min((int)((values[i] - min) / range * nBins), nBins - 1);

        return bins;
    }

    private double ComputeConditionalEntropy(int[] x, int[] y, int n, int nBins)
    {
        // H(Y|X) = sum over x of P(x) * H(Y|X=x)
        var jointCounts = new int[nBins, nBins];
        var xCounts = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            jointCounts[x[i], y[i]]++;
            xCounts[x[i]]++;
        }

        double condEntropy = 0;
        for (int xi = 0; xi < nBins; xi++)
        {
            if (xCounts[xi] == 0) continue;
            double px = (double)xCounts[xi] / n;

            double entropy = 0;
            for (int yi = 0; yi < nBins; yi++)
            {
                if (jointCounts[xi, yi] == 0) continue;
                double pyx = (double)jointCounts[xi, yi] / xCounts[xi];
                entropy -= pyx * Math.Log(pyx) / Math.Log(2);
            }
            condEntropy += px * entropy;
        }

        return condEntropy;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConditionalEntropySelector has not been fitted.");

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
        throw new NotSupportedException("ConditionalEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConditionalEntropySelector has not been fitted.");

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
