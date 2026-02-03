using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Conditional Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their ability to reduce uncertainty about the target,
/// measured by conditional entropy H(Y|X).
/// </para>
/// <para><b>For Beginners:</b> Conditional entropy measures how much uncertainty
/// remains about the target after knowing a feature's value. Features that reduce
/// uncertainty the most (lower conditional entropy) are most informative.
/// </para>
/// </remarks>
public class ConditionalEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _entropyReductions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? EntropyReductions => _entropyReductions;
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
        double yMin = y.Min();
        double yMax = y.Max();
        double yRange = yMax - yMin;
        var yBins = new int[n];
        for (int i = 0; i < n; i++)
            yBins[i] = yRange > 1e-10
                ? Math.Min(_nBins - 1, (int)((y[i] - yMin) / yRange * _nBins))
                : 0;

        // Compute H(Y)
        var yProbs = new double[_nBins];
        for (int i = 0; i < n; i++)
            yProbs[yBins[i]]++;
        for (int b = 0; b < _nBins; b++)
            yProbs[b] /= n;

        double hY = 0;
        for (int b = 0; b < _nBins; b++)
            if (yProbs[b] > 1e-10)
                hY -= yProbs[b] * Math.Log(yProbs[b]) / Math.Log(2);

        _entropyReductions = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double xMin = col.Min();
            double xMax = col.Max();
            double xRange = xMax - xMin;
            var xBins = new int[n];
            for (int i = 0; i < n; i++)
                xBins[i] = xRange > 1e-10
                    ? Math.Min(_nBins - 1, (int)((col[i] - xMin) / xRange * _nBins))
                    : 0;

            // Compute H(Y|X)
            var jointCounts = new int[_nBins, _nBins];
            var xCounts = new int[_nBins];
            for (int i = 0; i < n; i++)
            {
                jointCounts[xBins[i], yBins[i]]++;
                xCounts[xBins[i]]++;
            }

            double hYGivenX = 0;
            for (int xb = 0; xb < _nBins; xb++)
            {
                if (xCounts[xb] == 0) continue;
                double pX = (double)xCounts[xb] / n;

                for (int yb = 0; yb < _nBins; yb++)
                {
                    if (jointCounts[xb, yb] == 0) continue;
                    double pYGivenX = (double)jointCounts[xb, yb] / xCounts[xb];
                    hYGivenX -= pX * pYGivenX * Math.Log(pYGivenX) / Math.Log(2);
                }
            }

            // Information gain: I(Y;X) = H(Y) - H(Y|X)
            _entropyReductions[j] = hY - hYGivenX;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _entropyReductions[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
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
