using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Joint Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their joint entropy with the target, favoring
/// features that share information with the target.
/// </para>
/// <para><b>For Beginners:</b> Joint entropy measures the total randomness when
/// considering a feature and target together. Lower joint entropy relative to
/// individual entropies indicates the feature and target are related.
/// </para>
/// </remarks>
public class JointEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _normalizedMutualInfo;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? NormalizedMutualInfo => _normalizedMutualInfo;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public JointEntropySelector(
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
            "JointEntropySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _normalizedMutualInfo = new double[p];

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

            // Compute H(X)
            var xProbs = new double[_nBins];
            for (int i = 0; i < n; i++)
                xProbs[xBins[i]]++;
            for (int b = 0; b < _nBins; b++)
                xProbs[b] /= n;

            double hX = 0;
            for (int b = 0; b < _nBins; b++)
                if (xProbs[b] > 1e-10)
                    hX -= xProbs[b] * Math.Log(xProbs[b]) / Math.Log(2);

            // Compute H(X,Y) - joint entropy
            var jointProbs = new double[_nBins, _nBins];
            for (int i = 0; i < n; i++)
                jointProbs[xBins[i], yBins[i]]++;
            for (int xb = 0; xb < _nBins; xb++)
                for (int yb = 0; yb < _nBins; yb++)
                    jointProbs[xb, yb] /= n;

            double hXY = 0;
            for (int xb = 0; xb < _nBins; xb++)
                for (int yb = 0; yb < _nBins; yb++)
                    if (jointProbs[xb, yb] > 1e-10)
                        hXY -= jointProbs[xb, yb] * Math.Log(jointProbs[xb, yb]) / Math.Log(2);

            // Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
            double mi = hX + hY - hXY;

            // Normalized mutual information
            double avgEntropy = (hX + hY) / 2;
            _normalizedMutualInfo[j] = avgEntropy > 1e-10 ? mi / avgEntropy : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _normalizedMutualInfo[j])
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
            throw new InvalidOperationException("JointEntropySelector has not been fitted.");

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
        throw new NotSupportedException("JointEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JointEntropySelector has not been fitted.");

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
