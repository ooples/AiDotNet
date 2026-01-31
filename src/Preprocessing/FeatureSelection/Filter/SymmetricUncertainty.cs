using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Symmetric Uncertainty Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Symmetric Uncertainty is a normalized version of mutual information that
/// measures the correlation between features and targets. It ranges from 0
/// (no correlation) to 1 (perfect correlation) and is symmetric.
/// </para>
/// <para><b>For Beginners:</b> Think of this as measuring how well knowing
/// one value helps predict another. Unlike regular correlation, it works
/// for any type of relationship (not just linear) and always gives a value
/// between 0 and 1, making it easy to compare different features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SymmetricUncertainty<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _uncertaintyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? UncertaintyScores => _uncertaintyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SymmetricUncertainty(
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
            "SymmetricUncertainty requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _uncertaintyScores = new double[p];

        // Discretize target
        var targetBins = Discretize(target, n);
        double entropyY = ComputeEntropy(targetBins, n);

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            var featureValues = new double[n];
            for (int i = 0; i < n; i++)
                featureValues[i] = NumOps.ToDouble(data[i, j]);

            var featureBins = Discretize(featureValues);
            double entropyX = ComputeEntropy(featureBins, n);

            // Compute joint entropy
            var jointCounts = new Dictionary<(int, int), int>();
            for (int i = 0; i < n; i++)
            {
                var key = (featureBins[i], targetBins[i]);
                jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;
            }

            double jointEntropy = 0;
            foreach (int count in jointCounts.Values)
            {
                if (count > 0)
                {
                    double prob = (double)count / n;
                    jointEntropy -= prob * Math.Log(prob);
                }
            }

            // Symmetric uncertainty = 2 * MI / (H(X) + H(Y))
            double mi = entropyX + entropyY - jointEntropy;
            double denominator = entropyX + entropyY;

            _uncertaintyScores[j] = denominator > 1e-10 ? 2 * mi / denominator : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _uncertaintyScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] Discretize(Vector<T> values, int n)
    {
        var result = new int[n];
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;

        for (int i = 0; i < n; i++)
        {
            double v = NumOps.ToDouble(values[i]);
            minVal = Math.Min(minVal, v);
            maxVal = Math.Max(maxVal, v);
        }

        double range = maxVal - minVal;
        for (int i = 0; i < n; i++)
        {
            double v = NumOps.ToDouble(values[i]);
            result[i] = range > 1e-10
                ? Math.Min((int)((v - minVal) / range * (_nBins - 1)), _nBins - 1)
                : 0;
        }

        return result;
    }

    private int[] Discretize(double[] values)
    {
        int n = values.Length;
        var result = new int[n];
        double minVal = values.Min();
        double maxVal = values.Max();
        double range = maxVal - minVal;

        for (int i = 0; i < n; i++)
        {
            result[i] = range > 1e-10
                ? Math.Min((int)((values[i] - minVal) / range * (_nBins - 1)), _nBins - 1)
                : 0;
        }

        return result;
    }

    private double ComputeEntropy(int[] bins, int n)
    {
        var counts = new Dictionary<int, int>();
        foreach (int b in bins)
            counts[b] = counts.GetValueOrDefault(b) + 1;

        double entropy = 0;
        foreach (int count in counts.Values)
        {
            if (count > 0)
            {
                double prob = (double)count / n;
                entropy -= prob * Math.Log(prob);
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
            throw new InvalidOperationException("SymmetricUncertainty has not been fitted.");

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
        throw new NotSupportedException("SymmetricUncertainty does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SymmetricUncertainty has not been fitted.");

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
