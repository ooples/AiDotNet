using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Symmetric Uncertainty for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Symmetric Uncertainty is a normalized version of Mutual Information that ranges
/// from 0 to 1. It is symmetric, meaning SU(X,Y) = SU(Y,X), and corrects for the
/// bias toward features with many values.
/// </para>
/// <para><b>For Beginners:</b> Regular Mutual Information can be hard to interpret
/// because its scale depends on the data. Symmetric Uncertainty normalizes it to
/// always be between 0 (no relationship) and 1 (perfect relationship). It treats
/// both variables equally, so it doesn't matter which is the feature and which is
/// the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SymmetricUncertainty<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _suScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? SUScores => _suScores;
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

        // Compute target entropy H(Y)
        var classCount = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int c = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classCount.ContainsKey(c))
                classCount[c] = 0;
            classCount[c]++;
        }

        double targetEntropy = 0;
        foreach (var count in classCount.Values)
        {
            double p_c = (double)count / n;
            if (p_c > 0)
                targetEntropy -= p_c * Math.Log(p_c);
        }

        _suScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            var values = new double[n];
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                values[i] = NumOps.ToDouble(data[i, j]);
                min = Math.Min(min, values[i]);
                max = Math.Max(max, values[i]);
            }

            double binWidth = (max - min) / _nBins;
            if (binWidth < 1e-10) binWidth = 1;

            // Count frequencies
            var binClassCount = new Dictionary<int, Dictionary<int, int>>();
            var binCount = new Dictionary<int, int>();

            for (int i = 0; i < n; i++)
            {
                int bin = Math.Min((int)((values[i] - min) / binWidth), _nBins - 1);
                int c = (int)Math.Round(NumOps.ToDouble(target[i]));

                if (!binClassCount.ContainsKey(bin))
                    binClassCount[bin] = new Dictionary<int, int>();
                if (!binClassCount[bin].ContainsKey(c))
                    binClassCount[bin][c] = 0;
                binClassCount[bin][c]++;

                if (!binCount.ContainsKey(bin))
                    binCount[bin] = 0;
                binCount[bin]++;
            }

            // Feature entropy H(X)
            double featureEntropy = 0;
            foreach (var count in binCount.Values)
            {
                double p_bin = (double)count / n;
                if (p_bin > 0)
                    featureEntropy -= p_bin * Math.Log(p_bin);
            }

            // Conditional entropy H(Y|X)
            double conditionalEntropy = 0;
            foreach (var bin in binClassCount.Keys)
            {
                double binProb = (double)binCount[bin] / n;
                double binEntropy = 0;

                foreach (var count in binClassCount[bin].Values)
                {
                    double p_c_bin = (double)count / binCount[bin];
                    if (p_c_bin > 0)
                        binEntropy -= p_c_bin * Math.Log(p_c_bin);
                }

                conditionalEntropy += binProb * binEntropy;
            }

            // Mutual Information I(X;Y) = H(Y) - H(Y|X)
            double mi = targetEntropy - conditionalEntropy;

            // Symmetric Uncertainty = 2 * I(X;Y) / (H(X) + H(Y))
            double denominator = featureEntropy + targetEntropy;
            _suScores[j] = denominator > 1e-10 ? 2.0 * mi / denominator : 0;
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _suScores
            .Select((su, idx) => (SU: su, Index: idx))
            .OrderByDescending(x => x.SU)
            .Take(numToSelect)
            .Select(x => x.Index)
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
