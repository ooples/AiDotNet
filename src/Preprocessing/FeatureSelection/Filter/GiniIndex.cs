using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gini Index feature selection for measuring impurity reduction.
/// </summary>
/// <remarks>
/// <para>
/// The Gini Index measures the impurity of a split based on the probability of misclassifying
/// a randomly chosen element. Features that result in lower weighted Gini impurity after
/// splitting are considered more informative.
/// </para>
/// <para><b>For Beginners:</b> Imagine randomly guessing the class of an item. The Gini Index
/// measures how often you'd guess wrong. A feature that separates classes well will have
/// a low Gini Index (few wrong guesses). This is the same metric used by decision trees
/// like CART to choose splits.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GiniIndex<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _giniReductions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? GiniReductions => _giniReductions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GiniIndex(
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
            "GiniIndex requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute overall Gini impurity
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classCounts.ContainsKey(label))
                classCounts[label] = 0;
            classCounts[label]++;
        }

        double overallGini = 1.0;
        foreach (var count in classCounts.Values)
        {
            double prob = (double)count / n;
            overallGini -= prob * prob;
        }

        int nClasses = classCounts.Count;
        var classes = classCounts.Keys.OrderBy(x => x).ToArray();
        var classIndex = classes.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

        _giniReductions = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Bin the feature
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }

            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            var binCounts = new int[_nBins];
            var binClassCounts = new int[_nBins, nClasses];

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                int bin = Math.Min((int)(((val - minVal) / range) * _nBins), _nBins - 1);
                int classIdx = classIndex[(int)Math.Round(NumOps.ToDouble(target[i]))];

                binCounts[bin]++;
                binClassCounts[bin, classIdx]++;
            }

            // Weighted Gini impurity after split
            double weightedGini = 0;
            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] == 0) continue;

                double binGini = 1.0;
                for (int c = 0; c < nClasses; c++)
                {
                    double prob = (double)binClassCounts[b, c] / binCounts[b];
                    binGini -= prob * prob;
                }

                weightedGini += ((double)binCounts[b] / n) * binGini;
            }

            // Gini reduction (higher is better)
            _giniReductions[j] = Math.Max(0, overallGini - weightedGini);
        }

        // Select top features by Gini reduction
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _giniReductions
            .Select((gr, idx) => (GR: gr, Index: idx))
            .OrderByDescending(x => x.GR)
            .Take(nToSelect)
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
            throw new InvalidOperationException("GiniIndex has not been fitted.");

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
        throw new NotSupportedException("GiniIndex does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniIndex has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
