using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Gain;

/// <summary>
/// Gini Impurity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on Gini impurity reduction, the same criterion
/// used by CART decision trees to choose split features.
/// </para>
/// <para><b>For Beginners:</b> Gini impurity measures how "mixed" a set of labels is.
/// Pure sets (all same class) have Gini = 0. Features that best split data into
/// purer groups have higher Gini importance. This is fast to compute and widely
/// used in tree-based methods like Random Forest.
/// </para>
/// </remarks>
public class GiniImpuritySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _giniReductions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? GiniReductions => _giniReductions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GiniImpuritySelector(
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
            "GiniImpuritySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        _giniReductions = new double[p];

        // Base Gini impurity
        double baseGini = ComputeGini(y);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Find best split threshold
            double bestReduction = 0;

            double min = col.Min();
            double max = col.Max();
            double range = max - min;

            if (range < 1e-10)
            {
                _giniReductions[j] = 0;
                continue;
            }

            for (int b = 1; b < _nBins; b++)
            {
                double threshold = min + (range * b / _nBins);

                var leftY = new List<int>();
                var rightY = new List<int>();

                for (int i = 0; i < n; i++)
                {
                    if (col[i] <= threshold)
                        leftY.Add(y[i]);
                    else
                        rightY.Add(y[i]);
                }

                if (leftY.Count == 0 || rightY.Count == 0)
                    continue;

                double leftGini = ComputeGini([.. leftY]);
                double rightGini = ComputeGini([.. rightY]);

                double weightedGini = (leftY.Count * leftGini + rightY.Count * rightGini) / n;
                double reduction = baseGini - weightedGini;

                bestReduction = Math.Max(bestReduction, reduction);
            }

            _giniReductions[j] = bestReduction;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _giniReductions[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeGini(int[] labels)
    {
        if (labels.Length == 0) return 0;

        var counts = new Dictionary<int, int>();
        foreach (var label in labels)
        {
            if (!counts.ContainsKey(label))
                counts[label] = 0;
            counts[label]++;
        }

        double gini = 1;
        foreach (var count in counts.Values)
        {
            double p = (double)count / labels.Length;
            gini -= p * p;
        }

        return gini;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniImpuritySelector has not been fitted.");

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
        throw new NotSupportedException("GiniImpuritySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniImpuritySelector has not been fitted.");

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
