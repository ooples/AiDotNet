using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Classification;

/// <summary>
/// Gini Impurity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their ability to reduce Gini impurity when used
/// for splitting data, similar to decision tree feature importance.
/// </para>
/// <para><b>For Beginners:</b> Gini impurity measures how mixed up the classes are.
/// A pure group (all same class) has impurity 0. This selector keeps features that
/// best reduce impurity when used to split the data - the same criterion decision
/// trees use.
/// </para>
/// </remarks>
public class GiniImpuritySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSplits;

    private double[]? _giniReductions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? GiniReductions => _giniReductions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GiniImpuritySelector(
        int nFeaturesToSelect = 10,
        int nSplits = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSplits = nSplits;
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

        var classes = y.Distinct().OrderBy(c => c).ToList();
        double parentGini = ComputeGini(y, n, classes);

        _giniReductions = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double bestReduction = 0;

            // Try different split points
            double minVal = col.Min(), maxVal = col.Max();
            double step = (maxVal - minVal) / (_nSplits + 1);

            for (int s = 1; s <= _nSplits; s++)
            {
                double threshold = minVal + s * step;

                // Split data
                var leftIndices = new List<int>();
                var rightIndices = new List<int>();

                for (int i = 0; i < n; i++)
                {
                    if (col[i] <= threshold)
                        leftIndices.Add(i);
                    else
                        rightIndices.Add(i);
                }

                if (leftIndices.Count == 0 || rightIndices.Count == 0) continue;

                // Compute child Gini values
                var leftY = leftIndices.Select(i => y[i]).ToArray();
                var rightY = rightIndices.Select(i => y[i]).ToArray();

                double leftGini = ComputeGini(leftY, leftY.Length, classes);
                double rightGini = ComputeGini(rightY, rightY.Length, classes);

                double weightedChildGini = (leftY.Length * leftGini + rightY.Length * rightGini) / n;
                double reduction = parentGini - weightedChildGini;

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

    private double ComputeGini(int[] labels, int n, List<int> classes)
    {
        if (n == 0) return 0;

        double gini = 1.0;
        foreach (var c in classes)
        {
            double p = (double)labels.Count(l => l == c) / n;
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
