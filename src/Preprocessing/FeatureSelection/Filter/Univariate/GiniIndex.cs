using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Gini Index for feature selection based on impurity reduction.
/// </summary>
/// <remarks>
/// <para>
/// The Gini Index measures impurity in classification. Features that split
/// data into purer groups (lower Gini) are more informative for prediction.
/// </para>
/// <para><b>For Beginners:</b> Gini measures how mixed a group is. A group
/// with all same-class items has Gini=0 (pure). A 50/50 split has Gini=0.5.
/// Good features create pure groups when you split by them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GiniIndex<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _giniGains;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? GiniGains => _giniGains;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GiniIndex(int nFeaturesToSelect = 10, int nBins = 10, int[]? columnIndices = null)
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
            "GiniIndex requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Count class frequencies
        var classCounts = new Dictionary<double, int>();
        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            if (!classCounts.ContainsKey(y)) classCounts[y] = 0;
            classCounts[y]++;
        }

        // Compute parent Gini
        double parentGini = ComputeGini(classCounts.Values.ToArray(), n);

        _giniGains = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract and discretize feature
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            double min = values.Min();
            double max = values.Max();
            double range = max - min;
            if (range < 1e-10) range = 1;

            // Find best split
            double bestGiniReduction = 0;

            for (int b = 1; b < _nBins; b++)
            {
                double threshold = min + b * range / _nBins;

                // Count classes in left and right splits
                var leftCounts = new Dictionary<double, int>();
                var rightCounts = new Dictionary<double, int>();
                int leftTotal = 0, rightTotal = 0;

                for (int i = 0; i < n; i++)
                {
                    double y = NumOps.ToDouble(target[i]);
                    if (values[i] <= threshold)
                    {
                        if (!leftCounts.ContainsKey(y)) leftCounts[y] = 0;
                        leftCounts[y]++;
                        leftTotal++;
                    }
                    else
                    {
                        if (!rightCounts.ContainsKey(y)) rightCounts[y] = 0;
                        rightCounts[y]++;
                        rightTotal++;
                    }
                }

                if (leftTotal == 0 || rightTotal == 0) continue;

                double leftGini = ComputeGini(leftCounts.Values.ToArray(), leftTotal);
                double rightGini = ComputeGini(rightCounts.Values.ToArray(), rightTotal);

                double weightedGini = (double)leftTotal / n * leftGini + (double)rightTotal / n * rightGini;
                double giniReduction = parentGini - weightedGini;

                if (giniReduction > bestGiniReduction)
                    bestGiniReduction = giniReduction;
            }

            _giniGains[j] = bestGiniReduction;
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _giniGains
            .Select((g, idx) => (Gain: g, Index: idx))
            .OrderByDescending(x => x.Gain)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeGini(int[] counts, int total)
    {
        if (total == 0) return 0;

        double gini = 1.0;
        foreach (int count in counts)
        {
            double p = (double)count / total;
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
