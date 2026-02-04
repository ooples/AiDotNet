using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gini Impurity-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Measures how much each feature reduces impurity (Gini impurity) when used
/// to split the data. Features that create purer groups after splitting are
/// considered more important.
/// </para>
/// <para><b>For Beginners:</b> Gini impurity measures how "mixed" a group is.
/// If splitting data by a feature creates groups where samples are mostly
/// the same class (pure groups), that feature is valuable. This method finds
/// features that best separate different classes from each other.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GiniImpuritySelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nThresholds;

    private double[]? _giniReductionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? GiniReductionScores => _giniReductionScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GiniImpuritySelection(
        int nFeaturesToSelect = 10,
        int nThresholds = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nThresholds < 1)
            throw new ArgumentException("Number of thresholds must be at least 1.", nameof(nThresholds));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nThresholds = nThresholds;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GiniImpuritySelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _giniReductionScores = new double[p];

        // Compute parent Gini
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            classCounts[label] = classCounts.GetValueOrDefault(label) + 1;
        }

        double parentGini = 1.0;
        foreach (int count in classCounts.Values)
        {
            double prob = (double)count / n;
            parentGini -= prob * prob;
        }

        for (int j = 0; j < p; j++)
        {
            // Get feature values
            var values = new (double value, int label)[n];
            for (int i = 0; i < n; i++)
                values[i] = (NumOps.ToDouble(data[i, j]), (int)Math.Round(NumOps.ToDouble(target[i])));

            var sorted = values.OrderBy(v => v.value).ToArray();

            // Try multiple thresholds
            double minVal = sorted[0].value;
            double maxVal = sorted[n - 1].value;
            double range = maxVal - minVal;

            if (range < 1e-10)
            {
                _giniReductionScores[j] = 0;
                continue;
            }

            double bestReduction = 0;

            for (int t = 1; t <= _nThresholds; t++)
            {
                double threshold = minVal + (range * t) / (_nThresholds + 1);

                var leftCounts = new Dictionary<int, int>();
                var rightCounts = new Dictionary<int, int>();
                int nLeft = 0, nRight = 0;

                foreach (var (value, label) in sorted)
                {
                    if (value <= threshold)
                    {
                        leftCounts[label] = leftCounts.GetValueOrDefault(label) + 1;
                        nLeft++;
                    }
                    else
                    {
                        rightCounts[label] = rightCounts.GetValueOrDefault(label) + 1;
                        nRight++;
                    }
                }

                if (nLeft == 0 || nRight == 0)
                    continue;

                double leftGini = 1.0;
                foreach (int count in leftCounts.Values)
                {
                    double prob = (double)count / nLeft;
                    leftGini -= prob * prob;
                }

                double rightGini = 1.0;
                foreach (int count in rightCounts.Values)
                {
                    double prob = (double)count / nRight;
                    rightGini -= prob * prob;
                }

                double weightedGini = ((double)nLeft / n) * leftGini + ((double)nRight / n) * rightGini;
                double reduction = parentGini - weightedGini;

                bestReduction = Math.Max(bestReduction, reduction);
            }

            _giniReductionScores[j] = bestReduction;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _giniReductionScores[j])
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
            throw new InvalidOperationException("GiniImpuritySelection has not been fitted.");

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
        throw new NotSupportedException("GiniImpuritySelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniImpuritySelection has not been fitted.");

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
