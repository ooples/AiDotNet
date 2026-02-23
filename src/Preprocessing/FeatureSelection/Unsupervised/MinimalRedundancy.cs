using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Minimal Redundancy feature selection for unsupervised learning.
/// </summary>
/// <remarks>
/// <para>
/// Minimal Redundancy selects a diverse set of features that are not highly
/// correlated with each other. This ensures the selected features provide
/// complementary rather than overlapping information.
/// </para>
/// <para><b>For Beginners:</b> If two features are almost identical (highly
/// correlated), keeping both is wasteful. This method picks features that are
/// as different from each other as possible, giving you maximum diversity.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MinimalRedundancy<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _correlationThreshold;

    private double[,]? _correlationMatrix;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CorrelationThreshold => _correlationThreshold;
    public double[,]? CorrelationMatrix => _correlationMatrix;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MinimalRedundancy(
        int nFeaturesToSelect = 10,
        double correlationThreshold = 0.9,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _correlationThreshold = correlationThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute correlation matrix
        _correlationMatrix = new double[p, p];
        var means = new double[p];
        var stds = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - means[j];
                stds[j] += diff * diff;
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        for (int j1 = 0; j1 < p; j1++)
        {
            _correlationMatrix[j1, j1] = 1.0;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double corr = 0;
                for (int i = 0; i < n; i++)
                {
                    double z1 = (NumOps.ToDouble(data[i, j1]) - means[j1]) / stds[j1];
                    double z2 = (NumOps.ToDouble(data[i, j2]) - means[j2]) / stds[j2];
                    corr += z1 * z2;
                }
                corr /= n;
                _correlationMatrix[j1, j2] = corr;
                _correlationMatrix[j2, j1] = corr;
            }
        }

        // Compute variance for each feature
        var variances = new double[p];
        for (int j = 0; j < p; j++)
            variances[j] = stds[j] * stds[j];

        // Greedy selection: pick features with high variance and low redundancy
        var selected = new List<int>();
        var available = Enumerable.Range(0, p).ToList();

        while (selected.Count < _nFeaturesToSelect && available.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            foreach (int j in available)
            {
                // Score = variance - average correlation with selected features
                double redundancy = 0;
                if (selected.Count > 0)
                {
                    foreach (int s in selected)
                        redundancy += Math.Abs(_correlationMatrix[j, s]);
                    redundancy /= selected.Count;
                }

                double score = variances[j] - redundancy;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                available.Remove(bestFeature);

                // Remove features too correlated with the newly selected one
                var toRemove = new List<int>();
                foreach (int j in available)
                {
                    if (Math.Abs(_correlationMatrix[bestFeature, j]) > _correlationThreshold)
                        toRemove.Add(j);
                }
                foreach (int j in toRemove)
                    available.Remove(j);
            }
            else
            {
                break;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        // If not enough selected, add remaining by variance
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var remaining = Enumerable.Range(0, p)
                .Where(j => !_selectedIndices.Contains(j))
                .OrderByDescending(j => variances[j])
                .Take(_nFeaturesToSelect - _selectedIndices.Length);

            _selectedIndices = _selectedIndices.Concat(remaining).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MinimalRedundancy has not been fitted.");

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
        throw new NotSupportedException("MinimalRedundancy does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MinimalRedundancy has not been fitted.");

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
