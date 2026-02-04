using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Correlation-based Feature Removal.
/// </summary>
/// <remarks>
/// <para>
/// Removes highly correlated features to reduce multicollinearity, keeping
/// one feature from each group of highly correlated features.
/// </para>
/// <para><b>For Beginners:</b> When two features are highly correlated, they
/// provide similar information. This method finds pairs of highly correlated
/// features and removes one from each pair, keeping your dataset compact
/// without losing much information.
/// </para>
/// </remarks>
public class CorrelationRemovalSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    private double[,]? _correlationMatrix;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public double[,]? CorrelationMatrix => _correlationMatrix;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CorrelationRemovalSelector(
        double threshold = 0.9,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Compute correlation matrix
        _correlationMatrix = new double[p, p];
        var means = new double[p];
        var stds = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
        }

        for (int j1 = 0; j1 < p; j1++)
        {
            _correlationMatrix[j1, j1] = 1.0;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double cov = 0;
                for (int i = 0; i < n; i++)
                    cov += (X[i, j1] - means[j1]) * (X[i, j2] - means[j2]);
                cov /= (n - 1);
                double corr = cov / (stds[j1] * stds[j2]);
                _correlationMatrix[j1, j2] = corr;
                _correlationMatrix[j2, j1] = corr;
            }
        }

        // Select features by removing highly correlated ones
        var toRemove = new HashSet<int>();
        for (int j1 = 0; j1 < p; j1++)
        {
            if (toRemove.Contains(j1)) continue;

            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                if (toRemove.Contains(j2)) continue;

                if (Math.Abs(_correlationMatrix[j1, j2]) > _threshold)
                {
                    // Remove the later feature
                    toRemove.Add(j2);
                }
            }
        }

        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => !toRemove.Contains(j))
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationRemovalSelector has not been fitted.");

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
        throw new NotSupportedException("CorrelationRemovalSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationRemovalSelector has not been fitted.");

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
