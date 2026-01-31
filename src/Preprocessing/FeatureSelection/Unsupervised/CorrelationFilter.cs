using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Correlation Filter for removing highly correlated features.
/// </summary>
/// <remarks>
/// <para>
/// Identifies pairs of features with correlation above a threshold and removes
/// one from each pair. This reduces multicollinearity and feature redundancy.
/// </para>
/// <para><b>For Beginners:</b> If two features are very similar (highly correlated),
/// keeping both is redundant. This filter finds such pairs and keeps only one,
/// reducing data size without losing much information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CorrelationFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;
    private readonly KeepStrategy _strategy;

    public enum KeepStrategy { KeepFirst, KeepHigherVariance }

    private double[,]? _correlationMatrix;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public double[,]? CorrelationMatrix => _correlationMatrix;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CorrelationFilter(
        double threshold = 0.9,
        KeepStrategy strategy = KeepStrategy.KeepHigherVariance,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
        _strategy = strategy;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute means and standard deviations
        var means = new double[p];
        var stds = new double[p];
        var variances = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;
        }

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - means[j];
                variances[j] += diff * diff;
            }
            stds[j] = Math.Sqrt(variances[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        // Compute correlation matrix
        _correlationMatrix = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            _correlationMatrix[i, i] = 1.0;
            for (int j = i + 1; j < p; j++)
            {
                double cov = 0;
                for (int k = 0; k < n; k++)
                {
                    double di = NumOps.ToDouble(data[k, i]) - means[i];
                    double dj = NumOps.ToDouble(data[k, j]) - means[j];
                    cov += di * dj;
                }
                cov /= n;
                double corr = cov / (stds[i] * stds[j]);
                _correlationMatrix[i, j] = corr;
                _correlationMatrix[j, i] = corr;
            }
        }

        // Find features to drop
        var toDrop = new HashSet<int>();
        for (int i = 0; i < p; i++)
        {
            if (toDrop.Contains(i)) continue;

            for (int j = i + 1; j < p; j++)
            {
                if (toDrop.Contains(j)) continue;

                if (Math.Abs(_correlationMatrix[i, j]) > _threshold)
                {
                    // Drop one of the pair
                    if (_strategy == KeepStrategy.KeepFirst)
                    {
                        toDrop.Add(j);
                    }
                    else
                    {
                        // Keep higher variance
                        if (variances[i] >= variances[j])
                            toDrop.Add(j);
                        else
                            toDrop.Add(i);
                    }
                }
            }
        }

        _selectedIndices = Enumerable.Range(0, p)
            .Where(i => !toDrop.Contains(i))
            .ToArray();

        if (_selectedIndices.Length == 0)
            _selectedIndices = new[] { 0 };

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationFilter has not been fitted.");

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
        throw new NotSupportedException("CorrelationFilter does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationFilter has not been fitted.");

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
