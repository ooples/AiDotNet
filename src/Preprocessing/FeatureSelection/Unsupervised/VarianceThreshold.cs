using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Variance Threshold for removing low-variance features.
/// </summary>
/// <remarks>
/// <para>
/// Removes features with variance below a specified threshold. Features with
/// very low variance carry little information and are often constant or
/// near-constant.
/// </para>
/// <para><b>For Beginners:</b> If a feature has the same value for almost all
/// samples, it can't help distinguish between them. This method removes such
/// uninformative features automatically.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VarianceThreshold<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    private double[]? _variances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public double[]? Variances => _variances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public VarianceThreshold(double threshold = 0.0, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0)
            throw new ArgumentException("Threshold must be non-negative.", nameof(threshold));

        _threshold = threshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _variances = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute mean
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= n;

            // Compute variance
            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            _variances[j] = variance / n;
        }

        // Select features above threshold
        _selectedIndices = _variances
            .Select((v, idx) => (Variance: v, Index: idx))
            .Where(x => x.Variance > _threshold)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        // If nothing selected, keep highest variance
        if (_selectedIndices.Length == 0)
        {
            int maxIdx = 0;
            double maxVar = _variances[0];
            for (int j = 1; j < p; j++)
            {
                if (_variances[j] > maxVar)
                {
                    maxVar = _variances[j];
                    maxIdx = j;
                }
            }
            _selectedIndices = new[] { maxIdx };
        }

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
            throw new InvalidOperationException("VarianceThreshold has not been fitted.");

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
        throw new NotSupportedException("VarianceThreshold does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VarianceThreshold has not been fitted.");

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
