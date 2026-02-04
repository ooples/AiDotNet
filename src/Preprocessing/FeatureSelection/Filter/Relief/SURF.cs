using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// Spatially Uniform ReliefF (SURF) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SURF is a variant of ReliefF that uses distance thresholds instead of k
/// nearest neighbors. It considers all instances within a threshold distance,
/// making it more robust and parameter-free for the neighbor count.
/// </para>
/// <para><b>For Beginners:</b> Unlike ReliefF which looks at exactly k neighbors,
/// SURF looks at all neighbors within a certain distance. This is more natural
/// because it considers the actual distribution of your data - areas with more
/// data points contribute more comparisons.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SURF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _distanceThreshold;

    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double DistanceThreshold => _distanceThreshold;
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SURF(
        int nFeaturesToSelect = 10,
        double distanceThreshold = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (distanceThreshold <= 0)
            throw new ArgumentException("Distance threshold must be positive.", nameof(distanceThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _distanceThreshold = distanceThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SURF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute feature ranges for normalization
        var featureRanges = new double[p];
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < min) min = val;
                if (val > max) max = val;
            }
            featureRanges[j] = max - min;
        }

        // Compute average distance for threshold
        double avgDist = ComputeAverageDistance(data, n, p, featureRanges);
        double threshold = avgDist * _distanceThreshold;

        // Initialize weights
        _featureWeights = new double[p];

        // Process each instance
        for (int i = 0; i < n; i++)
        {
            int iClass = (int)Math.Round(NumOps.ToDouble(target[i]));

            // Find all neighbors within threshold
            var nearHits = new List<int>();
            var nearMisses = new List<int>();

            for (int k = 0; k < n; k++)
            {
                if (k == i) continue;

                double dist = ComputeNormalizedDistance(data, i, k, p, featureRanges);

                if (dist <= threshold)
                {
                    int kClass = (int)Math.Round(NumOps.ToDouble(target[k]));
                    if (kClass == iClass)
                        nearHits.Add(k);
                    else
                        nearMisses.Add(k);
                }
            }

            // Update weights
            foreach (int hitIdx in nearHits)
            {
                for (int j = 0; j < p; j++)
                {
                    double diff = Diff(data, i, hitIdx, j, featureRanges[j]);
                    _featureWeights[j] -= diff / (n * Math.Max(1, nearHits.Count));
                }
            }

            foreach (int missIdx in nearMisses)
            {
                for (int j = 0; j < p; j++)
                {
                    double diff = Diff(data, i, missIdx, j, featureRanges[j]);
                    _featureWeights[j] += diff / (n * Math.Max(1, nearMisses.Count));
                }
            }
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureWeights[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeAverageDistance(Matrix<T> data, int n, int p, double[] ranges)
    {
        int sampleSize = Math.Min(100, n);
        double totalDist = 0;
        int count = 0;

        for (int i = 0; i < sampleSize; i++)
        {
            for (int j = i + 1; j < sampleSize; j++)
            {
                totalDist += ComputeNormalizedDistance(data, i, j, p, ranges);
                count++;
            }
        }

        return count > 0 ? totalDist / count : 1.0;
    }

    private double ComputeNormalizedDistance(Matrix<T> data, int i1, int i2, int p, double[] ranges)
    {
        double dist = 0;
        for (int j = 0; j < p; j++)
        {
            double diff = Diff(data, i1, i2, j, ranges[j]);
            dist += diff * diff;
        }
        return Math.Sqrt(dist);
    }

    private double Diff(Matrix<T> data, int i1, int i2, int j, double range)
    {
        if (range < 1e-10) return 0;
        return Math.Abs(NumOps.ToDouble(data[i1, j]) - NumOps.ToDouble(data[i2, j])) / range;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SURF has not been fitted.");

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
        throw new NotSupportedException("SURF does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SURF has not been fitted.");

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
