using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// MultiSURF algorithm with adaptive distance thresholds per instance.
/// </summary>
/// <remarks>
/// <para>
/// MultiSURF improves SURF by using instance-specific distance thresholds
/// based on the local density around each instance. This adapts to varying
/// data densities across the feature space.
/// </para>
/// <para><b>For Beginners:</b> While SURF uses one global distance threshold,
/// MultiSURF calculates a custom threshold for each sample based on how
/// spread out its neighbors are. This works better for datasets with
/// varying density regions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MultiSURF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MultiSURF(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MultiSURF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Normalize features
        var minVals = new double[p];
        var maxVals = new double[p];
        for (int j = 0; j < p; j++)
        {
            minVals[j] = double.MaxValue;
            maxVals[j] = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double v = NumOps.ToDouble(data[i, j]);
                if (v < minVals[j]) minVals[j] = v;
                if (v > maxVals[j]) maxVals[j] = v;
            }
        }

        _featureWeights = new double[p];

        // Precompute all distances
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = ComputeDistance(data, i, j, minVals, maxVals, p);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        for (int ri = 0; ri < n; ri++)
        {
            double riClass = NumOps.ToDouble(target[ri]);

            // Compute mean and std of distances from ri
            double sumDist = 0;
            for (int i = 0; i < n; i++)
                if (i != ri) sumDist += distances[ri, i];
            double meanDist = sumDist / (n - 1);

            double sumSqDiff = 0;
            for (int i = 0; i < n; i++)
                if (i != ri) sumSqDiff += Math.Pow(distances[ri, i] - meanDist, 2);
            double stdDist = Math.Sqrt(sumSqDiff / (n - 1));

            // Threshold: mean - 0.5 * std (closer than average)
            double threshold = meanDist - 0.5 * stdDist;

            // Find near neighbors
            var nearHits = new List<int>();
            var nearMisses = new List<int>();

            for (int i = 0; i < n; i++)
            {
                if (i == ri) continue;
                if (distances[ri, i] > threshold) continue;

                double iClass = NumOps.ToDouble(target[i]);
                if (Math.Abs(iClass - riClass) < 1e-10)
                    nearHits.Add(i);
                else
                    nearMisses.Add(i);
            }

            // Update weights
            for (int j = 0; j < p; j++)
            {
                double range = maxVals[j] - minVals[j];
                if (range < 1e-10) continue;

                double ri_val = NumOps.ToDouble(data[ri, j]);

                // Contribution from hits
                foreach (int hitIdx in nearHits)
                {
                    double hit_val = NumOps.ToDouble(data[hitIdx, j]);
                    double diffHit = Math.Abs(ri_val - hit_val) / range;
                    _featureWeights[j] -= diffHit / (n * Math.Max(1, nearHits.Count));
                }

                // Contribution from misses
                foreach (int missIdx in nearMisses)
                {
                    double miss_val = NumOps.ToDouble(data[missIdx, j]);
                    double diffMiss = Math.Abs(ri_val - miss_val) / range;
                    _featureWeights[j] += diffMiss / (n * Math.Max(1, nearMisses.Count));
                }
            }
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureWeights
            .Select((w, idx) => (Weight: w, Index: idx))
            .OrderByDescending(x => x.Weight)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeDistance(Matrix<T> data, int i1, int i2, double[] minVals, double[] maxVals, int p)
    {
        double dist = 0;
        for (int j = 0; j < p; j++)
        {
            double range = maxVals[j] - minVals[j];
            if (range < 1e-10) continue;

            double v1 = NumOps.ToDouble(data[i1, j]);
            double v2 = NumOps.ToDouble(data[i2, j]);
            double diff = (v1 - v2) / range;
            dist += diff * diff;
        }
        return Math.Sqrt(dist);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiSURF has not been fitted.");

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
        throw new NotSupportedException("MultiSURF does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiSURF has not been fitted.");

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
