using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// Relief algorithm for instance-based feature weighting.
/// </summary>
/// <remarks>
/// <para>
/// Relief estimates feature quality by sampling instances and comparing
/// to nearest hits (same class) and misses (different class). Features
/// differentiating classes get higher weights.
/// </para>
/// <para><b>For Beginners:</b> For each sample, Relief looks at its nearest
/// neighbor from the same class (hit) and different class (miss). Good features
/// should be similar to hits and different from misses. This builds up a score
/// for each feature.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Relief<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public Relief(
        int nFeaturesToSelect = 10,
        int nIterations = -1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "Relief requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int m = _nIterations > 0 ? Math.Min(_nIterations, n) : n;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Normalize features for distance computation
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

        // Sample m instances
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(m).ToList();

        foreach (int ri in sampleIndices)
        {
            double riClass = NumOps.ToDouble(target[ri]);

            // Find nearest hit and miss
            int nearestHit = -1;
            int nearestMiss = -1;
            double minHitDist = double.MaxValue;
            double minMissDist = double.MaxValue;

            for (int i = 0; i < n; i++)
            {
                if (i == ri) continue;

                double dist = ComputeDistance(data, ri, i, minVals, maxVals, p);
                double iClass = NumOps.ToDouble(target[i]);

                if (Math.Abs(iClass - riClass) < 1e-10)
                {
                    if (dist < minHitDist)
                    {
                        minHitDist = dist;
                        nearestHit = i;
                    }
                }
                else
                {
                    if (dist < minMissDist)
                    {
                        minMissDist = dist;
                        nearestMiss = i;
                    }
                }
            }

            // Update weights
            if (nearestHit >= 0 && nearestMiss >= 0)
            {
                for (int j = 0; j < p; j++)
                {
                    double range = maxVals[j] - minVals[j];
                    if (range < 1e-10) continue;

                    double ri_val = NumOps.ToDouble(data[ri, j]);
                    double hit_val = NumOps.ToDouble(data[nearestHit, j]);
                    double miss_val = NumOps.ToDouble(data[nearestMiss, j]);

                    double diffHit = Math.Abs(ri_val - hit_val) / range;
                    double diffMiss = Math.Abs(ri_val - miss_val) / range;

                    _featureWeights[j] += (diffMiss - diffHit) / m;
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
            throw new InvalidOperationException("Relief has not been fitted.");

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
        throw new NotSupportedException("Relief does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("Relief has not been fitted.");

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
