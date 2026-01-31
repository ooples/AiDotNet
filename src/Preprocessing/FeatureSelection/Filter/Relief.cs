using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Relief algorithm for feature selection based on instance-based learning.
/// </summary>
/// <remarks>
/// <para>
/// Relief estimates feature quality by sampling instances and computing the
/// difference between distances to nearest hits (same class) and nearest misses
/// (different class). Features that differentiate classes get higher weights.
/// </para>
/// <para><b>For Beginners:</b> Relief works by looking at individual examples.
/// For each example, it finds the nearest example from the same class (hit) and
/// the nearest from a different class (miss). Good features should have similar
/// values for hits and different values for misses. This intuitive approach
/// works well for detecting local feature relevance.
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
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

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

        // Normalize features
        var normalizedData = new double[n, p];
        var minVals = new double[p];
        var ranges = new double[p];

        for (int j = 0; j < p; j++)
        {
            minVals[j] = double.MaxValue;
            double maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVals[j] = Math.Min(minVals[j], val);
                maxVal = Math.Max(maxVal, val);
            }
            ranges[j] = maxVal - minVals[j];
            if (ranges[j] < 1e-10) ranges[j] = 1;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                normalizedData[i, j] = (NumOps.ToDouble(data[i, j]) - minVals[j]) / ranges[j];

        // Get class labels
        var targetLabels = new int[n];
        for (int i = 0; i < n; i++)
            targetLabels[i] = (int)Math.Round(NumOps.ToDouble(target[i]));

        // Initialize weights
        _featureWeights = new double[p];

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int iterations = Math.Min(_nIterations, n);

        for (int iter = 0; iter < iterations; iter++)
        {
            // Sample a random instance
            int ri = rand.Next(n);
            int riClass = targetLabels[ri];

            // Find nearest hit (same class) and nearest miss (different class)
            int nearestHit = -1;
            int nearestMiss = -1;
            double nearestHitDist = double.MaxValue;
            double nearestMissDist = double.MaxValue;

            for (int j = 0; j < n; j++)
            {
                if (j == ri) continue;

                double dist = 0;
                for (int f = 0; f < p; f++)
                    dist += Math.Abs(normalizedData[ri, f] - normalizedData[j, f]);

                if (targetLabels[j] == riClass)
                {
                    if (dist < nearestHitDist)
                    {
                        nearestHitDist = dist;
                        nearestHit = j;
                    }
                }
                else
                {
                    if (dist < nearestMissDist)
                    {
                        nearestMissDist = dist;
                        nearestMiss = j;
                    }
                }
            }

            // Update weights
            if (nearestHit >= 0 && nearestMiss >= 0)
            {
                for (int f = 0; f < p; f++)
                {
                    double diffHit = Math.Abs(normalizedData[ri, f] - normalizedData[nearestHit, f]);
                    double diffMiss = Math.Abs(normalizedData[ri, f] - normalizedData[nearestMiss, f]);
                    _featureWeights[f] += (diffMiss - diffHit) / iterations;
                }
            }
        }

        // Select top features
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
