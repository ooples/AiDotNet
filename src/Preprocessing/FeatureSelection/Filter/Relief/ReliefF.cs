using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// ReliefF algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// ReliefF is an extension of the Relief algorithm that handles multi-class
/// problems and uses k nearest neighbors. It evaluates features based on their
/// ability to distinguish between instances of different classes.
/// </para>
/// <para><b>For Beginners:</b> ReliefF works by looking at each data point and
/// comparing it to its nearest neighbors. Features are scored based on how well
/// they separate a point from neighbors of different classes while keeping it
/// close to neighbors of the same class. It's like finding features that cluster
/// similar items together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nSamples;
    private readonly int? _randomState;

    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public int NSamples => _nSamples;
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int nSamples = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        if (nSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1.", nameof(nSamples));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _nSamples = nSamples;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ReliefF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Group instances by class
        var classSamples = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classSamples.ContainsKey(label))
                classSamples[label] = new List<int>();
            classSamples[label].Add(i);
        }

        // Compute class probabilities
        var classProbabilities = new Dictionary<int, double>();
        foreach (var kvp in classSamples)
            classProbabilities[kvp.Key] = (double)kvp.Value.Count / n;

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

        // Initialize weights
        _featureWeights = new double[p];

        // Sample instances
        int sampleCount = Math.Min(_nSamples, n);
        var sampleIndices = Enumerable.Range(0, n)
            .OrderBy(_ => random.Next())
            .Take(sampleCount)
            .ToList();

        foreach (int idx in sampleIndices)
        {
            int sampleClass = (int)Math.Round(NumOps.ToDouble(target[idx]));

            // Find k nearest hits (same class)
            var hits = FindNearestNeighbors(data, idx, classSamples[sampleClass], p, featureRanges);

            // Find k nearest misses (different classes, weighted by probability)
            foreach (var classKvp in classSamples)
            {
                if (classKvp.Key == sampleClass) continue;

                double classWeight = classProbabilities[classKvp.Key] / (1 - classProbabilities[sampleClass]);
                var misses = FindNearestNeighbors(data, idx, classKvp.Value, p, featureRanges);

                foreach (int missIdx in misses)
                {
                    for (int j = 0; j < p; j++)
                    {
                        double diff = Diff(data, idx, missIdx, j, featureRanges[j]);
                        _featureWeights[j] += classWeight * diff / (sampleCount * _nNeighbors);
                    }
                }
            }

            // Subtract hit contribution
            foreach (int hitIdx in hits)
            {
                for (int j = 0; j < p; j++)
                {
                    double diff = Diff(data, idx, hitIdx, j, featureRanges[j]);
                    _featureWeights[j] -= diff / (sampleCount * _nNeighbors);
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

    private List<int> FindNearestNeighbors(Matrix<T> data, int idx, List<int> candidates, int p, double[] ranges)
    {
        var distances = new List<(int Index, double Distance)>();

        foreach (int candIdx in candidates)
        {
            if (candIdx == idx) continue;

            double dist = 0;
            for (int j = 0; j < p; j++)
            {
                double diff = Diff(data, idx, candIdx, j, ranges[j]);
                dist += diff * diff;
            }
            distances.Add((candIdx, dist));
        }

        return distances
            .OrderBy(d => d.Distance)
            .Take(_nNeighbors)
            .Select(d => d.Index)
            .ToList();
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
            throw new InvalidOperationException("ReliefF has not been fitted.");

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
        throw new NotSupportedException("ReliefF does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ReliefF has not been fitted.");

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
