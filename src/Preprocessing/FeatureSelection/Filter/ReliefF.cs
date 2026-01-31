using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// ReliefF algorithm for feature selection based on nearest neighbor differences.
/// </summary>
/// <remarks>
/// <para>
/// ReliefF estimates feature relevance by sampling instances and comparing them to their
/// nearest neighbors of the same class (hits) and different classes (misses). Features
/// that differentiate between classes get higher weights.
/// </para>
/// <para><b>For Beginners:</b> ReliefF picks random samples and looks at their nearest
/// neighbors. If a feature differs between a sample and its nearby "enemies" (different class)
/// but stays similar to nearby "friends" (same class), it's useful. The algorithm rewards
/// features that help separate classes and penalizes those that don't.
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
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int nSamples = -1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

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

        // Convert data to double array for easier processing
        var dataDouble = new double[n, p];
        var targetInt = new int[n];
        for (int i = 0; i < n; i++)
        {
            targetInt[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                dataDouble[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Normalize features to [0, 1]
        var minVals = new double[p];
        var maxVals = new double[p];
        for (int j = 0; j < p; j++)
        {
            minVals[j] = double.MaxValue;
            maxVals[j] = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                minVals[j] = Math.Min(minVals[j], dataDouble[i, j]);
                maxVals[j] = Math.Max(maxVals[j], dataDouble[i, j]);
            }
            double range = maxVals[j] - minVals[j];
            if (range > 1e-10)
            {
                for (int i = 0; i < n; i++)
                    dataDouble[i, j] = (dataDouble[i, j] - minVals[j]) / range;
            }
        }

        // Group instances by class
        var classGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            if (!classGroups.ContainsKey(targetInt[i]))
                classGroups[targetInt[i]] = [];
            classGroups[targetInt[i]].Add(i);
        }

        // Class priors
        var classPriors = new Dictionary<int, double>();
        foreach (var kvp in classGroups)
            classPriors[kvp.Key] = (double)kvp.Value.Count / n;

        // Initialize weights
        _featureWeights = new double[p];

        // Number of samples to process
        int nSamplesToUse = _nSamples > 0 ? Math.Min(_nSamples, n) : n;
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(nSamplesToUse).ToArray();

        foreach (int idx in sampleIndices)
        {
            int targetClass = targetInt[idx];

            // Find nearest hits (same class)
            var nearestHits = FindNearestNeighbors(dataDouble, idx, classGroups[targetClass], p);

            // Find nearest misses for each other class
            foreach (var kvp in classGroups)
            {
                if (kvp.Key == targetClass) continue;

                var nearestMisses = FindNearestNeighbors(dataDouble, idx, kvp.Value, p);
                double classWeight = classPriors[kvp.Key] / (1 - classPriors[targetClass]);

                // Update weights
                for (int j = 0; j < p; j++)
                {
                    double hitDiff = 0, missDiff = 0;

                    foreach (int h in nearestHits)
                        hitDiff += Math.Abs(dataDouble[idx, j] - dataDouble[h, j]);

                    foreach (int m in nearestMisses)
                        missDiff += Math.Abs(dataDouble[idx, j] - dataDouble[m, j]);

                    hitDiff /= _nNeighbors;
                    missDiff /= _nNeighbors;

                    _featureWeights[j] += classWeight * (missDiff - hitDiff) / nSamplesToUse;
                }
            }
        }

        // Select top features by weight
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureWeights
            .Select((w, idx) => (Weight: w, Index: idx))
            .OrderByDescending(x => x.Weight)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] FindNearestNeighbors(double[,] data, int queryIdx, List<int> candidates, int p)
    {
        if (candidates.Count == 0)
            return [];

        var distances = new List<(int Index, double Distance)>();

        foreach (int candidate in candidates)
        {
            if (candidate == queryIdx) continue;

            double dist = 0;
            for (int j = 0; j < p; j++)
            {
                double diff = data[queryIdx, j] - data[candidate, j];
                dist += diff * diff;
            }
            distances.Add((candidate, dist));
        }

        return distances
            .OrderBy(x => x.Distance)
            .Take(Math.Min(_nNeighbors, distances.Count))
            .Select(x => x.Index)
            .ToArray();
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
