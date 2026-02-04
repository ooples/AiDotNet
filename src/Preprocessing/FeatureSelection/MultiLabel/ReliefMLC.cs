using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Relief for Multi-Label Classification (ReliefMLC).
/// </summary>
/// <remarks>
/// <para>
/// Extends the Relief algorithm to handle multi-label problems. Uses Hamming
/// distance on label sets to determine similarity between instances.
/// </para>
/// <para><b>For Beginners:</b> Standard Relief compares single labels to find
/// hits/misses. ReliefMLC compares label sets, accounting for partial overlap
/// in multi-label scenarios.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ReliefMLC<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ReliefMLC(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int nIterations = -1,
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
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ReliefMLC requires target values. Use Fit(Matrix<T> data, Matrix<T> targets) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> targets)
    {
        if (data.Rows != targets.Rows)
            throw new ArgumentException("Target rows must match data rows.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nLabels = targets.Columns;
        int m = _nIterations > 0 ? Math.Min(_nIterations, n) : n;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

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

        // Extract label sets
        var labelSets = new bool[n][];
        for (int i = 0; i < n; i++)
        {
            labelSets[i] = new bool[nLabels];
            for (int l = 0; l < nLabels; l++)
                labelSets[i][l] = NumOps.ToDouble(targets[i, l]) > 0.5;
        }

        _featureWeights = new double[p];

        // Sample m instances
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(m).ToList();

        foreach (int ri in sampleIndices)
        {
            // Find k nearest neighbors based on feature distance
            var neighbors = FindKNearest(data, ri, n, _nNeighbors, minVals, maxVals, p);

            foreach (var (neighborIdx, dist) in neighbors)
            {
                // Compute label similarity using Jaccard or Hamming
                double labelSimilarity = ComputeLabelSimilarity(labelSets[ri], labelSets[neighborIdx], nLabels);

                // Weight: similar labels = hit, different labels = miss
                double weight = labelSimilarity;  // 1 = perfect match, 0 = no overlap

                for (int j = 0; j < p; j++)
                {
                    double range = maxVals[j] - minVals[j];
                    if (range < 1e-10) continue;

                    double ri_val = NumOps.ToDouble(data[ri, j]);
                    double neighbor_val = NumOps.ToDouble(data[neighborIdx, j]);
                    double featureDiff = Math.Abs(ri_val - neighbor_val) / range;

                    // Hit contribution (similar labels = subtract diff)
                    // Miss contribution (different labels = add diff)
                    _featureWeights[j] += (1 - weight) * featureDiff - weight * featureDiff;
                }
            }
        }

        // Normalize weights
        for (int j = 0; j < p; j++)
            _featureWeights[j] /= m * _nNeighbors;

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

    private List<(int Index, double Distance)> FindKNearest(Matrix<T> data, int targetIdx, int n, int k,
        double[] minVals, double[] maxVals, int p)
    {
        var distances = new List<(int Index, double Distance)>();

        for (int i = 0; i < n; i++)
        {
            if (i == targetIdx) continue;
            double dist = ComputeDistance(data, targetIdx, i, minVals, maxVals, p);
            distances.Add((i, dist));
        }

        return distances
            .OrderBy(d => d.Distance)
            .Take(Math.Min(k, distances.Count))
            .ToList();
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

    private double ComputeLabelSimilarity(bool[] labels1, bool[] labels2, int nLabels)
    {
        // Jaccard similarity
        int intersection = 0;
        int union = 0;

        for (int l = 0; l < nLabels; l++)
        {
            if (labels1[l] && labels2[l]) intersection++;
            if (labels1[l] || labels2[l]) union++;
        }

        return union > 0 ? (double)intersection / union : 1.0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> targets)
    {
        Fit(data, targets);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ReliefMLC has not been fitted.");

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
        throw new NotSupportedException("ReliefMLC does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ReliefMLC has not been fitted.");

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
