using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// RReliefF algorithm for regression problems.
/// </summary>
/// <remarks>
/// <para>
/// RReliefF extends ReliefF to handle continuous targets (regression).
/// It uses distance-weighted contributions based on target value differences.
/// </para>
/// <para><b>For Beginners:</b> Standard Relief uses class labels (hit/miss).
/// RReliefF works with numeric targets by weighting neighbors based on
/// how similar their target values are to the sample's target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RReliefF(
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
            "RReliefF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Normalize target
        double yMin = double.MaxValue, yMax = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }
        double yRange = yMax - yMin;
        if (yRange < 1e-10) yRange = 1;

        _featureWeights = new double[p];
        var nDc = new double[p];  // Numerator for each feature
        double nD = 0;            // Denominator (cumulative target differences)

        // Sample m instances
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(m).ToList();

        foreach (int ri in sampleIndices)
        {
            double riY = (NumOps.ToDouble(target[ri]) - yMin) / yRange;

            // Find k nearest neighbors
            var neighbors = FindKNearest(data, ri, n, _nNeighbors, minVals, maxVals, p);

            foreach (var (neighborIdx, dist) in neighbors)
            {
                double weight = 1.0 / (dist + 1e-10);  // Distance weight

                double neighborY = (NumOps.ToDouble(target[neighborIdx]) - yMin) / yRange;
                double yDiff = Math.Abs(riY - neighborY);

                nD += weight * yDiff;

                for (int j = 0; j < p; j++)
                {
                    double range = maxVals[j] - minVals[j];
                    if (range < 1e-10) continue;

                    double ri_val = NumOps.ToDouble(data[ri, j]);
                    double neighbor_val = NumOps.ToDouble(data[neighborIdx, j]);
                    double fDiff = Math.Abs(ri_val - neighbor_val) / range;

                    nDc[j] += weight * yDiff * fDiff;
                }
            }
        }

        // Compute final weights
        for (int j = 0; j < p; j++)
        {
            if (nD > 1e-10)
                _featureWeights[j] = nDc[j] / nD;
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

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RReliefF has not been fitted.");

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
        throw new NotSupportedException("RReliefF does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RReliefF has not been fitted.");

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
