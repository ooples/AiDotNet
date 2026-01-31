using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Fisher-Relief Hybrid feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines Fisher Score (parametric, class separability) with ReliefF
/// (instance-based, neighbor relationships) to leverage both perspectives.
/// </para>
/// <para><b>For Beginners:</b> Fisher Score looks at class averages while
/// ReliefF looks at individual sample relationships. Combining them gives
/// you features that work well from both viewpoints - good class separation
/// AND good at distinguishing similar vs different samples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FisherReliefHybrid<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _fisherWeight;
    private readonly int _nNeighbors;

    private double[]? _fisherScores;
    private double[]? _reliefScores;
    private double[]? _hybridScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double FisherWeight => _fisherWeight;
    public double[]? FisherScores => _fisherScores;
    public double[]? ReliefScores => _reliefScores;
    public double[]? HybridScores => _hybridScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FisherReliefHybrid(
        int nFeaturesToSelect = 10,
        double fisherWeight = 0.5,
        int nNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (fisherWeight < 0 || fisherWeight > 1)
            throw new ArgumentException("Fisher weight must be between 0 and 1.", nameof(fisherWeight));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _fisherWeight = fisherWeight;
        _nNeighbors = nNeighbors;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FisherReliefHybrid requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute Fisher Scores
        _fisherScores = ComputeFisherScores(data, target, n, p);

        // Compute ReliefF Scores
        _reliefScores = ComputeReliefScores(data, target, n, p);

        // Normalize both score arrays
        var normalizedFisher = Normalize(_fisherScores);
        var normalizedRelief = Normalize(_reliefScores);

        // Combine scores
        _hybridScores = new double[p];
        for (int j = 0; j < p; j++)
            _hybridScores[j] = _fisherWeight * normalizedFisher[j] + (1 - _fisherWeight) * normalizedRelief[j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _hybridScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeFisherScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];

        var class0 = new List<int>();
        var class1 = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        if (class0.Count < 2 || class1.Count < 2)
            return scores;

        for (int j = 0; j < p; j++)
        {
            double mean0 = class0.Sum(i => NumOps.ToDouble(data[i, j])) / class0.Count;
            double mean1 = class1.Sum(i => NumOps.ToDouble(data[i, j])) / class1.Count;

            double var0 = class0.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2)) / class0.Count;
            double var1 = class1.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2)) / class1.Count;

            double denom = var0 + var1;
            scores[j] = denom > 1e-10 ? Math.Pow(mean0 - mean1, 2) / denom : 0;
        }

        return scores;
    }

    private double[] ComputeReliefScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        int sampleSize = Math.Min(50, n);

        for (int s = 0; s < sampleSize; s++)
        {
            int idx = s * n / sampleSize;
            double targetVal = NumOps.ToDouble(target[idx]);

            var distances = new List<(int Index, double Dist, bool SameClass)>();
            for (int i = 0; i < n; i++)
            {
                if (i == idx) continue;

                double dist = 0;
                for (int j = 0; j < p; j++)
                {
                    double diff = NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[i, j]);
                    dist += diff * diff;
                }

                bool sameClass = Math.Abs(NumOps.ToDouble(target[i]) - targetVal) < 0.5;
                distances.Add((i, dist, sameClass));
            }

            var hits = distances.Where(d => d.SameClass).OrderBy(d => d.Dist).Take(_nNeighbors).ToList();
            var misses = distances.Where(d => !d.SameClass).OrderBy(d => d.Dist).Take(_nNeighbors).ToList();

            foreach (var hit in hits)
            {
                for (int j = 0; j < p; j++)
                {
                    double diff = Math.Abs(NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[hit.Index, j]));
                    scores[j] -= diff / (sampleSize * _nNeighbors);
                }
            }

            foreach (var miss in misses)
            {
                for (int j = 0; j < p; j++)
                {
                    double diff = Math.Abs(NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[miss.Index, j]));
                    scores[j] += diff / (sampleSize * _nNeighbors);
                }
            }
        }

        return scores;
    }

    private double[] Normalize(double[] scores)
    {
        double min = scores.Min();
        double max = scores.Max();
        double range = max - min;

        if (range < 1e-10)
            return scores.Select(_ => 0.5).ToArray();

        return scores.Select(s => (s - min) / range).ToArray();
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherReliefHybrid has not been fitted.");

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
        throw new NotSupportedException("FisherReliefHybrid does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherReliefHybrid has not been fitted.");

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
