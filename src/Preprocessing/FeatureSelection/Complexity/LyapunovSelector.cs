using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Lyapunov Exponent based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their largest Lyapunov exponent, which measures
/// sensitivity to initial conditions (chaos) in dynamical systems.
/// </para>
/// <para><b>For Beginners:</b> The Lyapunov exponent measures how fast nearby
/// trajectories diverge over time. Positive values indicate chaos (butterfly effect),
/// zero indicates marginal stability, negative indicates convergence. Features with
/// positive Lyapunov exponents may contain complex nonlinear dynamics.
/// </para>
/// </remarks>
public class LyapunovSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _embeddingDimension;
    private readonly int _delay;

    private double[]? _lyapunovValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int EmbeddingDimension => _embeddingDimension;
    public int Delay => _delay;
    public double[]? LyapunovValues => _lyapunovValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LyapunovSelector(
        int nFeaturesToSelect = 10,
        int embeddingDimension = 3,
        int delay = 1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (embeddingDimension < 2)
            throw new ArgumentException("Embedding dimension must be at least 2.", nameof(embeddingDimension));
        if (delay < 1)
            throw new ArgumentException("Delay must be at least 1.", nameof(delay));

        _nFeaturesToSelect = nFeaturesToSelect;
        _embeddingDimension = embeddingDimension;
        _delay = delay;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        int minLength = (_embeddingDimension - 1) * _delay + 20;
        if (n < minLength)
            throw new ArgumentException($"Need at least {minLength} samples.");

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _lyapunovValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _lyapunovValues[j] = EstimateLyapunov(col);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _lyapunovValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EstimateLyapunov(double[] data)
    {
        int n = data.Length;
        int numVectors = n - (_embeddingDimension - 1) * _delay;
        if (numVectors < 10) return 0;

        // Create embedded vectors
        var vectors = new double[numVectors][];
        for (int i = 0; i < numVectors; i++)
        {
            vectors[i] = new double[_embeddingDimension];
            for (int d = 0; d < _embeddingDimension; d++)
                vectors[i][d] = data[i + d * _delay];
        }

        // Rosenstein's algorithm (simplified)
        var divergences = new List<double>();

        // For each point, find nearest neighbor (not too close in time)
        int minTimeSep = _embeddingDimension * _delay;

        for (int i = 0; i < numVectors - 10; i++)
        {
            double minDist = double.MaxValue;
            int nearestIdx = -1;

            for (int k = 0; k < numVectors; k++)
            {
                if (Math.Abs(k - i) < minTimeSep) continue;

                double dist = 0;
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    double diff = vectors[i][d] - vectors[k][d];
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);

                if (dist > 0 && dist < minDist)
                {
                    minDist = dist;
                    nearestIdx = k;
                }
            }

            if (nearestIdx < 0 || nearestIdx >= numVectors - 5) continue;

            // Track divergence over time steps
            for (int step = 1; step <= 5; step++)
            {
                if (i + step >= numVectors || nearestIdx + step >= numVectors)
                    break;

                double dist = 0;
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    double diff = vectors[i + step][d] - vectors[nearestIdx + step][d];
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);

                if (dist > 0 && minDist > 0)
                {
                    divergences.Add(Math.Log(dist / minDist) / step);
                }
            }
        }

        if (divergences.Count == 0)
            return 0;

        return divergences.Average();
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LyapunovSelector has not been fitted.");

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
        throw new NotSupportedException("LyapunovSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LyapunovSelector has not been fitted.");

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
