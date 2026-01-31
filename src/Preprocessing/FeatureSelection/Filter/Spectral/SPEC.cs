using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// SPEC (Spectral Feature Selection) for unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SPEC ranks features based on their consistency with the structure of the data
/// as captured by a similarity graph's spectral representation. It uses the
/// eigenvectors of the graph Laplacian to evaluate feature quality.
/// </para>
/// <para><b>For Beginners:</b> SPEC builds a graph where similar data points are
/// connected. It then looks at the graph's "shape" (via eigenvectors) and asks:
/// which features align well with this structure? Features that respect the natural
/// groupings in the data get high scores.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SPEC<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _sigma;
    private readonly int _nEigenvectors;

    private double[]? _specScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SpecScores => _specScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SPEC(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double sigma = 1.0,
        int nEigenvectors = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _sigma = sigma;
        _nEigenvectors = nEigenvectors;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Build affinity matrix
        var W = BuildAffinityMatrix(data, n, p);

        // Compute degree matrix D
        var D = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                D[i] += W[i, j];
        }

        // Compute normalized Laplacian L = I - D^(-1/2) W D^(-1/2)
        var L = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                    L[i, j] = 1.0;
                else if (D[i] > 1e-10 && D[j] > 1e-10)
                    L[i, j] = -W[i, j] / Math.Sqrt(D[i] * D[j]);
            }
        }

        // Compute eigenvectors using power iteration (simplified)
        var eigenvectors = ComputeTopEigenvectors(L, n, Math.Min(_nEigenvectors, n - 1));

        // Compute SPEC score for each feature
        _specScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract and normalize feature
            var f = new double[n];
            double sum = 0, sqSum = 0;
            for (int i = 0; i < n; i++)
            {
                f[i] = NumOps.ToDouble(data[i, j]);
                sum += f[i];
            }
            double mean = sum / n;

            for (int i = 0; i < n; i++)
            {
                f[i] -= mean;
                sqSum += f[i] * f[i];
            }
            double std = Math.Sqrt(sqSum / n);
            if (std > 1e-10)
            {
                for (int i = 0; i < n; i++)
                    f[i] /= std;
            }

            // Compute feature's alignment with eigenvectors
            double score = 0;
            for (int k = 0; k < eigenvectors.Count; k++)
            {
                var ev = eigenvectors[k];
                double dot = 0;
                for (int i = 0; i < n; i++)
                    dot += f[i] * ev[i];
                score += dot * dot;
            }

            _specScores[j] = score;
        }

        // Select features with highest SPEC scores
        _selectedIndices = _specScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(_nFeaturesToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] BuildAffinityMatrix(Matrix<T> data, int n, int p)
    {
        var W = new double[n, n];

        // Compute pairwise distances
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int k = 0; k < p; k++)
                {
                    double diff = NumOps.ToDouble(data[i, k]) - NumOps.ToDouble(data[j, k]);
                    dist += diff * diff;
                }
                distances[i, j] = Math.Sqrt(dist);
                distances[j, i] = distances[i, j];
            }
        }

        // Build k-NN graph with Gaussian kernel
        for (int i = 0; i < n; i++)
        {
            var neighbors = new List<(int Index, double Distance)>();
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                    neighbors.Add((j, distances[i, j]));
            }

            var kNearest = neighbors
                .OrderBy(x => x.Distance)
                .Take(_nNeighbors)
                .ToList();

            foreach (var neighbor in kNearest)
            {
                double weight = Math.Exp(-distances[i, neighbor.Index] * distances[i, neighbor.Index] /
                    (2 * _sigma * _sigma));
                W[i, neighbor.Index] = weight;
                W[neighbor.Index, i] = weight;
            }
        }

        return W;
    }

    private List<double[]> ComputeTopEigenvectors(double[,] L, int n, int k)
    {
        var eigenvectors = new List<double[]>();

        // Use power iteration for smallest eigenvectors (start from largest, then deflate)
        for (int ev = 0; ev < k; ev++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
                v[i] = 1.0 / Math.Sqrt(n);

            // Power iteration (50 iterations)
            for (int iter = 0; iter < 50; iter++)
            {
                var Lv = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Lv[i] += L[i, j] * v[j];

                // Orthogonalize against previous eigenvectors
                foreach (var prev in eigenvectors)
                {
                    double dot = 0;
                    for (int i = 0; i < n; i++)
                        dot += Lv[i] * prev[i];
                    for (int i = 0; i < n; i++)
                        Lv[i] -= dot * prev[i];
                }

                // Normalize
                double norm = 0;
                for (int i = 0; i < n; i++)
                    norm += Lv[i] * Lv[i];
                norm = Math.Sqrt(norm);

                if (norm > 1e-10)
                {
                    for (int i = 0; i < n; i++)
                        v[i] = Lv[i] / norm;
                }
            }

            eigenvectors.Add((double[])v.Clone());
        }

        return eigenvectors;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SPEC has not been fitted.");

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
        throw new NotSupportedException("SPEC does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SPEC has not been fitted.");

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
