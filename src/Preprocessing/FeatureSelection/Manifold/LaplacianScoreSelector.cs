using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Laplacian Score Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Laplacian Score to select features that best preserve the local
/// structure of the data. Features with lower Laplacian scores are considered
/// more important.
/// </para>
/// <para><b>For Beginners:</b> The Laplacian score measures how much a feature
/// changes between nearby points. Features that are smooth (don't change much
/// between neighbors) have low Laplacian scores and are considered more
/// important for preserving local structure.
/// </para>
/// </remarks>
public class LaplacianScoreSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _sigma;

    private double[]? _laplacianScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? LaplacianScores => _laplacianScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LaplacianScoreSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        double sigma = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _sigma = sigma;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        int k = Math.Min(_nNeighbors, n - 1);

        // Build k-NN similarity graph
        var W = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            var distances = new List<(int idx, double dist)>();
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double dist = 0;
                for (int d = 0; d < p; d++)
                    dist += (X[i, d] - X[j, d]) * (X[i, d] - X[j, d]);
                distances.Add((j, dist));
            }

            var neighbors = distances.OrderBy(x => x.dist).Take(k).ToList();
            foreach (var (idx, dist) in neighbors)
            {
                double sim = Math.Exp(-dist / (2 * _sigma * _sigma));
                W[i, idx] = sim;
                W[idx, i] = sim; // Make symmetric
            }
        }

        // Compute degree matrix
        var D = new double[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                D[i] += W[i, j];

        double Dsum = D.Sum();

        // Compute Laplacian score for each feature
        _laplacianScores = new double[p];
        for (int d = 0; d < p; d++)
        {
            // Get feature column
            var f = new double[n];
            for (int i = 0; i < n; i++) f[i] = X[i, d];

            // Compute f~ = f - (f'D1 / 1'D1) * 1
            double fD1 = 0;
            for (int i = 0; i < n; i++) fD1 += f[i] * D[i];
            double alpha = fD1 / (Dsum + 1e-10);

            var fTilde = new double[n];
            for (int i = 0; i < n; i++) fTilde[i] = f[i] - alpha;

            // Laplacian score = f~'Lf~ / f~'Df~
            // where L = D - W
            double numerator = 0;
            double denominator = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j)
                        numerator += fTilde[i] * D[i] * fTilde[i];
                    numerator -= fTilde[i] * W[i, j] * fTilde[j];
                }
                denominator += fTilde[i] * D[i] * fTilde[i];
            }

            _laplacianScores[d] = denominator > 1e-10 ? numerator / denominator : double.MaxValue;
        }

        // Select features with LOWEST Laplacian scores
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderBy(j => _laplacianScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScoreSelector has not been fitted.");

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
        throw new NotSupportedException("LaplacianScoreSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScoreSelector has not been fitted.");

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
