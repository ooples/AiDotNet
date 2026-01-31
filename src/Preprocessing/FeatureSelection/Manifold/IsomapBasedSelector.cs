using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Isomap-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Isomap (Isometric Mapping) concepts to identify features that
/// best preserve geodesic distances along the data manifold.
/// </para>
/// <para><b>For Beginners:</b> Isomap measures distances along the surface
/// of your data (like measuring distance along a curved road rather than
/// cutting through). Features that help preserve these "along the surface"
/// distances capture the true structure of your data.
/// </para>
/// </remarks>
public class IsomapBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public IsomapBasedSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
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

        // Compute pairwise Euclidean distances
        var euclideanDist = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int d = 0; d < p; d++)
                    dist += (X[i, d] - X[j, d]) * (X[i, d] - X[j, d]);
                euclideanDist[i, j] = Math.Sqrt(dist);
                euclideanDist[j, i] = euclideanDist[i, j];
            }
        }

        // Build k-NN graph with distances
        var graphDist = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                graphDist[i, j] = double.PositiveInfinity;
            graphDist[i, i] = 0;

            // Find k nearest neighbors
            var neighbors = Enumerable.Range(0, n)
                .Where(j => j != i)
                .OrderBy(j => euclideanDist[i, j])
                .Take(k)
                .ToList();

            foreach (int j in neighbors)
            {
                graphDist[i, j] = euclideanDist[i, j];
                graphDist[j, i] = euclideanDist[i, j];
            }
        }

        // Floyd-Warshall for geodesic distances
        for (int k_iter = 0; k_iter < n; k_iter++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (graphDist[i, k_iter] + graphDist[k_iter, j] < graphDist[i, j])
                        graphDist[i, j] = graphDist[i, k_iter] + graphDist[k_iter, j];

        // Compute feature importance based on contribution to geodesic preservation
        _featureScores = new double[p];
        for (int d = 0; d < p; d++)
        {
            // Compute distances using only this feature
            double correlation = 0;
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (double.IsInfinity(graphDist[i, j])) continue;

                    double singleFeatDist = Math.Abs(X[i, d] - X[j, d]);
                    double geodesic = graphDist[i, j];

                    // Use rank correlation idea
                    correlation += singleFeatDist * geodesic;
                    count++;
                }
            }

            _featureScores[d] = count > 0 ? correlation / count : 0;
        }

        // Normalize scores
        double maxScore = _featureScores.Max();
        if (maxScore > 0)
            for (int d = 0; d < p; d++)
                _featureScores[d] /= maxScore;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IsomapBasedSelector has not been fitted.");

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
        throw new NotSupportedException("IsomapBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IsomapBasedSelector has not been fitted.");

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
