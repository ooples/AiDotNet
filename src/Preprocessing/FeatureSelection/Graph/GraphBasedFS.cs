using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Graph;

/// <summary>
/// Graph-based Feature Selection using feature relationships.
/// </summary>
/// <remarks>
/// <para>
/// Constructs a graph where nodes are features and edges represent
/// relationships (like correlation). Uses graph centrality measures
/// to identify important features.
/// </para>
/// <para><b>For Beginners:</b> Imagine features as people in a social network.
/// Features that are "well connected" (correlated with many other features)
/// might be key features. We use social network analysis techniques
/// to find the most "influential" features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GraphBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _edgeThreshold;
    private readonly string _centralityMeasure;

    private double[]? _centralityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CentralityScores => _centralityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GraphBasedFS(
        int nFeaturesToSelect = 10,
        double edgeThreshold = 0.3,
        string centralityMeasure = "pagerank",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _edgeThreshold = edgeThreshold;
        _centralityMeasure = centralityMeasure.ToLowerInvariant();
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GraphBasedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Compute feature-target correlations
        var targetCorrelations = ComputeTargetCorrelations(X, y, n, p);

        // Build adjacency matrix based on feature correlations
        var adjacency = BuildAdjacencyMatrix(X, n, p);

        // Compute centrality scores
        _centralityScores = _centralityMeasure switch
        {
            "pagerank" => ComputePageRank(adjacency, p),
            "degree" => ComputeDegreeCentrality(adjacency, p),
            "eigenvector" => ComputeEigenvectorCentrality(adjacency, p),
            "betweenness" => ComputeBetweennessCentrality(adjacency, p),
            _ => ComputePageRank(adjacency, p)
        };

        // Combine centrality with target correlation
        for (int j = 0; j < p; j++)
            _centralityScores[j] *= Math.Abs(targetCorrelations[j]);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _centralityScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeTargetCorrelations(double[,] X, double[] y, int n, int p)
    {
        var correlations = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = X[i, j] - xMean;
                double yDiff = y[i] - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            correlations[j] = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
        }

        return correlations;
    }

    private double[,] BuildAdjacencyMatrix(double[,] X, int n, int p)
    {
        var adj = new double[p, p];
        var means = new double[p];
        var stds = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = X[i, j] - means[j];
                stds[j] += diff * diff;
            }
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
        }

        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += (X[i, j1] - means[j1]) * (X[i, j2] - means[j2]);
                double corr = sum / ((n - 1) * stds[j1] * stds[j2]);

                if (Math.Abs(corr) >= _edgeThreshold)
                {
                    adj[j1, j2] = Math.Abs(corr);
                    adj[j2, j1] = Math.Abs(corr);
                }
            }
        }

        return adj;
    }

    private double[] ComputePageRank(double[,] adj, int p, double damping = 0.85, int maxIter = 100)
    {
        var scores = new double[p];
        for (int j = 0; j < p; j++)
            scores[j] = 1.0 / p;

        var outDegrees = new double[p];
        for (int j = 0; j < p; j++)
            for (int k = 0; k < p; k++)
                outDegrees[j] += adj[j, k];

        for (int iter = 0; iter < maxIter; iter++)
        {
            var newScores = new double[p];
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < p; k++)
                    if (outDegrees[k] > 1e-10)
                        sum += adj[k, j] * scores[k] / outDegrees[k];
                newScores[j] = (1 - damping) / p + damping * sum;
            }
            scores = newScores;
        }

        return scores;
    }

    private double[] ComputeDegreeCentrality(double[,] adj, int p)
    {
        var scores = new double[p];
        for (int j = 0; j < p; j++)
            for (int k = 0; k < p; k++)
                if (adj[j, k] > 0)
                    scores[j]++;
        return scores;
    }

    private double[] ComputeEigenvectorCentrality(double[,] adj, int p, int maxIter = 100)
    {
        var scores = new double[p];
        for (int j = 0; j < p; j++)
            scores[j] = 1.0;

        for (int iter = 0; iter < maxIter; iter++)
        {
            var newScores = new double[p];
            for (int j = 0; j < p; j++)
                for (int k = 0; k < p; k++)
                    newScores[j] += adj[j, k] * scores[k];

            double norm = Math.Sqrt(newScores.Sum(s => s * s)) + 1e-10;
            for (int j = 0; j < p; j++)
                newScores[j] /= norm;

            scores = newScores;
        }

        return scores;
    }

    private double[] ComputeBetweennessCentrality(double[,] adj, int p)
    {
        // Simplified betweenness approximation
        var scores = new double[p];
        for (int s = 0; s < p; s++)
        {
            for (int t = s + 1; t < p; t++)
            {
                // Find shortest path (BFS-like for unweighted)
                var visited = new bool[p];
                var prev = new List<int>[p];
                for (int j = 0; j < p; j++)
                    prev[j] = new List<int>();

                var queue = new Queue<int>();
                queue.Enqueue(s);
                visited[s] = true;

                while (queue.Count > 0)
                {
                    int curr = queue.Dequeue();
                    if (curr == t) break;

                    for (int next = 0; next < p; next++)
                    {
                        if (adj[curr, next] > 0 && !visited[next])
                        {
                            visited[next] = true;
                            prev[next].Add(curr);
                            queue.Enqueue(next);
                        }
                    }
                }

                // Count nodes on shortest paths
                if (visited[t])
                {
                    var pathNodes = new HashSet<int>();
                    var pathQueue = new Queue<int>();
                    pathQueue.Enqueue(t);
                    while (pathQueue.Count > 0)
                    {
                        int curr = pathQueue.Dequeue();
                        foreach (int node in prev[curr])
                        {
                            if (node != s)
                                pathNodes.Add(node);
                            pathQueue.Enqueue(node);
                        }
                    }

                    foreach (int node in pathNodes)
                        scores[node]++;
                }
            }
        }

        return scores;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GraphBasedFS has not been fitted.");

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
        throw new NotSupportedException("GraphBasedFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GraphBasedFS has not been fitted.");

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
