using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Hierarchical Clustering-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses hierarchical (agglomerative) clustering to build a tree of feature
/// relationships, then cuts the tree to create groups and selects
/// representative features from each group.
/// </para>
/// <para><b>For Beginners:</b> This method builds a family tree of features,
/// starting with each feature as its own group, then gradually merging the
/// most similar ones. When we cut this tree at a certain height, we get
/// groups of related features. We pick the best feature from each group.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HierarchicalFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly string _linkage;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HierarchicalFS(
        int nFeaturesToSelect = 10,
        string linkage = "average",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _linkage = linkage.ToLowerInvariant();
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 2)
            throw new ArgumentException("HierarchicalFS requires at least 2 rows to compute variance/correlation.", nameof(data));

        // Convert to arrays
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Compute feature distance matrix based on correlation
        var distances = ComputeFeatureDistances(X, n, p);

        // Perform hierarchical clustering
        var clusterAssignments = AgglomerativeClustering(distances, p, _nFeaturesToSelect);

        // Compute feature scores (variance)
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += X[i, j];
            mean /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = X[i, j] - mean;
                _featureScores[j] += diff * diff;
            }
            _featureScores[j] /= (n - 1);
        }

        // Select best feature from each cluster
        int numClusters = clusterAssignments.Max() + 1;
        var selectedList = new List<int>();

        for (int c = 0; c < numClusters; c++)
        {
            var clusterFeatures = Enumerable.Range(0, p)
                .Where(j => clusterAssignments[j] == c)
                .ToList();

            if (clusterFeatures.Count > 0)
            {
                int bestFeature = clusterFeatures
                    .OrderByDescending(j => _featureScores[j])
                    .First();
                selectedList.Add(bestFeature);
            }
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double[,] ComputeFeatureDistances(double[,] X, int n, int p)
    {
        var distances = new double[p, p];
        var means = new double[p];
        var stds = new double[p];

        // Compute means and stds
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

        // Compute distance = 1 - |correlation|
        for (int j1 = 0; j1 < p; j1++)
        {
            distances[j1, j1] = 0;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += (X[i, j1] - means[j1]) * (X[i, j2] - means[j2]);
                double corr = sum / ((n - 1) * stds[j1] * stds[j2]);
                distances[j1, j2] = 1 - Math.Abs(corr);
                distances[j2, j1] = distances[j1, j2];
            }
        }

        return distances;
    }

    private int[] AgglomerativeClustering(double[,] distances, int p, int nClusters)
    {
        var assignments = Enumerable.Range(0, p).ToArray();
        var clusterSizes = new int[p];
        for (int j = 0; j < p; j++)
            clusterSizes[j] = 1;

        var clusterDistances = (double[,])distances.Clone();
        int numActiveClusters = p;

        while (numActiveClusters > nClusters)
        {
            // Find closest pair of clusters
            double minDist = double.MaxValue;
            int merge1 = -1, merge2 = -1;

            for (int c1 = 0; c1 < p; c1++)
            {
                if (clusterSizes[c1] == 0) continue;
                for (int c2 = c1 + 1; c2 < p; c2++)
                {
                    if (clusterSizes[c2] == 0) continue;
                    if (clusterDistances[c1, c2] < minDist)
                    {
                        minDist = clusterDistances[c1, c2];
                        merge1 = c1;
                        merge2 = c2;
                    }
                }
            }

            if (merge1 < 0) break;

            // Merge clusters
            for (int j = 0; j < p; j++)
            {
                if (assignments[j] == merge2)
                    assignments[j] = merge1;
            }

            int newSize = clusterSizes[merge1] + clusterSizes[merge2];

            // Update distances using linkage method
            for (int c = 0; c < p; c++)
            {
                if (c == merge1 || c == merge2 || clusterSizes[c] == 0) continue;

                double newDist = _linkage switch
                {
                    "single" => Math.Min(clusterDistances[merge1, c], clusterDistances[merge2, c]),
                    "complete" => Math.Max(clusterDistances[merge1, c], clusterDistances[merge2, c]),
                    "average" => (clusterDistances[merge1, c] * clusterSizes[merge1] +
                                 clusterDistances[merge2, c] * clusterSizes[merge2]) / newSize,
                    _ => (clusterDistances[merge1, c] * clusterSizes[merge1] +
                         clusterDistances[merge2, c] * clusterSizes[merge2]) / newSize
                };

                clusterDistances[merge1, c] = newDist;
                clusterDistances[c, merge1] = newDist;
            }

            clusterSizes[merge1] = newSize;
            clusterSizes[merge2] = 0;
            numActiveClusters--;
        }

        // Renumber clusters from 0 to n-1
        var uniqueClusters = assignments.Distinct().OrderBy(c => c).ToList();
        var renumbering = new Dictionary<int, int>();
        for (int i = 0; i < uniqueClusters.Count; i++)
            renumbering[uniqueClusters[i]] = i;

        return assignments.Select(c => renumbering[c]).ToArray();
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HierarchicalFS has not been fitted.");

        if (data.Columns != _nInputFeatures)
            throw new ArgumentException($"Expected {_nInputFeatures} columns but got {data.Columns}.", nameof(data));

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
        throw new NotSupportedException("HierarchicalFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HierarchicalFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HierarchicalFS has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (inputFeatureNames.Length < _nInputFeatures)
            throw new ArgumentException(
                $"Expected at least {_nInputFeatures} feature names, but got {inputFeatureNames.Length}.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
