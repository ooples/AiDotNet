using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Subspace;

/// <summary>
/// CLIQUE (CLustering In QUEst) subspace clustering algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLIQUE is a density-based, grid-based algorithm for finding subspace clusters.
/// It automatically identifies the relevant subspaces where clusters exist.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Partition each dimension into equal-width intervals
/// 2. Find dense 1-D units (cells with points >= threshold)
/// 3. Use Apriori principle to find dense k-D units
/// 4. Merge connected dense units into clusters
/// </para>
/// <para><b>For Beginners:</b> CLIQUE solves "the curse of dimensionality":
///
/// Problem: In high-dimensional data, distances become meaningless,
/// and traditional clustering fails.
///
/// Solution: Find clusters in subspaces (subsets of dimensions) where
/// the data is actually clustered.
///
/// Example: In a 100-feature dataset, a cluster might only be visible
/// when looking at features 3, 17, and 42 together.
///
/// CLIQUE automatically discovers these hidden subspaces!
/// </para>
/// </remarks>
public class CLIQUE<T> : ClusteringBase<T>
{
    private readonly CLIQUEOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private List<SubspaceClusterInternal>? _subspaceClustersInternal;
    private double[] _minBounds = Array.Empty<double>();
    private double[] _maxBounds = Array.Empty<double>();
    private double[] _intervalWidths = Array.Empty<double>();

    /// <summary>
    /// Internal cluster representation with dense units for prediction.
    /// </summary>
    private class SubspaceClusterInternal
    {
        public int ClusterId { get; set; }
        public int[] Dimensions { get; set; } = Array.Empty<int>();
        public List<DenseUnit> DenseUnits { get; set; } = new List<DenseUnit>();
        public HashSet<int> Points { get; set; } = new HashSet<int>();
    }

    /// <summary>
    /// Initializes a new CLIQUE instance.
    /// </summary>
    /// <param name="options">The CLIQUE configuration options.</param>
    public CLIQUE(CLIQUEOptions<T>? options = null)
        : base(options ?? new CLIQUEOptions<T>())
    {
        _options = options ?? new CLIQUEOptions<T>();
    }

    /// <summary>
    /// Gets the discovered subspace clusters.
    /// </summary>
    public IReadOnlyList<SubspaceCluster>? SubspaceClusters => _subspaceClustersInternal?
        .Select(c => new SubspaceCluster
        {
            ClusterId = c.ClusterId,
            Dimensions = c.Dimensions,
            Points = c.Points,
            NumUnits = c.DenseUnits.Count
        }).ToList().AsReadOnly();

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new CLIQUE<T>(new CLIQUEOptions<T>
        {
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            RandomState = _options.RandomState,
            NumIntervals = _options.NumIntervals,
            DensityThreshold = _options.DensityThreshold,
            MinPoints = _options.MinPoints,
            MaxSubspaceDimensions = _options.MaxSubspaceDimensions,
            UseAprioriPruning = _options.UseAprioriPruning
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (CLIQUE<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputData(x);

        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Compute data bounds
        ComputeBounds(x);

        // Compute density threshold as absolute count
        int densityThreshold = _options.MinPoints > 0
            ? _options.MinPoints
            : Math.Max(1, (int)(n * _options.DensityThreshold));

        // Find dense 1-D units
        var denseUnits = FindDenseUnits1D(x, densityThreshold);

        // Find dense k-D units using Apriori principle
        int maxDim = _options.MaxSubspaceDimensions > 0
            ? Math.Min(_options.MaxSubspaceDimensions, d)
            : d;

        var allDenseUnits = new List<DenseUnit>(denseUnits);
        var currentLevelUnits = denseUnits;

        for (int k = 2; k <= maxDim && currentLevelUnits.Count > 0; k++)
        {
            var nextLevelUnits = FindDenseUnitsKD(x, currentLevelUnits, k, densityThreshold);
            if (nextLevelUnits.Count == 0) break;

            allDenseUnits.AddRange(nextLevelUnits);
            currentLevelUnits = nextLevelUnits;
        }

        // Group dense units into clusters
        _subspaceClustersInternal = MergeIntoSubspaceClusters(allDenseUnits);

        // Assign labels based on highest-dimensional subspace cluster membership
        Labels = AssignLabels(x, _subspaceClustersInternal);

        // Count unique clusters
        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)NumOps.ToDouble(Labels[i]);
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }
        NumClusters = uniqueLabels.Count;

        // Compute cluster centers
        ComputeClusterCenters(x);

        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (_subspaceClustersInternal is null || _subspaceClustersInternal.Count == 0)
        {
            var noise = new Vector<T>(x.Rows);
            for (int i = 0; i < x.Rows; i++)
            {
                noise[i] = NumOps.FromDouble(-1);
            }
            return noise;
        }

        return AssignLabelsForNewData(x);
    }

    /// <summary>
    /// Assigns labels to new data by checking if points fall within stored dense units.
    /// </summary>
    private Vector<T> AssignLabelsForNewData(Matrix<T> x)
    {
        int n = x.Rows;
        var labels = new Vector<T>(n);

        // Initialize as noise
        for (int i = 0; i < n; i++)
        {
            labels[i] = NumOps.FromDouble(-1);
        }

        // Sort clusters by dimensionality (prefer higher dimensional matches)
        var sortedClusters = _subspaceClustersInternal!
            .OrderByDescending(c => c.Dimensions.Length)
            .ToList();

        // For each new point, check if it falls within any cluster's dense units
        for (int i = 0; i < n; i++)
        {
            foreach (var cluster in sortedClusters)
            {
                // Check if the point falls within any dense unit of this cluster
                bool inCluster = false;

                foreach (var denseUnit in cluster.DenseUnits)
                {
                    bool inUnit = true;

                    for (int j = 0; j < denseUnit.Dimensions.Length; j++)
                    {
                        int dim = denseUnit.Dimensions[j];
                        double val = NumOps.ToDouble(x[i, dim]);
                        int cell = GetCellIndex(val, dim);

                        if (cell != denseUnit.Cells[j])
                        {
                            inUnit = false;
                            break;
                        }
                    }

                    if (inUnit)
                    {
                        inCluster = true;
                        break;
                    }
                }

                if (inCluster)
                {
                    labels[i] = NumOps.FromDouble(cluster.ClusterId);
                    break; // Assigned to highest-dimensional cluster
                }
            }
        }

        return labels;
    }

    private void ComputeBounds(Matrix<T> x)
    {
        int n = x.Rows;
        int d = x.Columns;

        _minBounds = new double[d];
        _maxBounds = new double[d];
        _intervalWidths = new double[d];

        for (int j = 0; j < d; j++)
        {
            _minBounds[j] = double.MaxValue;
            _maxBounds[j] = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(x[i, j]);
                _minBounds[j] = Math.Min(_minBounds[j], val);
                _maxBounds[j] = Math.Max(_maxBounds[j], val);
            }

            double range = _maxBounds[j] - _minBounds[j];
            _intervalWidths[j] = range > 0 ? range / _options.NumIntervals : 1;
        }
    }

    private int GetCellIndex(double value, int dimension)
    {
        if (_intervalWidths[dimension] == 0)
        {
            return 0;
        }

        int idx = (int)((value - _minBounds[dimension]) / _intervalWidths[dimension]);
        return Math.Max(0, Math.Min(_options.NumIntervals - 1, idx));
    }

    private List<DenseUnit> FindDenseUnits1D(Matrix<T> x, int densityThreshold)
    {
        int n = x.Rows;
        int d = x.Columns;
        var denseUnits = new List<DenseUnit>();

        for (int dim = 0; dim < d; dim++)
        {
            // Count points in each cell
            var cellCounts = new int[_options.NumIntervals];
            var cellPoints = new List<int>[_options.NumIntervals];

            for (int i = 0; i < _options.NumIntervals; i++)
            {
                cellPoints[i] = new List<int>();
            }

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(x[i, dim]);
                int cell = GetCellIndex(val, dim);
                cellCounts[cell]++;
                cellPoints[cell].Add(i);
            }

            // Create dense units
            for (int cell = 0; cell < _options.NumIntervals; cell++)
            {
                if (cellCounts[cell] >= densityThreshold)
                {
                    denseUnits.Add(new DenseUnit
                    {
                        Dimensions = new[] { dim },
                        Cells = new[] { cell },
                        Points = cellPoints[cell].ToList(),
                        Count = cellCounts[cell]
                    });
                }
            }
        }

        return denseUnits;
    }

    private List<DenseUnit> FindDenseUnitsKD(Matrix<T> x, List<DenseUnit> previousLevel, int k, int densityThreshold)
    {
        int n = x.Rows;
        var candidates = GenerateCandidates(previousLevel, k);
        var denseUnits = new List<DenseUnit>();

        foreach (var candidate in candidates)
        {
            // Count points in this k-D cell
            var points = new List<int>();

            for (int i = 0; i < n; i++)
            {
                bool inCell = true;

                for (int j = 0; j < candidate.Dimensions.Length; j++)
                {
                    int dim = candidate.Dimensions[j];
                    double val = NumOps.ToDouble(x[i, dim]);
                    int cell = GetCellIndex(val, dim);

                    if (cell != candidate.Cells[j])
                    {
                        inCell = false;
                        break;
                    }
                }

                if (inCell)
                {
                    points.Add(i);
                }
            }

            if (points.Count >= densityThreshold)
            {
                candidate.Points = points;
                candidate.Count = points.Count;
                denseUnits.Add(candidate);
            }
        }

        return denseUnits;
    }

    private List<DenseUnit> GenerateCandidates(List<DenseUnit> previousLevel, int k)
    {
        var candidates = new List<DenseUnit>();
        var seen = new HashSet<string>();

        for (int i = 0; i < previousLevel.Count; i++)
        {
            for (int j = i + 1; j < previousLevel.Count; j++)
            {
                var unit1 = previousLevel[i];
                var unit2 = previousLevel[j];

                // Check if they share k-2 dimensions
                var mergedDims = unit1.Dimensions.Union(unit2.Dimensions).OrderBy(x => x).ToArray();

                if (mergedDims.Length == k)
                {
                    // Generate the cell indices for merged dimensions
                    var cells = new int[k];
                    for (int m = 0; m < k; m++)
                    {
                        int dim = mergedDims[m];
                        int idx1 = Array.IndexOf(unit1.Dimensions, dim);
                        int idx2 = Array.IndexOf(unit2.Dimensions, dim);

                        if (idx1 >= 0)
                        {
                            cells[m] = unit1.Cells[idx1];
                        }
                        else if (idx2 >= 0)
                        {
                            cells[m] = unit2.Cells[idx2];
                        }
                    }

                    // Avoid duplicates
                    string key = string.Join(",", mergedDims) + "|" + string.Join(",", cells);
                    if (!seen.Contains(key))
                    {
                        seen.Add(key);
                        candidates.Add(new DenseUnit
                        {
                            Dimensions = mergedDims,
                            Cells = cells
                        });
                    }
                }
            }
        }

        return candidates;
    }

    private List<SubspaceClusterInternal> MergeIntoSubspaceClusters(List<DenseUnit> denseUnits)
    {
        // Group by subspace (dimensions)
        var subspaceGroups = denseUnits
            .GroupBy(u => string.Join(",", u.Dimensions))
            .ToDictionary(g => g.Key, g => g.ToList());

        var clusters = new List<SubspaceClusterInternal>();

        foreach (var group in subspaceGroups)
        {
            var dims = group.Value[0].Dimensions;
            var unitsInSubspace = group.Value;

            // Connect adjacent dense units
            var visited = new bool[unitsInSubspace.Count];

            for (int i = 0; i < unitsInSubspace.Count; i++)
            {
                if (visited[i]) continue;

                // BFS to find connected components
                var cluster = new SubspaceClusterInternal
                {
                    ClusterId = clusters.Count,
                    Dimensions = dims.ToArray(),
                    DenseUnits = new List<DenseUnit>(),
                    Points = new HashSet<int>()
                };

                var queue = new Queue<int>();
                queue.Enqueue(i);
                visited[i] = true;

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    var unit = unitsInSubspace[current];
                    cluster.DenseUnits.Add(unit);

                    foreach (int pt in unit.Points)
                    {
                        cluster.Points.Add(pt);
                    }

                    // Find adjacent units
                    for (int j = 0; j < unitsInSubspace.Count; j++)
                    {
                        if (visited[j]) continue;

                        if (AreAdjacent(unit, unitsInSubspace[j]))
                        {
                            visited[j] = true;
                            queue.Enqueue(j);
                        }
                    }
                }

                clusters.Add(cluster);
            }
        }

        return clusters;
    }

    private bool AreAdjacent(DenseUnit unit1, DenseUnit unit2)
    {
        // Two units are adjacent if they differ by at most 1 in exactly one dimension
        if (!unit1.Dimensions.SequenceEqual(unit2.Dimensions))
        {
            return false;
        }

        int diffCount = 0;
        for (int i = 0; i < unit1.Cells.Length; i++)
        {
            int diff = Math.Abs(unit1.Cells[i] - unit2.Cells[i]);
            if (diff > 1) return false;
            if (diff == 1) diffCount++;
        }

        return diffCount == 1;
    }

    private Vector<T> AssignLabels(Matrix<T> x, List<SubspaceClusterInternal> clusters)
    {
        int n = x.Rows;
        var labels = new Vector<T>(n);

        // Initialize as noise
        for (int i = 0; i < n; i++)
        {
            labels[i] = NumOps.FromDouble(-1);
        }

        if (clusters.Count == 0)
        {
            return labels;
        }

        // Sort clusters by dimensionality (prefer higher dimensional)
        var sortedClusters = clusters.OrderByDescending(c => c.Dimensions.Length).ToList();

        // Assign points to highest-dimensional cluster they belong to
        for (int i = 0; i < n; i++)
        {
            foreach (var cluster in sortedClusters)
            {
                if (cluster.Points.Contains(i))
                {
                    labels[i] = NumOps.FromDouble(cluster.ClusterId);
                    break;
                }
            }
        }

        return labels;
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (NumClusters == 0)
        {
            ClusterCenters = null;
            return;
        }

        int d = x.Columns;
        ClusterCenters = new Matrix<T>(NumClusters, d);
        var counts = new int[NumClusters];

        int n = x.Rows;
        for (int i = 0; i < n; i++)
        {
            int label = (int)NumOps.ToDouble(Labels![i]);
            if (label >= 0 && label < NumClusters)
            {
                counts[label]++;
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[label, j] = NumOps.Add(ClusterCenters[label, j], x[i, j]);
                }
            }
        }

        for (int k = 0; k < NumClusters; k++)
        {
            if (counts[k] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[k, j] = NumOps.Divide(ClusterCenters[k, j], NumOps.FromDouble(counts[k]));
                }
            }
        }
    }

    private void ValidateInputData(Matrix<T> x)
    {
        if (x.Rows == 0 || x.Columns == 0)
        {
            throw new ArgumentException("Input data cannot be empty.");
        }
    }

    private void ValidatePredictInput(Matrix<T> x)
    {
        if (x.Columns != NumFeatures)
        {
            throw new ArgumentException($"Expected {NumFeatures} features, got {x.Columns}.");
        }
    }

    private class DenseUnit
    {
        public int[] Dimensions { get; set; } = Array.Empty<int>();
        public int[] Cells { get; set; } = Array.Empty<int>();
        public List<int> Points { get; set; } = new List<int>();
        public int Count { get; set; }
    }
}

/// <summary>
/// Represents a cluster discovered in a subspace.
/// </summary>
public class SubspaceCluster
{
    /// <summary>
    /// Unique identifier for this cluster.
    /// </summary>
    public int ClusterId { get; set; }

    /// <summary>
    /// The dimensions (feature indices) that define this subspace.
    /// </summary>
    public int[] Dimensions { get; set; } = Array.Empty<int>();

    /// <summary>
    /// The number of dense units in this cluster.
    /// </summary>
    public int NumUnits { get; set; }

    /// <summary>
    /// The indices of points belonging to this cluster.
    /// </summary>
    public HashSet<int> Points { get; set; } = new HashSet<int>();

    /// <summary>
    /// Number of points in this cluster.
    /// </summary>
    public int Size => Points.Count;

    /// <summary>
    /// Dimensionality of the subspace.
    /// </summary>
    public int Dimensionality => Dimensions.Length;
}
