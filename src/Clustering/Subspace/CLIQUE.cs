using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new CLIQUE&lt;double&gt;(new CLIQUEOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Automatic Subspace Clustering of High Dimensional Data for Data Mining Applications", "https://doi.org/10.1145/276304.276314", Year = 1998, Authors = "Rakesh Agrawal, Johannes Gehrke, Dimitrios Gunopulos, Prabhakar Raghavan")]
public partial class CLIQUE<T> : ClusteringBase<T>
{
    private readonly CLIQUEOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private List<SubspaceClusterInternal>? _subspaceClustersInternal;
    private Vector<T> _minBounds = new Vector<T>(0);
    private Vector<T> _maxBounds = new Vector<T>(0);
    private Vector<T> _intervalWidths = new Vector<T>(0);

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
        : base(options ??= new CLIQUEOptions<T>())
    {
        _options = options;
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

        // Compute data bounds and the grid cell of every value
        ComputeBounds(x);
        var cells = CellIndices(x);

        // Compute density threshold as absolute count
        int densityThreshold = _options.MinPoints > 0
            ? _options.MinPoints
            : Math.Max(1, (int)(n * _options.DensityThreshold));

        // Find dense 1-D units
        var denseUnits = FindDenseUnits1D(cells, n, d, densityThreshold);

        // Find dense k-D units using Apriori principle
        int maxDim = _options.MaxSubspaceDimensions > 0
            ? Math.Min(_options.MaxSubspaceDimensions, d)
            : d;

        var allDenseUnits = new List<DenseUnit>(denseUnits);
        var currentLevelUnits = denseUnits;

        for (int k = 2; k <= maxDim && currentLevelUnits.Count > 0; k++)
        {
            var nextLevelUnits = FindDenseUnitsKD(cells, n, d, currentLevelUnits, k, densityThreshold);
            if (nextLevelUnits.Count == 0) break;

            allDenseUnits.AddRange(nextLevelUnits);
            currentLevelUnits = nextLevelUnits;
        }

        // Group dense units into clusters
        _subspaceClustersInternal = MergeIntoSubspaceClusters(allDenseUnits);

        // Assign labels based on highest-dimensional subspace cluster membership
        var labelIds = AssignLabels(n, _subspaceClustersInternal);
        Labels = ClusteringTensorOps.FromIntArray<T>(labelIds);

        // Count unique clusters
        NumClusters = labelIds.Where(label => label >= 0).Distinct().Count();

        // Compute cluster centers
        ComputeClusterCenters(x);

        int premergeK = NumClusters;
        MergeDegenerateClusters(x);

        // Update subspace cluster IDs after merge to match the post-merge label space
        if (NumClusters < premergeK && _subspaceClustersInternal is not null && Labels is not null)
        {
            // Build old→new label map from the merged Labels
            var mergedIds = ClusteringTensorOps.ToIntArray(Labels);
            var oldToNew = new Dictionary<int, int>();
            for (int i = 0; i < mergedIds.Length; i++)
            {
                int newLabel = mergedIds[i];
                // Find which subspace cluster originally owned this point
                foreach (var sc in _subspaceClustersInternal)
                {
                    if (sc.Points.Contains(i) && !oldToNew.ContainsKey(sc.ClusterId))
                    {
                        oldToNew[sc.ClusterId] = newLabel;
                        break;
                    }
                }
            }

            foreach (var sc in _subspaceClustersInternal)
            {
                if (oldToNew.TryGetValue(sc.ClusterId, out int newId))
                    sc.ClusterId = newId;
            }
        }

        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        // Return stored labels for in-sample prediction (preserves merge results)
        if (Labels is not null && ReferenceEquals(x, TrainingDataRef))
            return new Vector<T>(Labels);

        if (_subspaceClustersInternal is null || _subspaceClustersInternal.Count == 0)
        {
            return Engine.Fill(x.Rows, NumOps.FromDouble(-1));
        }

        return AssignLabelsForNewData(x);
    }

    /// <summary>
    /// Assigns labels to new data by checking if points fall within stored dense units.
    /// </summary>
    private Vector<T> AssignLabelsForNewData(Matrix<T> x)
    {
        int n = x.Rows;
        int d = x.Columns;
        var cells = CellIndices(x);

        // Initialize as noise
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1;
        }

        // Sort clusters by dimensionality (prefer higher dimensional matches)
        var subspaceClusters = _subspaceClustersInternal ?? throw new InvalidOperationException("_subspaceClustersInternal has not been initialized.");
        var sortedClusters = subspaceClusters
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
                    if (InUnit(cells, i, d, denseUnit.Dimensions, denseUnit.Cells))
                    {
                        inCluster = true;
                        break;
                    }
                }

                if (inCluster)
                {
                    labels[i] = cluster.ClusterId;
                    break; // Assigned to highest-dimensional cluster
                }
            }
        }

        return ClusteringTensorOps.FromIntArray<T>(labels);
    }

    /// <summary>True when point <paramref name="row"/> lies in the given cell of every listed dimension.</summary>
    private static bool InUnit(int[] cells, int row, int d, int[] dimensions, int[] unitCells)
    {
        for (int j = 0; j < dimensions.Length; j++)
        {
            if (cells[row * d + dimensions[j]] != unitCells[j])
            {
                return false;
            }
        }

        return true;
    }

    private void ComputeBounds(Matrix<T> x)
    {
        var data = Tensor<T>.FromMatrix(x);
        var minBounds = Engine.TensorAmin(data, new[] { 0 }, false);
        var maxBounds = Engine.ReduceMax(data, new[] { 0 }, false);
        var range = Engine.TensorSubtract(maxBounds, minBounds);

        // Width of one interval per dimension (1 for constant dimensions)
        var widths = Engine.TensorWhere(
            Engine.TensorGreaterThan(range, NumOps.Zero),
            Engine.TensorDivideScalar(range, NumOps.FromDouble(_options.NumIntervals)),
            Engine.TensorOnesLike(range));

        _minBounds = minBounds.ToVector();
        _maxBounds = maxBounds.ToVector();
        _intervalWidths = widths.ToVector();
    }

    /// <summary>
    /// Grid cell of every value, row-major [n * d]: floor((value - min) / width) clamped to [0, NumIntervals - 1].
    /// </summary>
    private int[] CellIndices(Matrix<T> x)
    {
        int n = x.Rows;
        int d = x.Columns;
        var shape = new[] { n, d };
        var offsets = Engine.TensorSubtract(
            Tensor<T>.FromMatrix(x), Engine.TensorBroadcastTo(Tensor<T>.FromVector(_minBounds, new[] { 1, d }), shape));
        var scaled = Engine.TensorDivide(
            offsets, Engine.TensorBroadcastTo(Tensor<T>.FromVector(_intervalWidths, new[] { 1, d }), shape));
        var cells = Engine.TensorClamp(Engine.TensorFloor(scaled), NumOps.Zero, NumOps.FromDouble(_options.NumIntervals - 1));
        return ClusteringTensorOps.ToIntArray(Engine.Reshape(cells, new[] { n * d }).ToVector());
    }

    private List<DenseUnit> FindDenseUnits1D(int[] cells, int n, int d, int densityThreshold)
    {
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
                int cell = cells[i * d + dim];
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

    private List<DenseUnit> FindDenseUnitsKD(int[] cells, int n, int d, List<DenseUnit> previousLevel, int k, int densityThreshold)
    {
        var candidates = GenerateCandidates(previousLevel, k);
        var denseUnits = new List<DenseUnit>();

        foreach (var candidate in candidates)
        {
            // Count points in this k-D cell
            var points = new List<int>();

            for (int i = 0; i < n; i++)
            {
                if (InUnit(cells, i, d, candidate.Dimensions, candidate.Cells))
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

    private int[] AssignLabels(int n, List<SubspaceClusterInternal> clusters)
    {
        // Initialize as noise
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1;
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
                    labels[i] = cluster.ClusterId;
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

        // Mean of each cluster's members (labels outside [0, NumClusters) are ignored, empty clusters stay zero)
        ClusterCenters = ClusteringTensorOps.ClusterMeans(
            x, Labels ?? throw new InvalidOperationException("Labels has not been initialized."), NumClusters, out _);
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
