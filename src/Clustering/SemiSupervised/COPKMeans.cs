using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.SemiSupervised;

/// <summary>
/// COP-KMeans (Constrained K-Means) implementation with must-link and cannot-link constraints.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// COP-KMeans extends K-Means by enforcing pairwise constraints:
/// - Must-link: Specified pairs must belong to the same cluster
/// - Cannot-link: Specified pairs must belong to different clusters
/// </para>
/// <para>
/// Algorithm:
/// 1. Initialize cluster centers
/// 2. Assign points to nearest cluster that doesn't violate constraints
/// 3. If no valid assignment exists, return failure
/// 4. Update centers and repeat until convergence
/// </para>
/// <para><b>For Beginners:</b> This is K-Means with rules about what can/cannot be together.
///
/// Think of seating people at wedding tables:
/// - Some couples must sit together (must-link)
/// - Some feuding relatives must be separated (cannot-link)
///
/// COP-KMeans finds the best clustering that respects ALL constraints.
/// If constraints are contradictory, the algorithm will fail.
///
/// Use cases:
/// - User feedback: "These items are the same"
/// - Domain knowledge: "These categories are mutually exclusive"
/// - Data linkage: "These records refer to the same entity"
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new COPKMeans&lt;double&gt;(new COPKMeansOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Constrained K-means Clustering with Background Knowledge", "https://doi.org/10.1145/944919.944935", Year = 2001, Authors = "Kiri Wagstaff, Claire Cardie, Seth Rogers, Stefan Schroedl")]
public partial class COPKMeans<T> : ClusteringBase<T>
{
    private readonly COPKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private HashSet<(int, int)>? _mustLinkClosure;
    private HashSet<(int, int)>? _cannotLinkClosure;

    /// <summary>
    /// Initializes a new COP-KMeans instance.
    /// </summary>
    /// <param name="options">The COP-KMeans options.</param>
    public COPKMeans(COPKMeansOptions<T>? options = null)
        : base(options ??= new COPKMeansOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets whether all constraints were satisfied.
    /// </summary>
    public bool ConstraintsSatisfied { get; private set; }

    /// <summary>
    /// Gets the number of constraint violations (if any).
    /// </summary>
    public int ConstraintViolations { get; private set; }

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (COPKMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        int k = _options.NumClusters;
        NumFeatures = d;
        NumClusters = k;

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Compute transitive closures of constraints
        ComputeConstraintClosures(n);

        // Initialize centers using K-Means++
        var centers = InitializeCenters(x, k, n, d, rand, metric);

        var labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = -1;

        // Main loop
        bool changed = true;
        int iterations = 0;

        while (changed && iterations < Options.MaxIterations)
        {
            changed = false;
            iterations++;

            // Centers are fixed during the assignment pass
            var distances = metric.ComputePairwise(x, centers);

            // Assign points to clusters respecting constraints (in order: each assignment constrains later points)
            var allowed = new List<int>(k);
            for (int i = 0; i < n; i++)
            {
                allowed.Clear();
                for (int c = 0; c < k; c++)
                {
                    if (!ViolatesConstraints(i, c, labels))
                    {
                        allowed.Add(c);
                    }
                }

                int bestCluster = NearestAllowed(distances.GetRow(i), allowed);
                if (bestCluster != -1 && bestCluster != labels[i])
                {
                    labels[i] = bestCluster;
                    changed = true;
                }
            }

            // Update centers (empty clusters keep their previous center)
            var means = ClusteringTensorOps.ClusterMeans(x, ClusteringTensorOps.FromIntArray<T>(labels), k, out var counts);
            centers = ClusteringTensorOps.RowsWherePositive(counts, means, centers);
        }

        // Count violations
        ConstraintViolations = CountViolations(labels);
        ConstraintsSatisfied = ConstraintViolations == 0;

        // Set results
        ClusterCenters = centers;
        Labels = ClusteringTensorOps.FromIntArray<T>(labels);

        MergeDegenerateClusters(x);

        // Recompute constraint status after merge since labels may have changed
        if (Labels is not null)
        {
            ConstraintViolations = CountViolations(ClusteringTensorOps.ToIntArray(Labels));
            ConstraintsSatisfied = ConstraintViolations == 0;
        }

        IsTrained = true;
    }

    /// <summary>
    /// The allowed cluster with the smallest distance (first on ties), or -1 when none is allowed or reachable.
    /// </summary>
    private int NearestAllowed(Vector<T> distances, List<int> allowed)
    {
        if (allowed.Count == 0)
        {
            return -1;
        }

        var candidates = ClusteringTensorOps.SelectElements(distances, allowed.ToArray());
        int best = ClusteringTensorOps.ArgMin(candidates);
        return NumOps.LessThan(candidates[best], NumOps.MaxValue) ? allowed[best] : -1;
    }

    private void ComputeConstraintClosures(int n)
    {
        _mustLinkClosure = new HashSet<(int, int)>();
        _cannotLinkClosure = new HashSet<(int, int)>();

        // Add original must-link constraints
        if (_options.MustLink is not null)
        {
            foreach (var (i, j) in _options.MustLink)
            {
                _mustLinkClosure.Add((Math.Min(i, j), Math.Max(i, j)));
            }
        }

        // Compute transitive closure for must-link
        if (_options.UseTransitiveClosure && _mustLinkClosure.Count > 0)
        {
            var unionFind = new int[n];
            for (int i = 0; i < n; i++) unionFind[i] = i;

            int Find(int x)
            {
                if (unionFind[x] != x)
                    unionFind[x] = Find(unionFind[x]);
                return unionFind[x];
            }

            void Union(int x, int y)
            {
                int px = Find(x), py = Find(y);
                if (px != py) unionFind[px] = py;
            }

            foreach (var (i, j) in _mustLinkClosure)
            {
                Union(i, j);
            }

            // Regenerate must-link from connected components
            _mustLinkClosure.Clear();
            var components = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int root = Find(i);
                if (!components.ContainsKey(root))
                {
                    components[root] = new List<int>();
                }
                components[root].Add(i);
            }

            foreach (var component in components.Values)
            {
                for (int i = 0; i < component.Count; i++)
                {
                    for (int j = i + 1; j < component.Count; j++)
                    {
                        _mustLinkClosure.Add((component[i], component[j]));
                    }
                }
            }
        }

        // Add cannot-link constraints (with transitive extension from must-link)
        if (_options.CannotLink is not null)
        {
            foreach (var (i, j) in _options.CannotLink)
            {
                _cannotLinkClosure.Add((Math.Min(i, j), Math.Max(i, j)));

                // If i must-link with k, then k cannot-link with j
                if (_options.UseTransitiveClosure)
                {
                    foreach (var (a, b) in _mustLinkClosure)
                    {
                        if (a == i || b == i)
                        {
                            int other = a == i ? b : a;
                            _cannotLinkClosure.Add((Math.Min(other, j), Math.Max(other, j)));
                        }
                        if (a == j || b == j)
                        {
                            int other = a == j ? b : a;
                            _cannotLinkClosure.Add((Math.Min(other, i), Math.Max(other, i)));
                        }
                    }
                }
            }
        }
    }

    private bool ViolatesConstraints(int pointIdx, int cluster, int[] labels)
    {
        // Check must-link constraints
        foreach (var (i, j) in (_mustLinkClosure ?? throw new InvalidOperationException("_mustLinkClosure is null where it is required.")))
        {
            if (i == pointIdx || j == pointIdx)
            {
                int other = i == pointIdx ? j : i;
                if (labels[other] >= 0 && labels[other] != cluster)
                {
                    return true; // Must-linked point is in different cluster
                }
            }
        }

        // Check cannot-link constraints
        foreach (var (i, j) in (_cannotLinkClosure ?? throw new InvalidOperationException("_cannotLinkClosure is null where it is required.")))
        {
            if (i == pointIdx || j == pointIdx)
            {
                int other = i == pointIdx ? j : i;
                if (labels[other] == cluster)
                {
                    return true; // Cannot-linked point is in same cluster
                }
            }
        }

        return false;
    }

    private int CountViolations(int[] labels)
    {
        int violations = 0;

        foreach (var (i, j) in (_mustLinkClosure ?? throw new InvalidOperationException("_mustLinkClosure is null where it is required.")))
        {
            if (labels[i] != labels[j])
            {
                violations++;
            }
        }

        foreach (var (i, j) in (_cannotLinkClosure ?? throw new InvalidOperationException("_cannotLinkClosure is null where it is required.")))
        {
            if (labels[i] == labels[j])
            {
                violations++;
            }
        }

        return violations;
    }

    private Matrix<T> InitializeCenters(Matrix<T> x, int k, int n, int d, Random rand, IDistanceMetric<T> metric)
    {
        if (k <= 0)
        {
            return new Matrix<T>(0, d);
        }

        // K-Means++ initialization
        var chosen = new int[k];
        chosen[0] = rand.Next(n);

        // Squared distance of every point to its nearest chosen center
        Vector<T>? distances = null;
        for (int c = 1; c < k; c++)
        {
            var toLast = ClusteringTensorOps.DistancesToPoint(metric, x, x.GetRow(chosen[c - 1]));
            var squared = Engine.Multiply(toLast, toLast);
            distances = distances is null ? squared : Engine.Min(distances, squared);
            chosen[c] = ClusteringTensorOps.SampleProportional(distances, rand);
        }

        return ClusteringTensorOps.SelectRows(x, chosen);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            return new Vector<T>(x.Rows);
        }

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centers = ClusterCenters.Rows == NumClusters
            ? ClusterCenters
            : ClusterCenters.GetSubMatrix(0, 0, NumClusters, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(metric, x, centers));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }
}
