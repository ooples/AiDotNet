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
public class COPKMeans<T> : ClusteringBase<T>
{
    private readonly COPKMeansOptions<T> _options;
    private HashSet<(int, int)>? _mustLinkClosure;
    private HashSet<(int, int)>? _cannotLinkClosure;

    /// <summary>
    /// Initializes a new COP-KMeans instance.
    /// </summary>
    /// <param name="options">The COP-KMeans options.</param>
    public COPKMeans(COPKMeansOptions<T>? options = null)
        : base(options ?? new COPKMeansOptions<T>())
    {
        _options = options ?? new COPKMeansOptions<T>();
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
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new COPKMeans<T>(new COPKMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            MustLink = _options.MustLink,
            CannotLink = _options.CannotLink,
            UseTransitiveClosure = _options.UseTransitiveClosure,
            MaxIterations = _options.MaxIterations,
            DistanceMetric = _options.DistanceMetric
        });
    }

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

        var rand = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
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

            // Assign points to clusters respecting constraints
            for (int i = 0; i < n; i++)
            {
                var point = GetRow(x, i);
                int bestCluster = -1;
                double bestDist = double.MaxValue;

                // Try each cluster
                for (int c = 0; c < k; c++)
                {
                    if (ViolatesConstraints(i, c, labels))
                    {
                        continue;
                    }

                    var center = new Vector<T>(d);
                    for (int j = 0; j < d; j++)
                    {
                        center[j] = NumOps.FromDouble(centers[c][j]);
                    }

                    double dist = NumOps.ToDouble(metric.Compute(point, center));
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestCluster = c;
                    }
                }

                if (bestCluster != -1 && bestCluster != labels[i])
                {
                    labels[i] = bestCluster;
                    changed = true;
                }
            }

            // Update centers
            var counts = new int[k];
            var newCenters = new double[k][];
            for (int c = 0; c < k; c++)
            {
                newCenters[c] = new double[d];
            }

            for (int i = 0; i < n; i++)
            {
                if (labels[i] >= 0)
                {
                    counts[labels[i]]++;
                    for (int j = 0; j < d; j++)
                    {
                        newCenters[labels[i]][j] += NumOps.ToDouble(x[i, j]);
                    }
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (counts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        centers[c][j] = newCenters[c][j] / counts[c];
                    }
                }
            }
        }

        // Count violations
        ConstraintViolations = CountViolations(labels);
        ConstraintsSatisfied = ConstraintViolations == 0;

        // Set results
        ClusterCenters = new Matrix<T>(k, d);
        Labels = new Vector<T>(n);

        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = NumOps.FromDouble(centers[c][j]);
            }
        }

        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(labels[i]);
        }

        IsTrained = true;
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
        foreach (var (i, j) in _mustLinkClosure!)
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
        foreach (var (i, j) in _cannotLinkClosure!)
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

        foreach (var (i, j) in _mustLinkClosure!)
        {
            if (labels[i] != labels[j])
            {
                violations++;
            }
        }

        foreach (var (i, j) in _cannotLinkClosure!)
        {
            if (labels[i] == labels[j])
            {
                violations++;
            }
        }

        return violations;
    }

    private double[][] InitializeCenters(Matrix<T> x, int k, int n, int d, Random rand, IDistanceMetric<T> metric)
    {
        var centers = new double[k][];

        // K-Means++ initialization
        int firstIdx = rand.Next(n);
        centers[0] = new double[d];
        for (int j = 0; j < d; j++)
        {
            centers[0][j] = NumOps.ToDouble(x[firstIdx, j]);
        }

        var distances = new double[n];
        for (int i = 0; i < n; i++)
        {
            distances[i] = double.MaxValue;
        }

        for (int c = 1; c < k; c++)
        {
            double totalWeight = 0;

            for (int i = 0; i < n; i++)
            {
                var point = GetRow(x, i);
                var center = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    center[j] = NumOps.FromDouble(centers[c - 1][j]);
                }

                double dist = NumOps.ToDouble(metric.Compute(point, center));
                distances[i] = Math.Min(distances[i], dist * dist);
                totalWeight += distances[i];
            }

            double threshold = rand.NextDouble() * totalWeight;
            double cumWeight = 0;
            int selected = 0;

            for (int i = 0; i < n; i++)
            {
                cumWeight += distances[i];
                if (cumWeight >= threshold)
                {
                    selected = i;
                    break;
                }
            }

            centers[c] = new double[d];
            for (int j = 0; j < d; j++)
            {
                centers[c][j] = NumOps.ToDouble(x[selected, j]);
            }
        }

        return centers;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        var labels = new Vector<T>(x.Rows);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            double minDist = double.MaxValue;
            int nearestCluster = 0;

            if (ClusterCenters is not null)
            {
                for (int c = 0; c < NumClusters; c++)
                {
                    var center = GetRow(ClusterCenters, c);
                    double dist = NumOps.ToDouble(metric.Compute(point, center));

                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCluster = c;
                    }
                }
            }

            labels[i] = NumOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }
}
