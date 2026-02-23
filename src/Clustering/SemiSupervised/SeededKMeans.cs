using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.SemiSupervised;

/// <summary>
/// Seeded K-Means implementation with labeled initialization points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Seeded K-Means uses pre-labeled data points to initialize cluster centers.
/// Instead of random initialization, it computes initial centers from known
/// cluster assignments, then proceeds with standard K-Means iterations.
/// </para>
/// <para>
/// Algorithm:
/// 1. Group seed points by their labels
/// 2. Compute initial centers as mean of each seed group
/// 3. Run standard K-Means from these initial centers
/// 4. Optionally constrain seed points to stay in original clusters
/// </para>
/// <para><b>For Beginners:</b> This is K-Means with a "head start."
///
/// Instead of guessing where clusters might be:
/// - You provide example points with known labels
/// - The algorithm learns where each cluster should be
/// - Then it clusters the rest of the data
///
/// Example: Classifying news articles
/// - Manually label 20 articles as "sports", "politics", "tech"
/// - Seeded K-Means learns what each category looks like
/// - Then automatically categorizes 10,000 more articles
///
/// Benefits:
/// - Better initialization = better results
/// - Incorporates domain knowledge
/// - Helps when random init gives poor results
/// </para>
/// </remarks>
public class SeededKMeans<T> : ClusteringBase<T>
{
    private readonly SeededKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new Seeded K-Means instance.
    /// </summary>
    /// <param name="options">The Seeded K-Means options.</param>
    public SeededKMeans(SeededKMeansOptions<T>? options = null)
        : base(options ?? new SeededKMeansOptions<T>())
    {
        _options = options ?? new SeededKMeansOptions<T>();
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SeededKMeans<T>(new SeededKMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            Seeds = _options.Seeds,
            ConstrainSeeds = _options.ConstrainSeeds,
            MaxIterations = _options.MaxIterations,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (SeededKMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Determine number of clusters from seeds or option
        var seeds = _options.Seeds ?? new Dictionary<int, int>();
        int k = _options.NumClusters > 0
            ? _options.NumClusters
            : seeds.Count > 0 ? seeds.Values.Max() + 1 : 3;

        NumClusters = k;

        // Initialize centers from seeds
        var centers = InitializeCentersFromSeeds(x, seeds, k, n, d);

        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = seeds.ContainsKey(i) ? seeds[i] : -1;
        }

        // Main K-Means loop
        bool changed = true;
        int iterations = 0;

        while (changed && iterations < Options.MaxIterations)
        {
            changed = false;
            iterations++;

            // Assign points to nearest cluster
            for (int i = 0; i < n; i++)
            {
                // Skip constrained seeds
                if (_options.ConstrainSeeds && seeds.ContainsKey(i))
                {
                    continue;
                }

                var point = GetRow(x, i);
                int bestCluster = 0;
                double bestDist = double.MaxValue;

                for (int c = 0; c < k; c++)
                {
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

                if (bestCluster != labels[i])
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
                int c = labels[i];
                counts[c]++;
                for (int j = 0; j < d; j++)
                {
                    newCenters[c][j] += NumOps.ToDouble(x[i, j]);
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

    private double[][] InitializeCentersFromSeeds(Matrix<T> x, Dictionary<int, int> seeds, int k, int n, int d)
    {
        var centers = new double[k][];
        var counts = new int[k];

        // Initialize arrays
        for (int c = 0; c < k; c++)
        {
            centers[c] = new double[d];
        }

        // Compute centers from seeds
        foreach (var kvp in seeds)
        {
            int pointIdx = kvp.Key;
            int clusterLabel = kvp.Value;

            if (clusterLabel >= 0 && clusterLabel < k && pointIdx >= 0 && pointIdx < n)
            {
                counts[clusterLabel]++;
                for (int j = 0; j < d; j++)
                {
                    centers[clusterLabel][j] += NumOps.ToDouble(x[pointIdx, j]);
                }
            }
        }

        // Average the sums
        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int c = 0; c < k; c++)
        {
            if (counts[c] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    centers[c][j] /= counts[c];
                }
            }
            else
            {
                // No seeds for this cluster, initialize randomly
                int randomIdx = rand.Next(n);
                for (int j = 0; j < d; j++)
                {
                    centers[c][j] = NumOps.ToDouble(x[randomIdx, j]);
                }
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
