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
/// <example>
/// <code>
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newData = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for seeded K-Means clustering
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SeededKMeans&lt;double&gt;(new SeededKMeansOptions&lt;double&gt;()));
///
/// var result = builder.Build(dataMatrix, labels);
/// var predictions = result.Predict(newData);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Seeded Region Growing", "https://doi.org/10.1109/34.295913", Year = 2000, Authors = "Sugato Basu, Arindam Banerjee, Raymond Mooney")]
public partial class SeededKMeans<T> : ClusteringBase<T>
{
    private readonly SeededKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new Seeded K-Means instance.
    /// </summary>
    /// <param name="options">The Seeded K-Means options.</param>
    public SeededKMeans(SeededKMeansOptions<T>? options = null)
        : base(options ??= new SeededKMeansOptions<T>())
    {
        _options = options;
    }

    /// <inheritdoc />

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
            var nearest = ClusteringTensorOps.NearestCenters(metric, x, centers);
            for (int i = 0; i < n; i++)
            {
                // Skip constrained seeds
                if (_options.ConstrainSeeds && seeds.ContainsKey(i))
                {
                    continue;
                }

                if (nearest[i] != labels[i])
                {
                    labels[i] = nearest[i];
                    changed = true;
                }
            }

            // Update centers (empty clusters keep their previous center)
            var means = ClusteringTensorOps.ClusterMeans(x, ClusteringTensorOps.FromIntArray<T>(labels), k, out var counts);
            centers = ClusteringTensorOps.RowsWherePositive(counts, means, centers);
        }

        // Set results
        ClusterCenters = centers;
        Labels = ClusteringTensorOps.FromIntArray<T>(labels);

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    private Matrix<T> InitializeCentersFromSeeds(Matrix<T> x, Dictionary<int, int> seeds, int k, int n, int d)
    {
        // Seed label per point (-1 for unseeded points and out-of-range seeds)
        var seedLabels = new int[n];
        for (int i = 0; i < n; i++)
        {
            seedLabels[i] = -1;
        }

        foreach (var kvp in seeds)
        {
            int pointIdx = kvp.Key;
            int clusterLabel = kvp.Value;

            if (clusterLabel >= 0 && clusterLabel < k && pointIdx >= 0 && pointIdx < n)
            {
                seedLabels[pointIdx] = clusterLabel;
            }
        }

        // Average the seeds of each cluster
        var means = ClusteringTensorOps.ClusterMeans(x, ClusteringTensorOps.FromIntArray<T>(seedLabels), k, out var counts);

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Clusters without seeds start at a random data point
        var seedCounts = ClusteringTensorOps.ToIntArray(counts);
        var fallbackRows = new int[k];
        for (int c = 0; c < k; c++)
        {
            fallbackRows[c] = seedCounts[c] > 0 ? 0 : rand.Next(n);
        }

        var fallback = n > 0 ? ClusteringTensorOps.SelectRows(x, fallbackRows) : new Matrix<T>(k, d);
        return ClusteringTensorOps.RowsWherePositive(counts, means, fallback);
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
