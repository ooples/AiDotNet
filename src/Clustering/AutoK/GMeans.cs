using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.AutoK;

/// <summary>
/// G-Means clustering with automatic K determination using Gaussian tests.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// G-Means tests whether data in each cluster comes from a Gaussian distribution.
/// If not, the cluster is split. It uses the Anderson-Darling test projected
/// onto the principal direction connecting cluster centers.
/// </para>
/// <para>
/// Algorithm:
/// 1. Start with K=1 (all data in one cluster)
/// 2. For each cluster, project data onto the main axis
/// 3. Test if projection is Gaussian (Anderson-Darling)
/// 4. If not Gaussian, split the cluster
/// 5. Repeat until all clusters are Gaussian or max K reached
/// </para>
/// <para><b>For Beginners:</b> G-Means assumes good clusters are Gaussian.
///
/// A Gaussian distribution is the famous "bell curve":
/// - Most points near the center
/// - Fewer points as you go farther out
/// - Symmetric left and right
///
/// If a cluster doesn't look like a bell curve:
/// - It might contain multiple groups
/// - Split it into two and check again
///
/// This works well when true clusters are roughly Gaussian.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new GMeans&lt;double&gt;(new GMeansOptions&lt;double&gt;()))
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
[ResearchPaper("Learning the k in k-means", "https://proceedings.neurips.cc/paper/2003/hash/234833147b97bb6aed53a8f4f1c7a7d8-Abstract.html", Year = 2004, Authors = "Greg Hamerly, Charles Elkan")]
public partial class GMeans<T> : ClusteringBase<T>
{
    private readonly GMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new GMeans instance.
    /// </summary>
    /// <param name="options">The GMeans options.</param>
    public GMeans(GMeansOptions<T>? options = null)
        : base(options ??= new GMeansOptions<T>())
    {
        _options = options;
    }

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (GMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Start with minimum clusters
        int currentK = Math.Max(1, _options.MinClusters);

        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = currentK,
            MaxIterations = Options.MaxIterations,
            Seed = Options.Seed
        });

        kmeans.Train(x);
        var kmeansLabels = kmeans.Labels ?? throw new InvalidOperationException("K-means has not been trained.");
        var currentLabels = ClusteringTensorOps.ToIntArray(kmeansLabels);
        var currentCenters = CenterRows(kmeans.ClusterCenters, currentK, d);

        // Iteratively test and split clusters
        bool improved = true;
        while (improved && currentCenters.Count < _options.MaxClusters)
        {
            improved = false;
            var newCenters = new List<Vector<T>>();
            var newLabels = new int[n];
            var clusterMapping = new Dictionary<int, int>();

            for (int c = 0; c < currentCenters.Count; c++)
            {
                // Get points in this cluster
                var clusterPoints = new List<int>();
                for (int i = 0; i < n; i++)
                {
                    if (currentLabels[i] == c)
                    {
                        clusterPoints.Add(i);
                    }
                }

                if (clusterPoints.Count < 4)
                {
                    // Too few points to test
                    clusterMapping[c] = newCenters.Count;
                    newCenters.Add(currentCenters[c]);
                    continue;
                }

                // Create sub-matrix for this cluster
                var subMatrix = ClusteringTensorOps.SelectRows(x, clusterPoints.ToArray());

                // Test if Gaussian using Anderson-Darling
                bool isGaussian = TestGaussian(subMatrix, clusterPoints.Count, d);

                if (!isGaussian && newCenters.Count + 1 < _options.MaxClusters)
                {
                    // Split this cluster
                    var subKMeans = new KMeans<T>(new KMeansOptions<T>
                    {
                        NumClusters = 2,
                        MaxIterations = Options.MaxIterations,
                        Seed = Options.Seed
                    });

                    subKMeans.Train(subMatrix);

                    // Add both children
                    newCenters.AddRange(CenterRows(subKMeans.ClusterCenters, 2, d));

                    // Update labels for points in this cluster
                    var subLabels = subKMeans.Labels ?? throw new InvalidOperationException("K-means has not been trained.");
                    var subIds = ClusteringTensorOps.ToIntArray(subLabels);
                    for (int i = 0; i < clusterPoints.Count; i++)
                    {
                        newLabels[clusterPoints[i]] = newCenters.Count - 2 + subIds[i];
                    }

                    improved = true;
                }
                else
                {
                    // Keep this cluster
                    clusterMapping[c] = newCenters.Count;
                    newCenters.Add(currentCenters[c]);

                    foreach (int i in clusterPoints)
                    {
                        newLabels[i] = newCenters.Count - 1;
                    }
                }
            }

            currentCenters = newCenters;
            currentLabels = newLabels;
        }

        // Set final results
        NumClusters = currentCenters.Count;
        ClusterCenters = NumClusters > 0
            ? Matrix<T>.FromRows(currentCenters.Cast<IEnumerable<T>>().ToArray())
            : new Matrix<T>(0, d);
        Labels = ClusteringTensorOps.FromIntArray<T>(currentLabels);

        IsTrained = true;
    }

    /// <summary>The first <paramref name="count"/> center rows (zero rows when the model has no centers).</summary>
    private static List<Vector<T>> CenterRows(Matrix<T>? centers, int count, int d)
    {
        var rows = new List<Vector<T>>(count);
        for (int c = 0; c < count; c++)
        {
            rows.Add(centers is not null && c < centers.Rows ? centers.GetRow(c) : new Vector<T>(d));
        }

        return rows;
    }

    private bool TestGaussian(Matrix<T> data, int n, int d)
    {
        if (n < 4) return true;

        // Project data onto principal axis
        // For simplicity, project onto first principal component
        var projections = ProjectToPrincipalAxis(data, n, d).AsTensor();

        // Standardize projections
        T mean = Engine.TensorMean(projections);
        var centered = Engine.TensorSubtractScalar(projections, mean);
        T std = NumOps.Sqrt(NumOps.Divide(Engine.TensorSum(Engine.TensorSquare(centered)), NumOps.FromDouble(n)));

        if (NumOps.LessThan(std, NumOps.FromDouble(1e-10))) return true; // Constant data

        var standardized = Engine.TensorMsort(Engine.TensorDivideScalar(centered, std));

        // Anderson-Darling test
        T ad = ComputeAndersonDarling(standardized, n);

        // Get critical value for significance level
        T criticalValue = NumOps.FromDouble(GetADCriticalValue(_options.SignificanceLevel));

        return !NumOps.GreaterThan(ad, criticalValue);
    }

    private Vector<T> ProjectToPrincipalAxis(Matrix<T> data, int n, int d)
    {
        // Center data
        var centered = ClusteringTensorOps.SubtractRow(data, ClusteringTensorOps.ColumnMeans(data));
        var centeredT = Engine.TensorTranspose(centered);

        // Compute first principal component using power iteration
        // Seeded start vector in [-0.5, 0.5)^d: d rand.NextDouble() draws, shifted by 0.5
        var rand = RandomHelper.CreateSeededRandom(42);
        var pc = Engine.TensorSubtractScalar(
            Tensor<T>.FromVector(Vector<T>.CreateRandom(rand, d), new[] { d, 1 }), NumOps.FromDouble(0.5));

        // Normalize
        pc = Engine.TensorDivideScalar(pc, NumOps.Sqrt(Engine.TensorSum(Engine.TensorSquare(pc))));

        // Power iteration
        for (int iter = 0; iter < 20; iter++)
        {
            // Compute X^T * X * pc
            var newPc = Engine.TensorMatMul(centeredT, Engine.TensorMatMul(centered, pc));

            // Normalize
            T norm = NumOps.Sqrt(Engine.TensorSum(Engine.TensorSquare(newPc)));
            if (IsTiny(norm)) break;

            pc = Engine.TensorDivideScalar(newPc, norm);
        }

        // Project data onto PC
        return Engine.TensorMatMul(centered, pc).ToVector();
    }

    /// <summary>True when a norm is too small to normalize by.</summary>
    private bool IsTiny(T norm) => NumOps.LessThan(norm, NumOps.FromDouble(1e-10));

    private T ComputeAndersonDarling(Tensor<T> sortedStandardized, int n)
    {
        // Anderson-Darling test statistic: -n - (1/n) sum (2i - 1)(ln F(z_i) + ln(1 - F(z_{n+1-i})))
        T low = NumOps.FromDouble(1e-10);
        T high = NumOps.FromDouble(1 - 1e-10);
        var phi = Engine.TensorClamp(NormalCdf(sortedStandardized), low, high);
        var phiReversed = Engine.TensorFlip(phi, new[] { 0 });
        var terms = Engine.TensorAdd(
            Engine.TensorLog(phi),
            Engine.TensorLog(Engine.ScalarMinusTensor(NumOps.One, phiReversed)));
        var weights = Engine.TensorArange<T>(1, 2 * n, 2);
        T sum = Engine.TensorSum(Engine.TensorMultiply(weights, terms));

        double nd = n;
        T ad = NumOps.Subtract(NumOps.FromDouble(-nd), NumOps.Divide(sum, NumOps.FromDouble(nd)));

        // Apply correction for small samples
        return NumOps.Multiply(ad, NumOps.FromDouble(1 + 4.0 / nd - 25.0 / (nd * nd)));
    }

    /// <summary>Standard normal CDF, 0.5 (1 + erf(z / sqrt 2)).</summary>
    private Tensor<T> NormalCdf(Tensor<T> z)
    {
        var erf = Engine.TensorErf(Engine.TensorMultiplyScalar(z, NumOps.FromDouble(1.0 / Math.Sqrt(2))));
        return Engine.TensorMultiplyScalar(Engine.TensorAddScalar(erf, NumOps.One), NumOps.FromDouble(0.5));
    }

    private double GetADCriticalValue(double alpha)
    {
        // Critical values for Anderson-Darling test
        // These are approximate values for normal distribution
        if (alpha >= 0.15) return 0.576;
        if (alpha >= 0.10) return 0.656;
        if (alpha >= 0.05) return 0.787;
        if (alpha >= 0.025) return 0.918;
        if (alpha >= 0.01) return 1.092;
        if (alpha >= 0.005) return 1.227;
        if (alpha >= 0.001) return 1.551;
        return 1.8; // Very strict
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is not null && x.Columns != ClusterCenters.Columns)
            throw new ArgumentException(
                $"Input has {x.Columns} features but model was trained with {ClusterCenters.Columns} features.");

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
