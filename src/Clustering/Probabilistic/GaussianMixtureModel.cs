using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Probabilistic;

/// <summary>
/// Gaussian Mixture Model clustering using Expectation-Maximization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GMM represents data as a mixture of Gaussian distributions. Each cluster is
/// characterized by its mean, covariance, and mixing weight. The EM algorithm
/// iteratively refines these parameters.
/// </para>
/// <para>
/// Algorithm steps (EM):
/// 1. Initialize means, covariances, and weights
/// 2. E-step: Compute responsibility of each component for each point
/// 3. M-step: Update parameters to maximize expected log-likelihood
/// 4. Repeat until convergence
/// </para>
/// <para><b>For Beginners:</b> GMM is like soft K-Means.
///
/// Instead of saying "this point belongs to cluster 2", GMM says
/// "this point has 70% chance of cluster 2, 25% chance of cluster 1, 5% chance of cluster 3".
///
/// This is useful when:
/// - Clusters overlap (points could belong to multiple groups)
/// - Clusters have different shapes (some wide, some narrow)
/// - You need uncertainty estimates for cluster assignments
///
/// The EM algorithm works by:
/// - E-step: For each point, estimate probability of belonging to each cluster
/// - M-step: Update cluster parameters based on these probabilities
/// - Repeat until stable
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new GMMOptions&lt;double&gt;();
/// var gaussianMixtureModel = new GaussianMixtureModel&lt;double&gt;(options);
/// gaussianMixtureModel.Fit(dataMatrix);
/// int[] labels = gaussianMixtureModel.Labels;
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
public class GaussianMixtureModel<T> : ClusteringBase<T>
{
    private readonly GMMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[]? _weights;
    private T[,]? _means;
    private T[,,]? _covariances;
    private T[,]? _responsibilities;
    private T _lowerBound = default(T) ?? MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Initializes a new GaussianMixtureModel instance.
    /// </summary>
    /// <param name="options">The GMM options.</param>
    public GaussianMixtureModel(GMMOptions<T>? options = null)
        : base(options ?? new GMMOptions<T>())
    {
        _options = options ?? new GMMOptions<T>();
        NumClusters = _options.NumComponents;
    }

    /// <summary>
    /// Gets the mixture weights.
    /// </summary>
    public T[]? Weights => _weights;

    /// <summary>
    /// Gets the component means.
    /// </summary>
    public T[,]? Means => _means;

    /// <summary>
    /// Gets the component covariances.
    /// </summary>
    public T[,,]? Covariances => _covariances;

    /// <summary>
    /// Gets the lower bound (ELBO) from the last training.
    /// </summary>
    public T LowerBound => _lowerBound;

    /// <inheritdoc />

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GaussianMixtureModel<T>(new GMMOptions<T>
        {
            NumComponents = _options.NumComponents,
            CovarianceType = _options.CovarianceType,
            Tolerance = _options.Tolerance,
            MaxIterations = _options.MaxIterations,
            NumInitializations = _options.NumInitializations,
            InitMethod = _options.InitMethod,
            RegularizationCovariance = _options.RegularizationCovariance
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (GaussianMixtureModel<T>)CreateNewInstance();
        clone.NumFeatures = NumFeatures;
        clone.NumClusters = NumClusters;
        clone.IsTrained = IsTrained;
        clone.Labels = Labels is not null ? new Vector<T>(Labels) : null;
        clone.Inertia = Inertia;

        if (ClusterCenters is not null)
        {
            clone.ClusterCenters = new Matrix<T>(ClusterCenters.Rows, ClusterCenters.Columns);
            for (int i = 0; i < ClusterCenters.Rows; i++)
                for (int j = 0; j < ClusterCenters.Columns; j++)
                    clone.ClusterCenters[i, j] = ClusterCenters[i, j];
        }

        if (_weights is not null)
            clone._weights = (T[])_weights.Clone();
        if (_means is not null)
            clone._means = (T[,])_means.Clone();
        if (_covariances is not null)
            clone._covariances = (T[,,])_covariances.Clone();

        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (GaussianMixtureModel<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        int k = _options.NumComponents;

        NumFeatures = d;

        if (n < k)
        {
            throw new ArgumentException($"Number of samples ({n}) must be >= number of components ({k}).");
        }

        T bestLowerBound = NumOps.MinValue;
        T[]? bestWeights = null;
        T[,]? bestMeans = null;
        T[,,]? bestCovariances = null;
        T[,]? bestResponsibilities = null;

        // Run multiple initializations
        for (int init = 0; init < _options.NumInitializations; init++)
        {
            // Initialize parameters
            InitializeParameters(x, n, d, k);

            T prevLowerBound = NumOps.MinValue;

            // EM iterations
            for (int iter = 0; iter < _options.MaxIterations; iter++)
            {
                // E-step: Compute responsibilities
                EStep(x, n, d, k);

                // M-step: Update parameters
                MStep(x, n, d, k);

                // Compute lower bound
                if (_options.ComputeLowerBound)
                {
                    _lowerBound = ComputeLowerBound(x, n, d, k);

                    // Check convergence
                    T diff = NumOps.Abs(NumOps.Subtract(_lowerBound, prevLowerBound));
                    if (NumOps.LessThan(diff, NumOps.FromDouble(_options.Tolerance)))
                    {
                        break;
                    }
                    prevLowerBound = _lowerBound;
                }
            }

            // Keep best result
            if (NumOps.GreaterThan(_lowerBound, bestLowerBound))
            {
                bestLowerBound = _lowerBound;
                bestWeights = (T[])_weights!.Clone();
                bestMeans = (T[,])_means!.Clone();
                bestCovariances = (T[,,])_covariances!.Clone();
                bestResponsibilities = (T[,])_responsibilities!.Clone();
            }
        }

        // Use best result
        _weights = bestWeights;
        _means = bestMeans;
        _covariances = bestCovariances;
        _responsibilities = bestResponsibilities;
        _lowerBound = bestLowerBound;

        // Assign labels based on maximum responsibility
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            int bestCluster = 0;
            T maxResp = _responsibilities![i, 0];
            for (int c = 1; c < k; c++)
            {
                if (NumOps.GreaterThan(_responsibilities[i, c], maxResp))
                {
                    maxResp = _responsibilities[i, c];
                    bestCluster = c;
                }
            }
            Labels[i] = NumOps.FromDouble(bestCluster);
        }

        // Set cluster centers
        ClusterCenters = new Matrix<T>(k, d);
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = _means![c, j];
            }
        }

        IsTrained = true;
    }

    private void InitializeParameters(Matrix<T> data, int n, int d, int k)
    {
        _weights = new T[k];
        _means = new T[k, d];
        _covariances = new T[k, d, d];
        _responsibilities = new T[n, k];

        // Initialize weights uniformly
        T uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(k));
        for (int c = 0; c < k; c++)
        {
            _weights[c] = uniformWeight;
        }

        // Initialize responsibilities to zero
        for (int i = 0; i < n; i++)
            for (int c = 0; c < k; c++)
                _responsibilities[i, c] = NumOps.Zero;

        // Initialize means based on method
        switch (_options.InitMethod)
        {
            case GMMInitMethod.KMeans:
                InitializeWithKMeans(data, n, d, k);
                break;
            case GMMInitMethod.KMeansPlusPlus:
                InitializeWithKMeansPlusPlus(data, n, d, k);
                break;
            case GMMInitMethod.Random:
            default:
                InitializeRandom(data, n, d, k);
                break;
        }

        // Initialize covariances
        InitializeCovariances(data, n, d, k);
    }

    private void InitializeWithKMeans(Matrix<T> data, int n, int d, int k)
    {
        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = k,
            MaxIterations = 10,
            Seed = _options.Seed
        });
        kmeans.Train(data);

        // Copy means from KMeans
        if (kmeans.ClusterCenters is not null)
        {
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < d; j++)
                {
                    _means![c, j] = kmeans.ClusterCenters[c, j];
                }
            }
        }
    }

    private void InitializeWithKMeansPlusPlus(Matrix<T> data, int n, int d, int k)
    {
        var rand = Random ?? RandomHelper.CreateSecureRandom();

        // First center: random
        int firstIdx = rand.Next(n);
        for (int j = 0; j < d; j++)
        {
            _means![0, j] = data[firstIdx, j];
        }

        var distances = new T[n];

        // Subsequent centers: probability proportional to squared distance
        for (int c = 1; c < k; c++)
        {
            T totalDist = NumOps.Zero;

            for (int i = 0; i < n; i++)
            {
                T minDist = NumOps.MaxValue;
                for (int prev = 0; prev < c; prev++)
                {
                    T dist = NumOps.Zero;
                    for (int j = 0; j < d; j++)
                    {
                        T diff = NumOps.Subtract(data[i, j], _means![prev, j]);
                        dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                    }
                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                    }
                }
                distances[i] = minDist;
                totalDist = NumOps.Add(totalDist, minDist);
            }

            // Sample next center (random boundary stays double)
            double target = rand.NextDouble() * NumOps.ToDouble(totalDist);
            double cumulative = 0;
            int nextIdx = n - 1;
            for (int i = 0; i < n; i++)
            {
                cumulative += NumOps.ToDouble(distances[i]);
                if (cumulative >= target)
                {
                    nextIdx = i;
                    break;
                }
            }

            for (int j = 0; j < d; j++)
            {
                _means![c, j] = data[nextIdx, j];
            }
        }
    }

    private void InitializeRandom(Matrix<T> data, int n, int d, int k)
    {
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var selectedIndices = new HashSet<int>();

        for (int c = 0; c < k; c++)
        {
            int idx;
            do
            {
                idx = rand.Next(n);
            } while (selectedIndices.Contains(idx) && selectedIndices.Count < n);
            selectedIndices.Add(idx);

            for (int j = 0; j < d; j++)
            {
                _means![c, j] = data[idx, j];
            }
        }
    }

    private void InitializeCovariances(Matrix<T> data, int n, int d, int k)
    {
        T reg = NumOps.FromDouble(_options.RegularizationCovariance);

        switch (_options.CovarianceType)
        {
            case CovarianceType.Full:
            case CovarianceType.Tied:
                // Initialize with empirical covariance + regularization
                var globalCov = ComputeEmpiricalCovariance(data, n, d);
                for (int c = 0; c < k; c++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            _covariances![c, i, j] = globalCov[i, j];
                        }
                        _covariances![c, i, i] = NumOps.Add(_covariances[c, i, i], reg);
                    }
                }
                break;

            case CovarianceType.Diagonal:
                // Initialize with variances + regularization
                var variances = ComputeVariances(data, n, d);
                for (int c = 0; c < k; c++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            _covariances![c, i, j] = (i == j) ? NumOps.Add(variances[i], reg) : NumOps.Zero;
                        }
                    }
                }
                break;

            case CovarianceType.Spherical:
                // Initialize with average variance + regularization
                T avgVar = NumOps.Add(ComputeAverageVariance(data, n, d), reg);
                for (int c = 0; c < k; c++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            _covariances![c, i, j] = (i == j) ? avgVar : NumOps.Zero;
                        }
                    }
                }
                break;
        }
    }

    private T[,] ComputeEmpiricalCovariance(Matrix<T> data, int n, int d)
    {
        var mean = new T[d];
        T nT = NumOps.FromDouble(n);
        for (int j = 0; j < d; j++)
        {
            mean[j] = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                mean[j] = NumOps.Add(mean[j], data[i, j]);
            }
            mean[j] = NumOps.Divide(mean[j], nT);
        }

        var cov = new T[d, d];
        T nMinus1 = NumOps.FromDouble(n - 1);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < n; s++)
                {
                    T diffI = NumOps.Subtract(data[s, i], mean[i]);
                    T diffJ = NumOps.Subtract(data[s, j], mean[j]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diffI, diffJ));
                }
                cov[i, j] = NumOps.Divide(sum, nMinus1);
            }
        }

        return cov;
    }

    private T[] ComputeVariances(Matrix<T> data, int n, int d)
    {
        var variances = new T[d];
        T nT = NumOps.FromDouble(n);
        T nMinus1 = NumOps.FromDouble(n - 1);

        for (int j = 0; j < d; j++)
        {
            T mean = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                mean = NumOps.Add(mean, data[i, j]);
            }
            mean = NumOps.Divide(mean, nT);

            T sumSq = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], mean);
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            }
            variances[j] = NumOps.Divide(sumSq, nMinus1);
        }

        return variances;
    }

    private T ComputeAverageVariance(Matrix<T> data, int n, int d)
    {
        var variances = ComputeVariances(data, n, d);
        T sum = NumOps.Zero;
        for (int j = 0; j < d; j++)
        {
            sum = NumOps.Add(sum, variances[j]);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(d));
    }

    private void EStep(Matrix<T> data, int n, int d, int k)
    {
        // Compute log probabilities for each sample and component
        var logProbs = new T[n, k];
        T halfT = NumOps.FromDouble(0.5);
        T log2Pi = NumOps.Log(NumOps.FromDouble(2 * Math.PI));

        for (int c = 0; c < k; c++)
        {
            T logWeight = NumOps.Log(_weights![c]);

            // Get covariance for this component
            var cov = new T[d, d];
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    cov[i, j] = _covariances![c, i, j];
                }
            }

            // Compute log determinant and inverse
            var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);

            T logNormConst = NumOps.Negate(NumOps.Multiply(halfT,
                NumOps.Add(NumOps.Multiply(NumOps.FromDouble(d), log2Pi), logDet)));

            for (int i = 0; i < n; i++)
            {
                // Compute Mahalanobis distance
                T mahal = NumOps.Zero;
                for (int p = 0; p < d; p++)
                {
                    T diffP = NumOps.Subtract(data[i, p], _means![c, p]);
                    for (int q = 0; q < d; q++)
                    {
                        T diffQ = NumOps.Subtract(data[i, q], _means![c, q]);
                        mahal = NumOps.Add(mahal, NumOps.Multiply(NumOps.Multiply(diffP, covInv[p, q]), diffQ));
                    }
                }

                logProbs[i, c] = NumOps.Add(NumOps.Add(logWeight, logNormConst),
                    NumOps.Negate(NumOps.Multiply(halfT, mahal)));
            }
        }

        // Convert to responsibilities using log-sum-exp for stability
        for (int i = 0; i < n; i++)
        {
            // Find max for stability
            T maxLog = logProbs[i, 0];
            for (int c = 1; c < k; c++)
            {
                if (NumOps.GreaterThan(logProbs[i, c], maxLog))
                {
                    maxLog = logProbs[i, c];
                }
            }

            // Compute log-sum-exp
            T sumExp = NumOps.Zero;
            for (int c = 0; c < k; c++)
            {
                sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(logProbs[i, c], maxLog)));
            }
            T logSumExp = NumOps.Add(maxLog, NumOps.Log(sumExp));

            // Compute responsibilities
            for (int c = 0; c < k; c++)
            {
                _responsibilities![i, c] = NumOps.Exp(NumOps.Subtract(logProbs[i, c], logSumExp));
            }
        }
    }

    private void MStep(Matrix<T> data, int n, int d, int k)
    {
        T reg = NumOps.FromDouble(_options.RegularizationCovariance);
        T nT = NumOps.FromDouble(n);
        T epsilon = NumOps.FromDouble(1e-10);
        T minWeight = NumOps.FromDouble(_options.MinWeight);

        // Compute effective number of points per component
        var nk = new T[k];
        for (int c = 0; c < k; c++)
        {
            nk[c] = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                nk[c] = NumOps.Add(nk[c], _responsibilities![i, c]);
            }
            if (NumOps.LessThan(nk[c], epsilon))
            {
                nk[c] = epsilon;
            }
        }

        // Update weights
        for (int c = 0; c < k; c++)
        {
            _weights![c] = NumOps.Divide(nk[c], nT);
            if (!_options.AllowLowWeights && NumOps.LessThan(_weights[c], minWeight))
            {
                _weights[c] = minWeight;
            }
        }

        // Normalize weights
        T sumWeights = NumOps.Zero;
        for (int c = 0; c < k; c++)
        {
            sumWeights = NumOps.Add(sumWeights, _weights![c]);
        }
        for (int c = 0; c < k; c++)
        {
            _weights![c] = NumOps.Divide(_weights[c], sumWeights);
        }

        // Update means
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_responsibilities![i, c], data[i, j]));
                }
                _means![c, j] = NumOps.Divide(sum, nk[c]);
            }
        }

        // Update covariances based on type
        switch (_options.CovarianceType)
        {
            case CovarianceType.Full:
                UpdateFullCovariance(data, n, d, k, nk, reg);
                break;
            case CovarianceType.Tied:
                UpdateTiedCovariance(data, n, d, k, nk, reg);
                break;
            case CovarianceType.Diagonal:
                UpdateDiagonalCovariance(data, n, d, k, nk, reg);
                break;
            case CovarianceType.Spherical:
                UpdateSphericalCovariance(data, n, d, k, nk, reg);
                break;
        }
    }

    private void UpdateFullCovariance(Matrix<T> data, int n, int d, int k, T[] nk, T reg)
    {
        for (int c = 0; c < k; c++)
        {
            for (int p = 0; p < d; p++)
            {
                for (int q = 0; q < d; q++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        T diffP = NumOps.Subtract(data[i, p], _means![c, p]);
                        T diffQ = NumOps.Subtract(data[i, q], _means![c, q]);
                        sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(_responsibilities![i, c], diffP), diffQ));
                    }
                    _covariances![c, p, q] = NumOps.Divide(sum, nk[c]);
                    if (p == q)
                    {
                        _covariances![c, p, q] = NumOps.Add(_covariances[c, p, q], reg);
                    }
                }
            }
        }
    }

    private void UpdateTiedCovariance(Matrix<T> data, int n, int d, int k, T[] nk, T reg)
    {
        // Compute weighted average covariance
        var sharedCov = new T[d, d];
        T totalWeight = NumOps.Zero;

        for (int p = 0; p < d; p++)
            for (int q = 0; q < d; q++)
                sharedCov[p, q] = NumOps.Zero;

        for (int c = 0; c < k; c++)
        {
            totalWeight = NumOps.Add(totalWeight, nk[c]);
            for (int p = 0; p < d; p++)
            {
                for (int q = 0; q < d; q++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        T diffP = NumOps.Subtract(data[i, p], _means![c, p]);
                        T diffQ = NumOps.Subtract(data[i, q], _means![c, q]);
                        sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(_responsibilities![i, c], diffP), diffQ));
                    }
                    sharedCov[p, q] = NumOps.Add(sharedCov[p, q], sum);
                }
            }
        }

        // Normalize and apply to all components
        for (int c = 0; c < k; c++)
        {
            for (int p = 0; p < d; p++)
            {
                for (int q = 0; q < d; q++)
                {
                    _covariances![c, p, q] = NumOps.Divide(sharedCov[p, q], totalWeight);
                    if (p == q)
                    {
                        _covariances![c, p, q] = NumOps.Add(_covariances[c, p, q], reg);
                    }
                }
            }
        }
    }

    private void UpdateDiagonalCovariance(Matrix<T> data, int n, int d, int k, T[] nk, T reg)
    {
        for (int c = 0; c < k; c++)
        {
            for (int p = 0; p < d; p++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    T diff = NumOps.Subtract(data[i, p], _means![c, p]);
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(_responsibilities![i, c], diff), diff));
                }
                _covariances![c, p, p] = NumOps.Add(NumOps.Divide(sum, nk[c]), reg);

                // Zero out off-diagonal elements
                for (int q = 0; q < d; q++)
                {
                    if (p != q)
                    {
                        _covariances![c, p, q] = NumOps.Zero;
                    }
                }
            }
        }
    }

    private void UpdateSphericalCovariance(Matrix<T> data, int n, int d, int k, T[] nk, T reg)
    {
        for (int c = 0; c < k; c++)
        {
            T totalVar = NumOps.Zero;
            for (int p = 0; p < d; p++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    T diff = NumOps.Subtract(data[i, p], _means![c, p]);
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(_responsibilities![i, c], diff), diff));
                }
                totalVar = NumOps.Add(totalVar, sum);
            }
            T avgVar = NumOps.Add(NumOps.Divide(totalVar, NumOps.Multiply(nk[c], NumOps.FromDouble(d))), reg);

            for (int p = 0; p < d; p++)
            {
                for (int q = 0; q < d; q++)
                {
                    _covariances![c, p, q] = (p == q) ? avgVar : NumOps.Zero;
                }
            }
        }
    }

    private (T logDet, T[,] inverse) ComputeLogDetAndInverse(T[,] matrix, int n)
    {
        // Use Cholesky decomposition for positive definite matrices
        var L = new T[n, n];
        T regVal = NumOps.FromDouble(1e-6);
        T two = NumOps.FromDouble(2.0);

        // Cholesky decomposition: matrix = L * L^T
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = matrix[i, j];
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Subtract(sum, NumOps.Multiply(L[i, k], L[j, k]));
                }
                if (i == j)
                {
                    if (!NumOps.GreaterThan(sum, NumOps.Zero))
                    {
                        // Not positive definite, add regularization
                        sum = regVal;
                    }
                    L[i, j] = NumOps.Sqrt(sum);
                }
                else
                {
                    L[i, j] = NumOps.Divide(sum, L[j, j]);
                }
            }
        }

        // Log determinant = 2 * sum(log(diagonal of L))
        T logDet = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            logDet = NumOps.Add(logDet, NumOps.Multiply(two, NumOps.Log(L[i, i])));
        }

        // Compute inverse using forward/backward substitution
        var inverse = new T[n, n];

        // Compute L^-1
        var Linv = new T[n, n];
        for (int i = 0; i < n; i++)
        {
            Linv[i, i] = NumOps.Divide(NumOps.One, L[i, i]);
            for (int j = 0; j < i; j++)
            {
                T sum = NumOps.Zero;
                for (int k = j; k < i; k++)
                {
                    sum = NumOps.Subtract(sum, NumOps.Multiply(L[i, k], Linv[k, j]));
                }
                Linv[i, j] = NumOps.Divide(sum, L[i, i]);
            }
        }

        // inverse = (L^-1)^T * L^-1
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T sum = NumOps.Zero;
                for (int k = Math.Max(i, j); k < n; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(Linv[k, i], Linv[k, j]));
                }
                inverse[i, j] = sum;
            }
        }

        return (logDet, inverse);
    }

    private T ComputeLowerBound(Matrix<T> data, int n, int d, int k)
    {
        T bound = NumOps.Zero;
        T halfT = NumOps.FromDouble(0.5);
        T log2Pi = NumOps.Log(NumOps.FromDouble(2 * Math.PI));
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < n; i++)
        {
            T sampleBound = NumOps.Zero;
            for (int c = 0; c < k; c++)
            {
                if (NumOps.GreaterThan(_responsibilities![i, c], epsilon))
                {
                    // Get covariance for this component
                    var cov = new T[d, d];
                    for (int p = 0; p < d; p++)
                    {
                        for (int q = 0; q < d; q++)
                        {
                            cov[p, q] = _covariances![c, p, q];
                        }
                    }

                    var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);

                    // Compute log probability
                    T mahal = NumOps.Zero;
                    for (int p = 0; p < d; p++)
                    {
                        T diffP = NumOps.Subtract(data[i, p], _means![c, p]);
                        for (int q = 0; q < d; q++)
                        {
                            T diffQ = NumOps.Subtract(data[i, q], _means![c, q]);
                            mahal = NumOps.Add(mahal, NumOps.Multiply(NumOps.Multiply(diffP, covInv[p, q]), diffQ));
                        }
                    }

                    T logProb = NumOps.Subtract(NumOps.Log(_weights![c]),
                        NumOps.Multiply(halfT, NumOps.Add(NumOps.Add(
                            NumOps.Multiply(NumOps.FromDouble(d), log2Pi), logDet), mahal)));
                    T entropy = NumOps.Negate(NumOps.Log(_responsibilities![i, c]));

                    sampleBound = NumOps.Add(sampleBound,
                        NumOps.Multiply(_responsibilities![i, c], NumOps.Add(logProb, entropy)));
                }
            }
            bound = NumOps.Add(bound, sampleBound);
        }

        return bound;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        var labels = new Vector<T>(n);

        // Compute responsibilities for new data
        var resp = PredictProba(x);

        // Assign to component with highest probability
        for (int i = 0; i < n; i++)
        {
            int bestCluster = 0;
            T maxProb = resp[i, 0];
            for (int c = 1; c < NumClusters; c++)
            {
                if (NumOps.GreaterThan(resp[i, c], maxProb))
                {
                    maxProb = resp[i, c];
                    bestCluster = c;
                }
            }
            labels[i] = NumOps.FromDouble(bestCluster);
        }

        return labels;
    }

    /// <summary>
    /// Predicts probability of each component for each sample.
    /// </summary>
    /// <param name="x">Input data.</param>
    /// <returns>Matrix of probabilities [samples x components].</returns>
    public T[,] PredictProba(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int d = x.Columns;
        int k = NumClusters;
        var resp = new T[n, k];
        T halfT = NumOps.FromDouble(0.5);
        T log2Pi = NumOps.Log(NumOps.FromDouble(2 * Math.PI));

        // Compute log probabilities
        var logProbs = new T[n, k];

        for (int c = 0; c < k; c++)
        {
            T logWeight = NumOps.Log(_weights![c]);

            var cov = new T[d, d];
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    cov[i, j] = _covariances![c, i, j];
                }
            }

            var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);
            T logNormConst = NumOps.Negate(NumOps.Multiply(halfT,
                NumOps.Add(NumOps.Multiply(NumOps.FromDouble(d), log2Pi), logDet)));

            for (int i = 0; i < n; i++)
            {
                T mahal = NumOps.Zero;
                for (int p = 0; p < d; p++)
                {
                    T diffP = NumOps.Subtract(x[i, p], _means![c, p]);
                    for (int q = 0; q < d; q++)
                    {
                        T diffQ = NumOps.Subtract(x[i, q], _means![c, q]);
                        mahal = NumOps.Add(mahal, NumOps.Multiply(NumOps.Multiply(diffP, covInv[p, q]), diffQ));
                    }
                }

                logProbs[i, c] = NumOps.Add(NumOps.Add(logWeight, logNormConst),
                    NumOps.Negate(NumOps.Multiply(halfT, mahal)));
            }
        }

        // Normalize using log-sum-exp
        for (int i = 0; i < n; i++)
        {
            T maxLog = logProbs[i, 0];
            for (int c = 1; c < k; c++)
            {
                if (NumOps.GreaterThan(logProbs[i, c], maxLog))
                {
                    maxLog = logProbs[i, c];
                }
            }

            T sumExp = NumOps.Zero;
            for (int c = 0; c < k; c++)
            {
                sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(logProbs[i, c], maxLog)));
            }
            T logSumExp = NumOps.Add(maxLog, NumOps.Log(sumExp));

            for (int c = 0; c < k; c++)
            {
                resp[i, c] = NumOps.Exp(NumOps.Subtract(logProbs[i, c], logSumExp));
            }
        }

        return resp;
    }

    /// <summary>
    /// Computes log-likelihood of data under the model.
    /// </summary>
    /// <param name="x">Input data.</param>
    /// <returns>Log-likelihood score.</returns>
    public T Score(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int d = x.Columns;
        int k = NumClusters;
        T logLikelihood = NumOps.Zero;
        T halfT = NumOps.FromDouble(0.5);
        T log2Pi = NumOps.Log(NumOps.FromDouble(2 * Math.PI));

        for (int i = 0; i < n; i++)
        {
            var logProbs = new T[k];

            for (int c = 0; c < k; c++)
            {
                T logWeight = NumOps.Log(_weights![c]);

                var cov = new T[d, d];
                for (int p = 0; p < d; p++)
                {
                    for (int q = 0; q < d; q++)
                    {
                        cov[p, q] = _covariances![c, p, q];
                    }
                }

                var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);
                T logNormConst = NumOps.Negate(NumOps.Multiply(halfT,
                    NumOps.Add(NumOps.Multiply(NumOps.FromDouble(d), log2Pi), logDet)));

                T mahal = NumOps.Zero;
                for (int p = 0; p < d; p++)
                {
                    T diffP = NumOps.Subtract(x[i, p], _means![c, p]);
                    for (int q = 0; q < d; q++)
                    {
                        T diffQ = NumOps.Subtract(x[i, q], _means![c, q]);
                        mahal = NumOps.Add(mahal, NumOps.Multiply(NumOps.Multiply(diffP, covInv[p, q]), diffQ));
                    }
                }

                logProbs[c] = NumOps.Add(NumOps.Add(logWeight, logNormConst),
                    NumOps.Negate(NumOps.Multiply(halfT, mahal)));
            }

            // Log-sum-exp
            T maxLog = logProbs[0];
            for (int c = 1; c < k; c++)
            {
                if (NumOps.GreaterThan(logProbs[c], maxLog))
                {
                    maxLog = logProbs[c];
                }
            }

            T sumExp = NumOps.Zero;
            for (int c = 0; c < k; c++)
            {
                sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(logProbs[c], maxLog)));
            }

            logLikelihood = NumOps.Add(logLikelihood, NumOps.Add(maxLog, NumOps.Log(sumExp)));
        }

        return logLikelihood;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// Samples from the fitted mixture model.
    /// </summary>
    /// <param name="numSamples">Number of samples to generate.</param>
    /// <returns>Generated samples.</returns>
    public Matrix<T> Sample(int numSamples)
    {
        ValidateIsTrained();

        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var samples = new Matrix<T>(numSamples, NumFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            // Choose component based on weights (random boundary stays double)
            double u = rand.NextDouble();
            double cumulative = 0;
            int component = NumClusters - 1;
            for (int c = 0; c < NumClusters; c++)
            {
                cumulative += NumOps.ToDouble(_weights![c]);
                if (u < cumulative)
                {
                    component = c;
                    break;
                }
            }

            // Sample from chosen component's Gaussian
            var sample = SampleGaussian(component, rand);
            for (int j = 0; j < NumFeatures; j++)
            {
                samples[i, j] = sample[j];
            }
        }

        return samples;
    }

    private T[] SampleGaussian(int component, Random rand)
    {
        int d = NumFeatures;
        T regVal = NumOps.FromDouble(1e-10);

        // Generate standard normal samples
        var z = new T[d];
        for (int i = 0; i < d; i++)
        {
            z[i] = NumOps.FromDouble(rand.NextGaussian());
        }

        // Cholesky decomposition of covariance
        var L = new T[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = _covariances![component, i, j];
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Subtract(sum, NumOps.Multiply(L[i, k], L[j, k]));
                }
                if (i == j)
                {
                    if (!NumOps.GreaterThan(sum, regVal))
                    {
                        sum = regVal;
                    }
                    L[i, j] = NumOps.Sqrt(sum);
                }
                else
                {
                    L[i, j] = NumOps.Divide(sum, L[j, j]);
                }
            }
        }

        // Transform: sample = mean + L * z
        var sample = new T[d];
        for (int i = 0; i < d; i++)
        {
            sample[i] = _means![component, i];
            for (int j = 0; j <= i; j++)
            {
                sample[i] = NumOps.Add(sample[i], NumOps.Multiply(L[i, j], z[j]));
            }
        }

        return sample;
    }
}
