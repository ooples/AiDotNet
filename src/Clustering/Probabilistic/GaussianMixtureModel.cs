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
public class GaussianMixtureModel<T> : ClusteringBase<T>
{
    private readonly GMMOptions<T> _options;
    private double[]? _weights;
    private double[,]? _means;
    private double[,,]? _covariances;
    private double[,]? _responsibilities;
    private double _lowerBound;

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
    public double[]? Weights => _weights;

    /// <summary>
    /// Gets the component means.
    /// </summary>
    public double[,]? Means => _means;

    /// <summary>
    /// Gets the component covariances.
    /// </summary>
    public double[,,]? Covariances => _covariances;

    /// <summary>
    /// Gets the lower bound (ELBO) from the last training.
    /// </summary>
    public double LowerBound => _lowerBound;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

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

        // Convert data to double for computation
        var data = ConvertToDouble(x);

        double bestLowerBound = double.NegativeInfinity;
        double[]? bestWeights = null;
        double[,]? bestMeans = null;
        double[,,]? bestCovariances = null;
        double[,]? bestResponsibilities = null;

        // Run multiple initializations
        for (int init = 0; init < _options.NumInitializations; init++)
        {
            // Initialize parameters
            InitializeParameters(data, n, d, k);

            double prevLowerBound = double.NegativeInfinity;

            // EM iterations
            for (int iter = 0; iter < _options.MaxIterations; iter++)
            {
                // E-step: Compute responsibilities
                EStep(data, n, d, k);

                // M-step: Update parameters
                MStep(data, n, d, k);

                // Compute lower bound
                if (_options.ComputeLowerBound)
                {
                    _lowerBound = ComputeLowerBound(data, n, d, k);

                    // Check convergence
                    if (Math.Abs(_lowerBound - prevLowerBound) < _options.Tolerance)
                    {
                        break;
                    }
                    prevLowerBound = _lowerBound;
                }
            }

            // Keep best result
            if (_lowerBound > bestLowerBound)
            {
                bestLowerBound = _lowerBound;
                bestWeights = (double[])_weights!.Clone();
                bestMeans = (double[,])_means!.Clone();
                bestCovariances = (double[,,])_covariances!.Clone();
                bestResponsibilities = (double[,])_responsibilities!.Clone();
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
            double maxResp = _responsibilities![i, 0];
            for (int c = 1; c < k; c++)
            {
                if (_responsibilities[i, c] > maxResp)
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
                ClusterCenters[c, j] = NumOps.FromDouble(_means![c, j]);
            }
        }

        IsTrained = true;
    }

    private void InitializeParameters(double[,] data, int n, int d, int k)
    {
        _weights = new double[k];
        _means = new double[k, d];
        _covariances = new double[k, d, d];
        _responsibilities = new double[n, k];

        // Initialize weights uniformly
        for (int c = 0; c < k; c++)
        {
            _weights[c] = 1.0 / k;
        }

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

    private void InitializeWithKMeans(double[,] data, int n, int d, int k)
    {
        // Convert to Matrix<T> for KMeans
        var dataMatrix = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                dataMatrix[i, j] = NumOps.FromDouble(data[i, j]);
            }
        }

        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = k,
            MaxIterations = 10,
            RandomState = _options.RandomState
        });
        kmeans.Train(dataMatrix);

        // Copy means from KMeans
        if (kmeans.ClusterCenters is not null)
        {
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < d; j++)
                {
                    _means![c, j] = NumOps.ToDouble(kmeans.ClusterCenters[c, j]);
                }
            }
        }
    }

    private void InitializeWithKMeansPlusPlus(double[,] data, int n, int d, int k)
    {
        var rand = Random ?? RandomHelper.CreateSecureRandom();

        // First center: random
        int firstIdx = rand.Next(n);
        for (int j = 0; j < d; j++)
        {
            _means![0, j] = data[firstIdx, j];
        }

        var distances = new double[n];

        // Subsequent centers: probability proportional to squared distance
        for (int c = 1; c < k; c++)
        {
            double totalDist = 0;

            for (int i = 0; i < n; i++)
            {
                double minDist = double.MaxValue;
                for (int prev = 0; prev < c; prev++)
                {
                    double dist = 0;
                    for (int j = 0; j < d; j++)
                    {
                        double diff = data[i, j] - _means![prev, j];
                        dist += diff * diff;
                    }
                    minDist = Math.Min(minDist, dist);
                }
                distances[i] = minDist;
                totalDist += minDist;
            }

            // Sample next center
            double target = rand.NextDouble() * totalDist;
            double cumulative = 0;
            int nextIdx = n - 1;
            for (int i = 0; i < n; i++)
            {
                cumulative += distances[i];
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

    private void InitializeRandom(double[,] data, int n, int d, int k)
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

    private void InitializeCovariances(double[,] data, int n, int d, int k)
    {
        double reg = _options.RegularizationCovariance;

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
                        _covariances![c, i, i] += reg;
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
                            _covariances![c, i, j] = (i == j) ? variances[i] + reg : 0;
                        }
                    }
                }
                break;

            case CovarianceType.Spherical:
                // Initialize with average variance + regularization
                double avgVar = ComputeAverageVariance(data, n, d) + reg;
                for (int c = 0; c < k; c++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            _covariances![c, i, j] = (i == j) ? avgVar : 0;
                        }
                    }
                }
                break;
        }
    }

    private double[,] ComputeEmpiricalCovariance(double[,] data, int n, int d)
    {
        var mean = new double[d];
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                mean[j] += data[i, j];
            }
            mean[j] /= n;
        }

        var cov = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int s = 0; s < n; s++)
                {
                    sum += (data[s, i] - mean[i]) * (data[s, j] - mean[j]);
                }
                cov[i, j] = sum / (n - 1);
            }
        }

        return cov;
    }

    private double[] ComputeVariances(double[,] data, int n, int d)
    {
        var variances = new double[d];

        for (int j = 0; j < d; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
            {
                mean += data[i, j];
            }
            mean /= n;

            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = data[i, j] - mean;
                sumSq += diff * diff;
            }
            variances[j] = sumSq / (n - 1);
        }

        return variances;
    }

    private double ComputeAverageVariance(double[,] data, int n, int d)
    {
        var variances = ComputeVariances(data, n, d);
        double sum = 0;
        for (int j = 0; j < d; j++)
        {
            sum += variances[j];
        }
        return sum / d;
    }

    private void EStep(double[,] data, int n, int d, int k)
    {
        // Compute log probabilities for each sample and component
        var logProbs = new double[n, k];

        for (int c = 0; c < k; c++)
        {
            double logWeight = Math.Log(_weights![c]);

            // Get covariance for this component
            var cov = new double[d, d];
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    cov[i, j] = _covariances![c, i, j];
                }
            }

            // Compute log determinant and inverse
            var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);

            double logNormConst = -0.5 * (d * Math.Log(2 * Math.PI) + logDet);

            for (int i = 0; i < n; i++)
            {
                // Compute Mahalanobis distance
                double mahal = 0;
                for (int p = 0; p < d; p++)
                {
                    double diffP = data[i, p] - _means![c, p];
                    for (int q = 0; q < d; q++)
                    {
                        double diffQ = data[i, q] - _means![c, q];
                        mahal += diffP * covInv[p, q] * diffQ;
                    }
                }

                logProbs[i, c] = logWeight + logNormConst - 0.5 * mahal;
            }
        }

        // Convert to responsibilities using log-sum-exp for stability
        for (int i = 0; i < n; i++)
        {
            // Find max for stability
            double maxLog = logProbs[i, 0];
            for (int c = 1; c < k; c++)
            {
                maxLog = Math.Max(maxLog, logProbs[i, c]);
            }

            // Compute log-sum-exp
            double sumExp = 0;
            for (int c = 0; c < k; c++)
            {
                sumExp += Math.Exp(logProbs[i, c] - maxLog);
            }
            double logSumExp = maxLog + Math.Log(sumExp);

            // Compute responsibilities
            for (int c = 0; c < k; c++)
            {
                _responsibilities![i, c] = Math.Exp(logProbs[i, c] - logSumExp);
            }
        }
    }

    private void MStep(double[,] data, int n, int d, int k)
    {
        double reg = _options.RegularizationCovariance;

        // Compute effective number of points per component
        var nk = new double[k];
        for (int c = 0; c < k; c++)
        {
            for (int i = 0; i < n; i++)
            {
                nk[c] += _responsibilities![i, c];
            }
            nk[c] = Math.Max(nk[c], 1e-10); // Prevent division by zero
        }

        // Update weights
        for (int c = 0; c < k; c++)
        {
            _weights![c] = nk[c] / n;
            if (!_options.AllowLowWeights && _weights[c] < _options.MinWeight)
            {
                _weights[c] = _options.MinWeight;
            }
        }

        // Normalize weights
        double sumWeights = 0;
        for (int c = 0; c < k; c++)
        {
            sumWeights += _weights![c];
        }
        for (int c = 0; c < k; c++)
        {
            _weights![c] /= sumWeights;
        }

        // Update means
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += _responsibilities![i, c] * data[i, j];
                }
                _means![c, j] = sum / nk[c];
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

    private void UpdateFullCovariance(double[,] data, int n, int d, int k, double[] nk, double reg)
    {
        for (int c = 0; c < k; c++)
        {
            for (int p = 0; p < d; p++)
            {
                for (int q = 0; q < d; q++)
                {
                    double sum = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double diffP = data[i, p] - _means![c, p];
                        double diffQ = data[i, q] - _means![c, q];
                        sum += _responsibilities![i, c] * diffP * diffQ;
                    }
                    _covariances![c, p, q] = sum / nk[c];
                    if (p == q)
                    {
                        _covariances![c, p, q] += reg;
                    }
                }
            }
        }
    }

    private void UpdateTiedCovariance(double[,] data, int n, int d, int k, double[] nk, double reg)
    {
        // Compute weighted average covariance
        var sharedCov = new double[d, d];
        double totalWeight = 0;

        for (int c = 0; c < k; c++)
        {
            totalWeight += nk[c];
            for (int p = 0; p < d; p++)
            {
                for (int q = 0; q < d; q++)
                {
                    double sum = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double diffP = data[i, p] - _means![c, p];
                        double diffQ = data[i, q] - _means![c, q];
                        sum += _responsibilities![i, c] * diffP * diffQ;
                    }
                    sharedCov[p, q] += sum;
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
                    _covariances![c, p, q] = sharedCov[p, q] / totalWeight;
                    if (p == q)
                    {
                        _covariances![c, p, q] += reg;
                    }
                }
            }
        }
    }

    private void UpdateDiagonalCovariance(double[,] data, int n, int d, int k, double[] nk, double reg)
    {
        for (int c = 0; c < k; c++)
        {
            for (int p = 0; p < d; p++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    double diff = data[i, p] - _means![c, p];
                    sum += _responsibilities![i, c] * diff * diff;
                }
                _covariances![c, p, p] = sum / nk[c] + reg;

                // Zero out off-diagonal elements
                for (int q = 0; q < d; q++)
                {
                    if (p != q)
                    {
                        _covariances![c, p, q] = 0;
                    }
                }
            }
        }
    }

    private void UpdateSphericalCovariance(double[,] data, int n, int d, int k, double[] nk, double reg)
    {
        for (int c = 0; c < k; c++)
        {
            double totalVar = 0;
            for (int p = 0; p < d; p++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    double diff = data[i, p] - _means![c, p];
                    sum += _responsibilities![i, c] * diff * diff;
                }
                totalVar += sum;
            }
            double avgVar = totalVar / (nk[c] * d) + reg;

            for (int p = 0; p < d; p++)
            {
                for (int q = 0; q < d; q++)
                {
                    _covariances![c, p, q] = (p == q) ? avgVar : 0;
                }
            }
        }
    }

    private (double logDet, double[,] inverse) ComputeLogDetAndInverse(double[,] matrix, int n)
    {
        // Use Cholesky decomposition for positive definite matrices
        var L = new double[n, n];

        // Cholesky decomposition: matrix = L * L^T
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = matrix[i, j];
                for (int k = 0; k < j; k++)
                {
                    sum -= L[i, k] * L[j, k];
                }
                if (i == j)
                {
                    if (sum <= 0)
                    {
                        // Not positive definite, add regularization
                        sum = 1e-6;
                    }
                    L[i, j] = Math.Sqrt(sum);
                }
                else
                {
                    L[i, j] = sum / L[j, j];
                }
            }
        }

        // Log determinant = 2 * sum(log(diagonal of L))
        double logDet = 0;
        for (int i = 0; i < n; i++)
        {
            logDet += 2 * Math.Log(L[i, i]);
        }

        // Compute inverse using forward/backward substitution
        var inverse = new double[n, n];

        // Compute L^-1
        var Linv = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            Linv[i, i] = 1.0 / L[i, i];
            for (int j = 0; j < i; j++)
            {
                double sum = 0;
                for (int k = j; k < i; k++)
                {
                    sum -= L[i, k] * Linv[k, j];
                }
                Linv[i, j] = sum / L[i, i];
            }
        }

        // inverse = (L^-1)^T * L^-1
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int k = Math.Max(i, j); k < n; k++)
                {
                    sum += Linv[k, i] * Linv[k, j];
                }
                inverse[i, j] = sum;
            }
        }

        return (logDet, inverse);
    }

    private double ComputeLowerBound(double[,] data, int n, int d, int k)
    {
        double bound = 0;

        for (int i = 0; i < n; i++)
        {
            double sampleBound = 0;
            for (int c = 0; c < k; c++)
            {
                if (_responsibilities![i, c] > 1e-10)
                {
                    // Get covariance for this component
                    var cov = new double[d, d];
                    for (int p = 0; p < d; p++)
                    {
                        for (int q = 0; q < d; q++)
                        {
                            cov[p, q] = _covariances![c, p, q];
                        }
                    }

                    var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);

                    // Compute log probability
                    double mahal = 0;
                    for (int p = 0; p < d; p++)
                    {
                        double diffP = data[i, p] - _means![c, p];
                        for (int q = 0; q < d; q++)
                        {
                            double diffQ = data[i, q] - _means![c, q];
                            mahal += diffP * covInv[p, q] * diffQ;
                        }
                    }

                    double logProb = Math.Log(_weights![c]) - 0.5 * (d * Math.Log(2 * Math.PI) + logDet + mahal);
                    double entropy = -Math.Log(_responsibilities![i, c]);

                    sampleBound += _responsibilities![i, c] * (logProb + entropy);
                }
            }
            bound += sampleBound;
        }

        return bound;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        var data = ConvertToDouble(x);
        var labels = new Vector<T>(n);

        // Compute responsibilities for new data
        var resp = PredictProba(x);

        // Assign to component with highest probability
        for (int i = 0; i < n; i++)
        {
            int bestCluster = 0;
            double maxProb = resp[i, 0];
            for (int c = 1; c < NumClusters; c++)
            {
                if (resp[i, c] > maxProb)
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
    public double[,] PredictProba(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int d = x.Columns;
        int k = NumClusters;
        var data = ConvertToDouble(x);
        var resp = new double[n, k];

        // Compute log probabilities
        var logProbs = new double[n, k];

        for (int c = 0; c < k; c++)
        {
            double logWeight = Math.Log(_weights![c]);

            var cov = new double[d, d];
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    cov[i, j] = _covariances![c, i, j];
                }
            }

            var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);
            double logNormConst = -0.5 * (d * Math.Log(2 * Math.PI) + logDet);

            for (int i = 0; i < n; i++)
            {
                double mahal = 0;
                for (int p = 0; p < d; p++)
                {
                    double diffP = data[i, p] - _means![c, p];
                    for (int q = 0; q < d; q++)
                    {
                        double diffQ = data[i, q] - _means![c, q];
                        mahal += diffP * covInv[p, q] * diffQ;
                    }
                }

                logProbs[i, c] = logWeight + logNormConst - 0.5 * mahal;
            }
        }

        // Normalize using log-sum-exp
        for (int i = 0; i < n; i++)
        {
            double maxLog = logProbs[i, 0];
            for (int c = 1; c < k; c++)
            {
                maxLog = Math.Max(maxLog, logProbs[i, c]);
            }

            double sumExp = 0;
            for (int c = 0; c < k; c++)
            {
                sumExp += Math.Exp(logProbs[i, c] - maxLog);
            }
            double logSumExp = maxLog + Math.Log(sumExp);

            for (int c = 0; c < k; c++)
            {
                resp[i, c] = Math.Exp(logProbs[i, c] - logSumExp);
            }
        }

        return resp;
    }

    /// <summary>
    /// Computes log-likelihood of data under the model.
    /// </summary>
    /// <param name="x">Input data.</param>
    /// <returns>Log-likelihood score.</returns>
    public double Score(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int d = x.Columns;
        int k = NumClusters;
        var data = ConvertToDouble(x);
        double logLikelihood = 0;

        for (int i = 0; i < n; i++)
        {
            var logProbs = new double[k];

            for (int c = 0; c < k; c++)
            {
                double logWeight = Math.Log(_weights![c]);

                var cov = new double[d, d];
                for (int p = 0; p < d; p++)
                {
                    for (int q = 0; q < d; q++)
                    {
                        cov[p, q] = _covariances![c, p, q];
                    }
                }

                var (logDet, covInv) = ComputeLogDetAndInverse(cov, d);
                double logNormConst = -0.5 * (d * Math.Log(2 * Math.PI) + logDet);

                double mahal = 0;
                for (int p = 0; p < d; p++)
                {
                    double diffP = data[i, p] - _means![c, p];
                    for (int q = 0; q < d; q++)
                    {
                        double diffQ = data[i, q] - _means![c, q];
                        mahal += diffP * covInv[p, q] * diffQ;
                    }
                }

                logProbs[c] = logWeight + logNormConst - 0.5 * mahal;
            }

            // Log-sum-exp
            double maxLog = logProbs[0];
            for (int c = 1; c < k; c++)
            {
                maxLog = Math.Max(maxLog, logProbs[c]);
            }

            double sumExp = 0;
            for (int c = 0; c < k; c++)
            {
                sumExp += Math.Exp(logProbs[c] - maxLog);
            }

            logLikelihood += maxLog + Math.Log(sumExp);
        }

        return logLikelihood;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
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
            // Choose component based on weights
            double u = rand.NextDouble();
            double cumulative = 0;
            int component = NumClusters - 1;
            for (int c = 0; c < NumClusters; c++)
            {
                cumulative += _weights![c];
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
                samples[i, j] = NumOps.FromDouble(sample[j]);
            }
        }

        return samples;
    }

    private double[] SampleGaussian(int component, Random rand)
    {
        int d = NumFeatures;

        // Generate standard normal samples
        var z = new double[d];
        for (int i = 0; i < d; i++)
        {
            z[i] = rand.NextGaussian();
        }

        // Cholesky decomposition of covariance
        var L = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = _covariances![component, i, j];
                for (int k = 0; k < j; k++)
                {
                    sum -= L[i, k] * L[j, k];
                }
                if (i == j)
                {
                    L[i, j] = Math.Sqrt(Math.Max(sum, 1e-10));
                }
                else
                {
                    L[i, j] = sum / L[j, j];
                }
            }
        }

        // Transform: sample = mean + L * z
        var sample = new double[d];
        for (int i = 0; i < d; i++)
        {
            sample[i] = _means![component, i];
            for (int j = 0; j <= i; j++)
            {
                sample[i] += L[i, j] * z[j];
            }
        }

        return sample;
    }

    private double[,] ConvertToDouble(Matrix<T> x)
    {
        var result = new double[x.Rows, x.Columns];
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                result[i, j] = NumOps.ToDouble(x[i, j]);
            }
        }
        return result;
    }
}
