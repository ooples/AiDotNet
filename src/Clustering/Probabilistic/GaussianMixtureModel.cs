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
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new GaussianMixtureModel&lt;double&gt;(new GMMOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // one cluster index per row
/// var assignments = result.Predict(dataMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Maximum Likelihood from Incomplete Data via the EM Algorithm", "https://doi.org/10.1111/j.2517-6161.1977.tb01600.x")]
public partial class GaussianMixtureModel<T> : ClusteringBase<T>
{
    private readonly GMMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _weights;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _means;
    [AiDotNet.Attributes.FittedParameter]
    private Tensor<T>? _covariances;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _responsibilities;
    private T _lowerBound = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Initializes a new GaussianMixtureModel instance.
    /// </summary>
    /// <param name="options">The GMM options.</param>
    public GaussianMixtureModel(GMMOptions<T>? options = null)
        : base(options ??= new GMMOptions<T>())
    {
        _options = options;
        NumClusters = _options.NumComponents;
    }

    private Vector<T> FittedWeights => _weights ?? throw new InvalidOperationException("Model not fitted. Call Fit() or FitPredict() first.");
    private Matrix<T> FittedMeans => _means ?? throw new InvalidOperationException("Model not fitted. Call Fit() or FitPredict() first.");
    private Tensor<T> FittedCovariances => _covariances ?? throw new InvalidOperationException("Model not fitted. Call Fit() or FitPredict() first.");
    private Matrix<T> FittedResponsibilities => _responsibilities ?? throw new InvalidOperationException("Model not fitted. Call Fit() or FitPredict() first.");

    /// <summary>
    /// Gets the mixture weights.
    /// </summary>
    public Vector<T>? Weights => _weights;

    /// <summary>
    /// Gets the component means.
    /// </summary>
    public Matrix<T>? Means => _means;

    /// <summary>
    /// Gets the component covariances.
    /// </summary>
    public Tensor<T>? Covariances => _covariances;

    /// <summary>
    /// Gets the lower bound (ELBO) from the last training.
    /// </summary>
    public T LowerBound => _lowerBound;

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
        Vector<T>? bestWeights = null;
        Matrix<T>? bestMeans = null;
        Tensor<T>? bestCovariances = null;
        Matrix<T>? bestResponsibilities = null;

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
                    if (HasConverged(_lowerBound, prevLowerBound))
                    {
                        break;
                    }
                    prevLowerBound = _lowerBound;
                }
            }

            // Keep best result
            if (IsHigher(_lowerBound, bestLowerBound))
            {
                bestLowerBound = _lowerBound;
                bestWeights = AiDotNet.Training.HostStateCheckpoint<T>.Capture(FittedWeights, bestWeights);
                bestMeans = AiDotNet.Training.HostStateCheckpoint<T>.Capture(FittedMeans, bestMeans);
                if (bestCovariances is null)
                    bestCovariances = AiDotNet.Training.TensorCheckpoint<T>.Capture(new[] { FittedCovariances })[0];
                else
                    AiDotNet.Training.TensorCheckpoint<T>.CopyValues(FittedCovariances, bestCovariances);
                bestResponsibilities = AiDotNet.Training.HostStateCheckpoint<T>.Capture(FittedResponsibilities, bestResponsibilities);
            }
        }

        // Use best result
        _weights = bestWeights;
        _means = bestMeans;
        _covariances = bestCovariances;
        _responsibilities = bestResponsibilities;
        _lowerBound = bestLowerBound;

        // Assign labels based on maximum responsibility (first component on ties)
        Labels = ClusteringTensorOps.FromIntArray<T>(
            Engine.TensorArgMax(Tensor<T>.FromMatrix(FittedResponsibilities), 1).ToArray());

        // Set cluster centers
        ClusterCenters = FittedMeans.GetSubMatrix(0, 0, k, d);

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    /// <summary>True when the lower bound moved by less than the configured tolerance.</summary>
    private bool HasConverged(T current, T previous)
        => NumOps.LessThan(NumOps.Abs(NumOps.Subtract(current, previous)), NumOps.FromDouble(_options.Tolerance));

    /// <summary>True when <paramref name="value"/> is strictly greater than <paramref name="best"/>.</summary>
    private bool IsHigher(T value, T best) => NumOps.GreaterThan(value, best);

    private void InitializeParameters(Matrix<T> data, int n, int d, int k)
    {
        // Uniform weights and zero responsibilities
        _weights = Engine.Fill(k, NumOps.Divide(NumOps.One, NumOps.FromDouble(k)));
        _means = new Matrix<T>(k, d);
        _covariances = new Tensor<T>([k, d, d]);
        _responsibilities = new Matrix<T>(n, k);

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

        // Copy means from KMeans (may have fewer clusters after degenerate merge)
        if (kmeans.ClusterCenters is not null && _means is not null)
        {
            int actualK = Math.Min(k, kmeans.ClusterCenters.Rows);
            var centers = actualK > 0 ? kmeans.ClusterCenters.GetSubMatrix(0, 0, actualK, d) : new Matrix<T>(0, d);
            if (actualK >= k)
            {
                _means = centers;
                return;
            }

            // If KMeans returned fewer clusters, fill remaining rows c with mean_0 + c * 0.01
            int missing = k - actualK;
            var first = actualK > 0 ? centers.GetRow(0) : new Vector<T>(d);
            var offsets = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(actualK, k, 1), NumOps.FromDouble(0.01));
            var padding = Engine.TensorAdd(
                ClusteringTensorOps.TileRow(first, missing),
                Engine.TensorBroadcastTo(Engine.Reshape(offsets, new[] { missing, 1 }), new[] { missing, d }));
            _means = actualK > 0
                ? Engine.TensorConcatenate(new[] { Tensor<T>.FromMatrix(centers), padding }, 0).ToMatrix()
                : padding.ToMatrix();
        }
    }

    private void InitializeWithKMeansPlusPlus(Matrix<T> data, int n, int d, int k)
    {
        var rand = Random ?? RandomHelper.CreateSecureRandom();

        // First center: random
        var chosen = new int[k];
        chosen[0] = rand.Next(n);
        var minDistances = ClusteringTensorOps.SquaredDistancesToPoint(data.GetRow(chosen[0]), data);

        // Subsequent centers: probability proportional to the squared distance to the nearest chosen center
        for (int c = 1; c < k; c++)
        {
            int next = ClusteringTensorOps.SampleProportionalOrNone(minDistances, rand);
            chosen[c] = next >= 0 ? next : n - 1;
            minDistances = Engine.Min(minDistances, ClusteringTensorOps.SquaredDistancesToPoint(data.GetRow(chosen[c]), data));
        }

        _means = ClusteringTensorOps.SelectRows(data, chosen);
    }

    private void InitializeRandom(Matrix<T> data, int n, int d, int k)
    {
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var selectedIndices = new HashSet<int>();
        var chosen = new int[k];

        for (int c = 0; c < k; c++)
        {
            int idx;
            do
            {
                idx = rand.Next(n);
            } while (selectedIndices.Contains(idx) && selectedIndices.Count < n);
            selectedIndices.Add(idx);
            chosen[c] = idx;
        }

        _means = ClusteringTensorOps.SelectRows(data, chosen);
    }

    private void InitializeCovariances(Matrix<T> data, int n, int d, int k)
    {
        T reg = NumOps.FromDouble(_options.RegularizationCovariance);
        T nMinus1 = NumOps.FromDouble(n - 1);
        var centered = ClusteringTensorOps.SubtractRow(data, ClusteringTensorOps.ColumnMeans(data));
        var identity = Engine.TensorEye<T>(d);

        // Sample variances of each feature (n - 1 denominator)
        Tensor<T> Variances() => Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorSquare(centered), new[] { 0 }, false), nMinus1);

        Tensor<T> shared;
        switch (_options.CovarianceType)
        {
            case CovarianceType.Full:
            case CovarianceType.Tied:
                // Empirical covariance + regularization
                shared = Engine.TensorAdd(
                    Engine.TensorDivideScalar(Engine.TensorMatMul(Engine.TensorTranspose(centered), centered), nMinus1),
                    Engine.TensorMultiplyScalar(identity, reg));
                break;

            case CovarianceType.Diagonal:
                // Variances + regularization
                shared = DiagonalBlocks(Engine.Reshape(Engine.TensorAddScalar(Variances(), reg), new[] { 1, d }));
                break;

            case CovarianceType.Spherical:
                // Average variance + regularization
                shared = Engine.TensorMultiplyScalar(identity, NumOps.Add(Engine.TensorMean(Variances()), reg));
                break;

            default:
                return;
        }

        _covariances = Engine.TensorTile(Engine.Reshape(shared, new[] { 1, d, d }), new[] { k, 1, 1 });
    }

    /// <summary>[m, d, d] diagonal matrices whose diagonals are the rows of <paramref name="diagonals"/> [m, d].</summary>
    private Tensor<T> DiagonalBlocks(Tensor<T> diagonals)
    {
        int m = diagonals.Shape[0];
        int d = diagonals.Shape[1];
        var shape = new[] { m, d, d };
        var onDiagonal = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorEye<T>(d), new[] { 1, d, d }), shape);
        var values = Engine.TensorBroadcastTo(Engine.Reshape(diagonals, new[] { m, 1, d }), shape);
        return Engine.TensorWhere(onDiagonal, values, Tensor<T>.CreateZeros(shape));
    }

    /// <summary>The d x d covariance matrix of one component.</summary>
    private Matrix<T> ComponentCovariance(int component)
    {
        int d = FittedCovariances.Shape[1];
        return Engine.Reshape(
            Engine.TensorSlice(FittedCovariances, new[] { component, 0, 0 }, new[] { 1, d, d }), new[] { d, d }).ToMatrix();
    }

    private void EStep(Matrix<T> data, int n, int d, int k)
    {
        // Responsibilities from the per-component log probabilities (log-sum-exp for stability)
        _responsibilities = Normalize(LogProbabilities(data, k), out _).ToMatrix();
    }

    /// <summary>
    /// [n, k] log(w_c) + log N(x_i | mu_c, Sigma_c) for the first <paramref name="k"/> components, with the
    /// Mahalanobis term evaluated through the Cholesky factor: (x - mu)^T Sigma^-1 (x - mu) = ||L^-1 (x - mu)||^2.
    /// </summary>
    private Tensor<T> LogProbabilities(Matrix<T> x, int k)
    {
        int n = x.Rows;
        int d = x.Columns;
        T half = NumOps.FromDouble(0.5);
        T dLog2Pi = NumOps.Multiply(NumOps.FromDouble(d), NumOps.Log(NumOps.FromDouble(2 * Math.PI)));
        T choleskyFloor = NumOps.FromDouble(1e-6);
        T zero = NumOps.Zero;

        var mahalanobis = new Tensor<T>[k];
        var halfLogDets = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
        {
            var factor = Cholesky(ComponentCovariance(c), zero, choleskyFloor);

            // log det Sigma = 2 * sum(log diag(L))
            halfLogDets[c] = Engine.ReduceSum(Engine.TensorLog(Engine.TensorDiagonal(Tensor<T>.FromMatrix(factor))), new[] { 0 }, true);

            var centered = ClusteringTensorOps.SubtractRow(x, FittedMeans.GetRow(c));
            var whitened = Engine.TensorMatMul(centered, Engine.TensorTranspose(Tensor<T>.FromMatrix(InverseLower(factor))));
            mahalanobis[c] = Engine.ReduceSum(Engine.TensorSquare(whitened), new[] { 1 }, true);
        }

        // log w_c - (d log(2 pi) + log det Sigma_c) / 2
        var logDets = Engine.TensorMultiplyScalar(Engine.TensorConcatenate(halfLogDets, 0), NumOps.FromDouble(2.0));
        var logWeights = Engine.TensorLog(Engine.TensorSlice(FittedWeights.AsTensor(), new[] { 0 }, new[] { k }));
        var offsets = Engine.TensorSubtract(logWeights, Engine.TensorMultiplyScalar(Engine.TensorAddScalar(logDets, dLog2Pi), half));

        var shape = new[] { n, k };
        return Engine.TensorSubtract(
            Engine.TensorBroadcastTo(Engine.Reshape(offsets, new[] { 1, k }), shape),
            Engine.TensorMultiplyScalar(Engine.TensorConcatenate(mahalanobis, 1), half));
    }

    /// <summary>Row-wise softmax of [n, k] log probabilities; <paramref name="logSumExp"/> is the [n, 1] normaliser.</summary>
    private Tensor<T> Normalize(Tensor<T> logProbs, out Tensor<T> logSumExp)
    {
        var shape = new[] { logProbs.Shape[0], logProbs.Shape[1] };
        var maxLog = Engine.ReduceMax(logProbs, new[] { 1 }, true);
        var sumExp = Engine.ReduceSum(
            Engine.TensorExp(Engine.TensorSubtract(logProbs, Engine.TensorBroadcastTo(maxLog, shape))), new[] { 1 }, true);
        logSumExp = Engine.TensorAdd(maxLog, Engine.TensorLog(sumExp));
        return Engine.TensorExp(Engine.TensorSubtract(logProbs, Engine.TensorBroadcastTo(logSumExp, shape)));
    }

    /// <summary>
    /// Lower Cholesky factor (matrix = L L^T), built column by column. A pivot that is not above
    /// <paramref name="threshold"/> is replaced by <paramref name="floor"/>, keeping near-singular covariances usable.
    /// </summary>
    private Matrix<T> Cholesky(Matrix<T> matrix, T threshold, T floor)
    {
        int d = matrix.Rows;
        var factor = new Matrix<T>(d, d);
        var identity = Engine.TensorEye<T>(d).ToMatrix();
        var strictlyBelow = Engine.TensorTranspose(Engine.TensorTriu(Engine.TensorOnesLike(Tensor<T>.FromMatrix(matrix)), 1)).ToMatrix();

        for (int j = 0; j < d; j++)
        {
            // s_i = A_ij - sum_{m<j} L_im L_jm (columns >= j of L are still zero)
            var s = Engine.TensorSubtract(
                matrix.GetColumn(j).AsTensor(),
                Engine.Reshape(
                    Engine.TensorMatMul(Tensor<T>.FromMatrix(factor), Engine.Reshape(factor.GetRow(j).AsTensor(), new[] { d, 1 })),
                    new[] { d }));

            var roots = Engine.TensorSqrt(Engine.TensorWhere(Engine.TensorGreaterThan(s, threshold), s, Engine.TensorFullLike(s, floor)));
            T pivot = Engine.DotProduct(roots.ToVector(), identity.GetColumn(j));

            // L_jj = pivot, L_ij = s_i / pivot below the diagonal, zero above
            var column = Engine.TensorAdd(
                Engine.TensorMultiply(Engine.TensorDivideScalar(s, pivot), strictlyBelow.GetColumn(j).AsTensor()),
                Engine.TensorMultiplyScalar(identity.GetColumn(j).AsTensor(), pivot));
            Engine.SetColumn(factor, j, column.ToVector());
        }

        return factor;
    }

    /// <summary>Inverse of a lower-triangular matrix with a non-zero diagonal, by forward substitution.</summary>
    private Matrix<T> InverseLower(Matrix<T> lower)
    {
        int d = lower.Rows;
        var shape = new[] { d, d };
        var lowerTensor = Tensor<T>.FromMatrix(lower);
        var inverseDiagonal = Engine.TensorReciprocal(Engine.TensorDiagonal(lowerTensor));

        // L = D U with U unit lower-triangular, so L^-1 = U^-1 D^-1
        var unit = Engine.TensorMultiply(lowerTensor, Engine.TensorBroadcastTo(Engine.Reshape(inverseDiagonal, new[] { d, 1 }), shape)).ToMatrix();
        var identity = Engine.TensorEye<T>(d).ToMatrix();
        var unitInverse = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            // Row i of U^-1 = e_i - U[i, :] U^-1 (rows >= i of U^-1 are still zero)
            var row = Engine.TensorSubtract(
                identity.GetRow(i).AsTensor(),
                Engine.Reshape(
                    Engine.TensorMatMul(Engine.Reshape(unit.GetRow(i).AsTensor(), new[] { 1, d }), Tensor<T>.FromMatrix(unitInverse)),
                    new[] { d }));
            Engine.SetRow(unitInverse, i, row.ToVector());
        }

        return Engine.TensorMultiply(
            Tensor<T>.FromMatrix(unitInverse),
            Engine.TensorBroadcastTo(Engine.Reshape(inverseDiagonal, new[] { 1, d }), shape)).ToMatrix();
    }

    private void MStep(Matrix<T> data, int n, int d, int k)
    {
        T reg = NumOps.FromDouble(_options.RegularizationCovariance);
        T nT = NumOps.FromDouble(n);
        T epsilon = NumOps.FromDouble(1e-10);
        T minWeight = NumOps.FromDouble(_options.MinWeight);
        var resp = Tensor<T>.FromMatrix(FittedResponsibilities);

        // Effective number of points per component (floored at epsilon)
        var nk = Engine.TensorMax(Engine.ReduceSum(resp, new[] { 0 }, false), epsilon);

        // Update weights, then normalize
        var weights = Engine.TensorDivideScalar(nk, nT);
        if (!_options.AllowLowWeights)
        {
            weights = Engine.TensorMax(weights, minWeight);
        }
        _weights = Engine.TensorDivideScalar(weights, Engine.TensorSum(weights)).ToVector();

        // Update means: responsibility-weighted averages
        var nkColumn = Engine.Reshape(nk, new[] { k, 1 });
        _means = Engine.TensorDivide(
            Engine.TensorMatMul(Engine.TensorTranspose(resp), Tensor<T>.FromMatrix(data)),
            Engine.TensorBroadcastTo(nkColumn, new[] { k, d })).ToMatrix();

        // Update covariances based on type
        var blockShape = new[] { k, d, d };
        var regIdentity = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorMultiplyScalar(Engine.TensorEye<T>(d), reg), new[] { 1, d, d }), blockShape);
        switch (_options.CovarianceType)
        {
            case CovarianceType.Full:
                _covariances = Engine.TensorAdd(
                    Engine.TensorDivide(WeightedScatter(data, resp), Engine.TensorBroadcastTo(Engine.Reshape(nk, new[] { k, 1, 1 }), blockShape)),
                    regIdentity);
                break;

            case CovarianceType.Tied:
            {
                // Weighted average covariance shared by all components
                var shared = Engine.TensorDivideScalar(
                    Engine.ReduceSum(WeightedScatter(data, resp), new[] { 0 }, false), Engine.TensorSum(nk));
                _covariances = Engine.TensorAdd(
                    Engine.TensorBroadcastTo(Engine.Reshape(shared, new[] { 1, d, d }), blockShape), regIdentity);
                break;
            }

            case CovarianceType.Diagonal:
                _covariances = DiagonalBlocks(Engine.TensorAddScalar(
                    Engine.TensorDivide(WeightedSquares(data, resp), Engine.TensorBroadcastTo(nkColumn, new[] { k, d })), reg));
                break;

            case CovarianceType.Spherical:
            {
                // Average weighted variance over the features
                var total = Engine.ReduceSum(WeightedSquares(data, resp), new[] { 1 }, true);
                var average = Engine.TensorAddScalar(
                    Engine.TensorDivide(total, Engine.TensorMultiplyScalar(nkColumn, NumOps.FromDouble(d))), reg);
                _covariances = DiagonalBlocks(Engine.TensorBroadcastTo(average, new[] { k, d }));
                break;
            }
        }
    }

    /// <summary>[k, d, d] responsibility-weighted scatter sum_i r_ic (x_i - mu_c)(x_i - mu_c)^T per component.</summary>
    private Tensor<T> WeightedScatter(Matrix<T> data, Tensor<T> resp)
    {
        int k = resp.Shape[1];
        var respMatrix = resp.ToMatrix();
        var blocks = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
        {
            var centered = ClusteringTensorOps.SubtractRow(data, FittedMeans.GetRow(c));
            blocks[c] = Engine.TensorMatMul(Engine.TensorTranspose(WeightRows(centered, respMatrix.GetColumn(c))), centered);
        }

        return Engine.TensorStack(blocks, 0);
    }

    /// <summary>[k, d] responsibility-weighted squared deviations sum_i r_ic (x_ip - mu_cp)^2 per component.</summary>
    private Tensor<T> WeightedSquares(Matrix<T> data, Tensor<T> resp)
    {
        int k = resp.Shape[1];
        var respMatrix = resp.ToMatrix();
        var rows = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
        {
            var centered = ClusteringTensorOps.SubtractRow(data, FittedMeans.GetRow(c));
            rows[c] = Engine.ReduceSum(WeightRows(Engine.TensorSquare(centered), respMatrix.GetColumn(c)), new[] { 0 }, false);
        }

        return Engine.TensorStack(rows, 0);
    }

    /// <summary>Scales row i of an [n, d] tensor by weights[i].</summary>
    private Tensor<T> WeightRows(Tensor<T> rows, Vector<T> weights)
        => Engine.TensorMultiply(
            Engine.TensorBroadcastTo(Tensor<T>.FromVector(weights, new[] { weights.Length, 1 }), new[] { rows.Shape[0], rows.Shape[1] }),
            rows);

    private T ComputeLowerBound(Matrix<T> data, int n, int d, int k)
    {
        // sum over (i, c) with r_ic > epsilon of r_ic * (log p(x_i, c) - log r_ic)
        T epsilon = NumOps.FromDouble(1e-10);
        var resp = Tensor<T>.FromMatrix(FittedResponsibilities);
        var terms = Engine.TensorMultiply(
            resp, Engine.TensorSubtract(LogProbabilities(data, k), Engine.TensorLog(Engine.TensorMax(resp, epsilon))));
        var counted = Engine.TensorGreaterThan(resp, epsilon);
        return Engine.TensorSum(Engine.TensorWhere(counted, terms, Engine.TensorFullLike(terms, NumOps.Zero)));
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        // Assign to component with highest probability (first on ties)
        return ClusteringTensorOps.FromIntArray<T>(
            Engine.TensorArgMax(Tensor<T>.FromMatrix(PredictProba(x)), 1).ToArray());
    }

    /// <summary>
    /// Predicts probability of each component for each sample.
    /// </summary>
    /// <param name="x">Input data.</param>
    /// <returns>Matrix of probabilities [samples x components].</returns>
    public Matrix<T> PredictProba(Matrix<T> x)
    {
        ValidateIsTrained();

        return Normalize(LogProbabilities(x, NumClusters), out _).ToMatrix();
    }

    /// <summary>
    /// Computes log-likelihood of data under the model.
    /// </summary>
    /// <param name="x">Input data.</param>
    /// <returns>Log-likelihood score.</returns>
    public T Score(Matrix<T> x)
    {
        ValidateIsTrained();

        // Sum over samples of log sum_c w_c N(x_i | mu_c, Sigma_c)
        Normalize(LogProbabilities(x, NumClusters), out var logSumExp);
        return Engine.TensorSum(logSumExp);
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
        int k = NumClusters;
        int d = NumFeatures;

        // Per sample, one uniform draw (component choice) followed by d standard normal draws, consumed
        // from the caller's Random in that order so seeded sampling stays reproducible.
        var noise = new Matrix<T>(numSamples, d + 1);
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j <= d; j++)
            {
                noise[i, j] = NumOps.FromDouble(j == 0 ? rand.NextDouble() : rand.NextGaussian());
            }
        }

        if (numSamples == 0 || k == 0)
        {
            return new Matrix<T>(numSamples, d);
        }

        // Component: first c with u < cumulative weight c (k - 1 when none), i.e. k - #{c : u < cumulative_c}
        var noiseDraws = Tensor<T>.FromMatrix(noise);
        var cumulative = Engine.TensorCumSum(Engine.TensorSlice(FittedWeights.AsTensor(), new[] { 0 }, new[] { k }), 0);
        var shape = new[] { numSamples, k };
        var above = Engine.TensorGreaterThan(
            Engine.TensorBroadcastTo(Engine.Reshape(cumulative, new[] { 1, k }), shape),
            Engine.TensorBroadcastTo(Engine.TensorSlice(noiseDraws, new[] { 0, 0 }, new[] { numSamples, 1 }), shape));
        var components = Engine.TensorMin(
            Engine.ScalarMinusTensor(NumOps.FromDouble(k), Engine.ReduceSum(above, new[] { 1 }, true)), NumOps.FromDouble(k - 1));
        var chosen = Engine.TensorEquals(
            Engine.TensorBroadcastTo(components, shape),
            Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<T>(0, k, 1), new[] { 1, k }), shape)).ToMatrix();

        // sample = mean + L z for the chosen component's Cholesky factor L
        T floor = NumOps.FromDouble(1e-10);
        var z = Engine.TensorSlice(noiseDraws, new[] { 0, 1 }, new[] { numSamples, d });
        var samples = Tensor<T>.CreateZeros(new[] { numSamples, d });
        for (int c = 0; c < k; c++)
        {
            var factor = Cholesky(ComponentCovariance(c), floor, floor);
            var drawn = Engine.TensorAdd(
                Engine.TensorMatMul(z, Engine.TensorTranspose(Tensor<T>.FromMatrix(factor))),
                ClusteringTensorOps.TileRow(FittedMeans.GetRow(c), numSamples));
            var mask = Engine.TensorBroadcastTo(Tensor<T>.FromVector(chosen.GetColumn(c), new[] { numSamples, 1 }), new[] { numSamples, d });
            samples = Engine.TensorWhere(mask, drawn, samples);
        }

        return samples.ToMatrix();
    }
}
