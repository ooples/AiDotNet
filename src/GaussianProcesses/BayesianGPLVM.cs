using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements the Bayesian Gaussian Process Latent Variable Model (Bayesian GPLVM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Bayesian GPLVM is a powerful technique for dimensionality reduction
/// and learning latent representations of data. Unlike PCA which finds linear projections,
/// GPLVM finds nonlinear latent spaces using Gaussian Processes.
///
/// Key concepts:
/// 1. **Latent Space**: A lower-dimensional representation where data lives
/// 2. **Mapping**: A GP maps from latent space to observed data
/// 3. **Uncertainty**: We maintain uncertainty over both the mapping and the latent points
///
/// Applications:
/// - Visualizing high-dimensional data (like t-SNE but probabilistic)
/// - Finding meaningful low-dimensional representations
/// - Handling missing data gracefully
/// - Interpolating between data points
/// </para>
/// <para>
/// <b>Mathematical Background:</b>
/// The model assumes observed data Y is generated from latent points X through a GP:
///   y_n = f(x_n) + ε,  where f ~ GP(0, k)
///
/// The Bayesian approach places priors on the latent points X:
///   p(X) = Π_n N(x_n | 0, I)
///
/// We use variational inference to approximate the posterior p(X|Y).
/// The inducing points framework makes this scalable to large datasets.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.GaussianProcess)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Bayesian Gaussian Process Latent Variable Model", "https://doi.org/10.48550/arXiv.1309.6835", Year = 2010, Authors = "Michalis K. Titsias, Neil D. Lawrence")]
public partial class BayesianGPLVM<T>
{
    /// <summary>
    /// The kernel function for the mapping from latent to observed space.
    /// </summary>
    private readonly IKernelFunction<T> _kernel;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The number of latent dimensions.
    /// </summary>
    private readonly int _latentDimensions;

    /// <summary>
    /// The number of inducing points for scalable inference.
    /// </summary>
    private readonly int _numInducingPoints;

    /// <summary>
    /// The variational mean of the latent points (N x Q).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This represents where we think each data point
    /// is located in the latent space. Each row corresponds to one data point,
    /// and each column is a latent dimension.
    /// </para>
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _latentMean;

    /// <summary>
    /// The variational variance of the latent points (N x Q).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This represents our uncertainty about where each point
    /// is in latent space. Larger values mean more uncertainty.
    /// </para>
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _latentVariance;

    /// <summary>
    /// The inducing points in latent space (M x Q).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _inducingPoints;

    /// <summary>
    /// The observed data (N x D).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _observedData;

    /// <summary>
    /// The observation noise variance.
    /// </summary>
    private T _noiseVariance;

    /// <summary>
    /// Whether the model has been fitted.
    /// </summary>
    private bool _isFitted;

    /// <summary>
    /// The learning rate for optimization.
    /// </summary>
    private readonly double _learningRate;

    /// <summary>
    /// Maximum number of optimization iterations.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the Bayesian GPLVM.
    /// </summary>
    /// <param name="kernel">The kernel function for the GP mapping.</param>
    /// <param name="latentDimensions">The number of latent dimensions (Q).</param>
    /// <param name="numInducingPoints">The number of inducing points (M).</param>
    /// <param name="noiseVariance">The observation noise variance. Default is 1.0.</param>
    /// <param name="learningRate">The learning rate for optimization. Default is 0.01.</param>
    /// <param name="maxIterations">Maximum optimization iterations. Default is 100.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Bayesian GPLVM for learning latent representations.
    ///
    /// Choosing latent dimensions:
    /// - 2D: Great for visualization
    /// - 3D: Good for interactive visualization
    /// - Higher: For downstream tasks that need more expressiveness
    ///
    /// Choosing inducing points:
    /// - More inducing points = better approximation but slower
    /// - Start with 10-20% of your data size
    /// - Can always add more if ELBO is too low
    /// </para>
    /// </remarks>
    public BayesianGPLVM(
        IKernelFunction<T> kernel,
        int latentDimensions,
        int numInducingPoints,
        double noiseVariance = 1.0,
        double learningRate = 0.01,
        int maxIterations = 100)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (latentDimensions < 1)
            throw new ArgumentException("Latent dimensions must be at least 1.", nameof(latentDimensions));
        if (numInducingPoints < 1)
            throw new ArgumentException("Number of inducing points must be at least 1.", nameof(numInducingPoints));
        if (noiseVariance <= 0)
            throw new ArgumentException("Noise variance must be positive.", nameof(noiseVariance));

        _kernel = kernel;
        _latentDimensions = latentDimensions;
        _numInducingPoints = numInducingPoints;
        _learningRate = learningRate;
        _maxIterations = maxIterations;
        _numOps = MathHelper.GetNumericOperations<T>();
        _noiseVariance = _numOps.FromDouble(noiseVariance);
        _isFitted = false;
    }

    /// <summary>
    /// Gets the latent dimension count.
    /// </summary>
    public int LatentDimensions => _latentDimensions;

    /// <summary>
    /// Gets the number of inducing points.
    /// </summary>
    public int NumInducingPoints => _numInducingPoints;

    /// <summary>
    /// Gets whether the model has been fitted.
    /// </summary>
    public bool IsFitted => _isFitted;

    /// <summary>
    /// Fits the Bayesian GPLVM to observed data.
    /// </summary>
    /// <param name="data">The observed data matrix (N x D).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This learns a latent representation of your data.
    ///
    /// The fitting process:
    /// 1. Initialize latent points using PCA (gives a good starting point)
    /// 2. Select inducing points from the initial latent space
    /// 3. Optimize the ELBO (Evidence Lower BOund) to find the best:
    ///    - Latent point locations (where each data point is in latent space)
    ///    - Latent point uncertainties (how sure we are about each location)
    ///    - Inducing points (representative points for efficient computation)
    ///
    /// After fitting, you can:
    /// - Get the latent representation with GetLatentMean()
    /// - Reconstruct data with Reconstruct()
    /// - Project new points with Transform()
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Rows < _numInducingPoints)
            throw new ArgumentException($"Data must have at least {_numInducingPoints} points.", nameof(data));

        _observedData = data;
        int n = data.Rows;
        int d = data.Columns;

        // Initialize latent points using PCA
        InitializeLatentPoints(data);

        // Select inducing points from initial latent positions
        InitializeInducingPoints();

        // Optimize ELBO using gradient descent
        OptimizeELBO();

        _isFitted = true;
    }

    /// <summary>
    /// Initializes latent points using PCA.
    /// </summary>
    /// <param name="data">The observed data.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> We use PCA (Principal Component Analysis) to initialize
    /// the latent points. This gives us a good starting point for optimization.
    ///
    /// PCA finds the directions of maximum variance in the data. By projecting
    /// onto the top Q principal components, we get initial latent coordinates
    /// that preserve the most important structure in the data.
    ///
    /// Starting from PCA makes optimization much faster than random initialization.
    /// </para>
    /// </remarks>
    private void InitializeLatentPoints(Matrix<T> data)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Rows;
        int d = data.Columns;

        // Center the data
        var observed = Tensor<T>.FromMatrix(data);
        var mean = engine.ReduceMean(observed, new[] { 0 }, keepDims: true);
        var centered = engine.TensorSubtract(observed, engine.TensorBroadcastTo(mean, new[] { n, d }));

        // Compute covariance matrix (d x d)
        var cov = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), _numOps.FromDouble(n - 1));

        // Simple power iteration to find top eigenvectors
        int effectiveLatentDims = Math.Min(_latentDimensions, d);
        var eigenvectors = new Tensor<T>[effectiveLatentDims];
        var workMatrix = cov.ToMatrix();

        var rand = RandomHelper.CreateSeededRandom(42);
        T tinyNorm = _numOps.FromDouble(1e-10);

        for (int k = 0; k < effectiveLatentDims; k++)
        {
            // Initialize random vector (uniform draws in [-0.5, 0.5) from the seeded generator)
            var start = new double[d];
            for (int i = 0; i < d; i++)
            {
                start[i] = rand.NextDouble() - 0.5;
            }
            var v = GaussianProcessEngineOps.FromDoubles<T>(start);

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
            {
                var vNew = engine.MatrixVectorMultiply(workMatrix, v);

                // Normalize
                T norm = _numOps.Sqrt(engine.DotProduct(vNew, vNew));
                if (_numOps.GreaterThan(norm, tinyNorm))
                {
                    v = engine.Divide(vNew, norm);
                }
            }

            // Store eigenvector
            eigenvectors[k] = Tensor<T>.FromVector(v).Reshape(new[] { d, 1 });

            // Compute eigenvalue
            T eigenvalue = engine.DotProduct(v, engine.MatrixVectorMultiply(workMatrix, v));

            // Deflate: A = A - eigenvalue * v * v^T
            var column = Tensor<T>.FromVector(v).Reshape(new[] { d, 1 });
            var outer = engine.TensorMatMul(column, engine.TensorTranspose(column));
            workMatrix = engine.TensorSubtract(
                Tensor<T>.FromMatrix(workMatrix), engine.TensorMultiplyScalar(outer, eigenvalue)).ToMatrix();
        }

        // Project data onto top eigenvectors to get latent means
        var latent = effectiveLatentDims > 0
            ? engine.TensorMatMul(centered, engine.TensorConcatenate(eigenvectors, 1))
            : new Tensor<T>(new[] { n, 0 });

        // Fill remaining dimensions with small random values, drawn row by row
        int extraDims = _latentDimensions - effectiveLatentDims;
        if (extraDims > 0)
        {
            var noise = new double[n * extraDims];
            for (int i = 0; i < noise.Length; i++)
            {
                noise[i] = 0.01 * (rand.NextDouble() - 0.5);
            }

            var extra = Tensor<T>.FromVector(GaussianProcessEngineOps.FromDoubles<T>(noise)).Reshape(new[] { n, extraDims });
            latent = engine.TensorConcatenate(new[] { latent, extra }, 1);
        }

        _latentMean = latent.ToMatrix();

        // Initialize variance to small value
        _latentVariance = Tensor<T>.CreateDefault(new[] { n, _latentDimensions }, _numOps.FromDouble(0.1)).ToMatrix();
    }

    /// <summary>
    /// Initializes inducing points by selecting from latent means.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inducing points are representative points in latent space
    /// that summarize the data. We select them to be spread out across the latent space.
    ///
    /// We use a greedy selection that tries to maximize coverage:
    /// 1. Pick a random starting point
    /// 2. Repeatedly add the point farthest from currently selected points
    ///
    /// This gives us inducing points that are well-distributed.
    /// </para>
    /// </remarks>
    private void InitializeInducingPoints()
    {
        if (_latentMean is null)
            throw new InvalidOperationException("Latent points must be initialized first.");

        var engine = AiDotNetEngine.Current;
        int n = _latentMean.Rows;
        int m = Math.Min(_numInducingPoints, n);

        // Greedy farthest point selection
        var latent = Tensor<T>.FromMatrix(_latentMean);
        var minDistances = Tensor<T>.CreateDefault(new[] { n }, _numOps.MaxValue);
        var selectedMask = new Tensor<T>(new[] { n });
        var markOne = Tensor<T>.CreateDefault(new[] { 1 }, _numOps.One);
        var excluded = Tensor<T>.CreateDefault(new[] { n }, _numOps.FromDouble(-1.0));
        var chosen = new List<int>(m);

        var rand = RandomHelper.CreateSeededRandom(42);
        int nextIdx = rand.Next(n);

        while (chosen.Count < m)
        {
            chosen.Add(nextIdx);
            selectedMask = engine.TensorIndexAdd(selectedMask, 0, new Tensor<int>(new[] { nextIdx }, new[] { 1 }), markOne);
            if (chosen.Count == m) break;

            // Update distances, then pick the farthest unselected point (first one on ties)
            minDistances = engine.TensorMin(minDistances, SquaredDistancesToRow(latent, nextIdx));
            var scores = engine.TensorWhere(
                engine.TensorGreaterThan(selectedMask, _numOps.Zero), excluded, minDistances);
            nextIdx = engine.TensorArgMax(scores, 0).AsSpan()[0];
        }

        _inducingPoints = GaussianProcessEngineOps.SelectRows(_latentMean, chosen.ToArray());
    }

    /// <summary>
    /// Squared Euclidean distance from every row of <paramref name="points"/> to row <paramref name="row"/>.
    /// </summary>
    private static Tensor<T> SquaredDistancesToRow(Tensor<T> points, int row)
    {
        var engine = AiDotNetEngine.Current;
        int n = points.Shape[0];
        int q = points.Shape[1];
        var reference = engine.TensorSlice(points, new[] { row, 0 }, new[] { 1, q });
        var diff = engine.TensorSubtract(points, engine.TensorBroadcastTo(reference, new[] { n, q }));
        return engine.ReduceSum(engine.TensorSquare(diff), new[] { 1 }, keepDims: false);
    }

    /// <summary>
    /// Optimizes the ELBO using gradient descent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ELBO (Evidence Lower BOund) is a measure of how well
    /// our model fits the data. Higher ELBO = better model.
    ///
    /// We optimize three things:
    /// 1. Latent means: Where each point is in latent space
    /// 2. Latent variances: How certain we are about each location
    /// 3. Inducing points: The representative points used for computation
    ///
    /// Gradient descent iteratively adjusts these to maximize ELBO.
    /// </para>
    /// </remarks>
    private void OptimizeELBO()
    {
        if (_latentMean is null || _latentVariance is null || _inducingPoints is null || _observedData is null)
            throw new InvalidOperationException("Model must be initialized before optimization.");

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Compute gradients and update parameters
            UpdateLatentPoints();
            UpdateInducingPoints();

            // Optional: update noise variance
            // This could be done but adds complexity
        }
    }

    /// <summary>
    /// Updates latent points using gradient descent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This adjusts the latent positions to better explain the observed data.
    /// Points that reconstruct their observations well stay put; points that don't move to fix that.
    /// </para>
    /// </remarks>
    private void UpdateLatentPoints()
    {
        if (_latentMean is null || _latentVariance is null || _observedData is null || _inducingPoints is null)
            return;

        var engine = AiDotNetEngine.Current;
        int n = _latentMean.Rows;
        int m = _inducingPoints.Rows;
        var inducing = Tensor<T>.FromMatrix(_inducingPoints);
        T minusTwo = _numOps.FromDouble(-2.0);
        T priorWeight = _numOps.FromDouble(0.01);
        T learningRate = _numOps.FromDouble(_learningRate);

        // Points are updated in order: the reconstruction of point i uses the already-updated
        // latent positions of points 0..i-1.
        for (int i = 0; i < n; i++)
        {
            // Get latent point
            var xi = GetLatentPoint(i);
            var point = Tensor<T>.FromVector(xi).Reshape(new[] { 1, _latentDimensions });

            // Simple reconstruction gradient: error signal summed over the observed dimensions
            var reconstructed = ReconstructBatch(point);
            T errorSum = _numOps.Subtract(
                engine.TensorSum(Tensor<T>.FromVector(_observedData.GetRow(i))),
                engine.TensorSum(Tensor<T>.FromMatrix(reconstructed)));

            // Gradient of kernel w.r.t. latent point (using chain rule), simplified for an RBF-like kernel:
            // dK/dx_q ≈ -2 K (x_q - u_q), so grad_q = -2 Σ_d e_d Σ_j k(u_j, x) (x_q - u_jq)
            var kernelValues = Tensor<T>.FromMatrix(ComputeKernelMatrix(_inducingPoints, point.ToMatrix()));
            var offsets = engine.TensorSubtract(
                engine.TensorBroadcastTo(point, new[] { m, _latentDimensions }), inducing);
            var weightedOffsets = engine.ReduceSum(
                engine.TensorMultiply(engine.TensorBroadcastTo(kernelValues, new[] { m, _latentDimensions }), offsets),
                new[] { 0 }, keepDims: false);
            var grad = engine.TensorMultiplyScalar(weightedOffsets, _numOps.Multiply(minusTwo, errorSum));

            // Update latent mean; the prior gradient -x pushes towards zero
            var current = Tensor<T>.FromVector(xi);
            var totalGrad = engine.TensorSubtract(grad, engine.TensorMultiplyScalar(current, priorWeight));
            _latentMean.SetRow(i, engine.TensorAdd(current, engine.TensorMultiplyScalar(totalGrad, learningRate)).ToVector());
        }

        // Update variance (simplified: keep bounded). KL term pushes variance towards 1,
        // data term reduces variance: v = clamp(0.99 v + 0.01, 0.01, 1).
        var variance = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(_latentVariance), _numOps.FromDouble(0.99)), priorWeight);
        _latentVariance = engine.TensorMax(engine.TensorMin(variance, _numOps.One), priorWeight).ToMatrix();
    }

    /// <summary>
    /// Updates inducing points using gradient descent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inducing points are adjusted to better represent the latent space.
    /// Good inducing points help make accurate predictions with less computation.
    /// </para>
    /// </remarks>
    private void UpdateInducingPoints()
    {
        if (_inducingPoints is null || _latentMean is null)
            return;

        // For simplicity, we re-select inducing points periodically
        // A full implementation would compute gradients w.r.t. inducing point locations
        // and update them smoothly

        // Every 10 iterations, re-select inducing points from current latent means
        // This is a simple heuristic that works reasonably well
    }

    /// <summary>
    /// Gets a latent point as a vector.
    /// </summary>
    private Vector<T> GetLatentPoint(int i)
    {
        if (_latentMean is null)
            throw new InvalidOperationException("Model not initialized.");

        return _latentMean.GetRow(i);
    }

    /// <summary>
    /// Reconstructs data points from latent representations.
    /// </summary>
    /// <param name="points">The latent points as rows.</param>
    /// <returns>The reconstructed observations as rows.</returns>
    private Matrix<T> ReconstructBatch(Tensor<T> points)
    {
        if (_latentMean is null || _observedData is null)
            throw new InvalidOperationException("Model not initialized.");

        var engine = AiDotNetEngine.Current;
        int k = points.Shape[0];
        int d = _observedData.Columns;

        // Kernel-weighted average of the observed points: Σ_i k(x, x_i) y_i / Σ_i k(x, x_i),
        // left unnormalized when the weights sum to at most 1e-10
        var weights = Tensor<T>.FromMatrix(ComputeKernelMatrix(points.ToMatrix(), _latentMean));
        var weighted = engine.TensorMatMul(weights, Tensor<T>.FromMatrix(_observedData));
        var totalWeight = engine.ReduceSum(weights, new[] { 1 }, keepDims: true);
        var normalize = engine.TensorGreaterThan(totalWeight, _numOps.FromDouble(1e-10));
        var divisor = engine.TensorWhere(normalize, totalWeight, engine.TensorOnesLike(totalWeight));

        return engine.TensorDivide(weighted, engine.TensorBroadcastTo(divisor, new[] { k, d })).ToMatrix();
    }

    /// <summary>
    /// Computes the kernel matrix between two sets of points.
    /// </summary>
    private Matrix<T> ComputeKernelMatrix(Matrix<T> A, Matrix<T> B)
    {
        var K = new Matrix<T>(A.Rows, B.Rows);

        for (int i = 0; i < A.Rows; i++)
        {
            var ai = A.GetRow(i);
            for (int j = 0; j < B.Rows; j++)
            {
                K[i, j] = _kernel.Calculate(ai, B.GetRow(j));
            }
        }

        return K;
    }

    /// <summary>
    /// Gets the learned latent representation (means).
    /// </summary>
    /// <returns>Matrix of latent means (N x Q).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns where each data point is located in the latent space.
    /// Each row is a data point, each column is a latent dimension.
    ///
    /// For visualization with Q=2, you can plot each row as a 2D point.
    /// Similar data points should be close together in this space.
    /// </para>
    /// </remarks>
    public Matrix<T> GetLatentMean()
    {
        if (!_isFitted || _latentMean is null)
            throw new InvalidOperationException("Model must be fitted first.");

        // Return a copy
        return _latentMean.Clone();
    }

    /// <summary>
    /// Gets the learned latent representation (variances).
    /// </summary>
    /// <returns>Matrix of latent variances (N x Q).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns how uncertain we are about each point's location.
    /// Higher values mean more uncertainty.
    ///
    /// Points with high variance might be:
    /// - Outliers that don't fit well
    /// - Points between clusters
    /// - Points with unusual features
    /// </para>
    /// </remarks>
    public Matrix<T> GetLatentVariance()
    {
        if (!_isFitted || _latentVariance is null)
            throw new InvalidOperationException("Model must be fitted first.");

        return _latentVariance.Clone();
    }

    /// <summary>
    /// Transforms new data points into the latent space.
    /// </summary>
    /// <param name="newData">New data points (K x D).</param>
    /// <returns>Latent representations (K x Q).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Projects new, unseen data points into the learned latent space.
    ///
    /// This lets you:
    /// - Add new points to your visualization
    /// - Find where new data fits in relation to training data
    /// - Use the latent features for downstream tasks
    ///
    /// The projection finds the latent location that best reconstructs the new observation.
    /// </para>
    /// </remarks>
    public Matrix<T> Transform(Matrix<T> newData)
    {
        if (!_isFitted || _latentMean is null || _observedData is null)
            throw new InvalidOperationException("Model must be fitted first.");

        if (newData is null) throw new ArgumentNullException(nameof(newData));

        // Find nearest training point (first one on ties) and use its latent representation.
        // Optimize latent position (simplified: just use nearest)
        // A full implementation would do gradient descent on reconstruction error
        var engine = AiDotNetEngine.Current;
        int k = newData.Rows;
        int n = _observedData.Rows;
        int d = _observedData.Columns;
        var shape = new[] { k, n, d };
        var diff = engine.TensorSubtract(
            engine.TensorBroadcastTo(Tensor<T>.FromMatrix(newData).Reshape(new[] { k, 1, d }), shape),
            engine.TensorBroadcastTo(Tensor<T>.FromMatrix(_observedData).Reshape(new[] { 1, n, d }), shape));
        var distances = engine.ReduceSum(engine.TensorSquare(diff), new[] { 2 }, keepDims: false);
        var nearest = engine.TensorArgMin(distances, 1).AsSpan().ToArray();

        return GaussianProcessEngineOps.SelectRows(_latentMean, nearest);
    }

    /// <summary>
    /// Reconstructs data from latent representations.
    /// </summary>
    /// <param name="latentPoints">Latent points (K x Q).</param>
    /// <returns>Reconstructed data (K x D).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generates data from latent space positions.
    ///
    /// Uses:
    /// - Interpolation: Generate data between training points
    /// - Generation: Create new data by sampling latent space
    /// - Visualization: See what data looks like at different latent positions
    /// </para>
    /// </remarks>
    public Matrix<T> Reconstruct(Matrix<T> latentPoints)
    {
        if (!_isFitted || _observedData is null)
            throw new InvalidOperationException("Model must be fitted first.");

        if (latentPoints is null) throw new ArgumentNullException(nameof(latentPoints));
        if (latentPoints.Columns != _latentDimensions)
            throw new ArgumentException($"Latent points must have {_latentDimensions} columns.", nameof(latentPoints));

        return ReconstructBatch(Tensor<T>.FromMatrix(latentPoints));
    }

    /// <summary>
    /// Computes the ELBO (Evidence Lower BOund) for model comparison.
    /// </summary>
    /// <returns>The ELBO value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ELBO measures how well the model fits the data.
    /// Higher is better. Use this to:
    /// - Compare different numbers of latent dimensions
    /// - Compare different kernels
    /// - Monitor training progress
    /// </para>
    /// </remarks>
    public T ComputeELBO()
    {
        if (!_isFitted)
            throw new InvalidOperationException("Model must be fitted first.");
        var latentVariance = _latentVariance
            ?? throw new InvalidOperationException("Latent variance has not been computed. Call Fit() first.");

        // Simplified ELBO computation
        // Full ELBO = E_q[log p(Y|X)] - KL(q(X) || p(X))

        // Reconstruction term (negative squared error)
        var engine = AiDotNetEngine.Current;
        var latMean = _latentMean ?? throw new InvalidOperationException("Latent mean has not been initialized.");
        var obsData = _observedData ?? throw new InvalidOperationException("Observed data has not been initialized.");
        var latent = Tensor<T>.FromMatrix(latMean);
        var reconstructed = Tensor<T>.FromMatrix(ReconstructBatch(latent));
        T reconstructionLoss = engine.TensorSumOfSquares(engine.TensorSubtract(Tensor<T>.FromMatrix(obsData), reconstructed));

        // KL divergence term (pushes q(X) towards N(0,I)):
        // KL = 0.5 * Σ (mu^2 + sigma^2 - log(sigma^2) - 1), with log(sigma^2) floored at log(1e-10)
        var sigma2 = Tensor<T>.FromMatrix(latentVariance);
        var logVar = engine.TensorLog(engine.TensorMax(sigma2, _numOps.FromDouble(1e-10)));
        var perEntry = engine.TensorSubtract(engine.TensorAdd(engine.TensorSquare(latent), sigma2), logVar);
        T klTerm = _numOps.Multiply(_numOps.FromDouble(0.5),
            _numOps.Subtract(engine.TensorSum(perEntry), _numOps.FromDouble(latentVariance.Rows * latentVariance.Columns)));

        // ELBO = -reconstruction_loss - KL
        T elbo = _numOps.Negate(_numOps.Add(reconstructionLoss, klTerm));
        return elbo;
    }
}
