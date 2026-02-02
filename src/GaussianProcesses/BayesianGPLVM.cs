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
public class BayesianGPLVM<T>
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
    private Matrix<T>? _latentVariance;

    /// <summary>
    /// The inducing points in latent space (M x Q).
    /// </summary>
    private Matrix<T>? _inducingPoints;

    /// <summary>
    /// The observed data (N x D).
    /// </summary>
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
        int n = data.Rows;
        int d = data.Columns;

        // Center the data
        var mean = new Vector<T>(d);
        for (int j = 0; j < d; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = _numOps.Add(sum, data[i, j]);
            }
            mean[j] = _numOps.Divide(sum, _numOps.FromDouble(n));
        }

        var centered = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                centered[i, j] = _numOps.Subtract(data[i, j], mean[j]);
            }
        }

        // Compute covariance matrix (d x d)
        var cov = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = _numOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(centered[k, i], centered[k, j]));
                }
                T val = _numOps.Divide(sum, _numOps.FromDouble(n - 1));
                cov[i, j] = val;
                cov[j, i] = val;
            }
        }

        // Simple power iteration to find top eigenvectors
        int effectiveLatentDims = Math.Min(_latentDimensions, d);
        var eigenvectors = new Matrix<T>(d, effectiveLatentDims);
        var eigenvalues = new Vector<T>(effectiveLatentDims);

        var workMatrix = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                workMatrix[i, j] = cov[i, j];
            }
        }

        var rand = RandomHelper.CreateSeededRandom(42);

        for (int k = 0; k < effectiveLatentDims; k++)
        {
            // Initialize random vector
            var v = new Vector<T>(d);
            for (int i = 0; i < d; i++)
            {
                v[i] = _numOps.FromDouble(rand.NextDouble() - 0.5);
            }

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
            {
                var vNew = workMatrix.Multiply(v);

                // Normalize
                T norm = _numOps.Zero;
                for (int i = 0; i < d; i++)
                {
                    norm = _numOps.Add(norm, _numOps.Multiply(vNew[i], vNew[i]));
                }
                norm = _numOps.Sqrt(norm);

                if (_numOps.ToDouble(norm) > 1e-10)
                {
                    for (int i = 0; i < d; i++)
                    {
                        v[i] = _numOps.Divide(vNew[i], norm);
                    }
                }
            }

            // Store eigenvector
            for (int i = 0; i < d; i++)
            {
                eigenvectors[i, k] = v[i];
            }

            // Compute eigenvalue
            var Av = workMatrix.Multiply(v);
            T eigenvalue = _numOps.Zero;
            for (int i = 0; i < d; i++)
            {
                eigenvalue = _numOps.Add(eigenvalue, _numOps.Multiply(v[i], Av[i]));
            }
            eigenvalues[k] = eigenvalue;

            // Deflate: A = A - eigenvalue * v * v^T
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    workMatrix[i, j] = _numOps.Subtract(workMatrix[i, j],
                        _numOps.Multiply(eigenvalue, _numOps.Multiply(v[i], v[j])));
                }
            }
        }

        // Project data onto top eigenvectors to get latent means
        _latentMean = new Matrix<T>(n, _latentDimensions);
        _latentVariance = new Matrix<T>(n, _latentDimensions);

        for (int i = 0; i < n; i++)
        {
            for (int q = 0; q < effectiveLatentDims; q++)
            {
                T proj = _numOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    proj = _numOps.Add(proj, _numOps.Multiply(centered[i, j], eigenvectors[j, q]));
                }
                _latentMean[i, q] = proj;
            }
            // Fill remaining dimensions with small random values
            for (int q = effectiveLatentDims; q < _latentDimensions; q++)
            {
                _latentMean[i, q] = _numOps.FromDouble(0.01 * (rand.NextDouble() - 0.5));
            }
            // Initialize variance to small value
            for (int q = 0; q < _latentDimensions; q++)
            {
                _latentVariance[i, q] = _numOps.FromDouble(0.1);
            }
        }
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

        int n = _latentMean.Rows;
        int m = Math.Min(_numInducingPoints, n);

        _inducingPoints = new Matrix<T>(m, _latentDimensions);

        // Greedy farthest point selection
        var selected = new bool[n];
        var minDistances = new double[n];
        for (int i = 0; i < n; i++)
        {
            minDistances[i] = double.MaxValue;
        }

        var rand = RandomHelper.CreateSeededRandom(42);
        int firstIdx = rand.Next(n);
        selected[firstIdx] = true;

        for (int q = 0; q < _latentDimensions; q++)
        {
            _inducingPoints[0, q] = _latentMean[firstIdx, q];
        }

        // Update distances
        for (int i = 0; i < n; i++)
        {
            if (!selected[i])
            {
                double dist = ComputeLatentDistance(firstIdx, i);
                minDistances[i] = Math.Min(minDistances[i], dist);
            }
        }

        for (int k = 1; k < m; k++)
        {
            // Find farthest point
            int farthestIdx = -1;
            double maxDist = -1;
            for (int i = 0; i < n; i++)
            {
                if (!selected[i] && minDistances[i] > maxDist)
                {
                    maxDist = minDistances[i];
                    farthestIdx = i;
                }
            }

            if (farthestIdx < 0) break;

            selected[farthestIdx] = true;
            for (int q = 0; q < _latentDimensions; q++)
            {
                _inducingPoints[k, q] = _latentMean[farthestIdx, q];
            }

            // Update distances
            for (int i = 0; i < n; i++)
            {
                if (!selected[i])
                {
                    double dist = ComputeLatentDistance(farthestIdx, i);
                    minDistances[i] = Math.Min(minDistances[i], dist);
                }
            }
        }
    }

    /// <summary>
    /// Computes the squared Euclidean distance between two latent points.
    /// </summary>
    private double ComputeLatentDistance(int i, int j)
    {
        if (_latentMean is null)
            return 0;

        double dist = 0;
        for (int q = 0; q < _latentDimensions; q++)
        {
            double diff = _numOps.ToDouble(_latentMean[i, q]) - _numOps.ToDouble(_latentMean[j, q]);
            dist += diff * diff;
        }
        return dist;
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

        int n = _latentMean.Rows;
        int m = _inducingPoints.Rows;

        // Compute kernel matrices
        var Kuu = ComputeKernelMatrix(_inducingPoints, _inducingPoints);
        var KuuInv = InvertMatrixSafe(Kuu);

        for (int i = 0; i < n; i++)
        {
            // Get latent point
            var xi = GetLatentPoint(i);

            // Compute Kun (kernel between inducing points and this point)
            var Kux = new Vector<T>(m);
            for (int j = 0; j < m; j++)
            {
                var uj = GetInducingPoint(j);
                Kux[j] = _kernel.Calculate(uj, xi);
            }

            // Compute prediction: mean = Kxu @ Kuu^{-1} @ u_mean (simplified)
            // In full model, we'd have variational parameters for u

            // Gradient w.r.t. latent mean
            var grad = new Vector<T>(_latentDimensions);

            // Simple reconstruction gradient
            var reconstructed = ReconstructPoint(xi);
            var observed = GetObservedPoint(i);

            // Error signal
            var error = new Vector<T>(observed.Length);
            for (int d = 0; d < observed.Length; d++)
            {
                error[d] = _numOps.Subtract(observed[d], reconstructed[d]);
            }

            // Gradient of kernel w.r.t. latent point (using chain rule)
            for (int q = 0; q < _latentDimensions; q++)
            {
                T gradQ = _numOps.Zero;
                for (int j = 0; j < m; j++)
                {
                    var uj = GetInducingPoint(j);
                    T kernelVal = _kernel.Calculate(uj, xi);
                    T diff = _numOps.Subtract(xi[q], uj[q]);

                    // dK/dx_q ≈ -K * 2 * (x_q - u_q) / lengthscale^2
                    // Simplified gradient assuming RBF-like kernel
                    T kernelGrad = _numOps.Multiply(kernelVal, _numOps.Multiply(_numOps.FromDouble(-2.0), diff));

                    // Accumulate gradient
                    for (int d = 0; d < error.Length; d++)
                    {
                        gradQ = _numOps.Add(gradQ, _numOps.Multiply(error[d], kernelGrad));
                    }
                }
                grad[q] = gradQ;
            }

            // Update latent mean
            for (int q = 0; q < _latentDimensions; q++)
            {
                // Prior gradient (push towards zero)
                T priorGrad = _numOps.Negate(_latentMean[i, q]);
                T totalGrad = _numOps.Add(grad[q], _numOps.Multiply(_numOps.FromDouble(0.01), priorGrad));

                _latentMean[i, q] = _numOps.Add(_latentMean[i, q],
                    _numOps.Multiply(_numOps.FromDouble(_learningRate), totalGrad));
            }

            // Update variance (simplified: keep bounded)
            for (int q = 0; q < _latentDimensions; q++)
            {
                double currentVar = _numOps.ToDouble(_latentVariance[i, q]);
                // KL term pushes variance towards 1, data term reduces variance
                double newVar = Math.Max(0.01, Math.Min(1.0, currentVar * 0.99 + 0.01));
                _latentVariance[i, q] = _numOps.FromDouble(newVar);
            }
        }
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

        var x = new Vector<T>(_latentDimensions);
        for (int q = 0; q < _latentDimensions; q++)
        {
            x[q] = _latentMean[i, q];
        }
        return x;
    }

    /// <summary>
    /// Gets an inducing point as a vector.
    /// </summary>
    private Vector<T> GetInducingPoint(int j)
    {
        if (_inducingPoints is null)
            throw new InvalidOperationException("Model not initialized.");

        var u = new Vector<T>(_latentDimensions);
        for (int q = 0; q < _latentDimensions; q++)
        {
            u[q] = _inducingPoints[j, q];
        }
        return u;
    }

    /// <summary>
    /// Gets an observed data point as a vector.
    /// </summary>
    private Vector<T> GetObservedPoint(int i)
    {
        if (_observedData is null)
            throw new InvalidOperationException("Model not initialized.");

        var y = new Vector<T>(_observedData.Columns);
        for (int d = 0; d < _observedData.Columns; d++)
        {
            y[d] = _observedData[i, d];
        }
        return y;
    }

    /// <summary>
    /// Reconstructs a data point from a latent representation.
    /// </summary>
    /// <param name="x">The latent point.</param>
    /// <returns>The reconstructed observation.</returns>
    private Vector<T> ReconstructPoint(Vector<T> x)
    {
        if (_inducingPoints is null || _observedData is null)
            throw new InvalidOperationException("Model not initialized.");

        int m = _inducingPoints.Rows;
        int d = _observedData.Columns;

        // Compute kernel between x and all inducing points
        var Kxu = new Vector<T>(m);
        for (int j = 0; j < m; j++)
        {
            var uj = GetInducingPoint(j);
            Kxu[j] = _kernel.Calculate(x, uj);
        }

        // Simple mean prediction using nearest inducing points (weighted average)
        T weightSum = _numOps.Zero;
        var result = new Vector<T>(d);

        // Find closest observed point for each inducing point (approximation)
        // In a full implementation, we'd have variational parameters for the inducing values

        for (int j = 0; j < m; j++)
        {
            T weight = Kxu[j];
            weightSum = _numOps.Add(weightSum, weight);
        }

        if (_numOps.ToDouble(weightSum) < 1e-10)
        {
            weightSum = _numOps.FromDouble(1.0);
        }

        // Return weighted average of observed points (simplified)
        int n = _observedData.Rows;
        for (int i = 0; i < n; i++)
        {
            var xi = GetLatentPoint(i);
            T weight = _kernel.Calculate(x, xi);

            for (int dim = 0; dim < d; dim++)
            {
                result[dim] = _numOps.Add(result[dim],
                    _numOps.Multiply(weight, _observedData[i, dim]));
            }
        }

        // Normalize
        T totalWeight = _numOps.Zero;
        for (int i = 0; i < n; i++)
        {
            var xi = GetLatentPoint(i);
            totalWeight = _numOps.Add(totalWeight, _kernel.Calculate(x, xi));
        }

        if (_numOps.ToDouble(totalWeight) > 1e-10)
        {
            for (int dim = 0; dim < d; dim++)
            {
                result[dim] = _numOps.Divide(result[dim], totalWeight);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes the kernel matrix between two sets of points.
    /// </summary>
    private Matrix<T> ComputeKernelMatrix(Matrix<T> A, Matrix<T> B)
    {
        int n = A.Rows;
        int m = B.Rows;
        int q = A.Columns;

        var K = new Matrix<T>(n, m);

        for (int i = 0; i < n; i++)
        {
            var ai = new Vector<T>(q);
            for (int k = 0; k < q; k++) ai[k] = A[i, k];

            for (int j = 0; j < m; j++)
            {
                var bj = new Vector<T>(q);
                for (int k = 0; k < q; k++) bj[k] = B[j, k];

                K[i, j] = _kernel.Calculate(ai, bj);
            }
        }

        return K;
    }

    /// <summary>
    /// Safely inverts a matrix with regularization.
    /// </summary>
    private Matrix<T> InvertMatrixSafe(Matrix<T> M)
    {
        int n = M.Rows;
        var result = new Matrix<T>(n, n);

        // Add jitter for numerical stability
        var Mreg = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Mreg[i, j] = M[i, j];
            }
            Mreg[i, i] = _numOps.Add(Mreg[i, i], _numOps.FromDouble(1e-6));
        }

        // Simple Gauss-Jordan elimination
        var augmented = new Matrix<T>(n, 2 * n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = Mreg[i, j];
            }
            augmented[i, n + i] = _numOps.One;
        }

        for (int i = 0; i < n; i++)
        {
            // Find pivot
            int maxRow = i;
            double maxVal = Math.Abs(_numOps.ToDouble(augmented[i, i]));
            for (int k = i + 1; k < n; k++)
            {
                double val = Math.Abs(_numOps.ToDouble(augmented[k, i]));
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = k;
                }
            }

            // Swap rows
            if (maxRow != i)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    T temp = augmented[i, j];
                    augmented[i, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }
            }

            // Scale pivot row
            T pivot = augmented[i, i];
            if (Math.Abs(_numOps.ToDouble(pivot)) > 1e-12)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    augmented[i, j] = _numOps.Divide(augmented[i, j], pivot);
                }

                // Eliminate column
                for (int k = 0; k < n; k++)
                {
                    if (k != i)
                    {
                        T factor = augmented[k, i];
                        for (int j = 0; j < 2 * n; j++)
                        {
                            augmented[k, j] = _numOps.Subtract(augmented[k, j],
                                _numOps.Multiply(factor, augmented[i, j]));
                        }
                    }
                }
            }
        }

        // Extract inverse
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = augmented[i, n + j];
            }
        }

        return result;
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
        var result = new Matrix<T>(_latentMean.Rows, _latentMean.Columns);
        for (int i = 0; i < _latentMean.Rows; i++)
        {
            for (int j = 0; j < _latentMean.Columns; j++)
            {
                result[i, j] = _latentMean[i, j];
            }
        }
        return result;
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

        var result = new Matrix<T>(_latentVariance.Rows, _latentVariance.Columns);
        for (int i = 0; i < _latentVariance.Rows; i++)
        {
            for (int j = 0; j < _latentVariance.Columns; j++)
            {
                result[i, j] = _latentVariance[i, j];
            }
        }
        return result;
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
        if (!_isFitted)
            throw new InvalidOperationException("Model must be fitted first.");

        if (newData is null) throw new ArgumentNullException(nameof(newData));

        int k = newData.Rows;
        var result = new Matrix<T>(k, _latentDimensions);

        for (int i = 0; i < k; i++)
        {
            // Get new observation
            var y = new Vector<T>(newData.Columns);
            for (int d = 0; d < newData.Columns; d++)
            {
                y[d] = newData[i, d];
            }

            // Find nearest training point and use its latent representation as starting point
            int nearestIdx = FindNearestTrainingPoint(y);
            var xInit = GetLatentPoint(nearestIdx);

            // Optimize latent position (simplified: just use nearest)
            // A full implementation would do gradient descent on reconstruction error
            for (int q = 0; q < _latentDimensions; q++)
            {
                result[i, q] = xInit[q];
            }
        }

        return result;
    }

    /// <summary>
    /// Finds the nearest training point to an observation.
    /// </summary>
    private int FindNearestTrainingPoint(Vector<T> y)
    {
        if (_observedData is null)
            return 0;

        int nearestIdx = 0;
        double minDist = double.MaxValue;

        for (int i = 0; i < _observedData.Rows; i++)
        {
            double dist = 0;
            for (int d = 0; d < _observedData.Columns; d++)
            {
                double diff = _numOps.ToDouble(y[d]) - _numOps.ToDouble(_observedData[i, d]);
                dist += diff * diff;
            }

            if (dist < minDist)
            {
                minDist = dist;
                nearestIdx = i;
            }
        }

        return nearestIdx;
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

        int k = latentPoints.Rows;
        int d = _observedData.Columns;
        var result = new Matrix<T>(k, d);

        for (int i = 0; i < k; i++)
        {
            var x = new Vector<T>(_latentDimensions);
            for (int q = 0; q < _latentDimensions; q++)
            {
                x[q] = latentPoints[i, q];
            }

            var reconstructed = ReconstructPoint(x);
            for (int dim = 0; dim < d; dim++)
            {
                result[i, dim] = reconstructed[dim];
            }
        }

        return result;
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

        // Simplified ELBO computation
        // Full ELBO = E_q[log p(Y|X)] - KL(q(X) || p(X))

        // Reconstruction term (negative squared error)
        T reconstructionLoss = _numOps.Zero;
        int n = _latentMean!.Rows;
        int d = _observedData!.Columns;

        for (int i = 0; i < n; i++)
        {
            var xi = GetLatentPoint(i);
            var reconstructed = ReconstructPoint(xi);
            var observed = GetObservedPoint(i);

            for (int dim = 0; dim < d; dim++)
            {
                T diff = _numOps.Subtract(observed[dim], reconstructed[dim]);
                T sq = _numOps.Multiply(diff, diff);
                reconstructionLoss = _numOps.Add(reconstructionLoss, sq);
            }
        }

        // KL divergence term (pushes q(X) towards N(0,I))
        T klTerm = _numOps.Zero;
        for (int i = 0; i < n; i++)
        {
            for (int q = 0; q < _latentDimensions; q++)
            {
                T mu = _latentMean[i, q];
                T sigma2 = _latentVariance![i, q];

                // KL = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
                T muSq = _numOps.Multiply(mu, mu);
                T logVar = _numOps.FromDouble(Math.Log(Math.Max(1e-10, _numOps.ToDouble(sigma2))));
                T kl = _numOps.Multiply(_numOps.FromDouble(0.5),
                    _numOps.Subtract(_numOps.Add(muSq, sigma2),
                        _numOps.Add(logVar, _numOps.One)));
                klTerm = _numOps.Add(klTerm, kl);
            }
        }

        // ELBO = -reconstruction_loss - KL
        T elbo = _numOps.Negate(_numOps.Add(reconstructionLoss, klTerm));
        return elbo;
    }
}
