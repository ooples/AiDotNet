using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Kernel PCA reconstruction error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Kernel PCA extends PCA to capture non-linear patterns by mapping data
/// to a higher-dimensional feature space using a kernel function. Anomalies have high
/// reconstruction error when projected back from this space.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute kernel matrix (e.g., RBF kernel)
/// 2. Center the kernel matrix
/// 3. Extract principal components in kernel space
/// 4. Compute reconstruction error for each point
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Non-linear relationships in data
/// - When linear PCA doesn't capture patterns well
/// - Complex, curved cluster structures
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Kernel: RBF (Gaussian)
/// - Gamma: 1/n_features (auto)
/// - Number of components: 0.95 (95% variance)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Schölkopf, B., Smola, A., Müller, K.R. (1998). "Nonlinear Component Analysis
/// as a Kernel Eigenvalue Problem." Neural Computation.
/// </para>
/// </remarks>
public class KernelPCADetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _gamma;
    private readonly double _varianceRatio;
    private readonly KernelType _kernel;
    private double[][]? _trainingData;
    private double[,]? _kernelMatrix;
    private double[][]? _alphas; // Eigenvectors in kernel space
    private double[]? _lambdas; // Eigenvalues
    private int _nComponents;
    private double[]? _mean; // For centering

    /// <summary>
    /// Type of kernel function.
    /// </summary>
    public enum KernelType
    {
        /// <summary>RBF (Gaussian) kernel.</summary>
        RBF,
        /// <summary>Polynomial kernel.</summary>
        Polynomial,
        /// <summary>Linear kernel (equivalent to standard PCA).</summary>
        Linear
    }

    /// <summary>
    /// Gets the gamma parameter for RBF kernel.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the kernel type.
    /// </summary>
    public KernelType Kernel => _kernel;

    /// <summary>
    /// Creates a new Kernel PCA anomaly detector.
    /// </summary>
    /// <param name="gamma">
    /// Gamma parameter for RBF kernel. -1 means auto (1/n_features). Default is -1.
    /// </param>
    /// <param name="varianceRatio">
    /// Fraction of variance to retain (0-1). Default is 0.95 (95%).
    /// </param>
    /// <param name="kernel">Kernel type. Default is RBF.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public KernelPCADetector(double gamma = -1, double varianceRatio = 0.95,
        KernelType kernel = KernelType.RBF, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (varianceRatio <= 0 || varianceRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(varianceRatio),
                "VarianceRatio must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.95.");
        }

        _gamma = gamma;
        _varianceRatio = varianceRatio;
        _kernel = kernel;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        // Convert to double array
        _trainingData = new double[n][];
        for (int i = 0; i < n; i++)
        {
            _trainingData[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                _trainingData[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Set gamma if auto
        double effectiveGamma = _gamma > 0 ? _gamma : 1.0 / d;

        // Compute kernel matrix
        _kernelMatrix = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double k = ComputeKernel(_trainingData[i], _trainingData[j], effectiveGamma);
                _kernelMatrix[i, j] = k;
                _kernelMatrix[j, i] = k;
            }
        }

        // Center the kernel matrix
        CenterKernelMatrix(n);

        // Eigendecomposition (power iteration for top components)
        var (eigenvalues, eigenvectors) = ComputeEigendecomposition(n);

        // Select components based on variance ratio
        double totalVariance = eigenvalues.Sum();
        double cumulativeVariance = 0;
        _nComponents = 0;

        for (int i = 0; i < eigenvalues.Length; i++)
        {
            cumulativeVariance += eigenvalues[i];
            _nComponents++;
            if (cumulativeVariance / totalVariance >= _varianceRatio) break;
        }

        _lambdas = eigenvalues.Take(_nComponents).ToArray();
        _alphas = eigenvectors.Take(_nComponents).ToArray();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void CenterKernelMatrix(int n)
    {
        // K_c = K - 1_n K - K 1_n + 1_n K 1_n
        // where 1_n is matrix with all entries 1/n

        _mean = new double[n];
        double grandMean = 0;

        // Compute row means
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _mean[i] += _kernelMatrix![i, j];
            }
            _mean[i] /= n;
            grandMean += _mean[i];
        }
        grandMean /= n;

        // Center
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _kernelMatrix![i, j] = _kernelMatrix[i, j] - _mean[i] - _mean[j] + grandMean;
            }
        }
    }

    private (double[] eigenvalues, double[][] eigenvectors) ComputeEigendecomposition(int n)
    {
        int maxComponents = Math.Min(n, 50); // Limit for efficiency
        var eigenvalues = new List<double>();
        var eigenvectors = new List<double[]>();

        var matrix = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i, j] = _kernelMatrix![i, j];

        for (int c = 0; c < maxComponents; c++)
        {
            // Power iteration
            var v = new double[n];
            for (int i = 0; i < n; i++)
                v[i] = _random.NextDouble();

            Normalize(v);

            for (int iter = 0; iter < 100; iter++)
            {
                var vNew = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        vNew[i] += matrix[i, j] * v[j];
                    }
                }

                Normalize(vNew);

                // Check convergence
                double diff = 0;
                for (int i = 0; i < n; i++)
                    diff += Math.Abs(vNew[i] - v[i]);

                v = vNew;
                if (diff < 1e-8) break;
            }

            // Compute eigenvalue
            double lambda = 0;
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                    sum += matrix[i, j] * v[j];
                lambda += v[i] * sum;
            }

            if (lambda < 1e-10) break;

            eigenvalues.Add(lambda);
            eigenvectors.Add(v);

            // Deflate matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    matrix[i, j] -= lambda * v[i] * v[j];
                }
            }
        }

        return (eigenvalues.ToArray(), eigenvectors.ToArray());
    }

    private void Normalize(double[] v)
    {
        double norm = Math.Sqrt(v.Sum(x => x * x));
        if (norm > 1e-10)
        {
            for (int i = 0; i < v.Length; i++)
                v[i] /= norm;
        }
    }

    private double ComputeKernel(double[] a, double[] b, double gamma)
    {
        switch (_kernel)
        {
            case KernelType.RBF:
                double sqDist = 0;
                for (int i = 0; i < a.Length; i++)
                {
                    double diff = a[i] - b[i];
                    sqDist += diff * diff;
                }
                return Math.Exp(-gamma * sqDist);

            case KernelType.Polynomial:
                double dot = 0;
                for (int i = 0; i < a.Length; i++)
                    dot += a[i] * b[i];
                return Math.Pow(dot + 1, 3); // Degree 3 polynomial

            case KernelType.Linear:
                double linear = 0;
                for (int i = 0; i < a.Length; i++)
                    linear += a[i] * b[i];
                return linear;

            default:
                return 0;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        // Capture nullable fields in local variables (guaranteed non-null after EnsureFitted)
        var trainingData = _trainingData;
        var mean = _mean;
        var alphas = _alphas;
        var lambdas = _lambdas;

        if (trainingData == null || mean == null || alphas == null || lambdas == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int nTrain = trainingData.Length;
        double effectiveGamma = _gamma > 0 ? _gamma : 1.0 / X.Columns;

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // Compute kernel vector with training data
            var kVec = new double[nTrain];
            for (int t = 0; t < nTrain; t++)
            {
                kVec[t] = ComputeKernel(point, trainingData[t], effectiveGamma);
            }

            // Center the kernel vector
            double kMean = kVec.Average();
            double grandMean = mean.Average();
            for (int t = 0; t < nTrain; t++)
            {
                kVec[t] = kVec[t] - kMean - mean[t] + grandMean;
            }

            // Project to kernel PCA space and reconstruct
            double reconstructionError = ComputeKernel(point, point, effectiveGamma);

            for (int c = 0; c < _nComponents; c++)
            {
                // Projection coefficient
                double proj = 0;
                for (int t = 0; t < nTrain; t++)
                {
                    proj += alphas[c][t] * kVec[t];
                }

                // Subtract reconstructed variance
                reconstructionError -= proj * proj / lambdas[c];
            }

            scores[i] = NumOps.FromDouble(Math.Max(0, reconstructionError));
        }

        return scores;
    }
}
