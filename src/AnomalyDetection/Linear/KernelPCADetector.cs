using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Kernel)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Nonlinear Component Analysis as a Kernel Eigenvalue Problem", "https://doi.org/10.1162/089976698300017467", Year = 1998, Authors = "Bernhard Scholkopf, Alexander Smola, Klaus-Robert Muller")]
public class KernelPCADetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _gamma;
    private readonly double _varianceRatio;
    private readonly KernelType _kernel;
    private Matrix<T>? _trainingData;
    private Matrix<T>? _kernelMatrix;
    private Matrix<T>? _alphas; // Eigenvectors in kernel space (nComponents x n)
    private Vector<T>? _lambdas; // Eigenvalues
    private int _nComponents;
    private Vector<T>? _mean; // For centering

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

        _trainingData = X;

        // Set gamma if auto
        T effectiveGamma = _gamma > 0 ? NumOps.FromDouble(_gamma) : NumOps.Divide(NumOps.One, NumOps.FromDouble(d));

        // Compute kernel matrix
        _kernelMatrix = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            var pointI = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());
            for (int j = i; j < n; j++)
            {
                var pointJ = new Vector<T>(X.GetRowReadOnlySpan(j).ToArray());
                T k = ComputeKernel(pointI, pointJ, effectiveGamma);
                _kernelMatrix[i, j] = k;
                _kernelMatrix[j, i] = k;
            }
        }

        // Center the kernel matrix
        CenterKernelMatrix(n);

        // Eigendecomposition (power iteration for top components)
        var (eigenvalues, eigenvectors) = ComputeEigendecomposition(n);

        // Select components based on variance ratio
        T totalVariance = NumOps.Zero;
        for (int i = 0; i < eigenvalues.Length; i++)
            totalVariance = NumOps.Add(totalVariance, eigenvalues[i]);

        T cumulativeVariance = NumOps.Zero;
        T varThreshold = NumOps.FromDouble(_varianceRatio);
        _nComponents = 0;

        for (int i = 0; i < eigenvalues.Length; i++)
        {
            cumulativeVariance = NumOps.Add(cumulativeVariance, eigenvalues[i]);
            _nComponents++;
            if (NumOps.GreaterThan(totalVariance, NumOps.Zero) &&
                !NumOps.LessThan(NumOps.Divide(cumulativeVariance, totalVariance), varThreshold)) break;
        }

        _lambdas = new Vector<T>(_nComponents);
        _alphas = new Matrix<T>(_nComponents, n);
        for (int i = 0; i < _nComponents; i++)
        {
            _lambdas[i] = eigenvalues[i];
            for (int j = 0; j < n; j++)
            {
                _alphas[i, j] = eigenvectors[i][j];
            }
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void CenterKernelMatrix(int n)
    {
        var km = _kernelMatrix ?? throw new InvalidOperationException("Kernel matrix not computed.");
        T nT = NumOps.FromDouble(n);

        _mean = new Vector<T>(n);
        T grandMean = NumOps.Zero;

        // Compute row means
        for (int i = 0; i < n; i++)
        {
            T rowSum = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                rowSum = NumOps.Add(rowSum, km[i, j]);
            }
            _mean[i] = NumOps.Divide(rowSum, nT);
            grandMean = NumOps.Add(grandMean, _mean[i]);
        }
        grandMean = NumOps.Divide(grandMean, nT);

        // Center: K_c[i,j] = K[i,j] - mean[i] - mean[j] + grandMean
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                km[i, j] = NumOps.Add(NumOps.Subtract(NumOps.Subtract(km[i, j], _mean[i]), _mean[j]), grandMean);
            }
        }
    }

    private (Vector<T> eigenvalues, Vector<T>[] eigenvectors) ComputeEigendecomposition(int n)
    {
        int maxComponents = Math.Min(n, 50);
        var eigenvalues = new List<T>();
        var eigenvectors = new List<Vector<T>>();
        var km = _kernelMatrix ?? throw new InvalidOperationException("Kernel matrix not computed.");

        var matrix = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i, j] = km[i, j];

        T eps8 = NumOps.FromDouble(1e-8);
        T eps10 = NumOps.FromDouble(1e-10);

        for (int c = 0; c < maxComponents; c++)
        {
            // Power iteration
            var v = new Vector<T>(n);
            for (int i = 0; i < n; i++)
                v[i] = NumOps.FromDouble(_random.NextDouble());

            NormalizeVector(v);

            for (int iter = 0; iter < 100; iter++)
            {
                var vNew = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < n; j++)
                        sum = NumOps.Add(sum, NumOps.Multiply(matrix[i, j], v[j]));
                    vNew[i] = sum;
                }

                NormalizeVector(vNew);

                T diff = NumOps.Zero;
                for (int i = 0; i < n; i++)
                    diff = NumOps.Add(diff, NumOps.Abs(NumOps.Subtract(vNew[i], v[i])));

                v = vNew;
                if (NumOps.LessThan(diff, eps8)) break;
            }

            // Compute eigenvalue
            T lambda = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < n; j++)
                    sum = NumOps.Add(sum, NumOps.Multiply(matrix[i, j], v[j]));
                lambda = NumOps.Add(lambda, NumOps.Multiply(v[i], sum));
            }

            if (NumOps.LessThan(lambda, eps10)) break;

            eigenvalues.Add(lambda);
            eigenvectors.Add(v);

            // Deflate matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    matrix[i, j] = NumOps.Subtract(matrix[i, j], NumOps.Multiply(lambda, NumOps.Multiply(v[i], v[j])));
                }
            }
        }

        return (new Vector<T>(eigenvalues), eigenvectors.ToArray());
    }

    private void NormalizeVector(Vector<T> v)
    {
        T normSq = NumOps.Zero;
        for (int i = 0; i < v.Length; i++)
            normSq = NumOps.Add(normSq, NumOps.Multiply(v[i], v[i]));
        T norm = NumOps.Sqrt(normSq);
        T eps = NumOps.FromDouble(1e-10);
        if (NumOps.GreaterThan(norm, eps))
        {
            for (int i = 0; i < v.Length; i++)
                v[i] = NumOps.Divide(v[i], norm);
        }
    }

    private T ComputeKernel(Vector<T> a, Vector<T> b, T gamma)
    {
        switch (_kernel)
        {
            case KernelType.RBF:
                var diff = Engine.Subtract(a, b);
                T distSq = Engine.DotProduct(diff, diff);
                return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, distSq)));

            case KernelType.Polynomial:
                T dot = Engine.DotProduct(a, b);
                return NumOps.Power(NumOps.Add(dot, NumOps.One), NumOps.FromDouble(3));

            case KernelType.Linear:
                return Engine.DotProduct(a, b);

            default:
                return NumOps.Zero;
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

        int nTrain = trainingData.Rows;
        T effectiveGamma = _gamma > 0
            ? NumOps.FromDouble(_gamma)
            : NumOps.Divide(NumOps.One, NumOps.FromDouble(X.Columns));

        var scores = new Vector<T>(X.Rows);
        T eps10 = NumOps.FromDouble(1e-10);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            // Compute kernel vector with training data
            var kVec = new Vector<T>(nTrain);
            for (int t = 0; t < nTrain; t++)
            {
                var trainPt = new Vector<T>(trainingData.GetRowReadOnlySpan(t).ToArray());
                kVec[t] = ComputeKernel(point, trainPt, effectiveGamma);
            }

            // Center the kernel vector
            T kMean = NumOps.Zero;
            T grandMean = NumOps.Zero;
            for (int t = 0; t < nTrain; t++)
            {
                kMean = NumOps.Add(kMean, kVec[t]);
                grandMean = NumOps.Add(grandMean, mean[t]);
            }
            T nTrainT = NumOps.FromDouble(nTrain);
            kMean = NumOps.Divide(kMean, nTrainT);
            grandMean = NumOps.Divide(grandMean, nTrainT);

            for (int t = 0; t < nTrain; t++)
            {
                kVec[t] = NumOps.Add(NumOps.Subtract(NumOps.Subtract(kVec[t], kMean), mean[t]), grandMean);
            }

            // Project to kernel PCA space and compute scores
            T reconstructionError = ComputeKernel(point, point, effectiveGamma);
            T mahalanobis = NumOps.Zero;

            for (int c = 0; c < _nComponents; c++)
            {
                // Projection coefficient
                T proj = NumOps.Zero;
                for (int t = 0; t < nTrain; t++)
                {
                    proj = NumOps.Add(proj, NumOps.Multiply(alphas[c, t], kVec[t]));
                }

                if (NumOps.GreaterThan(lambdas[c], eps10))
                {
                    T projSq = NumOps.Multiply(proj, proj);
                    T projSqOverLambda = NumOps.Divide(projSq, lambdas[c]);
                    reconstructionError = NumOps.Subtract(reconstructionError, projSqOverLambda);
                    mahalanobis = NumOps.Add(mahalanobis, projSqOverLambda);
                }
            }

            // Combined score: Mahalanobis distance + reconstruction error
            T combined = NumOps.Add(mahalanobis,
                NumOps.GreaterThan(reconstructionError, NumOps.Zero) ? reconstructionError : NumOps.Zero);
            scores[i] = NumOps.Sqrt(combined);
        }

        return scores;
    }
}
