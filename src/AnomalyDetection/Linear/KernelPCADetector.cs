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
// Scholkopf 1998 gives kernel PCA itself; the NOVELTY SCORE this detector reports -- the
// reconstruction error in feature space -- is Hoffmann's, and that is the rule the scoring
// semantics have to be judged against.
[ResearchPaper("Kernel PCA for Novelty Detection", "https://doi.org/10.1016/j.patcog.2006.07.009", Year = 2007, Authors = "Heiko Hoffmann")]
public partial class KernelPCADetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _gamma;
    private readonly double _varianceRatio;
    private readonly KernelType _kernel;
    [Buffer]
    private Matrix<T>? _trainingData;
    [Buffer]
    private Matrix<T>? _kernelMatrix;
    private Matrix<T>? _alphas; // Eigenvectors in kernel space (nComponents x n)
    private Vector<T>? _lambdas; // Eigenvalues
    private int _nComponents;
    [Buffer]
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

        // Compute kernel matrix and center it
        var engine = Engine;
        var data = AnomalyTensorOps.ToTensor(X);
        var (centeredKernel, rowMeans) = CenterKernelMatrix(ComputeKernel(data, data, effectiveGamma));
        _kernelMatrix = centeredKernel.ToMatrix();
        _mean = rowMeans.ToVector();

        // Eigendecomposition: the leading eigenpairs (at most 50) down to the first eigenvalue below 1e-10
        var (allValues, allVectors) = AnomalyTensorOps.SymmetricEigenDescending(centeredKernel);
        int maxComponents = Math.Min(n, 50);
        var leading = engine.TensorSlice(allValues, new[] { 0 }, new[] { maxComponents });
        var belowFloor = engine.TensorLessThan(leading, NumOps.FromDouble(1e-10));
        int kept = NumOps.GreaterThan(engine.TensorSum(belowFloor), NumOps.Zero)
            ? AnomalyTensorOps.FirstArgMax(belowFloor)
            : maxComponents;

        // Select components based on variance ratio: the first count whose cumulative eigenvalue sum
        // reaches the ratio of the total (all kept components when it never does)
        _nComponents = kept;
        if (kept > 0)
        {
            var eigenvalues = engine.TensorSlice(allValues, new[] { 0 }, new[] { kept });
            var cumulative = engine.TensorCumSum(eigenvalues, 0);
            T totalVariance = engine.TensorSum(engine.TensorSlice(cumulative, new[] { kept - 1 }, new[] { 1 }));
            if (NumOps.GreaterThan(totalVariance, NumOps.Zero))
            {
                var reached = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(
                    engine.TensorDivideScalar(cumulative, totalVariance), NumOps.FromDouble(_varianceRatio)));
                if (NumOps.GreaterThan(engine.TensorSum(reached), NumOps.Zero))
                {
                    _nComponents = AnomalyTensorOps.FirstArgMax(reached) + 1;
                }
            }
        }

        _lambdas = engine.TensorSlice(allValues, new[] { 0 }, new[] { _nComponents }).ToVector();
        _alphas = engine.TensorSlice(allVectors, new[] { 0, 0 }, new[] { _nComponents, n }).ToMatrix();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    // K_c[i,j] = K[i,j] - mean[i] - mean[j] + grandMean, with mean the row means and grandMean their mean.
    private (Tensor<T> centered, Tensor<T> rowMeans) CenterKernelMatrix(Tensor<T> kernel)
    {
        var engine = Engine;
        int n = kernel.Shape[0];
        var rowMeans = engine.ReduceMean(kernel, new[] { 1 }, false);
        T grandMean = engine.TensorMean(rowMeans);
        var centered = engine.TensorAddScalar(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.SubtractColumn(kernel, rowMeans), rowMeans), grandMean);
        return (centered, rowMeans);
    }

    // Kernel between the rows of a ([m, d]) and b ([n, d]).
    private Tensor<T> ComputeKernel(Tensor<T> a, Tensor<T> b, T gamma)
    {
        var engine = Engine;
        switch (_kernel)
        {
            case KernelType.RBF:
                return engine.TensorExp(engine.TensorNegate(
                    engine.TensorMultiplyScalar(AnomalyTensorOps.PairwiseSquaredDistances(a, b), gamma)));
            case KernelType.Polynomial:
                return engine.TensorPower(
                    engine.TensorAddScalar(engine.TensorMatMul(a, engine.TensorTranspose(b)), NumOps.One), NumOps.FromDouble(3));
            case KernelType.Linear:
                return engine.TensorMatMul(a, engine.TensorTranspose(b));
            default:
                return Tensor<T>.CreateZeros(new[] { a.Shape[0], b.Shape[0] });
        }
    }

    // k(x, x) for every row of a.
    private Tensor<T> SelfKernel(Tensor<T> a)
    {
        var engine = Engine;
        int m = a.Shape[0];
        var squaredNorm = engine.ReduceSum(engine.TensorSquare(a), new[] { 1 }, false);
        switch (_kernel)
        {
            case KernelType.RBF:
                return Tensor<T>.CreateOnes(new[] { m });
            case KernelType.Polynomial:
                return engine.TensorPower(engine.TensorAddScalar(squaredNorm, NumOps.One), NumOps.FromDouble(3));
            case KernelType.Linear:
                return squaredNorm;
            default:
                return Tensor<T>.CreateZeros(new[] { m });
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

        var engine = Engine;
        var training = AnomalyTensorOps.ToTensor(trainingData);
        var meanTensor = mean.AsTensor();
        T grandMean = engine.TensorMean(meanTensor);
        T two = NumOps.FromDouble(2);
        int k = _nComponents;
        Tensor<T>? inverse = null;
        Tensor<T>? alphasT = null;
        if (k > 0)
        {
            var lambdaTensor = lambdas.AsTensor();
            var significant = engine.TensorGreaterThan(lambdaTensor, NumOps.FromDouble(1e-10));
            inverse = engine.TensorWhere(significant,
                engine.TensorReciprocal(engine.TensorWhere(significant, lambdaTensor, Tensor<T>.CreateOnes(new[] { k }))),
                Tensor<T>.CreateZeros(new[] { k }));
            alphasT = engine.TensorTranspose(AnomalyTensorOps.ToTensor(alphas));
        }

        // Query rows are scored in bounded blocks: each block's kernel with the training data is
        // [block, nTrain], never the whole [m, nTrain] matrix
        var scores = AnomalyTensorOps.MapRowBlocks(AnomalyTensorOps.ToTensor(X), nTrain,
            points => ScoreBlock(points, training, meanTensor, grandMean, two, effectiveGamma, inverse, alphasT));
        return scores.ToVector();
    }

    // Kernel PCA reconstruction-error scores of one block of query rows.
    private Tensor<T> ScoreBlock(Tensor<T> points, Tensor<T> training, Tensor<T> meanTensor, T grandMean, T two,
        T effectiveGamma, Tensor<T>? inverse, Tensor<T>? alphasT)
    {
        var engine = Engine;

        // Kernel vectors with the training data, centered like the training kernel matrix
        var kVec = ComputeKernel(points, training, effectiveGamma);
        var kMean = engine.ReduceMean(kVec, new[] { 1 }, false);
        var centered = engine.TensorAddScalar(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.SubtractColumn(kVec, kMean), meanTensor), grandMean);

        // Reconstruction error in the CENTERED kernel feature space is the KPCA novelty score
        // (Hoffmann, "Kernel PCA for novelty detection", Pattern Recognition 40(3):863-874, 2007):
        // ||phi~(x)||^2 = k(x,x) - 2 mean(k_vec) + grandMean, minus the squared projections
        // beta_c^2 = (u_c . k~vec)^2 / lambda_c onto the unit principal components (lambda_c > 1e-10).
        // A point far from the training distribution cannot be reconstructed from the learned principal
        // subspace, so its residual is large -> higher score = more anomalous.
        var reconstructionError = engine.TensorAddScalar(
            engine.TensorSubtract(SelfKernel(points), engine.TensorMultiplyScalar(kMean, two)), grandMean);
        if (inverse is not null && alphasT is not null)
        {
            var proj = engine.TensorMatMul(centered, alphasT);
            reconstructionError = engine.TensorSubtract(reconstructionError, engine.ReduceSum(
                AnomalyTensorOps.MultiplyRow(engine.TensorSquare(proj), inverse), new[] { 1 }, false));
        }

        // The residual is non-negative in exact arithmetic; clamp tiny negative values from numerical
        // error before taking the square root for a distance-like, well-scaled score.
        return engine.TensorSqrt(engine.TensorMax(reconstructionError, NumOps.Zero));
    }
}
