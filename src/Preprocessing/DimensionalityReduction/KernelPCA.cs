using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Kernel Principal Component Analysis for non-linear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Kernel PCA is an extension of PCA that uses a kernel function to map
/// data into a higher-dimensional feature space where non-linear relationships
/// become linear, then performs standard PCA in that space.
/// </para>
/// <para>
/// This allows capturing non-linear relationships that standard PCA cannot.
/// Common kernels include RBF (Gaussian), polynomial, and sigmoid.
/// </para>
/// <para><b>For Beginners:</b> Regular PCA finds straight-line patterns.
/// Kernel PCA can find curved patterns by mathematically "bending" the data:
/// - RBF kernel: Good for data with clusters or blobs
/// - Polynomial: Good for polynomial relationships
/// - Linear: Same as regular PCA
///
/// Think of it as finding principal components in a transformed space.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KernelPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly KernelType _kernel;
    private readonly double _gamma;
    private readonly double _degree;
    private readonly double _coef0;
    private readonly bool _fitInverseTransform;
    private readonly double _alpha;

    // Fitted parameters
    private Tensor<T>? _trainingData;
    private Tensor<T>? _alphas; // Eigenvectors in kernel space, scaled by 1/sqrt(lambda)
    private Vector<T>? _lambdas; // Eigenvalues
    private Tensor<T>? _inverseTransformMatrix;
    private Tensor<T>? _kernelColumnMeans; // [1, n] column means of the training kernel
    private T _kernelMean = MathHelper.GetNumericOperations<T>().Zero; // grand mean of the training kernel
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the kernel type.
    /// </summary>
    public KernelType Kernel => _kernel;

    /// <summary>
    /// Gets the gamma parameter for RBF and polynomial kernels.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the eigenvalues.
    /// </summary>
    public Vector<T>? Lambdas => _lambdas;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => _fitInverseTransform;

    /// <summary>
    /// Creates a new instance of <see cref="KernelPCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. Defaults to 2.</param>
    /// <param name="kernel">The kernel function to use. Defaults to RBF.</param>
    /// <param name="gamma">Kernel coefficient for RBF/poly/sigmoid. Defaults to 1.0.</param>
    /// <param name="degree">Polynomial degree. Defaults to 3.</param>
    /// <param name="coef0">Independent term in polynomial/sigmoid. Defaults to 1.0.</param>
    /// <param name="fitInverseTransform">Whether to learn inverse transformation. Defaults to false.</param>
    /// <param name="alpha">Regularization for inverse transform. Defaults to 1e-3.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public KernelPCA(
        int nComponents = 2,
        KernelType kernel = KernelType.RBF,
        double gamma = 1.0,
        double degree = 3.0,
        double coef0 = 1.0,
        bool fitInverseTransform = false,
        double alpha = 1e-3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _kernel = kernel;
        _gamma = gamma;
        _degree = degree;
        _coef0 = coef0;
        _fitInverseTransform = fitInverseTransform;
        _alpha = alpha;
    }

    /// <summary>
    /// Fits KernelPCA by computing the kernel matrix and its eigenvectors.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int k = Math.Min(_nComponents, n);

        var x = Tensor<T>.FromMatrix(data);
        _trainingData = x;

        // Compute and center the kernel matrix (training statistics are kept to center new rows the same way)
        var kernel = ComputeKernelMatrix(x, x);
        _kernelColumnMeans = engine.ReduceMean(kernel, new[] { 0 }, true);
        _kernelMean = engine.TensorMean(kernel);
        var centered = CenterKernelMatrix(kernel);

        // Leading eigenpairs of the centered kernel matrix
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(centered, descending: true);
        var lambdas = engine.TensorClampMin(engine.TensorNarrow(eigenvalues, 0, 0, k), NumOps.Zero);

        // Normalize eigenvectors: alpha = v / sqrt(lambda) (0 for lambda ~ 0)
        var scale = engine.TensorWhere(
            engine.TensorGreaterThan(lambdas, NumOps.FromDouble(1e-10)),
            engine.TensorRsqrt(lambdas),
            engine.TensorZerosLike(lambdas));

        _lambdas = lambdas.ToVector();
        _alphas = DimensionalityReductionOps.ScaleColumns(DimensionalityReductionOps.FirstColumns(eigenvectors, k), scale);

        // Fit inverse transform if requested
        if (_fitInverseTransform)
        {
            FitInverseTransform(x);
        }
    }

    private Tensor<T> ComputeKernelMatrix(Tensor<T> x1, Tensor<T> x2)
    {
        var engine = Engine;
        switch (_kernel)
        {
            case KernelType.Linear:
                return DimensionalityReductionOps.MatMulTranspose(x1, x2);

            case KernelType.RBF:
                return engine.TensorExp(engine.TensorMultiplyScalar(
                    DimensionalityReductionOps.SquaredDistances(x1, x2), NumOps.FromDouble(-_gamma)));

            case KernelType.Polynomial:
                return engine.TensorPower(
                    GammaDotPlusCoef0(x1, x2), NumOps.FromDouble(_degree));

            case KernelType.Sigmoid:
                return engine.TensorTanh(GammaDotPlusCoef0(x1, x2));

            default:
                throw new ArgumentException($"Unknown kernel type: {_kernel}");
        }
    }

    /// <summary>gamma * x1 x2^T + coef0.</summary>
    private Tensor<T> GammaDotPlusCoef0(Tensor<T> x1, Tensor<T> x2)
    {
        var engine = Engine;
        return engine.TensorAddScalar(
            engine.TensorMultiplyScalar(DimensionalityReductionOps.MatMulTranspose(x1, x2), NumOps.FromDouble(_gamma)),
            NumOps.FromDouble(_coef0));
    }

    /// <summary>
    /// Centers a kernel matrix in feature space against the training kernel:
    /// K - row means of K - training column means + training grand mean (scikit-learn's KernelCenterer).
    /// For the training kernel itself this is the usual double centering.
    /// </summary>
    private Tensor<T> CenterKernelMatrix(Tensor<T> kernel)
    {
        if (_kernelColumnMeans is null)
        {
            throw new InvalidOperationException("KernelPCA has not been fitted.");
        }

        var engine = Engine;
        var rowMeans = engine.ReduceMean(kernel, new[] { 1 }, true);
        return engine.TensorAddScalar(
            engine.TensorSubtract(engine.TensorSubtract(kernel, rowMeans), _kernelColumnMeans), _kernelMean);
    }

    private void FitInverseTransform(Tensor<T> x)
    {
        if (_alphas is null)
        {
            return;
        }

        var engine = Engine;
        int k = _alphas.Shape[1];

        // Get transformed training data
        var transformed = TransformInternal(x);

        // Solve the ridge system (T^T T + alpha I) W = T^T X for the linear pre-image map
        var ata = engine.TensorAdd(
            DimensionalityReductionOps.TransposeMatMul(transformed, transformed),
            engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(k), NumOps.FromDouble(_alpha)));
        var atx = DimensionalityReductionOps.TransposeMatMul(transformed, x);

        _inverseTransformMatrix = DimensionalityReductionOps.Solve(ata, atx);
    }

    private Tensor<T> TransformInternal(Tensor<T> x)
    {
        if (_trainingData is null || _alphas is null)
        {
            throw new InvalidOperationException("KernelPCA has not been fitted.");
        }

        // Kernel between new points and training points, centered like the training kernel
        var kernel = CenterKernelMatrix(ComputeKernelMatrix(x, _trainingData));

        // Project onto principal components
        return Engine.TensorMatMul(kernel, _alphas);
    }

    /// <summary>
    /// Transforms the data using kernel PCA.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        return TransformInternal(Tensor<T>.FromMatrix(data)).ToMatrix();
    }

    /// <summary>
    /// Transforms data back to approximate original space.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Approximate reconstruction in original space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (!_fitInverseTransform || _inverseTransformMatrix is null)
        {
            throw new NotSupportedException(
                "Inverse transform was not fitted. Set fitInverseTransform=true in constructor.");
        }

        int k = _inverseTransformMatrix.Shape[0];
        var scores = DimensionalityReductionOps.FirstColumns(Tensor<T>.FromMatrix(data), k);
        return Engine.TensorMatMul(scores, _inverseTransformMatrix).ToMatrix();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _alphas?.Shape[1] ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"KPC{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the kernel type for Kernel PCA.
/// </summary>
public enum KernelType
{
    /// <summary>
    /// Linear kernel: K(x, y) = x · y
    /// </summary>
    Linear,

    /// <summary>
    /// Radial Basis Function (Gaussian): K(x, y) = exp(-γ||x-y||²)
    /// </summary>
    RBF,

    /// <summary>
    /// Polynomial kernel: K(x, y) = (γ(x · y) + c₀)^d
    /// </summary>
    Polynomial,

    /// <summary>
    /// Sigmoid (hyperbolic tangent): K(x, y) = tanh(γ(x · y) + c₀)
    /// </summary>
    Sigmoid
}
