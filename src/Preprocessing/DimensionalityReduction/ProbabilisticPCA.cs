using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Probabilistic Principal Component Analysis (PPCA).
/// </summary>
/// <remarks>
/// <para>
/// Probabilistic PCA is a probabilistic formulation of PCA that models data as being
/// generated from a lower-dimensional latent space with added Gaussian noise. This
/// provides a proper likelihood model, enables handling of missing data, and allows
/// for Bayesian extensions.
/// </para>
/// <para>
/// The model:
/// x = W * z + μ + ε
/// where z ~ N(0, I), ε ~ N(0, σ²I)
/// </para>
/// <para>
/// The algorithm:
/// 1. Center the data
/// 2. Use EM algorithm or closed-form solution to estimate W and σ²
/// 3. Project data to latent space using posterior mean
/// </para>
/// <para><b>For Beginners:</b> PPCA extends standard PCA by:
/// - Providing a probabilistic model for the data
/// - Estimating noise variance σ² separately
/// - Enabling computation of data likelihoods
/// - Handling missing values naturally
///
/// Use cases:
/// - When you need uncertainty estimates
/// - Data with missing values
/// - Model comparison using likelihoods
/// - Bayesian machine learning pipelines
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ProbabilisticPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly int? _randomState;

    // Fitted parameters
    private Matrix<T>? _W; // Loading matrix (p x d)
    private Vector<T>? _mean;
    private T _noiseVariance = MathHelper.GetNumericOperations<T>().Zero;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the loading matrix W.
    /// </summary>
    public Matrix<T>? LoadingMatrix => _W;

    /// <summary>
    /// Gets the estimated noise variance.
    /// </summary>
    public T NoiseVariance => _noiseVariance;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="ProbabilisticPCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="maxIter">Maximum number of EM iterations. Defaults to 100.</param>
    /// <param name="tol">Tolerance for convergence. Defaults to 1e-4.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public ProbabilisticPCA(
        int nComponents = 2,
        int maxIter = 100,
        double tol = 1e-4,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _maxIter = maxIter;
        _tol = tol;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Probabilistic PCA by the closed-form maximum-likelihood solution (Tipping and Bishop, 1999).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;
        _nFeatures = p;

        // Center the data
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var xc = DimensionalityReductionOps.SubtractRow(x, mean);

        // Maximum-likelihood sample covariance S = Xc^T Xc / n
        var s = engine.TensorDivideScalar(DimensionalityReductionOps.TransposeMatMul(xc, xc), NumOps.FromDouble(n));

        // Eigen-decomposition, eigenvalues descending and clamped at zero
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(s, descending: true);
        eigenvalues = engine.TensorClampMin(eigenvalues, NumOps.Zero);

        // Noise variance: mean of the discarded eigenvalues (at least 1e-10)
        T noiseVariance = NumOps.Zero;
        int numSmall = p - _nComponents;
        if (numSmall > 0)
        {
            noiseVariance = NumOps.Divide(
                engine.TensorSum(engine.TensorNarrow(eigenvalues, 0, _nComponents, numSmall)), NumOps.FromDouble(numSmall));
        }

        T floor = NumOps.FromDouble(1e-10);
        _noiseVariance = NumOps.GreaterThan(noiseVariance, floor) ? noiseVariance : floor;

        // W = U_d * sqrt(max(lambda_d - sigma^2, 1e-10))
        var leading = engine.TensorNarrow(eigenvalues, 0, 0, _nComponents);
        var scale = engine.TensorSqrt(engine.TensorClampMin(engine.TensorSubtractScalar(leading, _noiseVariance), floor));
        _W = DimensionalityReductionOps.ScaleColumns(DimensionalityReductionOps.FirstColumns(eigenvectors, _nComponents), scale).ToMatrix();
        _mean = mean.ToVector();
    }

    /// <summary>
    /// Transforms data by projecting to the latent space.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_W is null || _mean is null)
        {
            throw new InvalidOperationException("ProbabilisticPCA has not been fitted.");
        }

        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {p} features but PPCA was fitted with {_nFeatures} features.",
                nameof(data));
        }

        var engine = Engine;
        var w = Tensor<T>.FromMatrix(_W);
        var xc = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));

        // Posterior mean of the latent variables: M^-1 W^T (x - mu) with M = W^T W + sigma^2 I
        var m = engine.TensorAdd(
            DimensionalityReductionOps.TransposeMatMul(w, w),
            engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(_nComponents), _noiseVariance));
        var wtx = DimensionalityReductionOps.TransposeMatMul(w, engine.TensorTranspose(xc));
        return engine.TensorTranspose(DimensionalityReductionOps.Solve(m, wtx)).ToMatrix();
    }

    /// <summary>
    /// Reconstructs data from the latent space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_W is null || _mean is null)
        {
            throw new InvalidOperationException("ProbabilisticPCA has not been fitted.");
        }

        int d = data.Columns;

        if (d != _nComponents)
        {
            throw new ArgumentException(
                $"Input has {d} dimensions but PPCA was fitted with {_nComponents} components.",
                nameof(data));
        }

        var reconstructed = DimensionalityReductionOps.MatMulTranspose(Tensor<T>.FromMatrix(data), Tensor<T>.FromMatrix(_W));
        return DimensionalityReductionOps.AddRow(reconstructed, Tensor<T>.FromVector(_mean)).ToMatrix();
    }

    /// <summary>
    /// Computes the log-likelihood of the data under the model.
    /// </summary>
    /// <param name="data">The data to evaluate.</param>
    /// <returns>The log-likelihood (summed over the rows) under N(mu, W W^T + sigma^2 I).</returns>
    public T Score(Matrix<T> data)
    {
        if (_W is null || _mean is null)
        {
            throw new InvalidOperationException("ProbabilisticPCA has not been fitted.");
        }

        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;
        var w = Tensor<T>.FromMatrix(_W);

        // Model covariance C = W W^T + sigma^2 I
        var c = engine.TensorAdd(
            DimensionalityReductionOps.MatMulTranspose(w, w),
            engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(p), _noiseVariance));

        // log|C| from the eigenvalues of the symmetric positive-definite C
        var (eigenvalues, _) = DimensionalityReductionOps.SymmetricEigen(c, descending: true);
        T logDet = engine.TensorSum(engine.TensorLog(engine.TensorClampMin(eigenvalues, NumOps.FromDouble(1e-10))));

        // Mahalanobis terms (x - mu)^T C^-1 (x - mu) for every row
        var xc = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        var solved = DimensionalityReductionOps.Solve(c, engine.TensorTranspose(xc));
        T mahalanobis = engine.TensorSum(engine.TensorMultiply(engine.TensorTranspose(xc), solved));

        // sum_i [-p/2 log(2 pi) - 1/2 log|C| - 1/2 mahal_i]
        T logConstant = NumOps.Subtract(
            NumOps.FromDouble(-0.5 * p * Math.Log(2 * Math.PI)),
            NumOps.Multiply(NumOps.FromDouble(0.5), logDet));
        return NumOps.Subtract(
            NumOps.Multiply(NumOps.FromDouble(n), logConstant),
            NumOps.Multiply(NumOps.FromDouble(0.5), mahalanobis));
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"PPCA{i + 1}";
        }
        return names;
    }
}
