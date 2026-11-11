using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Randomized PCA using randomized SVD for efficient computation.
/// </summary>
/// <remarks>
/// <para>
/// Randomized PCA uses randomized algorithms to efficiently compute principal components
/// without computing the full SVD. It's much faster than standard PCA for large datasets
/// while providing accurate approximations of the top components.
/// </para>
/// <para>
/// The algorithm:
/// 1. Generate random projection matrix
/// 2. Form sample matrix Y = A * Ω (project data to random subspace)
/// 3. Orthonormalize Y using QR decomposition
/// 4. Form B = Q^T * A (project to Q's range)
/// 5. Compute SVD of B to get principal components
/// </para>
/// <para><b>For Beginners:</b> Randomized PCA is faster because:
/// - It only computes the components you need (not all)
/// - Random projection preserves structure efficiently
/// - Power iteration improves accuracy if needed
/// - Works well for low-rank data
///
/// Use cases:
/// - Very large datasets where standard PCA is slow
/// - When you only need top few components
/// - Streaming or online scenarios
/// - Memory-constrained environments
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RandomizedPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nOversamples;
    private readonly int _nPowerIter;
    private readonly int? _randomState;

    // Fitted parameters
    private Matrix<T>? _components; // Principal components (d x p)
    private Vector<T>? _mean;
    private Vector<T>? _explainedVariance;
    private Vector<T>? _singularValues;
    private T _totalVariance = MathHelper.GetNumericOperations<T>().Zero;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the principal components (each row is a component).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public Vector<T>? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the singular values.
    /// </summary>
    public Vector<T>? SingularValues => _singularValues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="RandomizedPCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nOversamples">Additional random vectors for accuracy. Defaults to 10.</param>
    /// <param name="nPowerIter">Number of power iterations for accuracy. Defaults to 2.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public RandomizedPCA(
        int nComponents = 2,
        int nOversamples = 10,
        int nPowerIter = 2,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _nOversamples = nOversamples;
        _nPowerIter = nPowerIter;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Randomized PCA using randomized SVD.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;
        _nFeatures = p;

        // Validate minimum sample count (need at least 2 for variance calculation)
        if (n < 2)
        {
            throw new ArgumentException("RandomizedPCA requires at least 2 samples.", nameof(data));
        }

        // Validate that we can extract the requested number of components
        int maxComponents = Math.Min(n, p);
        if (_nComponents > maxComponents)
        {
            throw new ArgumentException(
                $"Cannot extract {_nComponents} components from data with dimensions {n}x{p}. " +
                $"Maximum available components: {maxComponents}.");
        }

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Center the data
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var xc = DimensionalityReductionOps.SubtractRow(x, mean);

        // Total variance: sum of the column sample variances
        _totalVariance = NumOps.Divide(engine.TensorSum(engine.TensorSquare(xc)), NumOps.FromDouble(n - 1));

        // Number of random projections (components + oversamples)
        int k = _nComponents + _nOversamples;
        k = Math.Min(k, Math.Min(n, p));

        // Random Gaussian test matrix Omega [p, k] and its range sample Y = Xc * Omega
        var omega = DimensionalityReductionOps.StandardNormal<T>(random, p, k);
        var y = engine.TensorMatMul(xc, omega);

        // Power iterations sharpen the spectrum: Y = Xc * orth(Xc^T * Y)
        for (int iter = 0; iter < _nPowerIter; iter++)
        {
            var z = DimensionalityReductionOps.OrthonormalizeColumns(DimensionalityReductionOps.TransposeMatMul(xc, y));
            y = engine.TensorMatMul(xc, z);
        }

        // Orthonormal basis of the range and the projected matrix B = Q^T * Xc [k, p]
        var q = DimensionalityReductionOps.OrthonormalizeColumns(y);
        var b = DimensionalityReductionOps.TransposeMatMul(q, xc);

        // SVD of the small matrix: right singular vectors are the eigenvectors of B^T B
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(
            DimensionalityReductionOps.TransposeMatMul(b, b), descending: true);
        // Signs as the original deflated power iteration converges to them from its seeded start vectors
        eigenvectors = DimensionalityReductionOps.MatchStartSigns(eigenvectors,
            DimensionalityReductionOps.PowerIterationStarts<T>(random, _nComponents, p, 0.5));
        var singularValues = engine.TensorSqrt(
            engine.TensorClampMin(engine.TensorNarrow(eigenvalues, 0, 0, _nComponents), NumOps.Zero));

        _mean = mean.ToVector();
        _components = engine.TensorTranspose(DimensionalityReductionOps.FirstColumns(eigenvectors, _nComponents)).ToMatrix();
        _singularValues = singularValues.ToVector();
        _explainedVariance = engine.TensorDivideScalar(engine.TensorSquare(singularValues), NumOps.FromDouble(n - 1)).ToVector();
    }

    /// <summary>
    /// Transforms data by projecting onto principal components.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_components is null || _mean is null)
        {
            throw new InvalidOperationException("RandomizedPCA has not been fitted.");
        }

        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {p} features but RandomizedPCA was fitted with {_nFeatures} features.",
                nameof(data));
        }

        var centered = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        return DimensionalityReductionOps.MatMulTranspose(centered, Tensor<T>.FromMatrix(_components)).ToMatrix();
    }

    /// <summary>
    /// Reconstructs data from the reduced representation.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_components is null || _mean is null)
        {
            throw new InvalidOperationException("RandomizedPCA has not been fitted.");
        }

        int d = data.Columns;

        if (d != _nComponents)
        {
            throw new ArgumentException(
                $"Input has {d} dimensions but RandomizedPCA was fitted with {_nComponents} components.",
                nameof(data));
        }

        var reconstructed = Engine.TensorMatMul(Tensor<T>.FromMatrix(data), Tensor<T>.FromMatrix(_components));
        return DimensionalityReductionOps.AddRow(reconstructed, Tensor<T>.FromVector(_mean)).ToMatrix();
    }

    /// <summary>
    /// Gets the total explained variance ratio (proportion of total variance explained by selected components).
    /// </summary>
    /// <returns>A value between 0 and 1 representing the fraction of total variance captured.</returns>
    public T GetExplainedVarianceRatio()
    {
        if (_explainedVariance is null)
        {
            throw new InvalidOperationException("RandomizedPCA has not been fitted.");
        }

        T explainedSum = Engine.TensorSum(Tensor<T>.FromVector(_explainedVariance));
        return NumOps.GreaterThan(_totalVariance, NumOps.Zero) ? NumOps.Divide(explainedSum, _totalVariance) : NumOps.Zero;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"RPCA{i + 1}";
        }
        return names;
    }
}
