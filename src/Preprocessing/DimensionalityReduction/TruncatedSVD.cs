using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Truncated Singular Value Decomposition for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// TruncatedSVD performs dimensionality reduction by computing the truncated
/// singular value decomposition. Unlike PCA, it does not center the data,
/// making it suitable for sparse matrices (e.g., TF-IDF from text).
/// </para>
/// <para>
/// Also known as Latent Semantic Analysis (LSA) when applied to document-term matrices.
/// </para>
/// <para><b>For Beginners:</b> SVD is similar to PCA but:
/// - Doesn't center the data (preserves sparsity in sparse matrices)
/// - Can be more memory-efficient for large sparse datasets
/// - Often used for text analysis (finding hidden topics in documents)
///
/// Example: In text analysis, TruncatedSVD can find that "car" and "automobile"
/// are related even if they never appear together in the same document.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TruncatedSVD<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _randomState;
    private readonly int _nIterations;

    // Fitted parameters
    private Matrix<T>? _components; // Right singular vectors (V^T)
    private Vector<T>? _singularValues;
    private Vector<T>? _explainedVariance;
    private Vector<T>? _explainedVarianceRatio;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of iterations for randomized SVD.
    /// </summary>
    public int NIterations => _nIterations;

    /// <summary>
    /// Gets the components (right singular vectors).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the singular values.
    /// </summary>
    public Vector<T>? SingularValues => _singularValues;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public Vector<T>? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the explained variance ratio for each component.
    /// </summary>
    public Vector<T>? ExplainedVarianceRatio => _explainedVarianceRatio;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="TruncatedSVD{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. Defaults to 2.</param>
    /// <param name="nIterations">Iteration budget kept for API compatibility (the decomposition is exact). Defaults to 5.</param>
    /// <param name="randomState">Random seed for reproducibility. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public TruncatedSVD(
        int nComponents = 2,
        int nIterations = 5,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nIterations < 1)
        {
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));
        }

        _nComponents = nComponents;
        _randomState = randomState;
        _nIterations = nIterations;
    }

    /// <summary>
    /// Fits TruncatedSVD by computing singular value decomposition.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        var x = Tensor<T>.FromMatrix(data);

        // Right singular vectors are the eigenvectors of X^T X; the singular values are sqrt(eigenvalues)
        var gram = DimensionalityReductionOps.TransposeMatMul(x, x);
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(gram, descending: true);
        // Signs as the original deflated power iteration converges to them from its seeded start vectors
        eigenvectors = DimensionalityReductionOps.MatchStartSigns(eigenvectors,
            DimensionalityReductionOps.PowerIterationStarts<T>(RandomHelper.CreateSeededRandom(_randomState), k, p, 0.5));
        var explainedVariance = engine.TensorClampMin(engine.TensorNarrow(eigenvalues, 0, 0, k), NumOps.Zero);

        T totalVariance = engine.TensorSum(engine.TensorSquare(x));
        var explainedVarianceRatio = NumOps.GreaterThan(totalVariance, NumOps.FromDouble(1e-10))
            ? engine.TensorDivideScalar(explainedVariance, totalVariance)
            : new Tensor<T>(new[] { k });

        _components = engine.TensorTranspose(DimensionalityReductionOps.FirstColumns(eigenvectors, k)).ToMatrix();
        _singularValues = engine.TensorSqrt(explainedVariance).ToVector();
        _explainedVariance = explainedVariance.ToVector();
        _explainedVarianceRatio = explainedVarianceRatio.ToVector();
    }

    /// <summary>
    /// Transforms the data by projecting onto singular vectors.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_components is null)
        {
            throw new InvalidOperationException("TruncatedSVD has not been fitted.");
        }

        return DimensionalityReductionOps.MatMulTranspose(Tensor<T>.FromMatrix(data), Tensor<T>.FromMatrix(_components)).ToMatrix();
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Data in original feature space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_components is null)
        {
            throw new InvalidOperationException("TruncatedSVD has not been fitted.");
        }

        int k = _components.Rows;
        var scores = DimensionalityReductionOps.FirstColumns(Tensor<T>.FromMatrix(data), k);
        return Engine.TensorMatMul(scores, Tensor<T>.FromMatrix(_components)).ToMatrix();
    }
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.Rows ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"Component{i + 1}";
        }
        return names;
    }
}
