using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Hessian Locally Linear Embedding for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Hessian LLE is an improvement over standard LLE that estimates the Hessian
/// (second derivative) of the embedding function. It uses a quadratic form to
/// measure local curvature and produces more globally coherent embeddings.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k-nearest neighbors for each point
/// 2. Compute local Hessian estimator using quadratic polynomials
/// 3. Build global Hessian matrix
/// 4. Find embedding by minimizing Hessian-based cost function
/// </para>
/// <para><b>For Beginners:</b> Hessian LLE improves on LLE by:
/// - Using curvature information (second derivatives)
/// - Producing more faithful global embeddings
/// - Better handling of manifolds with varying curvature
///
/// Use cases:
/// - When standard LLE produces distorted embeddings
/// - Manifolds with non-uniform curvature
/// - When you need more accurate distance preservation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HessianLLE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="HessianLLE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Should be > (n_components * (n_components + 3) / 2). Defaults to 10.</param>
    /// <param name="randomState">Kept for API compatibility; the eigen solver is deterministic.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public HessianLLE(
        int nComponents = 2,
        int nNeighbors = 10,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        // Hessian LLE requires k >= d(d+3)/2 + 1
        int minNeighbors = nComponents * (nComponents + 3) / 2 + 1;
        if (nNeighbors < minNeighbors)
        {
            throw new ArgumentException(
                $"For {nComponents} components, at least {minNeighbors} neighbors are required.",
                nameof(nNeighbors));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
    }

    /// <summary>
    /// Fits Hessian LLE and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nSamples = data.Rows;
        int n = data.Rows;
        int k = Math.Min(_nNeighbors, n - 1);
        int d = _nComponents;

        // Number of Hessian estimator columns: d(d+1)/2
        int dp = d * (d + 1) / 2;
        int nCols = 1 + d + dp;
        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Find k-nearest neighbors (ties in index order)
        var neighbors = DimensionalityReductionOps.NearestNeighbors(DimensionalityReductionOps.EuclideanDistances(x, x), k);

        // Step 2: Centered neighbourhoods [n, k, p]
        var rows = DimensionalityReductionOps.NeighborhoodRows(x, neighbors);
        var centered = engine.TensorSubtract(rows, engine.ReduceMean(rows, new[] { 1 }, true));

        // Local PCA: top d eigenvectors of X_i^T X_i, then tangent coordinates Y_i = X_i V_i [n, k, d]
        var scatter = DimensionalityReductionOps.BatchTransposeMatMul(centered, centered);
        var (_, principal) = DimensionalityReductionOps.SymmetricEigenEach(scatter, descending: true);
        var tangent = engine.TensorBatchMatMul(centered, engine.TensorNarrow(principal, 2, 0, d));

        // Hessian estimator G = [1, y_1..y_d, y_a y_b for a <= b] [n, k, nCols]
        var first = new int[dp];
        var second = new int[dp];
        int col = 0;
        for (int d1 = 0; d1 < d; d1++)
        {
            for (int d2 = d1; d2 < d; d2++)
            {
                first[col] = d1;
                second[col] = d2;
                col++;
            }
        }

        var quadratic = engine.TensorMultiply(
            engine.TensorIndexSelect(tangent, DimensionalityReductionOps.IndexTensor(first), 2),
            engine.TensorIndexSelect(tangent, DimensionalityReductionOps.IndexTensor(second), 2));
        var estimator = engine.TensorConcatenate(
            new[] { DimensionalityReductionOps.Full(new[] { n, k, 1 }, NumOps.One), tangent, quadratic }, 2);

        // Orthonormal factor Q of G (Gram-Schmidt); the columns after [1, y] span the Hessian part
        var q = DimensionalityReductionOps.OrthonormalizeColumnsEach(estimator);
        int available = Math.Min(k, nCols);
        Tensor<T> h;
        if (available > 1 + d)
        {
            var nullSpace = engine.TensorNarrow(q, 2, 1 + d, available - 1 - d);
            var local = DimensionalityReductionOps.BatchMatMulTranspose(nullSpace, nullSpace);
            h = DimensionalityReductionOps.ScatterAddBlocks(neighbors, local, n);
        }
        else
        {
            h = new Tensor<T>(new[] { n, n });
        }

        // Step 3: Eigenvectors of H in ascending eigenvalue order
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(h, descending: false);

        // Step 4: Create embedding, skipping the null eigenvector
        var columns = new int[_nComponents];
        for (int dd = 0; dd < _nComponents; dd++)
        {
            columns[dd] = dd + 1 < n ? dd + 1 : dd;
        }

        _embedding = DimensionalityReductionOps.SelectColumns(
            eigenvectors, DimensionalityReductionOps.IndexTensor(columns)).ToMatrix();
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("HessianLLE has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "HessianLLE does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HessianLLE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"HLLE{i + 1}";
        }
        return names;
    }
}
