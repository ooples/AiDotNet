using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Local Tangent Space Alignment for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// LTSA is a manifold learning algorithm that uses local tangent spaces to compute
/// a global embedding. It estimates the tangent space at each point using PCA on
/// neighbors, then aligns these local coordinate systems.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k-nearest neighbors for each point
/// 2. Compute local tangent space via PCA on each neighborhood
/// 3. Compute local coordinates in tangent space
/// 4. Align tangent spaces by minimizing reconstruction error
/// </para>
/// <para><b>For Beginners:</b> LTSA is an improved version of LLE that:
/// - Uses tangent spaces (local linear approximations) at each point
/// - Better handles points with different local geometries
/// - More mathematically principled than standard LLE
///
/// Use cases:
/// - When standard LLE produces poor results
/// - Manifolds with varying curvature
/// - When you need more stable embeddings
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LTSA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int? _randomState;
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
    /// Creates a new instance of <see cref="LTSA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Defaults to 5.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public LTSA(
        int nComponents = 2,
        int nNeighbors = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nComponents = nComponents;
        _randomState = randomState;
        _nNeighbors = nNeighbors;
    }

    /// <summary>
    /// Fits LTSA and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nNeighbors, n - 1);
        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Find k-nearest neighbors (ties in index order); each neighbourhood includes the point itself first
        var neighbors = DimensionalityReductionOps.NearestNeighbors(DimensionalityReductionOps.EuclideanDistances(x, x), k);
        var self = engine.Reshape(DimensionalityReductionOps.Range(n), new[] { n, 1 });
        var neighborhood = engine.TensorConcatenate(new[] { self, neighbors }, 1);   // [n, k + 1]

        // Step 2: Centered neighbourhoods [n, k + 1, p]
        var rows = DimensionalityReductionOps.NeighborhoodRows(x, neighborhood);
        var centered = engine.TensorSubtract(rows, engine.ReduceMean(rows, new[] { 1 }, true));

        // Local tangent space: top principal directions of X_i^T X_i, coordinates theta_i = X_i V_i
        var scatter = DimensionalityReductionOps.BatchTransposeMatMul(centered, centered);
        var (_, principal) = DimensionalityReductionOps.SymmetricEigenEach(scatter, descending: true);
        var theta = engine.TensorBatchMatMul(centered, engine.TensorNarrow(principal, 2, 0, Math.Min(_nComponents, p)));

        // G_i = [1 / sqrt(k + 1), theta_i], orthonormalized; W_i = I - Q_i Q_i^T
        var ones = DimensionalityReductionOps.Full(new[] { n, k + 1, 1 }, NumOps.FromDouble(1.0 / Math.Sqrt(k + 1)));
        var g = engine.TensorConcatenate(new[] { ones, theta }, 2);
        var q = DimensionalityReductionOps.OrthonormalizeColumnsEach(g, unitFallback: false);
        var identity = engine.Reshape(DimensionalityReductionOps.Identity<T>(k + 1), new[] { 1, k + 1, k + 1 });
        var local = engine.TensorSubtract(identity, DimensionalityReductionOps.BatchMatMulTranspose(q, q));

        // Alignment matrix B[I, I] += W_i
        var b = DimensionalityReductionOps.ScatterAddBlocks(neighborhood, local, n);

        // Step 3: Eigenvectors of B in ascending eigenvalue order
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(b, descending: false);
        // Signs as the original deflated power iteration converges to them from its seeded start vectors
        // (on shift I - M, so ascending order here is their order)
        eigenvectors = DimensionalityReductionOps.MatchStartSigns(eigenvectors,
            DimensionalityReductionOps.SeededPowerIterationStarts<T>(_randomState ?? 42, Math.Min(n, _nComponents + 2), n, 0.5));

        // Step 4: Create embedding (skip first eigenvector)
        var columns = new int[_nComponents];
        for (int d = 0; d < _nComponents; d++)
        {
            columns[d] = d + 1 < n ? d + 1 : d;
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
            throw new InvalidOperationException("LTSA has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "LTSA does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LTSA does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"LTSA{i + 1}";
        }
        return names;
    }
}
