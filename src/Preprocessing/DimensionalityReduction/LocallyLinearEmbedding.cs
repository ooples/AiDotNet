using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Locally Linear Embedding for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// LLE preserves local neighborhood structure by representing each point
/// as a weighted linear combination of its neighbors. The embedding is found
/// by preserving these reconstruction weights in lower dimensions.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k nearest neighbors for each point
/// 2. Compute reconstruction weights that best reconstruct each point from neighbors
/// 3. Find low-dimensional embedding that preserves these weights
/// </para>
/// <para><b>For Beginners:</b> LLE preserves local relationships:
/// - Each point is described by its neighbors
/// - The weights describe "how much" each neighbor contributes
/// - The embedding keeps these relationships intact
/// - Good for unfolding curved manifolds (like the Swiss roll)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LocallyLinearEmbedding<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double _reg;
    private readonly LLEMethod _method;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private Tensor<T>? _reconstructionWeights;
    private int _nSamples;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the regularization parameter.
    /// </summary>
    public double Regularization => _reg;

    /// <summary>
    /// Gets the LLE method.
    /// </summary>
    public LLEMethod Method => _method;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LocallyLinearEmbedding{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Defaults to 5.</param>
    /// <param name="reg">Regularization constant. Defaults to 0.001.</param>
    /// <param name="method">LLE algorithm variant. Defaults to Standard.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public LocallyLinearEmbedding(
        int nComponents = 2,
        int nNeighbors = 5,
        double reg = 0.001,
        LLEMethod method = LLEMethod.Standard,
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
        _nNeighbors = nNeighbors;
        _reg = reg;
        _method = method;
    }

    /// <summary>
    /// Fits LLE by computing reconstruction weights and embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int k = _nNeighbors;

        if (k >= n)
        {
            throw new ArgumentException($"n_neighbors ({k}) must be less than n_samples ({n}).");
        }

        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Find k nearest neighbors (excluding the point itself, ties in index order)
        var neighbors = DimensionalityReductionOps.NearestNeighbors(DimensionalityReductionOps.EuclideanDistances(x, x), k);

        // Step 2: Compute reconstruction weights
        _reconstructionWeights = ComputeReconstructionWeights(x, neighbors, n, k);

        // Step 3: Compute embedding
        var embedding = _method switch
        {
            LLEMethod.Modified => ComputeModifiedLLEEmbedding(x, neighbors, n),
            LLEMethod.HLLE => ComputeHLLEEmbedding(x, neighbors, n, k),
            LLEMethod.LTSA => ComputeLTSAEmbedding(x, neighbors, n),
            _ => ComputeStandardLLEEmbedding(n)
        };
        _embedding = embedding.ToMatrix();
    }

    /// <summary>
    /// Reconstruction weights: solve (C_i + reg_i I) w = 1 with C_i[j, l] = (x_i - x_j) . (x_i - x_l),
    /// reg_i = max(reg * trace(C_i) / k, max(reg, 1e-10)), then normalize w to sum one.
    /// </summary>
    private Tensor<T> ComputeReconstructionWeights(Tensor<T> x, Tensor<int> neighbors, int n, int k)
    {
        var engine = Engine;
        var differences = DimensionalityReductionOps.NeighborhoodDifferences(x, neighbors);
        var covariances = DimensionalityReductionOps.BatchMatMulTranspose(differences, differences);

        var regularization = engine.TensorClampMin(
            engine.TensorMultiplyScalar(DimensionalityReductionOps.BatchTrace(covariances), NumOps.FromDouble(_reg / k)),
            NumOps.FromDouble(Math.Max(_reg, 1e-10)));
        var regularized = engine.TensorAdd(
            covariances,
            engine.TensorMultiply(
                engine.Reshape(regularization, new[] { n, 1, 1 }),
                engine.Reshape(DimensionalityReductionOps.Identity<T>(k), new[] { 1, k, k })));

        var weights = DimensionalityReductionOps.SolveEach(regularized, DimensionalityReductionOps.Full(new[] { n, k }, NumOps.One));
        var sums = DimensionalityReductionOps.RowSums(weights);
        var nonZero = engine.TensorGreaterThan(engine.TensorAbs(sums), DimensionalityReductionOps.Tolerance<T>(1e-10));
        var divisor = engine.TensorWhere(nonZero, sums, DimensionalityReductionOps.Full(n, NumOps.One));
        weights = engine.TensorDivide(weights, DimensionalityReductionOps.AsColumn(divisor));

        return DimensionalityReductionOps.ScatterRows(neighbors, weights, n, NumOps.Zero);
    }

    private Tensor<T> ComputeStandardLLEEmbedding(int n)
    {
        if (_reconstructionWeights is null)
        {
            throw new InvalidOperationException("Reconstruction weights have not been computed.");
        }

        // Compute M = (I - W)' * (I - W)
        var iMinusW = Engine.TensorSubtract(DimensionalityReductionOps.Identity<T>(n), _reconstructionWeights);
        return SmallestNonTrivialEigenvectors(DimensionalityReductionOps.TransposeMatMul(iMinusW, iMinusW), n);
    }

    /// <summary>
    /// Modified LLE: per neighbourhood, W_i = I - U U^T where U holds the top d + 1 left singular vectors of the
    /// centered neighbourhood (vectors over the k neighbours), accumulated into M.
    /// </summary>
    private Tensor<T> ComputeModifiedLLEEmbedding(Tensor<T> x, Tensor<int> neighbors, int n)
    {
        var engine = Engine;
        int k = neighbors.Shape[1];
        var centered = CenteredNeighborhoods(x, neighbors);

        // Left singular vectors of X_i are the eigenvectors of X_i X_i^T
        int count = Math.Min(_nComponents + 1, Math.Min(k, x.Shape[1]));
        var (_, left) = DimensionalityReductionOps.SymmetricEigenEach(
            DimensionalityReductionOps.BatchMatMulTranspose(centered, centered), descending: true);
        var basis = engine.TensorNarrow(left, 2, 0, count);

        var identity = engine.Reshape(DimensionalityReductionOps.Identity<T>(k), new[] { 1, k, k });
        var local = engine.TensorSubtract(identity, DimensionalityReductionOps.BatchMatMulTranspose(basis, basis));
        return SmallestNonTrivialEigenvectors(DimensionalityReductionOps.ScatterAddBlocks(neighbors, local, n), n);
    }

    /// <summary>
    /// Hessian LLE: per neighbourhood, Y_i = [1, t, t_a t_b (a &lt;= b)] on the tangent coordinates t, and the
    /// projector onto the orthogonal complement of span(Y_i), I - Q Q^T, is accumulated into H.
    /// </summary>
    private Tensor<T> ComputeHLLEEmbedding(Tensor<T> x, Tensor<int> neighbors, int n, int k)
    {
        var engine = Engine;
        int d = _nComponents;
        int dpd1 = d * (d + 1) / 2; // Number of Hessian components
        int numHessianCols = 1 + d + dpd1;

        if (k < numHessianCols)
        {
            // Fall back to standard LLE if not enough neighbors
            return ComputeStandardLLEEmbedding(n);
        }

        if (k == numHessianCols)
        {
            // No null space in any neighbourhood
            return SmallestNonTrivialEigenvectors(new Tensor<T>(new[] { n, n }), n);
        }

        var centered = CenteredNeighborhoods(x, neighbors);
        var tangent = TangentCoordinates(centered, d);
        int dt = tangent.Shape[2];

        var first = new List<int>();
        var second = new List<int>();
        for (int c1 = 0; c1 < dt; c1++)
        {
            for (int c2 = c1; c2 < dt; c2++)
            {
                first.Add(c1);
                second.Add(c2);
            }
        }

        var quadratic = engine.TensorMultiply(
            engine.TensorIndexSelect(tangent, DimensionalityReductionOps.IndexTensor(first.ToArray()), 2),
            engine.TensorIndexSelect(tangent, DimensionalityReductionOps.IndexTensor(second.ToArray()), 2));
        var estimator = engine.TensorConcatenate(
            new[] { DimensionalityReductionOps.Full(new[] { n, k, 1 }, NumOps.One), tangent, quadratic }, 2);

        var q = DimensionalityReductionOps.OrthonormalizeColumnsEach(estimator, unitFallback: false);
        var identity = engine.Reshape(DimensionalityReductionOps.Identity<T>(k), new[] { 1, k, k });
        var local = engine.TensorSubtract(identity, DimensionalityReductionOps.BatchMatMulTranspose(q, q));
        return SmallestNonTrivialEigenvectors(DimensionalityReductionOps.ScatterAddBlocks(neighbors, local, n), n);
    }

    /// <summary>
    /// LTSA: per neighbourhood, W_i = Theta (Theta^T Theta + 1e-6 I)^-1 Theta^T on the tangent coordinates and
    /// G_i = I - W_i; G_i G_i^T is accumulated into B.
    /// </summary>
    private Tensor<T> ComputeLTSAEmbedding(Tensor<T> x, Tensor<int> neighbors, int n)
    {
        var engine = Engine;
        int k = neighbors.Shape[1];
        var centered = CenteredNeighborhoods(x, neighbors);
        var theta = TangentCoordinates(centered, _nComponents);
        int dt = theta.Shape[2];

        var gram = engine.TensorAdd(
            DimensionalityReductionOps.BatchTransposeMatMul(theta, theta),
            engine.Reshape(
                engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(dt), DimensionalityReductionOps.Tolerance<T>(1e-6)),
                new[] { 1, dt, dt }));
        var solved = DimensionalityReductionOps.SolveEach(gram, engine.TensorSwapAxes(theta, 1, 2));   // [n, dt, k]
        var w = engine.TensorBatchMatMul(theta, solved);                                                    // [n, k, k]

        var identity = engine.Reshape(DimensionalityReductionOps.Identity<T>(k), new[] { 1, k, k });
        var g = engine.TensorSubtract(identity, w);
        var local = DimensionalityReductionOps.BatchMatMulTranspose(g, g);
        return SmallestNonTrivialEigenvectors(DimensionalityReductionOps.ScatterAddBlocks(neighbors, local, n), n);
    }

    /// <summary>Neighbourhood rows centered on their own mean, [n, k, p].</summary>
    private Tensor<T> CenteredNeighborhoods(Tensor<T> x, Tensor<int> neighbors)
    {
        var rows = DimensionalityReductionOps.NeighborhoodRows(x, neighbors);
        return Engine.TensorSubtract(rows, Engine.ReduceMean(rows, new[] { 1 }, true));
    }

    /// <summary>
    /// Projection of centered neighbourhoods onto their top min(d, k, p) right singular vectors (the leading
    /// eigenvectors of X_i^T X_i), [n, k, min(d, k, p)].
    /// </summary>
    private Tensor<T> TangentCoordinates(Tensor<T> centered, int d)
    {
        var engine = Engine;
        int count = Math.Min(d, Math.Min(centered.Shape[1], centered.Shape[2]));
        var (_, right) = DimensionalityReductionOps.SymmetricEigenEach(
            DimensionalityReductionOps.BatchTransposeMatMul(centered, centered), descending: true);
        return engine.TensorBatchMatMul(centered, engine.TensorNarrow(right, 2, 0, count));
    }

    /// <summary>
    /// Eigenvectors 1..min(nComponents, n - 1) of a symmetric matrix in ascending eigenvalue order (skipping the
    /// zero-eigenvalue one), ties in index order.
    /// </summary>
    private Tensor<T> SmallestNonTrivialEigenvectors(Tensor<T> matrix, int n)
    {
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(matrix, descending: false);
        int nComp = Math.Min(_nComponents, n - 1);
        return Engine.TensorNarrow(eigenvectors, 1, 1, nComp);
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("LocallyLinearEmbedding has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "LLE does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LocallyLinearEmbedding does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"LLE{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the LLE algorithm variant.
/// </summary>
public enum LLEMethod
{
    /// <summary>
    /// Standard LLE algorithm.
    /// </summary>
    Standard,

    /// <summary>
    /// Modified LLE with multiple weight vectors.
    /// </summary>
    Modified,

    /// <summary>
    /// Hessian Locally Linear Embedding.
    /// </summary>
    HLLE,

    /// <summary>
    /// Local Tangent Space Alignment.
    /// </summary>
    LTSA
}
