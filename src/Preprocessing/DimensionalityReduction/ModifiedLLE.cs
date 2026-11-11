using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Modified Locally Linear Embedding with regularization.
/// </summary>
/// <remarks>
/// <para>
/// Modified LLE adds regularization to the standard LLE algorithm to improve
/// numerical stability when the number of neighbors exceeds the input dimensionality.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k-nearest neighbors for each point
/// 2. Compute reconstruction weights with regularization
/// 3. Compute embedding by minimizing reconstruction error
/// </para>
/// <para><b>For Beginners:</b> Modified LLE is more stable than standard LLE:
/// - Works better when you have many neighbors
/// - Less sensitive to noise
/// - Produces more consistent results
///
/// Use cases:
/// - When standard LLE is unstable
/// - High number of neighbors relative to dimensions
/// - Noisy data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ModifiedLLE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double _regParam;

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
    /// Creates a new instance of <see cref="ModifiedLLE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Defaults to 5.</param>
    /// <param name="regParam">Regularization parameter. Defaults to 1e-3.</param>
    /// <param name="randomState">Kept for API compatibility; the eigen solver is deterministic.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public ModifiedLLE(
        int nComponents = 2,
        int nNeighbors = 5,
        double regParam = 1e-3,
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

        if (regParam < 0)
        {
            throw new ArgumentException("Regularization parameter must be non-negative.", nameof(regParam));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _regParam = regParam;
    }

    /// <summary>
    /// Fits Modified LLE and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nSamples = data.Rows;
        int n = data.Rows;

        // Validate that we have enough samples for requested components
        // We need at least nComponents + 1 samples because we skip the null eigenvector
        if (_nComponents >= n)
        {
            throw new ArgumentException(
                $"nComponents ({_nComponents}) must be less than number of samples ({n}) for MLLE. " +
                $"The null eigenvector is skipped, so at least {_nComponents + 1} samples are required.");
        }

        int k = Math.Min(_nNeighbors, n - 1);
        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Find k-nearest neighbors (ties in index order)
        var neighbors = DimensionalityReductionOps.NearestNeighbors(DimensionalityReductionOps.EuclideanDistances(x, x), k);

        // Step 2: Local covariances C_i = (X_N(i) - x_i)(X_N(i) - x_i)^T, shape [n, k, k]
        var differences = DimensionalityReductionOps.NeighborhoodDifferences(x, neighbors);
        var covariances = DimensionalityReductionOps.BatchMatMulTranspose(differences, differences);

        // Regularization reg_i = regParam * trace(C_i) / k, replaced by 1e-3 when below 1e-10
        var reg = engine.TensorMultiplyScalar(
            DimensionalityReductionOps.BatchTrace(covariances), NumOps.FromDouble(_regParam / k));
        reg = engine.TensorWhere(
            engine.TensorLessThan(reg, NumOps.FromDouble(1e-10)), DimensionalityReductionOps.Full(n, NumOps.FromDouble(1e-3)), reg);
        var regularized = engine.TensorAdd(
            covariances,
            engine.TensorMultiply(
                engine.Reshape(reg, new[] { n, 1, 1 }),
                engine.Reshape(DimensionalityReductionOps.Identity<T>(k), new[] { 1, k, k })));

        // Solve C_i w_i = 1 and normalize to sum one where the sum is not ~0
        var weights = DimensionalityReductionOps.SolveEach(regularized, DimensionalityReductionOps.Full(new[] { n, k }, NumOps.One));
        var sums = DimensionalityReductionOps.RowSums(weights);
        var nonZero = engine.TensorGreaterThan(engine.TensorAbs(sums), NumOps.FromDouble(1e-10));
        var divisor = engine.TensorWhere(nonZero, sums, DimensionalityReductionOps.Full(n, NumOps.One));
        weights = engine.TensorDivide(weights, DimensionalityReductionOps.AsColumn(divisor));

        var w = DimensionalityReductionOps.ScatterRows(neighbors, weights, n, NumOps.Zero);

        // Step 3: Compute M = (I - W)^T * (I - W)
        var iMinusW = engine.TensorSubtract(DimensionalityReductionOps.Identity<T>(n), w);
        var m = DimensionalityReductionOps.TransposeMatMul(iMinusW, iMinusW);

        // Step 4: Eigenvectors of M in ascending eigenvalue order
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(m, descending: false);

        // Step 5: Create embedding, skipping the null eigenvector
        _embedding = engine.TensorNarrow(eigenvectors, 1, 1, _nComponents).ToMatrix();
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("ModifiedLLE has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "ModifiedLLE does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ModifiedLLE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"MLLE{i + 1}";
        }
        return names;
    }
}
