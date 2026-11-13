using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// PHATE: Potential of Heat-diffusion for Affinity-based Transition Embedding.
/// </summary>
/// <remarks>
/// <para>
/// PHATE is a dimensionality reduction method designed for visualizing high-dimensional
/// biological data, particularly single-cell data. It captures both local and global
/// structure by using diffusion-based distances and a special potential distance metric.
/// </para>
/// <para>
/// The algorithm:
/// 1. Compute local affinities using adaptive bandwidth Gaussian kernel
/// 2. Construct diffusion operator from affinities
/// 3. Diffuse for t steps to capture multi-scale structure
/// 4. Compute potential distance using log transform
/// 5. Embed using MDS on potential distances
/// </para>
/// <para><b>For Beginners:</b> PHATE excels at revealing data trajectories:
/// - Captures smooth progression paths in data (differentiation, time courses)
/// - Preserves both local clusters and global connectivity
/// - The potential distance emphasizes transitions between states
/// - Particularly effective for biological data
///
/// Use cases:
/// - Single-cell RNA sequencing visualization
/// - Developmental trajectories and cell differentiation
/// - Time-series data with smooth transitions
/// - Any data with underlying continuous processes
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PHATE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int? _randomState;
    private readonly int _nNeighbors;
    private readonly int _diffusionTime;
    private readonly double _decay;
    private readonly double _gamma;

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
    /// Gets the diffusion time parameter.
    /// </summary>
    public int DiffusionTime => _diffusionTime;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="PHATE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for affinity computation. Defaults to 5.</param>
    /// <param name="diffusionTime">Number of diffusion steps. Defaults to auto (-1) or use specific value.</param>
    /// <param name="decay">Alpha decay for kernel. Defaults to 40.</param>
    /// <param name="gamma">Informational distance parameter. Defaults to 1 (log potential).</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public PHATE(
        int nComponents = 2,
        int nNeighbors = 5,
        int diffusionTime = -1,
        double decay = 40,
        double gamma = 1,
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
        _diffusionTime = diffusionTime;
        _decay = decay;
        _gamma = gamma;
    }

    /// <summary>
    /// Fits PHATE and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Compute pairwise distances
        var distances = DimensionalityReductionOps.EuclideanDistances(x, x);

        // Step 2: Compute adaptive bandwidth kernel
        var kernel = ComputeAdaptiveKernel(distances, n);

        // Step 3: Normalize to get diffusion operator P
        var p = NormalizeToDiffusionOperator(kernel, n);

        // Step 4: Compute diffused operator P^t
        int t = _diffusionTime > 0 ? _diffusionTime : EstimateDiffusionTime(n);
        var pt = DiffuseOperator(p, t);

        // Step 5: Compute potential distance
        var potentialDistance = ComputePotentialDistance(pt);

        // Step 6: Embed using classical MDS (zero columns past the sample count)
        // Signs as the original deflated power iteration converges to them from its seeded start vectors
        var random = RandomHelper.CreateSeededRandom(_randomState ?? 42);
        var starts = DimensionalityReductionOps.PowerIterationStarts<T>(random, Math.Min(n, _nComponents + 2), n, 0.5);
        var embedding = DimensionalityReductionOps.ClassicalMds(potentialDistance, _nComponents, starts);
        if (embedding.Shape[1] < _nComponents)
        {
            embedding = Engine.TensorConcatenate(
                new[] { embedding, new Tensor<T>(new[] { n, _nComponents - embedding.Shape[1] }) }, 1);
        }

        _embedding = embedding.ToMatrix();
    }

    /// <summary>
    /// Alpha-decaying kernel exp(-(d_ij / sqrt(eps_i eps_j))^(decay / 10)) with eps_i the distance to the k-th
    /// nearest neighbour (at least 1e-10), zero diagonal, symmetrized as (K + K^T) / 2.
    /// </summary>
    private Tensor<T> ComputeAdaptiveKernel(Tensor<T> distances, int n)
    {
        var engine = Engine;
        int k = Math.Min(_nNeighbors, n - 1);

        // Compute local bandwidth for each point (k-th nearest neighbor distance)
        var masked = distances.Clone();
        engine.TensorFillDiagonalInPlace(masked, NumOps.FromDouble(double.PositiveInfinity));
        var (sorted, _) = engine.TensorSort(masked, 1, false);
        var epsilon = engine.TensorClampMin(engine.TensorNarrow(sorted, 1, k - 1, 1), DimensionalityReductionOps.Tolerance<T>(1e-10));   // [n, 1]

        // Adaptive bandwidth: geometric mean of bandwidths
        var bandwidth = engine.TensorSqrt(engine.TensorMatMul(epsilon, engine.TensorTranspose(epsilon)));
        var scaled = engine.TensorPow(engine.TensorDivide(distances, bandwidth), NumOps.FromDouble(_decay / 10.0));
        var kernel = engine.TensorExp(engine.TensorNegate(scaled));
        engine.TensorFillDiagonalInPlace(kernel, NumOps.Zero);

        // Make symmetric
        return engine.TensorMultiplyScalar(engine.TensorAdd(kernel, engine.TensorTranspose(kernel)), NumOps.FromDouble(0.5));
    }

    /// <summary>Row-normalized Markov matrix; rows summing to at most 1e-10 become a self-loop.</summary>
    private Tensor<T> NormalizeToDiffusionOperator(Tensor<T> kernel, int n)
    {
        var engine = Engine;
        var rowSum = engine.ReduceSum(kernel, new[] { 1 }, true);
        var valid = engine.TensorGreaterThan(rowSum, DimensionalityReductionOps.Tolerance<T>(1e-10));
        var normalized = engine.TensorDivide(kernel, engine.TensorWhere(valid, rowSum, engine.TensorOnesLike(rowSum)));
        var validRows = engine.TensorMultiply(engine.TensorOnesLike(kernel), valid);
        return engine.TensorWhere(validRows, normalized, DimensionalityReductionOps.Identity<T>(n));
    }

    private static int EstimateDiffusionTime(int n)
    {
        // Simple heuristic: use sqrt(n), between 1 and 100
        int maxT = 100;
        int optimalT = Math.Max(1, (int)Math.Sqrt(n));
        return Math.Min(optimalT, maxT);
    }

    /// <summary>Matrix power P^t by repeated multiplication.</summary>
    private Tensor<T> DiffuseOperator(Tensor<T> p, int t)
    {
        var pt = p;
        for (int step = 1; step < t; step++)
        {
            pt = Engine.TensorMatMul(pt, p);
        }

        return pt;
    }

    /// <summary>
    /// Potential distance: Euclidean distance between rows of log(P^t + 1e-10) (gamma = 1) or of
    /// (P^t + 1e-10)^(1 / gamma).
    /// </summary>
    private Tensor<T> ComputePotentialDistance(Tensor<T> pt)
    {
        var engine = Engine;
        var shifted = engine.TensorAddScalar(pt, DimensionalityReductionOps.Tolerance<T>(1e-10));
        var potential = _gamma == 1
            ? engine.TensorLog(shifted)
            : engine.TensorPow(shifted, NumOps.FromDouble(1.0 / _gamma));
        return DimensionalityReductionOps.EuclideanDistances(potential, potential);
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("PHATE has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "PHATE does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PHATE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"PHATE{i + 1}";
        }
        return names;
    }
}
