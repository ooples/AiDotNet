using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Diffusion Maps for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Diffusion Maps embeds data into a low-dimensional space where Euclidean distances
/// approximate diffusion distances on the data manifold. It simulates a random walk
/// on the data graph and uses the eigenvectors of the diffusion operator.
/// </para>
/// <para>
/// The algorithm:
/// 1. Constructs a kernel matrix (typically Gaussian)
/// 2. Normalizes to create a Markov transition matrix
/// 3. Computes eigenvectors of the diffusion operator
/// 4. Scales eigenvectors by eigenvalue powers for embedding
/// </para>
/// <para><b>For Beginners:</b> Diffusion Maps captures the underlying geometry by:
/// - Simulating how information "diffuses" through the data
/// - Points connected by many short paths are close in diffusion distance
/// - Robust to noise compared to geodesic distances
///
/// Use cases:
/// - Discovering underlying manifold structure
/// - Robust to noise in the data
/// - When you want distances to reflect connectivity, not just proximity
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DiffusionMaps<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int? _randomState;
    private readonly double _alpha;
    private readonly int _diffusionTime;
    private readonly double _epsilon;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private Vector<T>? _eigenvalues;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the diffusion time parameter.
    /// </summary>
    public int DiffusionTime => _diffusionTime;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets the eigenvalues.
    /// </summary>
    public Vector<T>? Eigenvalues => _eigenvalues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="DiffusionMaps{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="alpha">Normalization factor (0 to 1). Defaults to 0.5.</param>
    /// <param name="diffusionTime">Number of diffusion steps. Defaults to 1.</param>
    /// <param name="epsilon">Kernel bandwidth. If 0, computed automatically. Defaults to 0.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public DiffusionMaps(
        int nComponents = 2,
        double alpha = 0.5,
        int diffusionTime = 1,
        double epsilon = 0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0 || alpha > 1)
        {
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));
        }

        if (diffusionTime < 1)
        {
            throw new ArgumentException("Diffusion time must be at least 1.", nameof(diffusionTime));
        }

        _nComponents = nComponents;
        _randomState = randomState;
        _alpha = alpha;
        _diffusionTime = diffusionTime;
        _epsilon = epsilon;
    }

    /// <summary>
    /// Fits Diffusion Maps and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nSamples = data.Rows;
        int n = data.Rows;

        // Validate that we have enough samples for requested components
        // We need at least nComponents + 1 samples because we skip the first eigenvector
        if (_nComponents >= n)
        {
            throw new ArgumentException(
                $"nComponents ({_nComponents}) must be less than number of samples ({n}) for Diffusion Maps. " +
                $"The first eigenvector is skipped, so at least {_nComponents + 1} samples are required.");
        }

        // Step 1: Compute pairwise distances and squared distances
        var x = Tensor<T>.FromMatrix(data);
        var distances = DimensionalityReductionOps.EuclideanDistances(x, x);
        var distSq = engine.TensorSquare(distances);

        // Step 2: Compute epsilon (kernel bandwidth) if not provided
        T eps = NumOps.FromDouble(_epsilon);
        if (_epsilon <= 0)
        {
            // Median of the n(n-1)/2 pairwise distances, squared. With the diagonal at +inf every pair appears
            // twice in the sorted flattened matrix, so the upper-triangle element m / 2 sits at position 2 (m / 2).
            var masked = distances.Clone();
            engine.TensorFillDiagonalInPlace(masked, NumOps.FromDouble(double.PositiveInfinity));
            var (sorted, _) = engine.TensorSort(engine.TensorFlatten(masked), 0, false);
            int pairs = n * (n - 1) / 2;
            T median = sorted[2 * (pairs / 2)];
            eps = NumOps.Multiply(median, median);
        }

        // Step 3: Kernel K = exp(-d^2 / eps)
        var kernel = engine.TensorExp(engine.TensorNegate(engine.TensorDivideScalar(distSq, eps)));

        // Step 4-5: alpha-normalization K_alpha = K / (q_i q_j)^alpha with q = row sums (q_i >= K_ii = 1)
        var q = DimensionalityReductionOps.RowSums(kernel);
        var qAlpha = engine.TensorPow(q, NumOps.FromDouble(_alpha));
        var kAlpha = engine.TensorDivide(
            engine.TensorDivide(kernel, DimensionalityReductionOps.AsColumn(qAlpha)),
            DimensionalityReductionOps.AsRow(qAlpha));

        // Step 6-7: symmetric diffusion operator M_s = D^(-1/2) K_alpha D^(-1/2) (d_i >= K_alpha_ii > 0)
        var rootD = engine.TensorSqrt(DimensionalityReductionOps.RowSums(kAlpha));
        var ms = engine.TensorDivide(
            engine.TensorDivide(kAlpha, DimensionalityReductionOps.AsColumn(rootD)),
            DimensionalityReductionOps.AsRow(rootD));

        // Step 8: Eigen-decomposition (M_s is positive semi-definite, so descending order = magnitude order)
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(ms, descending: true);
        // Signs as the original deflated power iteration converges to them from its seeded start vectors
        eigenvectors = DimensionalityReductionOps.MatchStartSigns(eigenvectors,
            DimensionalityReductionOps.SeededPowerIterationStarts<T>(_randomState ?? 42, Math.Min(n, _nComponents + 2), n, 0.0));

        // Step 9: Embedding from eigenvectors 1..nComponents (skipping the trivial first one) scaled by lambda^t
        var selected = engine.TensorNarrow(eigenvalues, 0, 1, _nComponents);
        var scale = engine.TensorPow(selected, NumOps.FromDouble(_diffusionTime));
        _eigenvalues = selected.ToVector();
        _embedding = DimensionalityReductionOps.ScaleColumns(
            engine.TensorNarrow(eigenvectors, 1, 1, _nComponents), scale).ToMatrix();
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("DiffusionMaps has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "DiffusionMaps does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DiffusionMaps does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"DM{i + 1}";
        }
        return names;
    }
}
