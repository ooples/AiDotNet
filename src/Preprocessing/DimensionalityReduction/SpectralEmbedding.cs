using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Spectral Embedding for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Spectral Embedding forms an affinity matrix from the data and computes
/// the eigenvectors of the graph Laplacian. This provides a low-dimensional
/// representation that preserves local connectivity.
/// </para>
/// <para>
/// The algorithm constructs a similarity graph and uses spectral decomposition
/// of the Laplacian matrix to find coordinates that respect graph structure.
/// </para>
/// <para><b>For Beginners:</b> Spectral Embedding uses graph theory:
/// - Build a graph where similar points are connected
/// - Use the graph's structure to find good coordinates
/// - Similar to what's used in spectral clustering
/// - Good for data with cluster structure
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SpectralEmbedding<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int? _randomState;
    private readonly SpectralAffinity _affinity;
    private readonly double _gamma;
    private readonly int _nNeighbors;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private Matrix<T>? _affinityMatrix;
    private int _nSamples;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the affinity type.
    /// </summary>
    public SpectralAffinity Affinity => _affinity;

    /// <summary>
    /// Gets the gamma parameter for RBF kernel.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the number of neighbors for nearest neighbors affinity.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the affinity matrix.
    /// </summary>
    public Matrix<T>? AffinityMatrix => _affinityMatrix;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SpectralEmbedding{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="affinity">How to construct the affinity matrix. Defaults to NearestNeighbors.</param>
    /// <param name="gamma">Kernel coefficient for RBF. If null, uses 1/n_features.</param>
    /// <param name="nNeighbors">Number of neighbors for NN affinity, the sample itself included (scikit-learn's
    /// kneighbors_graph(include_self=True)). Defaults to 10.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public SpectralEmbedding(
        int nComponents = 2,
        SpectralAffinity affinity = SpectralAffinity.NearestNeighbors,
        double? gamma = null,
        int nNeighbors = 10,
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
        _affinity = affinity;
        _gamma = gamma ?? 0; // Will be computed during fit
        _nNeighbors = nNeighbors;
    }

    /// <summary>
    /// Fits Spectral Embedding by computing the graph Laplacian eigenvectors.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute gamma if not provided
        T gamma = NumOps.FromDouble(_gamma > 0 ? _gamma : 1.0 / p);

        // Step 1: Compute affinity matrix
        var affinity = ComputeAffinityMatrix(Tensor<T>.FromMatrix(data), n, gamma);
        _affinityMatrix = affinity.ToMatrix();

        // Steps 2 and 3: the bottom eigenvectors of the normalized graph Laplacian (scikit-learn spectral_embedding)
        _embedding = ComputeEmbedding(affinity, n).ToMatrix();
    }

    private Tensor<T> ComputeAffinityMatrix(Tensor<T> x, int n, T gamma)
    {
        var engine = Engine;
        Tensor<T> affinity;

        switch (_affinity)
        {
            case SpectralAffinity.RBF:
                // Radial Basis Function (Gaussian) kernel, rbf_kernel(X, gamma) (diagonal 1)
                affinity = engine.TensorExp(engine.TensorMultiplyScalar(
                    DimensionalityReductionOps.SquaredDistances(x, x), NumOps.Negate(gamma)));
                break;

            case SpectralAffinity.NearestNeighbors:
                // scikit-learn: A = kneighbors_graph(X, n_neighbors, include_self=True) (0/1 connectivity, the
                // sample itself counting as one of its n_neighbors), affinity = (A + A^T) / 2
                var directed = DimensionalityReductionOps.Full(new[] { n, n }, NumOps.Zero);
                int others = Math.Min(_nNeighbors - 1, n - 1);
                if (others > 0)
                {
                    var neighbors = DimensionalityReductionOps.NearestNeighbors(
                        DimensionalityReductionOps.EuclideanDistances(x, x), others);
                    directed = DimensionalityReductionOps.ScatterRows(
                        neighbors, DimensionalityReductionOps.Full(new[] { n, others }, NumOps.One), n, NumOps.Zero);
                }

                affinity = engine.TensorMultiplyScalar(
                    engine.TensorAdd(directed, engine.TensorTranspose(directed)), NumOps.FromDouble(0.5));
                break;

            case SpectralAffinity.Precomputed:
                throw new InvalidOperationException(
                    "Precomputed affinity requires passing the affinity matrix directly.");

            default:
                affinity = DimensionalityReductionOps.Full(new[] { n, n }, NumOps.Zero);
                break;
        }

        // Every sample is fully similar to itself (rbf_kernel's diagonal, include_self's self connection)
        engine.TensorFillDiagonalInPlace(affinity, NumOps.One);
        return affinity;
    }

    /// <summary>
    /// scikit-learn spectral_embedding(norm_laplacian=True, drop_first=True): the graph Laplacian ignores self loops
    /// (scipy csgraph.laplacian zeroes the diagonal), L = I - D^(-1/2) W D^(-1/2), its eigenvectors 1 .. nComponents in
    /// ascending eigenvalue order are divided by sqrt(degree) and each is signed so its largest-magnitude entry is
    /// positive (_deterministic_vector_sign_flip).
    /// </summary>
    private Tensor<T> ComputeEmbedding(Tensor<T> affinity, int n)
    {
        var engine = Engine;
        var w = affinity.Clone();
        engine.TensorFillDiagonalInPlace(w, NumOps.Zero);

        // D^(-1/2), zero where the degree is at most 1e-10
        var degree = DimensionalityReductionOps.RowSums(w);
        var positive = engine.TensorGreaterThan(degree, DimensionalityReductionOps.Tolerance<T>(1e-10));
        var safeDegree = engine.TensorWhere(positive, degree, DimensionalityReductionOps.Full(n, NumOps.One));
        var dInv = engine.TensorWhere(
            positive, engine.TensorRsqrt(safeDegree), DimensionalityReductionOps.Full(n, NumOps.Zero));

        // L = I - D^(-1/2) W D^(-1/2)
        var normalized = DimensionalityReductionOps.ScaleColumns(DimensionalityReductionOps.ScaleRows(w, dInv), dInv);
        var laplacian = engine.TensorSubtract(DimensionalityReductionOps.Identity<T>(n), normalized);

        // Eigenvectors of L in ascending eigenvalue order (ties in index order); skip the first (trivial) one
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(laplacian, descending: false);
        int nComp = Math.Min(_nComponents, n - 1);
        var selected = engine.TensorNarrow(eigenvectors, 1, 1, nComp);

        // Undo the symmetric normalisation (y = D^(-1/2) v), then fix the signs on the scaled vectors
        return DimensionalityReductionOps.AlignColumnSigns(DimensionalityReductionOps.ScaleRows(selected, dInv));
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("SpectralEmbedding has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "SpectralEmbedding does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpectralEmbedding does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"Spectral{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies how to construct the affinity matrix for Spectral Embedding.
/// </summary>
public enum SpectralAffinity
{
    /// <summary>
    /// K-nearest neighbors graph.
    /// </summary>
    NearestNeighbors,

    /// <summary>
    /// Radial Basis Function (Gaussian) kernel.
    /// </summary>
    RBF,

    /// <summary>
    /// Use a precomputed affinity matrix.
    /// </summary>
    Precomputed
}
