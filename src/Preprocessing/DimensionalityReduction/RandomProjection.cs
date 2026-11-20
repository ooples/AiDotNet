using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Random Projection for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Random projection reduces dimensionality by projecting data onto a random subspace.
/// Despite its simplicity, it has strong theoretical guarantees via the Johnson-Lindenstrauss lemma:
/// pairwise distances are approximately preserved with high probability.
/// </para>
/// <para>
/// Two projection types are supported:
/// - Gaussian: Random matrix with entries from N(0, 1/k)
/// - Sparse: Random matrix with mostly zeros (faster, memory efficient)
/// </para>
/// <para><b>For Beginners:</b> Random projection is surprisingly effective:
/// - It's very fast (just matrix multiplication)
/// - It preserves distances approximately (guaranteed by math!)
/// - Great for preprocessing before other algorithms
/// - Works well for very high-dimensional data
///
/// Use cases:
/// - Speeding up distance-based algorithms (kNN, clustering)
/// - Reducing memory for large datasets
/// - Preprocessing before other dimensionality reduction
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RandomProjection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int? _nComponents;
    private readonly double? _eps;
    private readonly RandomProjectionType _projectionType;
    private readonly double _density;
    private readonly int? _randomState;

    // Fitted parameters
    private Matrix<T>? _projectionMatrix;
    private int _nFeaturesIn;
    private int _nComponentsOut;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponentsOut;

    /// <summary>
    /// Gets the projection type.
    /// </summary>
    public RandomProjectionType ProjectionType => _projectionType;

    /// <summary>
    /// Gets the projection matrix.
    /// </summary>
    public Matrix<T>? ProjectionMatrix => _projectionMatrix;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="RandomProjection{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. If null, computed from eps.</param>
    /// <param name="eps">Maximum distortion ratio. Used to compute nComponents if not specified. Defaults to 0.1.</param>
    /// <param name="projectionType">Type of random projection. Defaults to Gaussian.</param>
    /// <param name="density">Density for sparse projection (proportion of non-zeros). Defaults to auto.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public RandomProjection(
        int? nComponents = null,
        double? eps = 0.1,
        RandomProjectionType projectionType = RandomProjectionType.Gaussian,
        double density = 0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents.HasValue && nComponents.Value < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (eps.HasValue && (eps.Value <= 0 || eps.Value >= 1))
        {
            throw new ArgumentException("Epsilon must be between 0 and 1.", nameof(eps));
        }

        if (density < 0 || density > 1)
        {
            throw new ArgumentException("Density must be between 0 and 1.", nameof(density));
        }

        _nComponents = nComponents;
        _eps = eps;
        _projectionType = projectionType;
        _density = density;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the random projection by generating the projection matrix.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;

        // Determine number of components
        if (_nComponents.HasValue)
        {
            _nComponentsOut = _nComponents.Value;
        }
        else if (_eps.HasValue)
        {
            // Johnson-Lindenstrauss bound
            _nComponentsOut = JohnsonLindenstraussMinDim(n, _eps.Value);
        }
        else
        {
            _nComponentsOut = Math.Min(_nFeaturesIn, 100);
        }

        _nComponentsOut = Math.Min(_nComponentsOut, _nFeaturesIn);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Generate projection matrix
        if (_projectionType == RandomProjectionType.Gaussian)
        {
            _projectionMatrix = GenerateGaussianMatrix(_nFeaturesIn, _nComponentsOut, random).ToMatrix();
        }
        else
        {
            double density = _density > 0 ? _density : 1.0 / Math.Sqrt(_nFeaturesIn);
            _projectionMatrix = GenerateSparseMatrix(_nFeaturesIn, _nComponentsOut, density, random).ToMatrix();
        }
    }

    private static int JohnsonLindenstraussMinDim(int nSamples, double eps)
    {
        // k >= 4 * log(n) / (eps^2 / 2 - eps^3 / 3)
        double denominator = eps * eps / 2 - eps * eps * eps / 3;
        if (denominator <= 0) denominator = eps * eps / 2;

        int k = (int)Math.Ceiling(4 * Math.Log(nSamples) / denominator);
        return Math.Max(k, 1);
    }

    private Tensor<T> GenerateGaussianMatrix(int nFeatures, int nComponents, Random random)
    {
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(nComponents));
        return Engine.TensorMultiplyScalar(DimensionalityReductionOps.StandardNormal<T>(random, nFeatures, nComponents), scale);
    }

    private Tensor<T> GenerateSparseMatrix(int nFeatures, int nComponents, double density, Random random)
    {
        var engine = Engine;
        T scale = NumOps.FromDouble(Math.Sqrt(1.0 / (nComponents * density)));
        var draws = DimensionalityReductionOps.Uniform<T>(random, nFeatures, nComponents);
        var zeros = new Tensor<T>(new[] { nFeatures, nComponents });

        // Sparse random projection: +scale, -scale, 0 with probabilities density/2, density/2, 1-density
        var negative = engine.TensorWhere(
            engine.TensorLessThan(draws, NumOps.FromDouble(density)),
            engine.TensorFullLike(zeros, NumOps.Negate(scale)),
            zeros);
        return engine.TensorWhere(
            engine.TensorLessThan(draws, NumOps.FromDouble(density / 2)),
            engine.TensorFullLike(zeros, scale),
            negative);
    }

    /// <summary>
    /// Transforms data using the random projection.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_projectionMatrix is null)
        {
            throw new InvalidOperationException("RandomProjection has not been fitted.");
        }

        // Matrix multiplication: X * R
        return Engine.TensorMatMul(Tensor<T>.FromMatrix(data), Tensor<T>.FromMatrix(_projectionMatrix)).ToMatrix();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RandomProjection does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponentsOut];
        for (int i = 0; i < _nComponentsOut; i++)
        {
            names[i] = $"RP{i + 1}";
        }
        return names;
    }
}
