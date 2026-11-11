using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Multidimensional Scaling for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// MDS finds a low-dimensional embedding that preserves pairwise distances between points.
/// Classical MDS uses eigendecomposition of the double-centered distance matrix.
/// Non-metric MDS uses iterative optimization to preserve distance rankings.
/// </para>
/// <para>
/// MDS is useful for:
/// - Visualizing similarity/dissimilarity data
/// - Preserving pairwise relationships
/// - When you have a distance matrix rather than feature vectors
/// </para>
/// <para><b>For Beginners:</b> MDS tries to place points in 2D/3D such that:
/// - Points that were close in high-D stay close in low-D
/// - Points that were far apart stay far apart
/// - Unlike t-SNE/UMAP, MDS tries to preserve actual distances, not just neighborhoods
///
/// Classical MDS: Preserves exact distances (works well when data is linear)
/// Non-metric MDS: Preserves distance rankings (more flexible, works better for complex data)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MDS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly MDSType _mdsType;
    private readonly MDSMetric _metric;
    private readonly int _maxIter;
    private readonly double _eps;
    private readonly bool _normalized;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private T _stress = MathHelper.GetNumericOperations<T>().Zero;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the MDS type (classical or non-metric).
    /// </summary>
    public MDSType MdsType => _mdsType;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets the final stress value (goodness of fit).
    /// </summary>
    public T Stress => _stress;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="MDS{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="mdsType">Type of MDS algorithm. Defaults to Classical.</param>
    /// <param name="metric">Distance metric. Defaults to Euclidean.</param>
    /// <param name="maxIter">Maximum iterations for non-metric MDS. Defaults to 300.</param>
    /// <param name="eps">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="normalized">Whether to normalize stress. Defaults to false.</param>
    /// <param name="randomState">Accepted for API compatibility; the embedding starts from classical MDS, so no random state is drawn.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public MDS(
        int nComponents = 2,
        MDSType mdsType = MDSType.Classical,
        MDSMetric metric = MDSMetric.Euclidean,
        int maxIter = 300,
        double eps = 1e-6,
        bool normalized = false,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _mdsType = mdsType;
        _metric = metric;
        _maxIter = maxIter;
        _eps = eps;
        _normalized = normalized;
    }

    /// <summary>
    /// Fits MDS and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;

        // Compute pairwise distance matrix
        var distances = ComputeDistanceMatrix(Tensor<T>.FromMatrix(data));

        var embedding = _mdsType == MDSType.Classical
            ? ClassicalMDS(distances)
            : NonMetricMDS(distances);

        _embedding = embedding.ToMatrix();
    }

    private Tensor<T> ComputeDistanceMatrix(Tensor<T> x)
    {
        return _metric switch
        {
            MDSMetric.SquaredEuclidean => DimensionalityReductionOps.SquaredDistances(x, x),
            MDSMetric.Manhattan => Engine.TensorCDist(x, x, 1.0),
            _ => DimensionalityReductionOps.EuclideanDistances(x, x)
        };
    }

    private Tensor<T> ClassicalMDS(Tensor<T> distances)
    {
        // Double-centre the squared distances and keep the leading scaled eigenvectors
        var embedding = DimensionalityReductionOps.ClassicalMds(distances, _nComponents);

        // Compute stress
        _stress = ComputeStress(distances, embedding);

        return embedding;
    }

    private Tensor<T> NonMetricMDS(Tensor<T> distances)
    {
        // Initialize with classical MDS
        var embedding = ClassicalMDS(distances);

        // SMACOF algorithm for non-metric MDS
        T prevStress = NumOps.MaxValue;
        T eps = NumOps.FromDouble(_eps);

        for (int iter = 0; iter < _maxIter; iter++)
        {
            // Compute current distances in embedding
            var embeddingDist = DimensionalityReductionOps.EuclideanDistances(embedding, embedding);

            // Monotonic regression (isotonic regression on distances)
            var disparities = IsotonicRegression(distances, embeddingDist);

            // Compute Guttman transform
            var newEmbedding = GuttmanTransform(embedding, disparities, embeddingDist);

            // Compute stress
            T stress = ComputeStress(disparities, newEmbedding);

            embedding = newEmbedding;
            _stress = stress;

            // Check convergence
            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(prevStress, stress)), eps))
            {
                break;
            }

            prevStress = stress;
        }

        return embedding;
    }

    /// <summary>
    /// Disparities for the upper-triangle pairs: the embedding distances ordered by original distance and made
    /// monotone by repeated adjacent averaging, mirrored to a symmetric matrix with a zero diagonal.
    /// </summary>
    /// <remarks>
    /// The pair ordering and the mapping back are engine ops. The monotone fit itself is a sequential sweep with
    /// no engine equivalent yet (TensorIsotonicRegression, see missing-ops), so it keeps the original algorithm.
    /// </remarks>
    private Tensor<T> IsotonicRegression(Tensor<T> originalDist, Tensor<T> embeddingDist)
    {
        var engine = Engine;
        int n = originalDist.Shape[0];

        // Flat positions (i * n + j) of the pairs i < j, row by row, and of their mirrors (j * n + i)
        var upper = new List<int>(n * (n - 1) / 2);
        var lower = new List<int>(n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                upper.Add(i * n + j);
                lower.Add(j * n + i);
            }
        }

        var upperIndex = DimensionalityReductionOps.IndexTensor(upper.ToArray());
        var lowerIndex = DimensionalityReductionOps.IndexTensor(lower.ToArray());
        var originalFlat = engine.TensorIndexSelect(engine.Reshape(originalDist, new[] { n * n }), upperIndex, 0);
        var embeddingFlat = engine.TensorIndexSelect(engine.Reshape(embeddingDist, new[] { n * n }), upperIndex, 0);

        // Sort by original distance (stable: equal distances keep pair order)
        var order = engine.TensorArgsort(originalFlat, 0, false);
        var fittedSorted = MonotoneByAdjacentAveraging(engine.TensorIndexSelect(embeddingFlat, order, 0));

        // Map back to matrix
        var disparities = new Tensor<T>(new[] { n * n });
        disparities = engine.TensorScatter(disparities, engine.TensorIndexSelect(upperIndex, order, 0), fittedSorted, 0);
        disparities = engine.TensorScatter(disparities, engine.TensorIndexSelect(lowerIndex, order, 0), fittedSorted, 0);
        return engine.Reshape(disparities, new[] { n, n });
    }

    /// <summary>
    /// Repeats left-to-right passes that replace each adjacent decreasing pair by its average until no pair
    /// decreases (the original monotone fit, kept unchanged until TensorIsotonicRegression exists).
    /// </summary>
    private static Tensor<T> MonotoneByAdjacentAveraging(Tensor<T> values)
    {
        var fitted = values.Cast<double>().AsSpan().ToArray();
        int m = fitted.Length;
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int i = 0; i < m - 1; i++)
            {
                if (fitted[i] > fitted[i + 1])
                {
                    double avg = (fitted[i] + fitted[i + 1]) / 2;
                    fitted[i] = avg;
                    fitted[i + 1] = avg;
                    changed = true;
                }
            }
        }

        return new Tensor<double>(fitted, new[] { m }).Cast<T>();
    }

    private Tensor<T> GuttmanTransform(Tensor<T> embedding, Tensor<T> disparities, Tensor<T> embeddingDist)
    {
        var engine = Engine;
        int n = embedding.Shape[0];

        // B[i, j] = -disparity / distance off the diagonal (0 where the distance is ~0), B[i, i] = -sum_j B[i, j]
        var ratio = engine.TensorWhere(
            engine.TensorGreaterThan(embeddingDist, NumOps.FromDouble(1e-10)),
            engine.TensorNegate(engine.TensorDivide(disparities, embeddingDist)),
            new Tensor<T>(new[] { n, n }));
        var b = engine.TensorSubtract(ratio, engine.TensorDiag(DimensionalityReductionOps.RowSums(ratio)));

        // New embedding: X_new = (1/n) * B * X_old
        return engine.TensorDivideScalar(engine.TensorMatMul(b, embedding), NumOps.FromDouble(n));
    }

    /// <summary>
    /// Kruskal stress of an embedding against target dissimilarities over the pairs i &lt; j:
    /// sqrt(sum (target - d)^2), or sqrt(sum (target - d)^2 / sum target^2) when normalized.
    /// </summary>
    private T ComputeStress(Tensor<T> target, Tensor<T> embedding)
    {
        var engine = Engine;
        var embeddingDist = DimensionalityReductionOps.EuclideanDistances(embedding, embedding);
        T stress = engine.TensorSum(engine.TensorTriu(engine.TensorSquare(engine.TensorSubtract(target, embeddingDist)), 1));
        T normalization = engine.TensorSum(engine.TensorTriu(engine.TensorSquare(target), 1));

        return _normalized && NumOps.GreaterThan(normalization, NumOps.Zero)
            ? NumOps.Sqrt(NumOps.Divide(stress, normalization))
            : NumOps.Sqrt(stress);
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("MDS has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "MDS does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MDS does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"MDS{i + 1}";
        }
        return names;
    }
}
