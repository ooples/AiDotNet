using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Mini-batch Sparse PCA using online dictionary learning.
/// </summary>
/// <remarks>
/// <para>
/// MiniBatchSparsePCA is a faster, memory-efficient version of SparsePCA that
/// processes data in mini-batches instead of using the full dataset. This makes
/// it suitable for large datasets that don't fit in memory.
/// </para>
/// <para>
/// The algorithm uses online dictionary learning with mini-batches, updating
/// the components incrementally as it processes each batch.
/// </para>
/// <para><b>For Beginners:</b> Think of this as SparsePCA on a budget:
/// - SparsePCA looks at ALL your data at once (memory intensive)
/// - MiniBatchSparsePCA looks at small pieces at a time (memory efficient)
/// - Results are similar, but mini-batch is faster for large datasets
/// - Trade-off: Slightly less accurate but much more scalable
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MiniBatchSparsePCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _alpha;
    private readonly double _ridge;
    private readonly int _batchSize;
    private readonly int _nIter;
    private readonly double _tol;
    private readonly int? _randomState;
    private readonly bool _shuffle;

    // Fitted parameters
    private Vector<T>? _mean;
    private Matrix<T>? _components;
    private int _nFeaturesIn;
    private int _nSamplesSeen;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the sparsity regularization parameter.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the ridge regularization parameter.
    /// </summary>
    public double Ridge => _ridge;

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize => _batchSize;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the sparse components (each row is a component).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the number of samples seen during fitting.
    /// </summary>
    public int NSamplesSeen => _nSamplesSeen;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="MiniBatchSparsePCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of sparse components. Defaults to 2.</param>
    /// <param name="alpha">Sparsity regularization parameter. Defaults to 1.0.</param>
    /// <param name="ridge">Ridge regularization for stability. Defaults to 0.01.</param>
    /// <param name="batchSize">Size of mini-batches. Defaults to 50.</param>
    /// <param name="nIter">Number of iterations over the full dataset. Defaults to 100.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="shuffle">Whether to shuffle data before each iteration. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public MiniBatchSparsePCA(
        int nComponents = 2,
        double alpha = 1.0,
        double ridge = 0.01,
        int batchSize = 50,
        int nIter = 100,
        double tol = 1e-6,
        bool shuffle = true,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0)
        {
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        }

        if (batchSize < 1)
        {
            throw new ArgumentException("Batch size must be at least 1.", nameof(batchSize));
        }

        _nComponents = nComponents;
        _alpha = alpha;
        _ridge = ridge;
        _batchSize = batchSize;
        _nIter = nIter;
        _tol = tol;
        _shuffle = shuffle;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Mini-batch Sparse PCA using online dictionary learning.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Compute mean and center the data once (each batch is gathered from the centered rows)
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var centered = DimensionalityReductionOps.SubtractRow(x, mean);
        _mean = mean.ToVector();

        // Initialize components randomly (uniform in [-0.5, 0.5), unit rows)
        var components = DimensionalityReductionOps.NormalizeRows(
            engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, k, p), NumOps.FromDouble(0.5)));

        // Accumulators for online dictionary learning: A = codes^T codes (ridge on the diagonal), B = codes^T X
        var a = engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(k), NumOps.FromDouble(_ridge));
        var b = new Tensor<T>(new[] { k, p });
        T ridge = NumOps.FromDouble(_ridge);
        T tol = NumOps.FromDouble(_tol);

        _nSamplesSeen = 0;

        // Create index array for shuffling
        var indices = Enumerable.Range(0, n).ToArray();

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Shuffle indices if requested
            if (_shuffle)
            {
                ShuffleArray(indices, random);
            }

            var prevComponents = components;

            // Process mini-batches
            for (int batchStart = 0; batchStart < n; batchStart += _batchSize)
            {
                int batchEnd = Math.Min(batchStart + _batchSize, n);
                int actualBatchSize = batchEnd - batchStart;

                // Gather the batch rows (already centered)
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batchStart, batchRows, 0, actualBatchSize);
                var batchData = DimensionalityReductionOps.SelectRows(centered, DimensionalityReductionOps.IndexTensor(batchRows));

                // Compute codes for this batch (ridge regression)
                var batchCodes = ComputeCodes(batchData, components);

                // Running averages A = (1 - w) A + w codes^T codes, B = (1 - w) B + w codes^T X
                double weight = 1.0 / (_nSamplesSeen + actualBatchSize);
                T keep = DimensionalityReductionOps.Constant<T>(1 - weight);
                T add = DimensionalityReductionOps.Constant<T>(weight);
                a = engine.TensorAdd(
                    engine.TensorMultiplyScalar(a, keep),
                    engine.TensorMultiplyScalar(DimensionalityReductionOps.TransposeMatMul(batchCodes, batchCodes), add));
                var diagonal = engine.TensorDiagonal(a);
                a = engine.TensorAdd(a, engine.TensorDiag(engine.TensorSubtract(engine.TensorClampMin(diagonal, ridge), diagonal)));
                b = engine.TensorAdd(
                    engine.TensorMultiplyScalar(b, keep),
                    engine.TensorMultiplyScalar(DimensionalityReductionOps.TransposeMatMul(batchCodes, batchData), add));

                _nSamplesSeen += actualBatchSize;

                // Update dictionary using block coordinate descent
                components = UpdateDictionaryFromAccumulators(a, b, components);
            }

            // Check convergence
            if (DimensionalityReductionOps.Below(DimensionalityReductionOps.MaxAbsDifference(components, prevComponents), tol))
            {
                break;
            }
        }

        _components = components.ToMatrix();
    }

    private static void ShuffleArray(int[] array, Random random)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    /// <summary>Ridge codes: X C^T (C C^T + ridge I)^-1.</summary>
    private Tensor<T> ComputeCodes(Tensor<T> data, Tensor<T> components)
    {
        var engine = Engine;
        int k = components.Shape[0];

        var gram = engine.TensorAdd(
            DimensionalityReductionOps.MatMulTranspose(components, components),
            engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(k), DimensionalityReductionOps.Constant<T>(_ridge)));

        var dataProj = DimensionalityReductionOps.MatMulTranspose(data, components);
        return engine.TensorTranspose(DimensionalityReductionOps.Solve(gram, engine.TensorTranspose(dataProj)));
    }

    /// <summary>
    /// One block coordinate descent sweep over the components from the accumulators:
    /// C[c] = normalize(S((B[c] - sum_{o != c} A[c, o] C[o]) / A[c, c], alpha / A[c, c])) with A[c, c] floored at 1e-10.
    /// </summary>
    private Tensor<T> UpdateDictionaryFromAccumulators(Tensor<T> a, Tensor<T> b, Tensor<T> components)
    {
        var engine = Engine;
        int k = components.Shape[0];
        var alpha = DimensionalityReductionOps.Scalar(DimensionalityReductionOps.Constant<T>(_alpha));
        T floor = DimensionalityReductionOps.Constant<T>(1e-10);

        for (int c = 0; c < k; c++)
        {
            var aRow = engine.TensorNarrow(a, 0, c, 1);                                // [1, k]
            var aDiagonal = engine.TensorNarrow(aRow, 1, c, 1);                        // [1, 1]
            var row = engine.TensorNarrow(components, 0, c, 1);                        // [1, p]

            // rhs = B[c] - A[c, :] C + A[c, c] C[c]
            var rhs = engine.TensorAdd(
                engine.TensorSubtract(engine.TensorNarrow(b, 0, c, 1), engine.TensorMatMul(aRow, components)),
                engine.TensorMultiply(aDiagonal, row));

            var denominator = engine.TensorClampMin(aDiagonal, floor);
            var updated = DimensionalityReductionOps.NormalizeRows(DimensionalityReductionOps.SoftThreshold(
                engine.TensorDivide(rhs, denominator), engine.TensorDivide(alpha, denominator)));
            components = DimensionalityReductionOps.WithRow(components, c, updated);
        }

        return components;
    }

    /// <summary>
    /// Transforms the data by projecting onto sparse components.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("MiniBatchSparsePCA has not been fitted.");
        }

        var centered = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        return DimensionalityReductionOps.MatMulTranspose(centered, Tensor<T>.FromMatrix(_components)).ToMatrix();
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("MiniBatchSparsePCA has not been fitted.");
        }

        int k = _components.Rows;
        var scores = DimensionalityReductionOps.FirstColumns(Tensor<T>.FromMatrix(data), k);
        var reconstructed = Engine.TensorMatMul(scores, Tensor<T>.FromMatrix(_components));
        return DimensionalityReductionOps.AddRow(reconstructed, Tensor<T>.FromVector(_mean)).ToMatrix();
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
            names[i] = $"MiniBatchSparsePC{i + 1}";
        }
        return names;
    }
}
