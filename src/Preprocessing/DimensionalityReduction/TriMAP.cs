using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// TriMAP: Large-scale Dimensionality Reduction Using Triplets.
/// </summary>
/// <remarks>
/// <para>
/// TriMAP is a dimensionality reduction method that uses triplet constraints to preserve
/// both local and global structure. It outperforms t-SNE and UMAP on many datasets while
/// being more robust to hyperparameter choices.
/// </para>
/// <para>
/// The algorithm:
/// 1. Generate triplets (anchor, positive, negative) based on distance relationships
/// 2. Initialize embedding using PCA
/// 3. Optimize using triplet loss to preserve distance rankings
/// 4. Weight triplets to balance local and global structure
/// </para>
/// <para><b>For Beginners:</b> TriMAP preserves distance relationships using triplets:
/// - A triplet (i, j, k) means point i is closer to j than to k
/// - Optimization ensures these relationships hold in the embedding
/// - More accurate than t-SNE/UMAP for many datasets
/// - Less sensitive to parameter tuning
///
/// Use cases:
/// - Large-scale visualization (millions of points)
/// - When t-SNE/UMAP produces poor results
/// - When you need faithful global structure
/// - Complex datasets with hierarchical structure
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TriMAP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nInliers;
    private readonly int _nOutliers;
    private readonly int _nRandom;
    private readonly int _nIter;
    private readonly double _learningRate;
    private readonly int? _randomState;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="TriMAP{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nInliers">Number of nearest neighbors for triplet generation. Defaults to 10.</param>
    /// <param name="nOutliers">Number of outliers per point. Defaults to 5.</param>
    /// <param name="nRandom">Number of random triplets per point. Defaults to 3.</param>
    /// <param name="nIter">Number of optimization iterations. Defaults to 400.</param>
    /// <param name="learningRate">Learning rate for optimization. Defaults to 100. Higher values may cause instability.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public TriMAP(
        int nComponents = 2,
        int nInliers = 10,
        int nOutliers = 5,
        int nRandom = 3,
        int nIter = 400,
        double learningRate = 100.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nInliers < 1)
        {
            throw new ArgumentException("Number of inliers must be at least 1.", nameof(nInliers));
        }

        if (learningRate <= 0 || learningRate > 10000)
        {
            throw new ArgumentException(
                "Learning rate must be positive and not exceed 10000 to avoid numerical instability.",
                nameof(learningRate));
        }

        if (nIter < 1)
        {
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIter));
        }

        if (nOutliers < 0)
        {
            throw new ArgumentException("Number of outliers cannot be negative.", nameof(nOutliers));
        }

        if (nRandom < 0)
        {
            throw new ArgumentException("Number of random triplets cannot be negative.", nameof(nRandom));
        }

        _nComponents = nComponents;
        _nInliers = nInliers;
        _nOutliers = nOutliers;
        _nRandom = nRandom;
        _nIter = nIter;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits TriMAP and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;

        // TriMAP requires at least 3 samples to form triplets (anchor, positive, negative)
        if (n < 3)
        {
            throw new ArgumentException("TriMAP requires at least 3 samples to form triplets.", nameof(data));
        }

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Compute pairwise distances
        var distances = DimensionalityReductionOps.EuclideanDistances(x, x);

        // Step 2: Generate candidate triplets (random ones are filtered by the weights below)
        var (anchors, positives, negatives, isRandom) = GenerateTriplets(distances, n, random);

        // Step 3: Compute triplet weights
        var weights = ComputeTripletWeights(anchors, positives, negatives, isRandom, distances, n);

        // Step 4: Initialize embedding with PCA
        var y = InitializeWithPCA(x, n, random);

        // Step 5: Optimize using triplet loss
        _embedding = OptimizeEmbedding(y, anchors, positives, negatives, weights).ToMatrix();
    }

    /// <summary>
    /// Inlier triplets (each of the k nearest neighbours as positive with the next-ranked points as negatives)
    /// and random triplets (random neighbour as positive, random point as negative). Random triplets are only
    /// kept when the positive is strictly closer than the negative; that test is applied as a zero weight in
    /// <see cref="ComputeTripletWeights"/>, flagged by the returned 0/1 tensor.
    /// </summary>
    private (Tensor<int> Anchors, Tensor<int> Positives, Tensor<int> Negatives, Tensor<T> IsRandom) GenerateTriplets(
        Tensor<T> distances, int n, Random random)
    {
        var engine = Engine;
        int k = Math.Min(_nInliers, n - 1);
        int nOut = Math.Min(_nOutliers, n - 1);

        // Full neighbour ranking of every row (the point itself last), ties in index order
        var masked = distances.Clone();
        engine.TensorFillDiagonalInPlace(masked, NumOps.FromDouble(double.PositiveInfinity));
        var sorted = AiDotNet.AnomalyDetection.AnomalyTensorOps.StableSortRows(masked, descending: false).Order.AsSpan().ToArray();

        var anchors = new List<int>();
        var positives = new List<int>();
        var negatives = new List<int>();
        var randomFlags = new List<int>();

        for (int i = 0; i < n; i++)
        {
            int rowStart = i * n;

            // Inlier triplets (nearest neighbors as positives)
            for (int j = 0; j < k; j++)
            {
                int positive = sorted[rowStart + j];

                // Mid-range negatives
                for (int m = 0; m < Math.Min(nOut, k); m++)
                {
                    int negIdx = Math.Min(k + m, n - 1);
                    int negative = sorted[rowStart + negIdx];
                    if (positive != negative)
                    {
                        anchors.Add(i);
                        positives.Add(positive);
                        negatives.Add(negative);
                        randomFlags.Add(0);
                    }
                }
            }

            // Random triplets for global structure
            for (int r = 0; r < _nRandom; r++)
            {
                if (k > 0)
                {
                    int posIdx = random.Next(k);
                    int positive = sorted[rowStart + posIdx];
                    int negative = random.Next(n);
                    while (negative == i || negative == positive)
                    {
                        negative = random.Next(n);
                    }

                    anchors.Add(i);
                    positives.Add(positive);
                    negatives.Add(negative);
                    randomFlags.Add(1);
                }
            }
        }

        return (
            DimensionalityReductionOps.IndexTensor(anchors.ToArray()),
            DimensionalityReductionOps.IndexTensor(positives.ToArray()),
            DimensionalityReductionOps.IndexTensor(negatives.ToArray()),
            new Tensor<int>(randomFlags.ToArray(), new[] { randomFlags.Count }).Cast<T>());
    }

    /// <summary>
    /// w = min(log(1 + dNeg / (dPos + 1e-10)) scale / (dNeg + 1e-10), 10) with scale the largest pairwise
    /// distance (1 if zero), zero for random triplets whose positive is not strictly closer, normalized to sum one.
    /// </summary>
    private Tensor<T> ComputeTripletWeights(
        Tensor<int> anchors, Tensor<int> positives, Tensor<int> negatives, Tensor<T> isRandom, Tensor<T> distances, int n)
    {
        var engine = Engine;
        int count = anchors.Length;
        if (count == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        var flat = engine.Reshape(distances, new[] { n * n });
        var anchorRows = engine.TensorMultiplyScalar(anchors, n);
        var dPos = engine.TensorIndexSelect(flat, engine.TensorAdd(anchorRows, positives), 0);
        var dNeg = engine.TensorIndexSelect(flat, engine.TensorAdd(anchorRows, negatives), 0);

        // Compute distance scale
        var maxDist = engine.ReduceMax(flat, new[] { 0 }, true);
        var scale = engine.TensorWhere(engine.TensorGreaterThan(maxDist, NumOps.Zero), maxDist, engine.TensorOnesLike(maxDist));

        var epsilon = DimensionalityReductionOps.Tolerance<T>(1e-10);
        var ratio = engine.TensorDivide(dNeg, engine.TensorAddScalar(dPos, epsilon));
        var weights = engine.TensorLog(engine.TensorAddScalar(ratio, NumOps.One));
        weights = engine.TensorDivide(engine.TensorMultiply(weights, scale), engine.TensorAddScalar(dNeg, epsilon));
        weights = engine.TensorClampMax(weights, NumOps.FromDouble(10.0));

        // Random triplets only count when d(anchor, positive) < d(anchor, negative)
        var keep = engine.TensorAdd(
            engine.TensorSubtract(engine.TensorOnesLike(isRandom), isRandom),
            engine.TensorMultiply(isRandom, engine.TensorLessThan(dPos, dNeg)));
        weights = engine.TensorMultiply(weights, keep);

        // Normalize weights
        var sum = engine.ReduceSum(weights, new[] { 0 }, true);
        var positive = engine.TensorGreaterThan(sum, NumOps.Zero);
        return engine.TensorDivide(weights, engine.TensorWhere(positive, sum, engine.TensorOnesLike(sum)));
    }

    /// <summary>Centered data projected on the top principal axes, each component scaled to unit variance.</summary>
    private Tensor<T> InitializeWithPCA(Tensor<T> x, int n, Random random)
    {
        var engine = Engine;
        int p = x.Shape[1];
        var centered = DimensionalityReductionOps.SubtractRow(x, DimensionalityReductionOps.ColumnMeans(x));
        var covariance = engine.TensorDivideScalar(
            DimensionalityReductionOps.TransposeMatMul(centered, centered), NumOps.FromDouble(n));

        // Top eigenvectors (zero columns past the feature count)
        var (_, vectors) = DimensionalityReductionOps.SymmetricEigen(covariance, descending: true);
        // Signs as the original deflated power iteration converges to them from its seeded start vectors
        vectors = DimensionalityReductionOps.MatchStartSigns(vectors,
            DimensionalityReductionOps.PowerIterationStarts<T>(random, _nComponents, x.Shape[1], 0.5));
        int available = Math.Min(_nComponents, p);
        var y = engine.TensorMatMul(centered, DimensionalityReductionOps.FirstColumns(vectors, available));
        if (available < _nComponents)
        {
            y = engine.TensorConcatenate(new[] { y, new Tensor<T>(new[] { n, _nComponents - available }) }, 1);
        }

        // Scale to unit variance (components with std <= 1e-10 are only centered)
        y = DimensionalityReductionOps.SubtractRow(y, DimensionalityReductionOps.ColumnMeans(y));
        var std = engine.TensorSqrt(DimensionalityReductionOps.ColumnMeans(engine.TensorSquare(y)));
        var divisor = engine.TensorWhere(
            engine.TensorGreaterThan(std, DimensionalityReductionOps.Tolerance<T>(1e-10)), std, engine.TensorOnesLike(std));
        return DimensionalityReductionOps.ScaleColumns(y, engine.TensorReciprocal(divisor));
    }

    /// <summary>
    /// Full-batch momentum descent on the weighted margin-1 triplet loss max(0, dPos - dNeg + 1) with
    /// d = sqrt(|.|^2 + 1e-10). The gradient sums every triplet against the same positions, so the original
    /// per-iteration shuffle only changed the summation order and is not repeated.
    /// </summary>
    private Tensor<T> OptimizeEmbedding(
        Tensor<T> y, Tensor<int> anchors, Tensor<int> positives, Tensor<int> negatives, Tensor<T> weights)
    {
        var engine = Engine;
        int n = y.Shape[0];
        double lr = _learningRate;
        var velocity = new Tensor<T>(new[] { n, _nComponents });
        var momentum = NumOps.FromDouble(0.5);
        var lateMomentum = NumOps.FromDouble(0.8);
        var epsilon = DimensionalityReductionOps.Tolerance<T>(1e-10);
        var margin = NumOps.FromDouble(1.0);

        for (int iter = 0; iter < _nIter; iter++)
        {
            var gradients = new Tensor<T>(new[] { n, _nComponents });

            if (anchors.Length > 0)
            {
                var anchorRows = DimensionalityReductionOps.SelectRows(y, anchors);
                var diffPos = engine.TensorSubtract(anchorRows, DimensionalityReductionOps.SelectRows(y, positives));
                var diffNeg = engine.TensorSubtract(anchorRows, DimensionalityReductionOps.SelectRows(y, negatives));
                var dPos = engine.TensorSqrt(engine.TensorAddScalar(DimensionalityReductionOps.RowSums(engine.TensorSquare(diffPos)), epsilon));
                var dNeg = engine.TensorSqrt(engine.TensorAddScalar(DimensionalityReductionOps.RowSums(engine.TensorSquare(diffNeg)), epsilon));

                // Active triplets: dPos - dNeg + margin > 0
                var active = engine.TensorGreaterThan(engine.TensorAddScalar(engine.TensorSubtract(dPos, dNeg), margin), NumOps.Zero);
                var activeWeights = engine.TensorMultiply(weights, active);

                var gradPos = DimensionalityReductionOps.ScaleRows(
                    diffPos, engine.TensorDivide(activeWeights, engine.TensorAddScalar(dPos, epsilon)));
                var gradNeg = DimensionalityReductionOps.ScaleRows(
                    diffNeg, engine.TensorDivide(activeWeights, engine.TensorAddScalar(dNeg, epsilon)));

                gradients = engine.TensorIndexAdd(gradients, 0, anchors, engine.TensorSubtract(gradPos, gradNeg));
                gradients = engine.TensorIndexAdd(gradients, 0, positives, engine.TensorNegate(gradPos));
                gradients = engine.TensorIndexAdd(gradients, 0, negatives, gradNeg);
            }

            // Update with momentum
            if (iter > 100)
            {
                momentum = lateMomentum;
            }

            velocity = engine.TensorSubtract(
                engine.TensorMultiplyScalar(velocity, momentum),
                engine.TensorMultiplyScalar(gradients, DimensionalityReductionOps.Constant<T>(lr)));
            y = engine.TensorAdd(y, velocity);

            // Decay learning rate
            if (iter > 0 && iter % 100 == 0)
            {
                lr *= 0.9;
            }
        }

        return y;
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("TriMAP has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "TriMAP does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TriMAP does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"TriMAP{i + 1}";
        }
        return names;
    }
}
