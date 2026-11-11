using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// PaCMAP: Pairwise Controlled Manifold Approximation.
/// </summary>
/// <remarks>
/// <para>
/// PaCMAP is a dimensionality reduction method designed to preserve both local and global
/// structure. It uses three types of point pairs with carefully controlled weights during
/// optimization to achieve better structure preservation than t-SNE or UMAP.
/// </para>
/// <para>
/// The algorithm:
/// 1. Create three types of pairs: nearby (local), mid-near (intermediate), further (global)
/// 2. Initialize embedding using PCA
/// 3. Optimize using attractive/repulsive forces with dynamic weighting
/// 4. Gradually shift from global to local focus during optimization
/// </para>
/// <para><b>For Beginners:</b> PaCMAP improves on UMAP/t-SNE by:
/// - Preserving global structure better through mid-near and further pairs
/// - Using controlled pair selection instead of random sampling
/// - Dynamically adjusting focus from global to local structure
/// - Being more robust to hyperparameter choices
///
/// Use cases:
/// - When you need faithful global structure
/// - When t-SNE/UMAP produces fragmented clusters
/// - When relative distances between clusters matter
/// - Biological data, image embeddings, document visualization
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PaCMAP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double _mnRatio;
    private readonly double _fpRatio;
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
    /// Creates a new instance of <see cref="PaCMAP{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for nearby pairs. Defaults to 10.</param>
    /// <param name="mnRatio">Ratio of mid-near pairs to neighbors. Defaults to 0.5.</param>
    /// <param name="fpRatio">Ratio of further pairs to neighbors. Defaults to 2.0.</param>
    /// <param name="nIter">Number of optimization iterations. Defaults to 450.</param>
    /// <param name="learningRate">Learning rate for optimization. Defaults to 1.0.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public PaCMAP(
        int nComponents = 2,
        int nNeighbors = 10,
        double mnRatio = 0.5,
        double fpRatio = 2.0,
        int nIter = 450,
        double learningRate = 1.0,
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

        if (mnRatio <= 0)
        {
            throw new ArgumentException("Mid-near ratio must be positive.", nameof(mnRatio));
        }

        if (fpRatio <= 0)
        {
            throw new ArgumentException("Further pair ratio must be positive.", nameof(fpRatio));
        }

        if (nIter < 1)
        {
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIter));
        }

        if (learningRate <= 0)
        {
            throw new ArgumentException("Learning rate must be positive.", nameof(learningRate));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _mnRatio = mnRatio;
        _fpRatio = fpRatio;
        _nIter = nIter;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits PaCMAP and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 2)
        {
            throw new ArgumentException("PaCMAP requires at least 2 samples.", nameof(data));
        }

        if (p < _nComponents)
        {
            throw new ArgumentException(
                $"Number of features ({p}) must be at least the number of components ({_nComponents}).",
                nameof(data));
        }

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Compute pairwise (squared) distances
        var distances = DimensionalityReductionOps.SquaredDistances(x, x);

        // Step 2: Generate pairs
        var (nearbyPairs, midNearPairs, furtherPairs) = GeneratePairs(distances, n, random);

        // Step 3: Initialize embedding with PCA
        var y = InitializeWithPCA(x, n);

        // Step 4: Optimize using PaCMAP loss
        _embedding = OptimizeEmbedding(y, nearbyPairs, midNearPairs, furtherPairs).ToMatrix();
    }

    /// <summary>
    /// Nearby pairs (the k nearest neighbours, ties in index order), mid-near pairs (random ranks in [k, 6k) of
    /// the sorted neighbour list) and further pairs (uniformly random other points), each as (heads, tails).
    /// </summary>
    private ((Tensor<int> Heads, Tensor<int> Tails) Nearby, (Tensor<int> Heads, Tensor<int> Tails) MidNear, (Tensor<int> Heads, Tensor<int> Tails) Further)
        GeneratePairs(Tensor<T> distances, int n, Random random)
    {
        int k = Math.Min(_nNeighbors, n - 1);
        int nMidNear = Math.Max(1, (int)(k * _mnRatio));
        int nFurther = Math.Max(1, (int)(k * _fpRatio));

        // Neighbour ranking of every row (the point itself excluded), copied once for the sequential sampling below
        int width = n - 1;
        var sorted = DimensionalityReductionOps.NearestNeighbors(distances, width).AsSpan().ToArray();

        var nearbyHeads = new List<int>();
        var nearbyTails = new List<int>();
        var midHeads = new List<int>();
        var midTails = new List<int>();
        var furtherHeads = new List<int>();
        var furtherTails = new List<int>();

        for (int i = 0; i < n; i++)
        {
            int rowStart = i * width;

            // Nearby pairs (k nearest neighbors)
            for (int j = 0; j < k && j < n - 1; j++)
            {
                nearbyHeads.Add(i);
                nearbyTails.Add(sorted[rowStart + j]);
            }

            // Mid-near pairs (neighbors at moderate distance)
            int midStart = k;
            int midEnd = Math.Min(6 * k, n - 1);
            for (int m = 0; m < nMidNear; m++)
            {
                if (midStart < midEnd)
                {
                    int idx = random.Next(midStart, midEnd);
                    if (idx < n - 1)
                    {
                        midHeads.Add(i);
                        midTails.Add(sorted[rowStart + idx]);
                    }
                }
            }

            // Further pairs (random sampling from distant points)
            for (int f = 0; f < nFurther; f++)
            {
                int idx = random.Next(n);
                while (idx == i)
                {
                    idx = random.Next(n);
                }

                furtherHeads.Add(i);
                furtherTails.Add(idx);
            }
        }

        return (
            (DimensionalityReductionOps.IndexTensor(nearbyHeads.ToArray()), DimensionalityReductionOps.IndexTensor(nearbyTails.ToArray())),
            (DimensionalityReductionOps.IndexTensor(midHeads.ToArray()), DimensionalityReductionOps.IndexTensor(midTails.ToArray())),
            (DimensionalityReductionOps.IndexTensor(furtherHeads.ToArray()), DimensionalityReductionOps.IndexTensor(furtherTails.ToArray())));
    }

    /// <summary>Centered data projected on the top principal axes, scaled by 1e-4.</summary>
    private Tensor<T> InitializeWithPCA(Tensor<T> x, int n)
    {
        var engine = Engine;
        var centered = DimensionalityReductionOps.SubtractRow(x, DimensionalityReductionOps.ColumnMeans(x));
        var covariance = engine.TensorDivideScalar(
            DimensionalityReductionOps.TransposeMatMul(centered, centered), NumOps.FromDouble(n));
        var (_, vectors) = DimensionalityReductionOps.SymmetricEigen(covariance, descending: true);
        var projected = engine.TensorMatMul(centered, DimensionalityReductionOps.FirstColumns(vectors, _nComponents));
        return engine.TensorMultiplyScalar(projected, NumOps.FromDouble(0.0001));
    }

    /// <summary>
    /// Full-batch gradient steps on the three pair sets with the phase weights of the original schedule, then a
    /// final re-centering.
    /// </summary>
    private Tensor<T> OptimizeEmbedding(
        Tensor<T> y,
        (Tensor<int> Heads, Tensor<int> Tails) nearbyPairs,
        (Tensor<int> Heads, Tensor<int> Tails) midNearPairs,
        (Tensor<int> Heads, Tensor<int> Tails) furtherPairs)
    {
        var engine = Engine;
        int n = y.Shape[0];
        double lr = _learningRate;

        // PaCMAP uses three phases with different weight schedules
        // Phase 1: Focus on global structure (further pairs)
        // Phase 2: Balance all pairs
        // Phase 3: Focus on local structure (nearby pairs)
        for (int iter = 0; iter < _nIter; iter++)
        {
            double t = (double)iter / _nIter;
            double wNearby, wMidNear, wFurther;

            if (t < 0.1)
            {
                // Early phase: emphasize global structure
                wNearby = 2.0;
                wMidNear = 3.0;
                wFurther = 1.0;
            }
            else if (t < 0.3)
            {
                // Transition phase
                wNearby = 3.0;
                wMidNear = 3.0;
                wFurther = 0.5;
            }
            else
            {
                // Late phase: emphasize local structure
                wNearby = 1.0;
                wMidNear = 0.5;
                wFurther = 0.1;
            }

            var gradients = new Tensor<T>(new[] { n, _nComponents });

            // Nearby: attraction 2w / (10 + d)^2; mid-near: attraction 2w 1e4 / (1e4 + d)^2; further: repulsion 2w / (1 + d)^2
            gradients = AccumulatePairGradient(gradients, y, nearbyPairs, 10.0, 2.0 * wNearby, attract: true);
            gradients = AccumulatePairGradient(gradients, y, midNearPairs, 10000.0, 2.0 * wMidNear * 10000.0, attract: true);
            gradients = AccumulatePairGradient(gradients, y, furtherPairs, 1.0, 2.0 * wFurther, attract: false);

            // Update embedding
            y = engine.TensorAdd(y, engine.TensorMultiplyScalar(gradients, DimensionalityReductionOps.Constant<T>(lr)));

            // Decay learning rate
            if (iter > 0 && iter % 50 == 0)
            {
                lr *= 0.95;
            }
        }

        // Center the final embedding
        return DimensionalityReductionOps.SubtractRow(y, DimensionalityReductionOps.ColumnMeans(y));
    }

    /// <summary>
    /// For every pair (h, t) with diff = y_h - y_t and c = numerator / (offset + |diff|^2)^2, adds -c diff to
    /// gradient h and +c diff to gradient t when attracting (the opposite signs when repelling).
    /// </summary>
    private Tensor<T> AccumulatePairGradient(
        Tensor<T> gradients, Tensor<T> y, (Tensor<int> Heads, Tensor<int> Tails) pairs, double offset, double numerator, bool attract)
    {
        if (pairs.Heads.Length == 0)
        {
            return gradients;
        }

        var engine = Engine;
        var diff = engine.TensorSubtract(
            DimensionalityReductionOps.SelectRows(y, pairs.Heads), DimensionalityReductionOps.SelectRows(y, pairs.Tails));
        var denom = engine.TensorAddScalar(
            DimensionalityReductionOps.RowSums(engine.TensorSquare(diff)), DimensionalityReductionOps.Constant<T>(offset));
        var coefficient = engine.TensorDivide(
            DimensionalityReductionOps.Full(denom.Length, DimensionalityReductionOps.Constant<T>(numerator)), engine.TensorSquare(denom));
        var step = DimensionalityReductionOps.ScaleRows(diff, coefficient);
        var headStep = attract ? engine.TensorNegate(step) : step;
        var tailStep = attract ? step : engine.TensorNegate(step);
        gradients = engine.TensorIndexAdd(gradients, 0, pairs.Heads, headStep);
        return engine.TensorIndexAdd(gradients, 0, pairs.Tails, tailStep);
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("PaCMAP has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "PaCMAP does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PaCMAP does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"PaCMAP{i + 1}";
        }
        return names;
    }
}
