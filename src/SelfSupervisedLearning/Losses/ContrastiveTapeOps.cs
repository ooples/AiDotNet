using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// The tape-connected building blocks the contrastive objectives share.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// ONE COPY OF EACH, because these had begun to multiply: row-wise L2 normalization existed in
/// BYOLLoss, InfoNCELoss and NTXentLoss, and the identity constant in BarlowTwinsLoss and
/// InfoNCELoss. Six implementations of one family drifting apart is what made the family
/// undifferentiable in the first place; duplicating their replacements would rebuild the same
/// problem one level down.
/// </para>
/// <para>
/// The constant matrices are CACHED. Barlow Twins runs at projector dimensions of 4096 or 8192,
/// where rebuilding a dim x dim tensor with a host loop on every training step costs two dim^2
/// allocations and a dim^2 fill for data that never changes. They are never mutated after
/// construction, so one instance serves every caller at that size.
/// </para>
/// </remarks>
internal static class ContrastiveTapeOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private static readonly ConcurrentDictionary<int, Tensor<T>> IdentityCache = new();
    private static readonly ConcurrentDictionary<(int Size, double Lambda), Tensor<T>> DiagonalWeightCache = new();
    private static readonly ConcurrentDictionary<int, Tensor<T>> SelfMaskCache = new();
    private static readonly ConcurrentDictionary<int, Tensor<T>> PositivePairCache = new();

    /// <summary>Epsilon folded inside the square root of every norm computed here.</summary>
    /// <remarks>
    /// Inside the root rather than added to the norm afterwards, so a zero row scales by a finite
    /// value instead of dividing by zero AND its derivative stays finite there too.
    /// </remarks>
    private const double NormEpsilon = 1e-12;

    /// <summary>Row-wise L2 normalization, on the tape.</summary>
    public static Tensor<T> L2NormalizeRows(Tensor<T> x)
    {
        var squaredNorm = Engine.ReduceSum(Engine.TensorMultiply(x, x), new[] { 1 }, keepDims: true);
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(squaredNorm, NumOps.FromDouble(NormEpsilon)));

        return Engine.TensorDivide(x, norm);
    }

    /// <summary>Constant <c>[size, size]</c> identity.</summary>
    public static Tensor<T> Identity(int size) => IdentityCache.GetOrAdd(size, s =>
    {
        var identity = new Tensor<T>(new[] { s, s });
        for (int i = 0; i < s; i++) identity[(i * s) + i] = NumOps.One;
        return identity;
    });

    /// <summary>Constant <c>[size, size]</c> weights: 1 on the diagonal, <paramref name="lambda"/> off it.</summary>
    public static Tensor<T> DiagonalWeights(int size, double lambda)
        => DiagonalWeightCache.GetOrAdd((size, lambda), key =>
        {
            var weights = new Tensor<T>(new[] { key.Size, key.Size });
            var off = NumOps.FromDouble(key.Lambda);
            for (int i = 0; i < key.Size; i++)
                for (int j = 0; j < key.Size; j++)
                    weights[(i * key.Size) + j] = i == j ? NumOps.One : off;

            return weights;
        });

    /// <summary>
    /// Constant additive mask that removes each anchor's similarity with itself.
    /// </summary>
    /// <remarks>
    /// Finite (-1e9) rather than negative infinity: inf - inf is NaN, which a fully-masked row
    /// would produce. Log-softmax drives these terms to zero, which is what an <c>i != j</c> guard
    /// achieves in a host loop.
    /// </remarks>
    public static Tensor<T> SelfSimilarityMask(int size) => SelfMaskCache.GetOrAdd(size, s =>
    {
        var mask = new Tensor<T>(new[] { s, s });
        var blocked = NumOps.FromDouble(-1e9);
        for (int i = 0; i < s; i++) mask[(i * s) + i] = blocked;
        return mask;
    });

    /// <summary>
    /// Constant one-hot selector pairing each anchor with its partner view: row <c>i</c> selects
    /// column <c>i + batchSize</c> for the first view and <c>i - batchSize</c> for the second.
    /// </summary>
    public static Tensor<T> PositivePairSelector(int batchSize)
        => PositivePairCache.GetOrAdd(batchSize, b =>
        {
            int total = 2 * b;
            var selector = new Tensor<T>(new[] { total, total });
            for (int i = 0; i < total; i++)
            {
                int positive = i < b ? i + b : i - b;
                selector[(i * total) + positive] = NumOps.One;
            }

            return selector;
        });

    /// <summary>
    /// Requires two views to be rank-2 and identically shaped.
    /// </summary>
    /// <remarks>
    /// SHAPES, not just ranks. A rank-only check lets [8, 196, 1] through against [8, 196, 768],
    /// and the engine's implicit broadcasting then returns a finite loss computed against the wrong
    /// target -- silently. Equally, a batch of 1 against a batch of 8 broadcasts rather than
    /// failing.
    /// </remarks>
    public static void RequireMatchingRank2(
        Tensor<T> view1, Tensor<T> view2, string lossName, string firstParameterName, string secondParameterName)
    {
        if (view1 is null) throw new ArgumentNullException(firstParameterName);
        if (view2 is null) throw new ArgumentNullException(secondParameterName);

        if (view1.Shape.Length != 2 || view2.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"{lossName} expects rank-2 [batch, dim] tensors; got ranks {view1.Shape.Length} "
                + $"and {view2.Shape.Length}.", firstParameterName);
        }

        if (view1.Shape[0] != view2.Shape[0] || view1.Shape[1] != view2.Shape[1])
        {
            throw new ArgumentException(
                $"{lossName} requires both views to have the same shape; got "
                + $"[{view1.Shape[0]}, {view1.Shape[1]}] and [{view2.Shape[0]}, {view2.Shape[1]}].",
                secondParameterName);
        }
    }
}
