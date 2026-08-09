using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// The shape precondition the contrastive objectives share.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This class used to also hold row-wise L2 normalization and four cached constant matrices --
/// identity, diagonal weights, a self-similarity mask and a positive-pair selector -- built to stop
/// the same helpers being written once per loss. <c>ObjectiveOps</c> now provides that maths for the
/// whole family, so those members were duplicates of it and are gone.
/// </para>
/// <para>
/// Removing them also removed what they cost: they were cached in <c>static readonly</c>
/// dictionaries on a generic type, so every closed construction kept its own set for the lifetime of
/// the process with nothing evicting them. At the projector dimensions that motivated the cache --
/// 4096 and 8192 -- a single identity is 8192x8192 elements, about 512 MB at <c>double</c>, retained
/// after the losses that asked for it were long collected. A cache of large dense constants keyed by
/// every size ever seen is a leak with a lookup table in front of it.
/// </para>
/// </remarks>
internal static class ContrastiveTapeOps<T>
{
    /// <summary>
    /// Requires two views to be rank-2 and identically shaped.
    /// </summary>
    /// <remarks>
    /// SHAPES, not just ranks. A rank-only check lets [8, 196, 1] through against [8, 196, 768], and
    /// the engine's implicit broadcasting then returns a finite loss computed against the wrong
    /// target -- silently. Equally, a batch of 1 against a batch of 8 broadcasts rather than failing.
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

        // Positive, not merely equal. A zero-length batch or width shares its shape with the other
        // view, so it passes the checks above and then produces a reduction over nothing -- a
        // meaningless zero loss rather than a failure.
        if (view1.Shape[0] <= 0 || view1.Shape[1] <= 0)
        {
            throw new ArgumentException(
                $"{lossName} requires a non-empty [batch, dim]; got "
                + $"[{view1.Shape[0]}, {view1.Shape[1]}].", firstParameterName);
        }
    }
}
