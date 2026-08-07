namespace AiDotNet.Attributes;

/// <summary>
/// The rank rule for a tensor layout, as a pure function over primitives.
/// </summary>
/// <remarks>
/// <para>
/// ONE RULE, ONE IMPLEMENTATION, TWO CALLERS. This decision used to be written twice — once in
/// <see cref="TensorLayoutAttribute.AcceptsRank"/> against a loaded attribute instance, and once in
/// <c>ShapeDeclarationValidationGenerator.Layout.AcceptedRanks</c> against attribute arguments read
/// as symbols. The two copies had already drifted APART IN OPPOSITE DIRECTIONS: the attribute lacked
/// the "more than one axis" guard, so a single-axis batch-optional layout accepted rank 0; the
/// generator lacked the "first axis is Batch" guard, so it reported a build ERROR for a rank the
/// runtime would have accepted. A rule duplicated in two places is a rule that will disagree with
/// itself, and a parity test between two copies only reports the disagreement after it happens.
/// </para>
/// <para>
/// The generator genuinely cannot call <see cref="TensorLayoutAttribute.AcceptsRank"/>: it runs
/// inside the compiler against symbols, and the attribute type is not loaded. But that constraint is
/// about the ATTRIBUTE INSTANCE, not about the rule — expressed over an axis count, a flag, and a
/// rank, the rule needs neither reflection nor Roslyn. This file is therefore compiled into both
/// assemblies (see the <c>Compile Include</c> in <c>AiDotNet.Generators.csproj</c>), so there is one
/// implementation and nothing left to keep in step.
/// </para>
/// </remarks>
internal static class TensorLayoutRank
{
    /// <summary>
    /// True when a tensor of <paramref name="rank"/> could match a layout of
    /// <paramref name="declaredAxisCount"/> axes, allowing for an omitted optional batch axis.
    /// </summary>
    /// <param name="declaredAxisCount">How many axes the layout declares.</param>
    /// <param name="batchOptional">Whether the layout marks its batch axis optional.</param>
    /// <param name="firstAxisIsBatch">Whether the layout's first declared axis is the batch axis.</param>
    /// <param name="rank">The rank of the tensor being checked.</param>
    /// <remarks>
    /// <paramref name="firstAxisIsBatch"/> is a parameter rather than read from an axis array because
    /// the generator sees axis names as strings and the attribute sees them as enum members; passing
    /// the answer keeps the rule free of both representations.
    ///
    /// The count must exceed one, not zero: dropping the batch axis from a one-axis layout leaves a
    /// rank-0 tensor, which is a scalar, not an unbatched form of anything.
    /// </remarks>
    public static bool Accepts(int declaredAxisCount, bool batchOptional, bool firstAxisIsBatch, int rank)
    {
        if (rank == declaredAxisCount) return true;

        return batchOptional
            && declaredAxisCount > 1
            && firstAxisIsBatch
            && rank == declaredAxisCount - 1;
    }
}
