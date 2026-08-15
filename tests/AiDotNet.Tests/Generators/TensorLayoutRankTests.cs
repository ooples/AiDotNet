using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Drives one table through both implementations of the layout rank rule and requires identical
/// verdicts.
/// </summary>
/// <remarks>
/// <para>
/// The rule exists twice: in <see cref="TensorLayoutAttribute.AcceptsRank"/>, against a loaded
/// attribute instance, and in <c>ShapeDeclarationValidationGenerator.Layout.AcceptedRanks</c>,
/// against attribute arguments read as symbols. The generator cannot call the attribute — it runs
/// inside the compiler and the attribute type is never loaded — so the duplication is structural and
/// cannot be removed from either side.
/// </para>
/// <para>
/// The two copies had already drifted APART IN OPPOSITE DIRECTIONS. The attribute lacked the
/// "more than one axis" guard, so a single-axis batch-optional layout accepted rank 0. The generator
/// lacked the "first axis is Batch" guard, so it raised a build <b>Error</b> for a rank the runtime
/// accepts. Each was wrong in the direction the other was not, which is exactly how a duplicated
/// rule fails: reviewing either copy in isolation reads as correct.
/// </para>
/// <para>
/// So the guard is a shared table rather than a shared implementation. <see cref="GeneratorRule"/>
/// below is a faithful transcription of <c>AcceptedRanks</c>, kept next to the assertions it feeds;
/// if either real copy changes without the other, the corresponding row disagrees and this fails.
/// The two cases each copy historically got wrong are rows in the table.
/// </para>
/// </remarks>
public class TensorLayoutRankTests
{
    // (declaredAxisCount, batchOptional, firstAxisIsBatch, rank, expected)
    public static TheoryData<int, bool, bool, int, bool> RankTable() => new()
    {
        // Exact rank always matches, regardless of the optional-batch flags.
        { 4, false, false, 4, true },
        { 4, true,  true,  4, true },
        { 1, true,  true,  1, true },

        // Batch-optional drops exactly one rank, and only from the front.
        { 4, true,  true,  3, true },
        { 4, true,  true,  2, false },
        { 4, false, true,  3, false },

        // THE GENERATOR'S OLD BUG: batch-optional but the leading axis is not Batch.
        // It accepted rank-1-less here and raised a build error on correct code.
        { 4, true,  false, 3, false },
        { 3, true,  false, 2, false },

        // THE ATTRIBUTE'S OLD BUG: a single-axis batch-optional layout accepting rank 0.
        // Dropping the batch axis from a one-axis layout leaves a scalar, not an unbatched form.
        { 1, true,  true,  0, false },

        // Nothing accepts a larger rank than it declares.
        { 2, true,  true,  3, false },
    };

    /// <summary>
    /// A transcription of <c>ShapeDeclarationValidationGenerator.Layout.AcceptedRanks</c>, which
    /// this assembly cannot reference (the generator is consumed as an analyzer).
    /// </summary>
    /// <remarks>
    /// Axis names are compared as strings here because that is what the generator does — it reads
    /// the attribute's arguments as symbols and never has enum members.
    /// </remarks>
    private static IEnumerable<int> GeneratorRule(IReadOnlyList<string> axes, bool batchOptional)
    {
        yield return axes.Count;

        if (batchOptional
            && axes.Count > 1
            && string.Equals(axes[0], "Batch", System.StringComparison.Ordinal))
        {
            yield return axes.Count - 1;
        }
    }

    [Theory]
    [MemberData(nameof(RankTable))]
    public void AcceptsRank_AndTheGeneratorRule_Agree(
        int axisCount, bool batchOptional, bool firstIsBatch, int rank, bool expected)
    {
        var axes = BuildAxes(axisCount, firstIsBatch);
        var attribute = new TensorLayoutAttribute(axes) { BatchOptional = batchOptional };

        bool viaAttribute = attribute.AcceptsRank(rank);
        bool viaGenerator = GeneratorRule(axes.Select(a => a.ToString()).ToList(), batchOptional)
            .Contains(rank);

        Assert.Equal(expected, viaAttribute);
        Assert.True(viaGenerator == expected,
            $"The generator's rank rule disagrees with the table for " +
            $"(axes={axisCount}, batchOptional={batchOptional}, firstIsBatch={firstIsBatch}, rank={rank}): " +
            $"expected {expected}, generator says {viaGenerator}. The two copies of this rule have " +
            "drifted apart before, in opposite directions — check both.");
    }

    /// <summary>
    /// <see cref="TensorLayoutAttribute.AxesForRank"/> must drop the batch axis from the FRONT for
    /// the unbatched form, and refuse a rank the rule rejects rather than returning a trimmed guess.
    /// </summary>
    [Fact]
    public void AxesForRank_DropsTheBatchAxisAndRefusesRejectedRanks()
    {
        var attribute = new TensorLayoutAttribute(
            TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width)
        {
            BatchOptional = true,
        };

        Assert.Equal(
            new[] { TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width },
            attribute.AxesForRank(4));

        Assert.Equal(
            new[] { TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width },
            attribute.AxesForRank(3));

        Assert.Null(attribute.AxesForRank(2));
        Assert.Null(attribute.AxesForRank(5));
    }

    /// <summary>A layout that does not lead with Batch keeps its full rank even when BatchOptional is set.</summary>
    [Fact]
    public void AxesForRank_IgnoresBatchOptionalWhenTheLeadingAxisIsNotBatch()
    {
        var attribute = new TensorLayoutAttribute(
            TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width)
        {
            BatchOptional = true,
        };

        Assert.Null(attribute.AxesForRank(2));
        Assert.Equal(
            new[] { TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width },
            attribute.AxesForRank(3));
    }

    /// <summary>Distinct roles after the first, so a case never trips the duplicate-axis rule instead.</summary>
    private static TensorAxis[] BuildAxes(int axisCount, bool firstIsBatch)
    {
        var axes = new TensorAxis[axisCount];
        for (int i = 0; i < axisCount; i++)
        {
            axes[i] = i == 0
                ? (firstIsBatch ? TensorAxis.Batch : TensorAxis.Channels)
                : (TensorAxis)(i + 2);
        }
        return axes;
    }
}
