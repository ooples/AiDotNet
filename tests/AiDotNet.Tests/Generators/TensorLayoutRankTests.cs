using AiDotNet.Attributes;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Pins the layout rank rule, and pins that <see cref="TensorLayoutAttribute.AcceptsRank"/> is a
/// caller of it rather than a second copy of it.
/// </summary>
/// <remarks>
/// <para>
/// The rule was implemented twice — here in the attribute, against a loaded instance, and in
/// <c>ShapeDeclarationValidationGenerator.Layout.AcceptedRanks</c>, against attribute arguments read
/// as symbols — and the two copies drifted apart in OPPOSITE directions. The attribute lacked the
/// "more than one axis" guard, so a single-axis batch-optional layout accepted rank 0. The generator
/// lacked the "first axis is Batch" guard, so it raised a build error for a rank the runtime accepts.
/// </para>
/// <para>
/// Both now call <see cref="TensorLayoutRank.Accepts"/>, one file compiled into both assemblies, so
/// the two cannot disagree by construction. What still needs a test is the rule itself and the
/// attribute's delegation to it — the cases below are the ones each copy used to get wrong.
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

    [Theory]
    [MemberData(nameof(RankTable))]
    public void Accepts_MatchesTheRule(int axisCount, bool batchOptional, bool firstIsBatch, int rank, bool expected)
    {
        Assert.Equal(expected, TensorLayoutRank.Accepts(axisCount, batchOptional, firstIsBatch, rank));
    }

    /// <summary>
    /// Drives the same table through the public attribute, so the delegation cannot be removed
    /// without this failing.
    /// </summary>
    [Theory]
    [MemberData(nameof(RankTable))]
    public void AcceptsRank_AgreesWithTheRule(int axisCount, bool batchOptional, bool firstIsBatch, int rank, bool expected)
    {
        var axes = new TensorAxis[axisCount];
        for (int i = 0; i < axisCount; i++)
        {
            // Distinct roles after the first, so this never accidentally exercises the
            // duplicate-axis path instead of the rank path.
            axes[i] = i == 0
                ? (firstIsBatch ? TensorAxis.Batch : TensorAxis.Channels)
                : (TensorAxis)(i + 2);
        }

        var attribute = new TensorLayoutAttribute(axes) { BatchOptional = batchOptional };

        Assert.Equal(expected, attribute.AcceptsRank(rank));
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
}
