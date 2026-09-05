using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Novelty;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs.Novelty;

public sealed class ProgramGenomeDistanceTests
{
    [Fact]
    public void TokenSetDistanceIsZeroForIdenticalPrograms()
    {
        var distance = new ProgramTokenSetDistance();
        var first = new ProgramGenome("def f(x):\n    return x + 1\n");
        var second = new ProgramGenome("def f(x):\r\n    return x + 1\r\n\r\n");

        Assert.Equal(0.0, distance.Distance(first, second));
    }

    [Fact]
    public void TokenSetDistanceIsOneForDisjointVocabularies()
    {
        var distance = new ProgramTokenSetDistance();
        var first = new ProgramGenome("alpha beta gamma");
        var second = new ProgramGenome("delta epsilon zeta");

        Assert.Equal(1.0, distance.Distance(first, second));
    }

    [Fact]
    public void TokenSetDistanceMatchesTheJaccardComplement()
    {
        // {a, b, c} against {b, c, d}: two shared of four distinct, so 1 - 2/4.
        Assert.Equal(0.5, ProgramTokenSetDistance.ComputeDistance("a b c", "b c d"), 10);
    }

    [Fact]
    public void TokenSetDistanceIsSymmetricAndBounded()
    {
        var distance = new ProgramTokenSetDistance();
        var first = new ProgramGenome("total = 0\nfor item in items:\n    total += item\n");
        var second = new ProgramGenome("total = 0\nfor value in values:\n    total = total + value\n");

        double forward = distance.Distance(first, second);
        double backward = distance.Distance(second, first);

        Assert.Equal(forward, backward);
        Assert.InRange(forward, 0.0, 1.0);
        Assert.True(forward > 0.0);
    }

    [Fact]
    public void TokenSetDistanceIgnoresPurelyCosmeticChange()
    {
        var distance = new ProgramTokenSetDistance();
        var compact = new ProgramGenome("x=1\ny=2\n");
        var spaced = new ProgramGenome("x = 1\ny  =  2   \n");

        Assert.Equal(0.0, distance.Distance(compact, spaced));
    }

    [Fact]
    public void LineEditDistanceCountsChangedLinesOverTheLongerProgram()
    {
        // Four lines, one of them different: one edit over four lines.
        double distance = ProgramLineEditDistance.ComputeDistance(
            "a\nb\nc\nd",
            "a\nb\nZ\nd");

        Assert.Equal(0.25, distance, 10);
    }

    [Fact]
    public void LineEditDistanceIsZeroForIdenticalProgramsAndOneForFullReplacement()
    {
        var distance = new ProgramLineEditDistance();
        var first = new ProgramGenome("a\nb\nc");
        var same = new ProgramGenome("a\nb\nc\n");
        var different = new ProgramGenome("x\ny\nz");

        Assert.Equal(0.0, distance.Distance(first, same));
        Assert.Equal(1.0, distance.Distance(first, different));
    }

    [Fact]
    public void LineEditDistanceIsSymmetricAndSeesReordering()
    {
        var edit = new ProgramLineEditDistance();
        var tokens = new ProgramTokenSetDistance();
        var first = new ProgramGenome("open()\nread()\nclose()");
        var reordered = new ProgramGenome("close()\nread()\nopen()");

        Assert.Equal(edit.Distance(first, reordered), edit.Distance(reordered, first));

        // The set metric cannot see a pure reordering; the sequence metric can. That is why both ship.
        Assert.Equal(0.0, tokens.Distance(first, reordered));
        Assert.True(edit.Distance(first, reordered) > 0.0);
    }

    [Fact]
    public void LineEditDistanceBoundsHowMuchOfAProgramItCompares()
    {
        var bounded = new ProgramLineEditDistance(maxComparedLines: 2);
        var first = new ProgramGenome("a\nb\nc\nd\ne");
        var second = new ProgramGenome("a\nb\nZ\nZ\nZ");

        // Only the first two lines are compared, and those agree.
        Assert.Equal(0.0, bounded.Distance(first, second));
        Assert.True(new ProgramLineEditDistance().Distance(first, second) > 0.0);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(100_001)]
    public void LineEditDistanceRejectsAnInvalidLineBound(int lines) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramLineEditDistance(lines));

    [Fact]
    public void BothStructuralMetricsExposeStableIdentitiesAndRejectNulls()
    {
        IGenomeDistance<ProgramGenome> tokens = new ProgramTokenSetDistance();
        IGenomeDistance<ProgramGenome> edit = new ProgramLineEditDistance();
        var genome = new ProgramGenome("a");

        Assert.Equal("program-token-set", tokens.Id);
        Assert.Equal("program-token-set-v1", tokens.VersionHash);
        Assert.Equal("program-line-edit", edit.Id);
        Assert.Equal(edit.VersionHash, new ProgramLineEditDistance().VersionHash);
        Assert.NotEqual(edit.VersionHash, new ProgramLineEditDistance(maxComparedLines: 7).VersionHash);

#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() => tokens.Distance(genome, null));
        Assert.Throws<ArgumentNullException>(() => edit.Distance(null, genome));
#pragma warning restore CS8625
    }

    [Fact]
    public void MemoizingTokenSetsChangesNothingButTheWork()
    {
        var memoized = new ProgramTokenSetDistance();
        var uncached = new ProgramTokenSetDistance(memoCapacity: 0);
        var tiny = new ProgramTokenSetDistance(memoCapacity: 1);

        var candidate = new ProgramGenome("def solve(n):\n    return sum(range(n))\n");
        var others = new List<ProgramGenome>();
        for (int index = 0; index < 8; index++)
        {
            others.Add(new ProgramGenome("def solve_" + index + "(n):\n    total = 0\n    return total\n"));
        }

        foreach (ProgramGenome other in others)
        {
            double expected = uncached.Distance(candidate, other);
            Assert.Equal(expected, memoized.Distance(candidate, other), 12);
            Assert.Equal(expected, memoized.Distance(candidate, other), 12);
            Assert.Equal(expected, tiny.Distance(candidate, other), 12);

            memoized.ClearMemo();
            Assert.Equal(expected, memoized.Distance(candidate, other), 12);
        }

        Assert.Equal(0, uncached.MemoCapacity);
        Assert.Equal(ProgramTokenSetDistance.DefaultMemoCapacity, memoized.MemoCapacity);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1_000_001)]
    public void TokenSetDistanceRejectsAnInvalidMemoCapacity(int capacity) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramTokenSetDistance(capacity));

    [Fact]
    public void StructuralMetricsAreDeterministicAcrossRepeatedCalls()
    {
        var tokens = new ProgramTokenSetDistance();
        var edit = new ProgramLineEditDistance();
        var first = new ProgramGenome("def solve(n):\n    return sum(range(n))\n");
        var second = new ProgramGenome("def solve(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n");

        double tokenDistance = tokens.Distance(first, second);
        double editDistance = edit.Distance(first, second);
        for (int repeat = 0; repeat < 25; repeat++)
        {
            Assert.Equal(tokenDistance, tokens.Distance(first, second));
            Assert.Equal(editDistance, edit.Distance(first, second));
        }
    }
}
