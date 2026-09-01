using AiDotNet.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class StableRandomTests
{
    [Fact]
    public void SequenceIsPinnedAcrossRuntimes()
    {
        var random = new StableRandom(42);

        uint[] actual = Enumerable.Range(0, 8).Select(_ => random.NextUInt32()).ToArray();

        Assert.Equal(new uint[]
        {
            565663470, 3244226384, 2504567229, 903561869,
            4026996297, 2722332799, 3032858066, 272411090
        }, actual);
    }

    [Fact]
    public void CapturedStateResumesBitIdentically()
    {
        var random = new StableRandom(123, 9);
        _ = random.NextUInt64();
        StableRandomState state = random.CaptureState();
        ulong[] expected = Enumerable.Range(0, 20).Select(_ => random.NextUInt64()).ToArray();

        StableRandom resumed = StableRandom.Restore(state);

        Assert.Equal(expected, Enumerable.Range(0, 20).Select(_ => resumed.NextUInt64()).ToArray());
    }

    [Fact]
    public void CandidateStreamsDoNotDependOnCreationOrder()
    {
        uint first = StableRandom.CreateStream(99, 7).NextUInt32();
        _ = StableRandom.CreateStream(99, 2).NextUInt64();
        uint second = StableRandom.CreateStream(99, 7).NextUInt32();

        Assert.Equal(first, second);
        Assert.NotEqual(first, StableRandom.CreateStream(99, 8).NextUInt32());
    }

    [Fact]
    public void NextIntStaysInsideRequestedBounds()
    {
        var random = new StableRandom(5);

        for (int i = 0; i < 10_000; i++)
        {
            int value = random.NextInt(-2_000_000_000, 2_000_000_000);
            Assert.InRange(value, -2_000_000_000, 1_999_999_999);
        }
    }
}
