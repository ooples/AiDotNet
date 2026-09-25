using AttributionSubject;
using Xunit;

namespace AttributionTests;

public sealed class SharedFixture : IDisposable
{
    public SharedFixture() => Assert.Equal(31, Operations.Setup());
    public void Dispose() => Assert.Equal(47, Operations.Cleanup());
}

public sealed class OwnershipTests : IClassFixture<SharedFixture>
{
    public OwnershipTests(SharedFixture fixture) => ArgumentNullException.ThrowIfNull(fixture);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public async Task First(int value)
    {
        await Task.Yield();
        bool swapped = Environment.GetEnvironmentVariable("ATTRIBUTION_SWAP") == "1";
        Assert.Equal(value + (swapped ? 23 : 11),
            await Task.Run(() => swapped ? Operations.Right(value) : Operations.Left(value)));
    }

    [Fact]
    public async Task Second()
    {
        await Task.Yield();
        bool swapped = Environment.GetEnvironmentVariable("ATTRIBUTION_SWAP") == "1";
        Assert.Equal(swapped ? 12 : 24,
            await Task.Run(() => swapped ? Operations.Left(1) : Operations.Right(1)));
    }
}
