using AttributionSubject;
using Xunit;

namespace PrototypeTests;

public sealed class DeferredTests
{
    public static IEnumerable<object[]> Rows()
    {
        yield return [1];
        yield return [2];
    }

    [Theory, MemberData(nameof(Rows), DisableDiscoveryEnumeration = true), Trait("Scenario", "Positive")]
    public void AllRows(int value) => Assert.Equal(value + 11, Operations.Left(value));
}
