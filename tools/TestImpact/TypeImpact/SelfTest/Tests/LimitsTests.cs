using Xunit;

namespace Fixture.Tests;

// Reads Limits.MaxDepth, a const: the compiler inlines its value, so no reference to Limits survives in the IL.
public class LimitsTests
{
    [Fact]
    public void ReadsTheDepth() => System.GC.KeepAlive(Limits.MaxDepth);
}
