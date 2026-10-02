using Xunit;

namespace Fixture.Tests;

public class StampedTests
{
    [Fact]
    public void Reads() => System.GC.KeepAlive(Stamped.Value());
}

public class SteadyTests
{
    [Fact]
    public void Reads() => System.GC.KeepAlive(Steady.Value());
}