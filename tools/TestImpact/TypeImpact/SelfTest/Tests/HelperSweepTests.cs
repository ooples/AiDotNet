using Xunit;

namespace Fixture.Tests;

public class HelperSweepTests
{
    [Fact]
    public void CountsModels() => System.GC.KeepAlive(TypeSweep.CountModels());
}
