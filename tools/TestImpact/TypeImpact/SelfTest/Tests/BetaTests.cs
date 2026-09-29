using Xunit;

namespace Fixture.Tests;

public class BetaTests
{
    [Fact]
    [Trait("Category", "Slow")]
    public void Predicts() => System.GC.KeepAlive(new Beta().Predict());
}
