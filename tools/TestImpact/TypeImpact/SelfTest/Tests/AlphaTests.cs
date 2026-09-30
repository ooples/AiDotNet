using Xunit;

namespace Fixture.Tests;

public class AlphaTests
{
    [Fact]
    public void Predicts() => System.GC.KeepAlive(new Alpha().Predict());
}
