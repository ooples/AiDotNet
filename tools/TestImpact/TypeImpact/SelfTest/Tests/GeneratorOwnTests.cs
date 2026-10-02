using Xunit;

namespace Fixture.Tests.Generator;

// Stands in for a generator's own tests, which drive it through a generator driver and so are reached
// by no reference from its output: --generator-tests selects them by namespace.
public class GeneratorOwnTests
{
    [Fact]
    public void Runs() => System.GC.KeepAlive(0);
}