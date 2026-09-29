using Xunit;

namespace Fixture.Tests;

public class CatalogTests
{
    [Fact]
    public void CreatesByName() => System.GC.KeepAlive(Catalog.Create("alpha"));
}
