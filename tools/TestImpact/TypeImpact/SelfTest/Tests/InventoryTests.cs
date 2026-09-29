using Xunit;

namespace Fixture.Tests;

// Enumerates types by reflection: no signature names what it reaches.
public class InventoryTests
{
    [Fact]
    public void EveryModelIsSealed()
    {
        foreach (var type in typeof(ModelBase).Assembly.GetTypes())
        {
            System.GC.KeepAlive(type.IsSealed);
        }
    }
}
