using Xunit;

namespace Fixture.Tests;

public class SurfaceTests
{
    [Theory]
    public void Reads(ISurface surface) => System.GC.KeepAlive(surface.Size);
}
