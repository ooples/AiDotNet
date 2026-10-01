using Xunit;

namespace Fixture.Tests;

public class GreetingTests
{
    [Fact]
    public void Greets() => System.GC.KeepAlive(Greeting.Hello());
}
