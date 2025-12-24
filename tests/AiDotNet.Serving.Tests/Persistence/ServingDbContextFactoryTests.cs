using AiDotNet.Serving.Persistence;
using Xunit;

namespace AiDotNet.Serving.Tests.Persistence;

public sealed class ServingDbContextFactoryTests
{
    [Fact]
    public void CreateDbContext_CreatesContext()
    {
        var factory = new ServingDbContextFactory();
        using var context = factory.CreateDbContext([]);

        Assert.NotNull(context);
        Assert.NotNull(context.Database);
    }
}

