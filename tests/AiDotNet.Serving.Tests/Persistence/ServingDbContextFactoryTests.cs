using AiDotNet.Serving.Persistence;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Tests.Persistence;

public sealed class ServingDbContextFactoryTests
{
    [Fact(Timeout = 60000)]
    public async Task CreateDbContext_CreatesContext()
    {
        var factory = new ServingDbContextFactory();
        using var context = factory.CreateDbContext([]);

        Assert.NotNull(context);
        Assert.NotNull(context.Database);
    }
}

