using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class ServingProgramSynthesisConcurrencyLimiterTests
{
    [Fact]
    public async Task AcquireAsync_EnforcesTierConcurrency()
    {
        var options = new ServingProgramSynthesisOptions
        {
            Free = new ServingProgramSynthesisLimitOptions { MaxConcurrentRequests = 1 },
            Premium = new ServingProgramSynthesisLimitOptions { MaxConcurrentRequests = 1 },
            Enterprise = new ServingProgramSynthesisLimitOptions { MaxConcurrentRequests = 1 }
        };

        var limiter = new ServingProgramSynthesisConcurrencyLimiter(Options.Create(options));

        using var lease1 = await limiter.AcquireAsync(ServingTier.Free, CancellationToken.None);

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        await Assert.ThrowsAsync<OperationCanceledException>(() => limiter.AcquireAsync(ServingTier.Free, cts.Token));
    }
}

