using AiDotNet.DistributedTraining;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

[Collection("ConfigureMethodCoverage")]
public class DistributedTestOwnershipTests
{
    [Fact]
    public void IndependentTestsCanOwnTheSameRankAndDisposeIndependently()
    {
        using var first = new Session();
        using var second = new Session();
        var a = first.Create(0, 1);
        var b = second.Create(0, 1);
        a.Initialize();
        b.Initialize();
        first.Dispose();
        Assert.False(a.IsInitialized);
        Assert.True(b.IsInitialized);
        second.Dispose();
        Assert.False(b.IsInitialized);
    }

    [Fact]
    public void SameSessionSupportsMultipleRanksButRejectsDuplicateActiveRank()
    {
        using var session = new Session();
        var first = session.Create(0, 2);
        var second = session.Create(1, 2);
        first.Initialize();
        second.Initialize();
        var duplicate = session.Create(0, 2);
        Assert.Throws<InvalidOperationException>(() => duplicate.Initialize());
        session.Dispose();
        Assert.False(first.IsInitialized);
        Assert.False(second.IsInitialized);
        Assert.False(duplicate.IsInitialized);
    }

    [Fact]
    public void CleanupAfterFailureReleasesSessionAndIsIdempotent()
    {
        var session = new Session();
        var backend = session.Create(0, 1);
        Action failAfterInitialization = () =>
        {
            using (session)
            {
                backend.Initialize();
                throw new InvalidOperationException("Simulated build failure after initialization.");
            }
        };
        Assert.Throws<InvalidOperationException>(failAfterInitialization);
        Assert.False(backend.IsInitialized);
        session.Dispose();
        var replacement = new InMemoryCommunicationBackend<float>(0, 1, session.EnvironmentId);
        try { replacement.Initialize(); }
        finally { replacement.Shutdown(); }
    }

    private sealed class Session : ConfigureMethodTestBase
    {
        public string EnvironmentId => DistributedEnvironmentId;

        public InMemoryCommunicationBackend<float> Create(int rank, int worldSize)
        {
            var backend = new InMemoryCommunicationBackend<float>(rank, worldSize, DistributedEnvironmentId);
            OwnCommunicationBackend(backend);
            return backend;
        }
    }
}
