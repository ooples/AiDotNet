#if NET5_0_OR_GREATER
using System;
using System.Threading.Tasks;
using AiDotNet.Agentic.Graph.Checkpointing;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Graph
{
    // Integration tests for the Postgres/Redis graph checkpointers. They require a live server and are
    // SKIPPED (not failed) when the corresponding connection-string environment variable is not set, so CI
    // stays green; a developer with a server runs them by setting the variable. The projects are net10-only
    // (the Npgsql/Redis drivers' TFMs), hence the net10 gate.
    public class DistributedCheckpointerTests
    {
        private const string PostgresEnv = "AIDOTNET_PG_CONNSTRING";
        private const string RedisEnv = "AIDOTNET_REDIS_CONNSTRING";

        [SkippableFact(Timeout = 60000)]
        public async Task Postgres_SavesAndRoundTripsCheckpoints()
        {
            var connectionString = Environment.GetEnvironmentVariable(PostgresEnv);
            Skip.If(string.IsNullOrWhiteSpace(connectionString), $"Set {PostgresEnv} to run the Postgres checkpointer test.");

            var checkpointer = new PostgresGraphCheckpointer<int>(connectionString);
            await RoundTripAsync(checkpointer);
        }

        [SkippableFact(Timeout = 60000)]
        public async Task Redis_SavesAndRoundTripsCheckpoints()
        {
            var connectionString = Environment.GetEnvironmentVariable(RedisEnv);
            Skip.If(string.IsNullOrWhiteSpace(connectionString), $"Set {RedisEnv} to run the Redis checkpointer test.");

            using var checkpointer = new RedisGraphCheckpointer<int>(connectionString);
            await RoundTripAsync(checkpointer);
        }

        private static async Task RoundTripAsync(IGraphCheckpointer<int> checkpointer)
        {
            var threadId = "test-" + Guid.NewGuid().ToString("N");

            await checkpointer.SaveAsync(new GraphCheckpoint<int>(threadId, "cp-1", 1, "next", 10));
            await checkpointer.SaveAsync(new GraphCheckpoint<int>(threadId, "cp-2", 2, "__end__", 20));

            var latest = await checkpointer.GetLatestAsync(threadId);
            Assert.NotNull(latest);
            Assert.Equal("cp-2", latest.CheckpointId);
            Assert.Equal(2, latest.Step);
            Assert.Equal(20, latest.State);
            Assert.True(latest.IsComplete);

            var byId = await checkpointer.GetAsync(threadId, "cp-1");
            Assert.NotNull(byId);
            Assert.Equal(10, byId.State);

            var history = await checkpointer.GetHistoryAsync(threadId);
            Assert.Equal(2, history.Count);
            Assert.Equal("cp-1", history[0].CheckpointId);
            Assert.Equal("cp-2", history[1].CheckpointId);
        }
    }
}
#endif
