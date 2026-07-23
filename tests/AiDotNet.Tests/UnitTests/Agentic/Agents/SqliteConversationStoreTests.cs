#if NET5_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Memory;
using AiDotNet.Agentic.Models;
using AiDotNet.Storage.Sqlite;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    // SqliteConversationStore works on both target frameworks, but the native SQLite library (e_sqlite3) is
    // not deployed to the .NET Framework (net471) shadow-copy test host in this environment — a pre-existing,
    // repo-wide condition that also affects SqliteSqlSyntaxValidatorTests. These tests therefore run on the
    // framework where the native dependency is available; the cross-instance durability contract is also
    // covered TFM-agnostically by JsonFileConversationStore tests.
    public class SqliteConversationStoreTests
    {
        [Fact(Timeout = 60000)]
        public async Task PersistsAndReloadsAcrossInstances()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            var connectionString = $"Data Source={path};Pooling=False";
            try
            {
                var writer = new SqliteConversationStore(connectionString);
                await writer.AppendAsync("conv", new[] { ChatMessage.User("hello"), ChatMessage.Assistant("world") });
                await writer.AppendAsync("conv", new[] { ChatMessage.User("again") });

                // A fresh instance over the same database reads the persisted, ordered history.
                var reader = new SqliteConversationStore(connectionString);
                var history = await reader.GetAsync("conv");

                Assert.Equal(3, history.Count);
                Assert.Equal(ChatRole.User, history[0].Role);
                Assert.Equal("hello", history[0].Text);
                Assert.Equal(ChatRole.Assistant, history[1].Role);
                Assert.Equal("world", history[1].Text);
                Assert.Equal("again", history[2].Text);

                Assert.Contains("conv", await reader.ListThreadsAsync());
            }
            finally
            {
                TryDelete(path);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task ThreadsAreIsolated_AndClearRemovesOne()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            var connectionString = $"Data Source={path};Pooling=False";
            try
            {
                var store = new SqliteConversationStore(connectionString);
                await store.AppendAsync("alice", new[] { ChatMessage.User("I am Alice.") });
                await store.AppendAsync("bob", new[] { ChatMessage.User("I am Bob.") });

                Assert.Single(await store.GetAsync("alice"));
                Assert.Single(await store.GetAsync("bob"));

                await store.ClearAsync("alice");
                Assert.Empty(await store.GetAsync("alice"));
                Assert.Single(await store.GetAsync("bob"));
            }
            finally
            {
                TryDelete(path);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task UnknownThread_ReturnsEmpty()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            var connectionString = $"Data Source={path};Pooling=False";
            try
            {
                var store = new SqliteConversationStore(connectionString);
                Assert.Empty(await store.GetAsync("nope"));
            }
            finally
            {
                TryDelete(path);
            }
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch (IOException)
            {
                // Best-effort cleanup of the temp database file.
            }
        }
    }
}
#endif
