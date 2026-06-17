using System.IO;
using System.Threading.Tasks;
using AiDotNet.Agentic.Graph;
using AiDotNet.Agentic.Graph.Checkpointing;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Graph
{
    public class GraphDurableCheckpointerTests
    {
        // a: +1, b: +10, a -> b -> End
        private static CompiledStateGraph<int> BuildGraph() =>
            new StateGraph<int>()
                .AddNode("a", s => s + 1)
                .AddNode("b", s => s + 10)
                .AddEdge("a", "b")
                .AddEdge("b", StateGraph<int>.End)
                .SetEntryPoint("a")
                .Compile();

        // Runs a checkpointed graph via 'writer', then verifies a *fresh* checkpointer instance over the
        // same backing store can read the history and resume — proving durability across instances/restarts.
        private static async Task AssertDurableAsync(IGraphCheckpointer<int> writer, IGraphCheckpointer<int> freshReader)
        {
            var graph = BuildGraph();

            var result = await graph.InvokeAsync(0, writer, "d1");
            Assert.Equal(11, result);

            var history = await freshReader.GetHistoryAsync("d1");
            Assert.Equal(3, history.Count);
            Assert.True(history[history.Count - 1].IsComplete);
            Assert.Equal(11, history[history.Count - 1].State);

            // Resuming a completed thread through the fresh instance returns the persisted final state.
            Assert.Equal(11, await graph.InvokeAsync(999, freshReader, "d1"));
        }

        [Fact(Timeout = 60000)]
        public async Task JsonFileCheckpointer_PersistsAcrossInstances()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            try
            {
                await AssertDurableAsync(
                    new JsonFileGraphCheckpointer<int>(path),
                    new JsonFileGraphCheckpointer<int>(path));
            }
            finally
            {
                TryDelete(path);
            }
        }

#if NET5_0_OR_GREATER
        // SqliteGraphCheckpointer works on both target frameworks, but the native SQLite library
        // (e_sqlite3) is not deployed to the .NET Framework (net471) shadow-copy test host in this
        // environment — a pre-existing, repo-wide condition that also affects SqliteSqlSyntaxValidatorTests.
        // The cross-TFM durability contract is covered by the JSON-file test above; this adds SQLite coverage
        // on the framework where the native dependency is available.
        [Fact(Timeout = 60000)]
        public async Task SqliteCheckpointer_PersistsAcrossInstances()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            var connectionString = $"Data Source={path};Pooling=False";
            try
            {
                await AssertDurableAsync(
                    new SqliteGraphCheckpointer<int>(connectionString),
                    new SqliteGraphCheckpointer<int>(connectionString));
            }
            finally
            {
                TryDelete(path);
            }
        }
#endif

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
                // Best-effort cleanup; the OS reclaims temp files eventually.
            }
        }
    }
}
