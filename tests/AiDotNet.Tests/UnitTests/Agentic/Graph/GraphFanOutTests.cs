using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Graph;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Graph
{
    public class GraphFanOutTests
    {
        // Fan out over 1..N, square each branch in parallel, reduce by summing into the state.
        private static CompiledStateGraph<int> BuildSquareSumGraph(int? maxParallelism = null) =>
            new StateGraph<int>()
                .AddFanOutNode<int, int>(
                    "squareSum",
                    map: s => Enumerable.Range(1, s),
                    branch: async (i, ct) => { await Task.Yield(); return i * i; },
                    reduce: (s, results) => results.Sum(),
                    maxDegreeOfParallelism: maxParallelism)
                .AddEdge("squareSum", StateGraph<int>.End)
                .SetEntryPoint("squareSum")
                .Compile();

        [Fact(Timeout = 60000)]
        public async Task FanOut_MapsBranchesAndReduces()
        {
            await Task.Yield();

            var graph = BuildSquareSumGraph();
            Assert.Equal(14, await graph.InvokeAsync(3)); // 1 + 4 + 9
        }

        [Fact(Timeout = 60000)]
        public async Task FanOut_RespectsParallelismCap_StillCorrect()
        {
            await Task.Yield();

            var graph = BuildSquareSumGraph(maxParallelism: 2);
            Assert.Equal(30, await graph.InvokeAsync(4)); // 1 + 4 + 9 + 16
        }

        [Fact(Timeout = 60000)]
        public async Task FanOut_EmptyItemSet_ReducesOverNothing()
        {
            await Task.Yield();

            var graph = BuildSquareSumGraph();
            Assert.Equal(0, await graph.InvokeAsync(0)); // no items -> Sum() == 0
        }

        [Fact(Timeout = 60000)]
        public async Task FanOut_ComposesWithOtherNodes()
        {
            await Task.Yield();

            var graph = new StateGraph<int>()
                .AddNode("seed", s => s + 2) // 1 -> 3
                .AddFanOutNode<int, int>(
                    "squareSum",
                    map: s => Enumerable.Range(1, s),
                    branch: (i, ct) => Task.FromResult(i * i),
                    reduce: (s, results) => results.Sum())
                .AddEdge("seed", "squareSum")
                .AddEdge("squareSum", StateGraph<int>.End)
                .SetEntryPoint("seed")
                .Compile();

            Assert.Equal(14, await graph.InvokeAsync(1)); // seed: 1->3, then 1+4+9
        }
    }
}
