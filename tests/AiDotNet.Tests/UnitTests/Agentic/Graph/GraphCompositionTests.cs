using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Graph;
using AiDotNet.Agentic.Graph.Checkpointing;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Graph
{
    public class GraphCompositionTests
    {
        [Fact(Timeout = 60000)]
        public async Task Subgraph_RunsAsSingleStep()
        {
            var inner = new StateGraph<int>()
                .AddNode("plus1", s => s + 1)
                .AddEdge("plus1", StateGraph<int>.End)
                .SetEntryPoint("plus1")
                .Compile();

            var outer = new StateGraph<int>()
                .AddNode("pre", s => s * 2)
                .AddSubgraph("sub", inner)
                .AddEdge("pre", "sub")
                .AddEdge("sub", StateGraph<int>.End)
                .SetEntryPoint("pre")
                .Compile();

            Assert.Equal(7, await outer.InvokeAsync(3)); // (3*2) then +1
        }

        [Fact(Timeout = 60000)]
        public async Task RewardGatedEdges_LoopUntilThresholdMet()
        {
            var graph = new StateGraph<int>()
                .AddNode("work", s => s + 1)
                .AddRewardGatedEdges("work", reward: s => s, threshold: 3, ifMeetsThreshold: StateGraph<int>.End, ifBelowThreshold: "work")
                .SetEntryPoint("work")
                .Compile();

            Assert.Equal(3, await graph.InvokeAsync(0)); // loops work until score >= 3
        }

        [Fact(Timeout = 60000)]
        public async Task Replay_ReproducesRecordedTrajectory_WithoutExecuting()
        {
            var graph = new StateGraph<int>()
                .AddNode("a", s => s + 1)
                .AddNode("b", s => s + 10)
                .AddEdge("a", "b")
                .AddEdge("b", StateGraph<int>.End)
                .SetEntryPoint("a")
                .Compile();
            var cp = new InMemoryGraphCheckpointer<int>();
            await graph.InvokeAsync(0, cp, "r1");

            var replay = new List<GraphStepUpdate<int>>();
            await foreach (var u in graph.ReplayAsync(cp, "r1"))
            {
                replay.Add(u);
            }

            Assert.Equal(new[] { "a", "b" }, replay.Select(u => u.NodeName));
            Assert.Equal(new[] { 1, 11 }, replay.Select(u => u.State));
            Assert.Equal(11, await graph.GetRecordedStateAsync(cp, "r1"));
        }
    }
}
