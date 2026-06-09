using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Graph;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Graph
{
    public class StateGraphTests
    {
        [Fact(Timeout = 60000)]
        public async Task Linear_RunsNodesInOrder_AndThreadsState()
        {
            var graph = new StateGraph<int>()
                .AddNode("inc", s => s + 1)
                .AddNode("double", s => s * 2)
                .AddEdge("inc", "double")
                .AddEdge("double", StateGraph<int>.End)
                .SetEntryPoint("inc")
                .Compile();

            var result = await graph.InvokeAsync(1);
            Assert.Equal(4, result); // (1 + 1) * 2
        }

        [Fact(Timeout = 60000)]
        public async Task ConditionalEdges_SupportCycles()
        {
            var graph = new StateGraph<int>()
                .AddNode("inc", s => s + 1)
                .AddConditionalEdges("inc", s => s < 3 ? "inc" : StateGraph<int>.End)
                .SetEntryPoint("inc")
                .Compile();

            var result = await graph.InvokeAsync(0);
            Assert.Equal(3, result); // loops inc until value reaches 3
        }

        [Fact(Timeout = 60000)]
        public async Task Stream_EmitsUpdatePerNode_WithFinalState()
        {
            var graph = new StateGraph<int>()
                .AddNode("a", s => s + 1)
                .AddNode("b", s => s + 10)
                .AddEdge("a", "b")
                .AddEdge("b", StateGraph<int>.End)
                .SetEntryPoint("a")
                .Compile();

            var updates = new List<GraphStepUpdate<int>>();
            await foreach (var u in graph.StreamAsync(0))
            {
                updates.Add(u);
            }

            Assert.Equal(new[] { "a", "b" }, updates.Select(u => u.NodeName));
            Assert.Equal(1, updates[0].State);
            Assert.Equal(11, updates[1].State);
        }

        [Fact(Timeout = 60000)]
        public async Task RecursionLimit_Throws_OnRunawayCycle()
        {
            var graph = new StateGraph<int>()
                .AddNode("loop", s => s + 1)
                .AddConditionalEdges("loop", _ => "loop") // never terminates
                .SetEntryPoint("loop")
                .Compile();

            var ex = await Assert.ThrowsAsync<GraphRecursionException>(
                () => graph.InvokeAsync(0, new GraphRunOptions { MaxSteps = 5 }));
            Assert.Equal(5, ex.MaxSteps);
        }

        [Fact(Timeout = 60000)]
        public async Task AsyncNode_IsAwaited()
        {
            var graph = new StateGraph<int>()
                .AddNode("delay", async (s, ct) => { await Task.Yield(); return s + 41; })
                .AddEdge("delay", StateGraph<int>.End)
                .SetEntryPoint("delay")
                .Compile();

            Assert.Equal(42, await graph.InvokeAsync(1));
        }

        [Fact(Timeout = 60000)]
        public async Task Compile_Validates_EntryPointEdgesAndExclusiveRouting()
        {
            // No entry point.
            Assert.Throws<InvalidOperationException>(() =>
                new StateGraph<int>().AddNode("a", s => s).Compile());

            // Edge to unknown node.
            Assert.Throws<InvalidOperationException>(() =>
                new StateGraph<int>().AddNode("a", s => s).AddEdge("a", "ghost").SetEntryPoint("a").Compile());

            // A node cannot have both a fixed edge and conditional edges.
            Assert.Throws<InvalidOperationException>(() =>
                new StateGraph<int>()
                    .AddNode("a", s => s)
                    .AddEdge("a", StateGraph<int>.End)
                    .AddConditionalEdges("a", _ => StateGraph<int>.End)
                    .SetEntryPoint("a")
                    .Compile());

            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task AddNode_RejectsDuplicateAndReservedNames()
        {
            var graph = new StateGraph<int>().AddNode("a", s => s);
            Assert.Throws<ArgumentException>(() => graph.AddNode("a", s => s));            // duplicate
            Assert.Throws<ArgumentException>(() => new StateGraph<int>().AddNode(StateGraph<int>.End, s => s)); // reserved
            await Task.CompletedTask;
        }
    }
}
