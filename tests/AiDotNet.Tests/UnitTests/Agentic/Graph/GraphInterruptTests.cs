using System.Threading.Tasks;
using AiDotNet.Agentic.Graph;
using AiDotNet.Agentic.Graph.Checkpointing;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Graph
{
    public class GraphInterruptTests
    {
        // draft: +1, approve: *10, draft -> approve -> End, pause before "approve".
        private static CompiledStateGraph<int> BuildApprovalGraph() =>
            new StateGraph<int>()
                .AddNode("draft", s => s + 1)
                .AddNode("approve", s => s * 10)
                .AddEdge("draft", "approve")
                .AddEdge("approve", StateGraph<int>.End)
                .AddInterruptBefore("approve")
                .SetEntryPoint("draft")
                .Compile();

        [Fact(Timeout = 60000)]
        public async Task RunAsync_PausesBeforeInterruptNode_ThenResumesToCompletion()
        {
            await Task.Yield();

            var graph = BuildApprovalGraph();
            var cp = new InMemoryGraphCheckpointer<int>();

            var first = await graph.RunAsync(1, cp, "t1");
            Assert.True(first.IsInterrupted);
            Assert.Equal("approve", first.InterruptedBefore);
            Assert.Equal(2, first.State); // draft ran (1 -> 2); approve did not

            var second = await graph.RunAsync(0, cp, "t1"); // initialState ignored on resume
            Assert.True(second.IsComplete);
            Assert.Equal(20, second.State); // approve ran (2 -> 20)
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_AppliesHumanEdit_OnResume()
        {
            await Task.Yield();

            var graph = BuildApprovalGraph();
            var cp = new InMemoryGraphCheckpointer<int>();

            var first = await graph.RunAsync(1, cp, "t2");
            Assert.True(first.IsInterrupted);

            // Human edits the state (adds 100) before approving.
            var second = await graph.RunAsync(0, cp, "t2", applyOnResume: s => s + 100);
            Assert.True(second.IsComplete);
            Assert.Equal(1020, second.State); // (2 + 100) * 10
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_CanInterruptBeforeEntryNode()
        {
            await Task.Yield();

            var graph = new StateGraph<int>()
                .AddNode("start", s => s + 1)
                .AddEdge("start", StateGraph<int>.End)
                .AddInterruptBefore("start")
                .SetEntryPoint("start")
                .Compile();
            var cp = new InMemoryGraphCheckpointer<int>();

            var first = await graph.RunAsync(5, cp, "t3");
            Assert.True(first.IsInterrupted);
            Assert.Equal("start", first.InterruptedBefore);
            Assert.Equal(5, first.State); // nothing ran yet

            var second = await graph.RunAsync(0, cp, "t3");
            Assert.True(second.IsComplete);
            Assert.Equal(6, second.State);
        }

        [Fact(Timeout = 60000)]
        public async Task InvokeAsync_RunsStraightThrough_IgnoringInterrupts()
        {
            await Task.Yield();

            var graph = BuildApprovalGraph();
            var result = await graph.InvokeAsync(1); // plain run, no checkpointer
            Assert.Equal(20, result); // (1 + 1) * 10 — did not pause
        }
    }
}
