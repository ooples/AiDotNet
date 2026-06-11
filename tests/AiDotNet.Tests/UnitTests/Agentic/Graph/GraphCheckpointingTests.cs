using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Graph;
using AiDotNet.Agentic.Graph.Checkpointing;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Graph
{
    public class GraphCheckpointingTests
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

        [Fact(Timeout = 60000)]
        public async Task CheckpointedRun_Completes_AndRecordsHistory()
        {
            var graph = BuildGraph();
            var cp = new InMemoryGraphCheckpointer<int>();

            var result = await graph.InvokeAsync(0, cp, "t1");
            Assert.Equal(11, result);

            var history = await cp.GetHistoryAsync("t1");
            // step0(next=a,0), step1(next=b,1), step2(next=End,11)
            Assert.Equal(3, history.Count);
            Assert.Equal(new[] { 0, 1, 2 }, history.Select(h => h.Step));
            Assert.Equal(StateGraph<int>.End, history[2].NextNode);
            Assert.True(history[2].IsComplete);
            Assert.Equal(11, history[2].State);
        }

        [Fact(Timeout = 60000)]
        public async Task Resume_PicksUpFromLatestCheckpoint_SkippingEarlierNodes()
        {
            var graph = BuildGraph();
            var cp = new InMemoryGraphCheckpointer<int>();

            // Simulate that node 'a' already ran (produced 100, next is 'b').
            await cp.SaveAsync(new GraphCheckpoint<int>("t2", "t2-1", 1, "b", 100));

            var result = await graph.InvokeAsync(initialState: 0, cp, "t2");
            // Resumes at 'b' with state 100 -> 110; 'a' is NOT re-run (would have given 11).
            Assert.Equal(110, result);
        }

        [Fact(Timeout = 60000)]
        public async Task ReinvokingCompletedThread_ReturnsFinalState_WithoutRerunning()
        {
            var graph = BuildGraph();
            var cp = new InMemoryGraphCheckpointer<int>();

            await graph.InvokeAsync(0, cp, "t3"); // completes at 11
            var again = await graph.InvokeAsync(999, cp, "t3"); // initialState ignored; thread is complete
            Assert.Equal(11, again);
        }

        [Fact(Timeout = 60000)]
        public async Task TimeTravel_ResumeFromEarlierCheckpoint_Replays()
        {
            var graph = BuildGraph();
            var cp = new InMemoryGraphCheckpointer<int>();
            await graph.InvokeAsync(0, cp, "t4"); // history: t4-0(a,0), t4-1(b,1), t4-2(End,11)

            // Rewind to the checkpoint after 'a' (next='b', state=1) and replay forward.
            var result = await graph.ResumeFromAsync(cp, "t4", "t4-1");
            Assert.Equal(11, result);
        }

        [Fact(Timeout = 60000)]
        public async Task ResumeFrom_UnknownCheckpoint_Throws()
        {
            var graph = BuildGraph();
            var cp = new InMemoryGraphCheckpointer<int>();
            await Assert.ThrowsAsync<InvalidOperationException>(
                () => graph.ResumeFromAsync(cp, "t5", "does-not-exist"));
        }
    }
}
