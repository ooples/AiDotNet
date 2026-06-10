using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.SelfImproving;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.SelfImproving
{
    public class TrajectoryCaptureTests
    {
        private static AgentExecutor<double> Agent(string name, string answer)
        {
            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text(answer));
            return new AgentExecutor<double>(client, tools: null, new AgentExecutorOptions { Name = name });
        }

        [Fact(Timeout = 60000)]
        public async Task TracingAgent_RecordsEachRun()
        {
            var store = new InMemoryTrajectoryStore();
            var traced = new TracingAgent<double>(Agent("writer", "Hello."), store);

            var result = await traced.RunAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("Hello.", result.FinalText); // behavior unchanged

            var all = await store.GetAllAsync();
            Assert.Single(all);
            Assert.Equal("writer", all[0].AgentName);
            Assert.Equal("Hello.", all[0].FinalText);
            Assert.Contains(all[0].Messages, m => m.Role == ChatRole.User && m.Text == "hi");
            Assert.Null(all[0].Reward);
        }

        [Fact(Timeout = 60000)]
        public async Task TracingAgent_AccumulatesAcrossRuns_AndIsQueryable()
        {
            var store = new InMemoryTrajectoryStore();
            await new TracingAgent<double>(Agent("a", "one"), store).RunAsync(new[] { ChatMessage.User("x") });
            await new TracingAgent<double>(Agent("b", "two"), store).RunAsync(new[] { ChatMessage.User("y") });

            Assert.Equal(2, (await store.GetAllAsync()).Count);

            var onlyB = await store.QueryAsync(t => t.AgentName == "b");
            Assert.Single(onlyB);
            Assert.Equal("two", onlyB[0].FinalText);
        }

        [Fact(Timeout = 60000)]
        public async Task TracingAgent_AttachesMetadata()
        {
            var store = new InMemoryTrajectoryStore();
            var metadata = new Dictionary<string, string> { ["experiment"] = "exp-1" };
            var traced = new TracingAgent<double>(Agent("a", "ok"), store, metadata);

            await traced.RunAsync(new[] { ChatMessage.User("x") });

            var all = await store.GetAllAsync();
            Assert.NotNull(all[0].Metadata);
            Assert.Equal("exp-1", all[0].Metadata["experiment"]);
        }

        [Fact(Timeout = 60000)]
        public async Task Store_GetById_And_RewardAnnotation()
        {
            var store = new InMemoryTrajectoryStore();
            var traced = new TracingAgent<double>(Agent("a", "answer"), store);
            await traced.RunAsync(new[] { ChatMessage.User("x") });

            var id = (await store.GetAllAsync())[0].Id;
            var fetched = await store.GetAsync(id);
            Assert.NotNull(fetched);

            // An evaluator grades the trajectory; the annotation is visible on subsequent reads.
            fetched.Reward = 0.75;
            var again = await store.GetAsync(id);
            Assert.NotNull(again);
            Assert.Equal(0.75, again.Reward);
        }

        [Fact(Timeout = 60000)]
        public async Task Store_Clear_Empties()
        {
            var store = new InMemoryTrajectoryStore();
            await new TracingAgent<double>(Agent("a", "ok"), store).RunAsync(new[] { ChatMessage.User("x") });
            await store.ClearAsync();
            Assert.Empty(await store.GetAllAsync());
        }
    }
}
