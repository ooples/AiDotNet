using System;
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
    public class LearnedRouterTests
    {
        private static AgentExecutor<double> Agent(string name) =>
            new(ScriptedChatClient<double>.Sequence(ChatResponses.Text(name + "-answer")),
                tools: null, new AgentExecutorOptions { Name = name });

        private static AgentTrajectory Graded(string agentName, double reward, string input = "task") =>
            new(Guid.NewGuid().ToString("N"), agentName,
                new List<ChatMessage> { ChatMessage.User(input) }, "out", iterations: 1, reward: reward);

        [Fact(Timeout = 60000)]
        public async Task Router_RoutesToHighestRewardAgent_AfterLearning()
        {
            await Task.Yield();

            var good = Agent("good");
            var bad = Agent("bad");
            var router = new LearnedAgentRouter<double>(new IAgent<double>[] { good, bad });

            router.LearnFrom(new[]
            {
                Graded("good", 1.0), Graded("good", 0.9),
                Graded("bad", 0.1), Graded("bad", 0.0),
            });

            var result = await router.RunAsync(new[] { ChatMessage.User("do it") });

            Assert.Equal("good", result.AgentName);
            Assert.Equal("good-answer", result.FinalText);
            Assert.Equal("good", router.SelectAgentName(new[] { ChatMessage.User("anything") }));
        }

        [Fact(Timeout = 60000)]
        public async Task Router_ExploresUnseenCandidateFirst()
        {
            await Task.Yield();

            var seen = Agent("seen");
            var unseen = Agent("unseen");
            var router = new LearnedAgentRouter<double>(new IAgent<double>[] { seen, unseen });

            // Only "seen" has data; the router should try the unseen candidate (optimistic exploration).
            router.LearnFrom(new[] { Graded("seen", 1.0) });

            Assert.Equal("unseen", router.SelectAgentName(new[] { ChatMessage.User("x") }));
        }

        [Fact(Timeout = 60000)]
        public async Task Router_LearnsPerContext()
        {
            await Task.Yield();

            var mathExpert = Agent("math");
            var writer = Agent("writer");
            // Context key = the first word of the latest user message.
            var router = new LearnedAgentRouter<double>(
                new IAgent<double>[] { mathExpert, writer },
                contextKey: msgs => msgs.Last().Text.Split(' ')[0]);

            router.LearnFrom(new[]
            {
                Graded("math", 1.0, "math solve 2+2"), Graded("writer", 0.0, "math solve 2+2"),
                Graded("writer", 1.0, "prose write a poem"), Graded("math", 0.0, "prose write a poem"),
            });

            Assert.Equal("math", router.SelectAgentName(new[] { ChatMessage.User("math integrate x") }));
            Assert.Equal("writer", router.SelectAgentName(new[] { ChatMessage.User("prose a story") }));
        }

        [Fact(Timeout = 60000)]
        public async Task Router_ClosedLoop_ImprovesWithTracing()
        {
            await Task.Yield();

            // Seed the policy so it already prefers "good", then route — demonstrating the policy drives runs.
            var router = new LearnedAgentRouter<double>(new IAgent<double>[] { Agent("good"), Agent("bad") });
            router.LearnFrom(new[] { Graded("good", 1.0), Graded("bad", 0.0) });

            var store = new InMemoryTrajectoryStore();
            var traced = new TracingAgent<double>(router, store);
            await traced.RunAsync(new[] { ChatMessage.User("go") });

            // The routed run was captured under the chosen agent's name.
            var all = await store.GetAllAsync();
            Assert.Single(all);
            Assert.Equal("good", all[0].AgentName);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_RejectsEmptyAndDuplicates()
        {
            await Task.Yield();

            Assert.Throws<ArgumentException>(() => new LearnedAgentRouter<double>(Array.Empty<IAgent<double>>()));
            Assert.Throws<ArgumentException>(() =>
                new LearnedAgentRouter<double>(new IAgent<double>[] { Agent("dup"), Agent("dup") }));
        }
    }
}
