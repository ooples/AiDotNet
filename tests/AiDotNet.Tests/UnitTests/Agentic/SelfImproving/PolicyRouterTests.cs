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
    public class PolicyRouterTests
    {
        private static AgentExecutor<double> Agent(string name) =>
            new(ScriptedChatClient<double>.Sequence(ChatResponses.Text(name + "-out")),
                tools: null, new AgentExecutorOptions { Name = name });

        [Fact(Timeout = 60000)]
        public async Task Policy_ShiftsProbabilityTowardHigherRewardAgent()
        {
            var router = new SoftmaxPolicyRouter<double>(new IAgent<double>[] { Agent("good"), Agent("bad") }, learningRate: 0.5);

            // Repeated REINFORCE updates: "good" earns 1.0, "bad" earns 0.0.
            for (var i = 0; i < 40; i++)
            {
                router.Update(null, "good", 1.0);
                router.Update(null, "bad", 0.0);
            }

            Assert.True(router.ProbabilityOf("good") > 0.8);
            Assert.Equal("good", router.SelectBestAgentName());
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task UntrainedPolicy_IsUniform()
        {
            var router = new SoftmaxPolicyRouter<double>(new IAgent<double>[] { Agent("a"), Agent("b") });
            Assert.Equal(0.5, router.ProbabilityOf("a"), 6);
            Assert.Equal(0.5, router.ProbabilityOf("b"), 6);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task LearnsFromTrajectories_AndRunsChosenAgent()
        {
            var router = new SoftmaxPolicyRouter<double>(new IAgent<double>[] { Agent("good"), Agent("bad") }, seed: 1);
            var trajectories = new List<AgentTrajectory>();
            for (var i = 0; i < 30; i++)
            {
                trajectories.Add(new AgentTrajectory(Guid.NewGuid().ToString("N"), "good",
                    new List<ChatMessage> { ChatMessage.User("t") }, "o", 1, reward: 1.0));
                trajectories.Add(new AgentTrajectory(Guid.NewGuid().ToString("N"), "bad",
                    new List<ChatMessage> { ChatMessage.User("t") }, "o", 1, reward: 0.0));
            }

            router.LearnFrom(trajectories);

            Assert.Equal("good", router.SelectBestAgentName());

            // RunAsync routes (stochastically) to a candidate and runs it.
            var result = await router.RunAsync(new[] { ChatMessage.User("go") });
            Assert.Contains(result.AgentName, new[] { "good", "bad" });
            Assert.EndsWith("-out", result.FinalText);
        }

        [Fact(Timeout = 60000)]
        public async Task LearnsPerContext()
        {
            var router = new SoftmaxPolicyRouter<double>(
                new IAgent<double>[] { Agent("math"), Agent("writer") },
                learningRate: 0.5,
                contextKey: msgs => msgs[msgs.Count - 1].Text.Split(' ')[0]);

            for (var i = 0; i < 30; i++)
            {
                router.Update(new[] { ChatMessage.User("math q") }, "math", 1.0);
                router.Update(new[] { ChatMessage.User("math q") }, "writer", 0.0);
                router.Update(new[] { ChatMessage.User("prose q") }, "writer", 1.0);
                router.Update(new[] { ChatMessage.User("prose q") }, "math", 0.0);
            }

            Assert.Equal("math", router.SelectBestAgentName(new[] { ChatMessage.User("math integrate") }));
            Assert.Equal("writer", router.SelectBestAgentName(new[] { ChatMessage.User("prose poem") }));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_Guards()
        {
            Assert.Throws<ArgumentException>(() => new SoftmaxPolicyRouter<double>(Array.Empty<IAgent<double>>()));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SoftmaxPolicyRouter<double>(new IAgent<double>[] { Agent("a") }, learningRate: 0));
            await Task.CompletedTask;
        }
    }
}
