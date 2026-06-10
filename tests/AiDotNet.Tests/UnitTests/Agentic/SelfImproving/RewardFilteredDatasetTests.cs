using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.SelfImproving;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.SelfImproving
{
    public class RewardFilteredDatasetTests
    {
        private static AgentTrajectory Run(string id, string question, string answer, double reward) =>
            new(id, "agent",
                new List<ChatMessage> { ChatMessage.User(question), ChatMessage.Assistant(answer) },
                finalText: answer, iterations: 1, reward: reward);

        [Fact(Timeout = 60000)]
        public async Task Builder_KeepsOnlyHighRewardRuns()
        {
            var builder = new RewardFilteredDatasetBuilder(minReward: 0.5);
            var dataset = builder.Build(new[]
            {
                Run("1", "2+2?", "4", 1.0),    // kept
                Run("2", "3+3?", "five", 0.0), // filtered out (low reward)
                Run("3", "5+5?", "10", 0.8),   // kept
            });

            Assert.Equal(2, dataset.Count);
            Assert.All(dataset.Examples, e => Assert.True(e.Reward >= 0.5));
            Assert.Equal(0.9, dataset.MeanReward, 3);

            var first = dataset.Examples.Single(e => e.Completion == "4");
            Assert.Contains("2+2?", first.Prompt);   // prompt holds the input context
            Assert.DoesNotContain("4", first.Prompt); // completion is excluded from the prompt
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Builder_SkipsUngradedAndDegenerateRuns()
        {
            var ungraded = new AgentTrajectory("u", "agent",
                new List<ChatMessage> { ChatMessage.User("q"), ChatMessage.Assistant("a") }, "a", 1);
            // reward is null -> skipped
            var tooShort = new AgentTrajectory("s", "agent",
                new List<ChatMessage> { ChatMessage.User("only one message") }, "", 1, reward: 1.0);

            var dataset = new RewardFilteredDatasetBuilder(0.5).Build(new[] { ungraded, tooShort });

            Assert.Empty(dataset.Examples);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Builder_ThresholdControlsInclusion()
        {
            var runs = new[]
            {
                Run("1", "q1", "a1", 0.3),
                Run("2", "q2", "a2", 0.6),
                Run("3", "q3", "a3", 0.9),
            };

            Assert.Equal(3, new RewardFilteredDatasetBuilder(0.0).Build(runs).Count);
            Assert.Equal(2, new RewardFilteredDatasetBuilder(0.5).Build(runs).Count);
            Assert.Equal(1, new RewardFilteredDatasetBuilder(0.85).Build(runs).Count);
            await Task.CompletedTask;
        }
    }
}
