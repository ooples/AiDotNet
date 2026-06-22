using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.SelfImproving;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.SelfImproving
{
    public class EvaluationHarnessTests
    {
        private static AgentTrajectory Trajectory(string id, string finalText) =>
            new(id, "agent", new List<ChatMessage> { ChatMessage.User("q") }, finalText, iterations: 1);

        [Fact(Timeout = 60000)]
        public async Task Runner_ScoresTrajectories_AndAnnotatesReward()
        {
            await Task.Yield();

            var t1 = Trajectory("1", "42");
            var t2 = Trajectory("2", "wrong");
            // Exact-match grader against the known answer "42".
            var evaluator = new DelegateTrajectoryEvaluator(t => t.FinalText == "42" ? 1.0 : 0.0);
            var runner = new TrajectoryEvaluationRunner(evaluator);

            var report = await runner.EvaluateAsync(new[] { t1, t2 });

            Assert.Equal(1.0, t1.Reward);
            Assert.Equal(0.0, t2.Reward);
            Assert.Equal(2, report.Count);
            Assert.Equal(0.5, report.MeanReward);
            Assert.Equal(0.0, report.MinReward);
            Assert.Equal(1.0, report.MaxReward);
        }

        [Fact(Timeout = 60000)]
        public async Task Report_PassRate_UsesThreshold()
        {
            await Task.Yield();

            var trajectories = new[]
            {
                Trajectory("1", "a"),
                Trajectory("2", "b"),
                Trajectory("3", "c"),
                Trajectory("4", "d"),
            };
            // Rewards: 0.2, 0.4, 0.6, 0.8 by id.
            var rewards = new Dictionary<string, double> { ["1"] = 0.2, ["2"] = 0.4, ["3"] = 0.6, ["4"] = 0.8 };
            var runner = new TrajectoryEvaluationRunner(new DelegateTrajectoryEvaluator(t => rewards[t.Id]));

            var report = await runner.EvaluateAsync(trajectories, passThreshold: 0.5);

            Assert.Equal(0.5, report.PassRate); // 2 of 4 reach >= 0.5
            Assert.Equal(0.5, report.PassThreshold);
            Assert.Equal(0.5, report.MeanReward, 3);
        }

        [Fact(Timeout = 60000)]
        public async Task Runner_EvaluatesWholeStore()
        {
            await Task.Yield();

            var store = new InMemoryTrajectoryStore();
            await store.AddAsync(Trajectory("1", "good"));
            await store.AddAsync(Trajectory("2", "good"));
            await store.AddAsync(Trajectory("3", "bad"));

            var runner = new TrajectoryEvaluationRunner(new DelegateTrajectoryEvaluator(t => t.FinalText == "good" ? 1.0 : 0.0));
            var report = await runner.EvaluateStoreAsync(store, passThreshold: 1.0);

            Assert.Equal(3, report.Count);
            Assert.Equal(2.0 / 3.0, report.MeanReward, 3);
            Assert.Equal(2.0 / 3.0, report.PassRate, 3);

            // Rewards were written back to the stored trajectories.
            var graded = await store.GetAsync("1");
            Assert.NotNull(graded);
            Assert.Equal(1.0, graded.Reward);
        }

        [Fact(Timeout = 60000)]
        public async Task Runner_EmptySet_ReturnsZeroReport()
        {
            await Task.Yield();

            var runner = new TrajectoryEvaluationRunner(new DelegateTrajectoryEvaluator(_ => 1.0));
            var report = await runner.EvaluateAsync(Array.Empty<AgentTrajectory>());
            Assert.Equal(0, report.Count);
            Assert.Equal(0.0, report.MeanReward);
        }
    }
}
