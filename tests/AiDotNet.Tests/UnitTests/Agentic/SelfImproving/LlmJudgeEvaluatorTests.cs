using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.SelfImproving;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.SelfImproving
{
    public class LlmJudgeEvaluatorTests
    {
        private static AgentTrajectory Trajectory(string answer) =>
            new("t", "agent", new List<ChatMessage> { ChatMessage.User("What is 2+2?") }, answer, iterations: 1);

        [Fact(Timeout = 60000)]
        public async Task ParsesPlainNumberScore()
        {
            await Task.Yield();

            var judge = ScriptedChatClient<double>.Sequence(ChatResponses.Text("0.8"));
            var evaluator = new ChatClientTrajectoryEvaluator<double>(judge);

            var score = await evaluator.EvaluateAsync(Trajectory("4"));

            Assert.Equal(0.8, score);
        }

        [Fact(Timeout = 60000)]
        public async Task ExtractsNumberFromProse()
        {
            await Task.Yield();

            var judge = ScriptedChatClient<double>.Sequence(ChatResponses.Text("I rate this 0.95 out of 1."));
            var evaluator = new ChatClientTrajectoryEvaluator<double>(judge);

            Assert.Equal(0.95, await evaluator.EvaluateAsync(Trajectory("4")));
        }

        [Fact(Timeout = 60000)]
        public async Task ClampsOutOfRangeAndHandlesNonNumeric()
        {
            await Task.Yield();

            var high = new ChatClientTrajectoryEvaluator<double>(ScriptedChatClient<double>.Sequence(ChatResponses.Text("5")));
            Assert.Equal(1.0, await high.EvaluateAsync(Trajectory("x")));

            var negative = new ChatClientTrajectoryEvaluator<double>(ScriptedChatClient<double>.Sequence(ChatResponses.Text("-2")));
            Assert.Equal(0.0, await negative.EvaluateAsync(Trajectory("x")));

            var prose = new ChatClientTrajectoryEvaluator<double>(ScriptedChatClient<double>.Sequence(ChatResponses.Text("no number here")));
            Assert.Equal(0.0, await prose.EvaluateAsync(Trajectory("x")));
        }

        [Fact(Timeout = 60000)]
        public async Task IntegratesWithEvaluationRunner()
        {
            await Task.Yield();

            // The LLM judge plugs into the same runner as any ITrajectoryEvaluator.
            var judge = ScriptedChatClient<double>.Sequence(ChatResponses.Text("1.0"));
            var runner = new TrajectoryEvaluationRunner(new ChatClientTrajectoryEvaluator<double>(judge));

            var report = await runner.EvaluateAsync(new[] { Trajectory("4") });

            Assert.Equal(1.0, report.MeanReward);
            Assert.Equal(1.0, report.PassRate);
        }
    }
}
