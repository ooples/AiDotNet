using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.SelfImproving;
using AiDotNetTests.UnitTests.Agentic.Agents;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.SelfImproving
{
    public class PromptOptimizerTests
    {
        // Builds a real AgentExecutor whose scripted model answers correctly only under the "calc" system
        // prompt — so different prompts genuinely score differently, with no manual result construction.
        private static IAgent<double> Factory(string prompt)
        {
            var answers = new Dictionary<string, string> { ["2+2"] = "4", ["3+3"] = "6" };
            var client = new ScriptedChatClient<double>((callIndex, messages) =>
            {
                var system = messages.FirstOrDefault(m => m.Role == ChatRole.System)?.Text ?? string.Empty;
                var input = messages.Last(m => m.Role == ChatRole.User).Text;
                var text = system == "calc" && answers.TryGetValue(input, out var a) ? a : "I am not sure.";
                return ChatResponses.Text(text);
            });
            return new AgentExecutor<double>(client, tools: null, new AgentExecutorOptions { SystemPrompt = prompt });
        }

        private static readonly IReadOnlyList<PromptEvalCase> Cases = new[]
        {
            new PromptEvalCase("2+2", "4"),
            new PromptEvalCase("3+3", "6"),
        };

        [Fact(Timeout = 60000)]
        public async Task Optimizer_SelectsBestPrompt()
        {
            await Task.Yield();

            var optimizer = new PromptOptimizer<double>();

            var result = await optimizer.OptimizeAsync(new[] { "vague", "calc" }, Factory, Cases);

            Assert.Equal("calc", result.BestPrompt);
            Assert.Equal(1.0, result.BestScore);

            var vague = result.Candidates.Single(c => c.Prompt == "vague");
            Assert.Equal(0.0, vague.Score);
            // Candidates are ranked best-first.
            Assert.Equal("calc", result.Candidates[0].Prompt);
        }

        [Fact(Timeout = 60000)]
        public async Task Optimizer_UsesCustomScorer()
        {
            await Task.Yield();

            // Custom scorer rewards shorter answers; "calc" yields "4"/"6" vs "vague"'s long sentence.
            var optimizer = new PromptOptimizer<double>((run, _) => 1.0 / (1 + run.FinalText.Length));

            var result = await optimizer.OptimizeAsync(new[] { "calc", "vague" }, Factory, Cases);

            Assert.Equal("calc", result.BestPrompt);
        }

        [Fact(Timeout = 60000)]
        public async Task Optimizer_SingleCandidate_ReturnsIt()
        {
            await Task.Yield();

            var optimizer = new PromptOptimizer<double>();
            var result = await optimizer.OptimizeAsync(new[] { "calc" }, Factory, Cases);
            Assert.Equal("calc", result.BestPrompt);
            Assert.Single(result.Candidates);
        }

        [Fact(Timeout = 60000)]
        public async Task Optimizer_RejectsEmptyInputs()
        {
            await Task.Yield();

            var optimizer = new PromptOptimizer<double>();
            await Assert.ThrowsAsync<ArgumentException>(() =>
                optimizer.OptimizeAsync(Array.Empty<string>(), Factory, Cases));
            await Assert.ThrowsAsync<ArgumentException>(() =>
                optimizer.OptimizeAsync(new[] { "calc" }, Factory, Array.Empty<PromptEvalCase>()));
        }
    }
}
