using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    public class SupervisorAgentTests
    {
        // A worker that always answers with a fixed string (one model call, no tools).
        private static AgentExecutor<double> FixedWorker(string name, string answer, string? description = null)
        {
            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text(answer));
            return new AgentExecutor<double>(client, tools: null, new AgentExecutorOptions
            {
                Name = name,
                Description = description ?? $"Always answers '{answer}'.",
            });
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_RoutesToWorker_ThenComposesFinalAnswer()
        {
            var math = FixedWorker("math", "5", "Solves arithmetic.");

            // Coordinator: call 0 hands off to math; call 1 produces the final answer.
            var coordinator = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("c1", "transfer_to_math", "{\"task\":\"2+3\"}"),
                ChatResponses.Text("The result is 5."));

            var supervisor = new SupervisorAgent<double>(coordinator, new[] { math });

            var result = await supervisor.RunAsync("What is 2+3?");

            Assert.True(result.Completed);
            Assert.Equal("The result is 5.", result.FinalText);

            // The coordinator's second turn saw the worker's answer ("5") as a tool-result message.
            var secondRequest = coordinator.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Tool
                && m.Contents.OfType<ToolResultContent>().Any(r => r.Result == "5"));
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_RoutesToCorrectWorker_AmongSeveral()
        {
            var math = FixedWorker("math", "42", "Solves arithmetic.");
            var writer = FixedWorker("writer", "Once upon a time...", "Writes prose.");

            var coordinator = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("c1", "transfer_to_writer", "{\"task\":\"write a story\"}"),
                ChatResponses.Text("Here is your story."));

            var supervisor = new SupervisorAgent<double>(coordinator, new[] { math, writer });

            var result = await supervisor.RunAsync("Write me a story.");

            Assert.True(result.Completed);
            // The writer's output flowed back to the coordinator; the math agent was never engaged.
            var secondRequest = coordinator.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Tool
                && m.Contents.OfType<ToolResultContent>().Any(r => r.Result == "Once upon a time..."));
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_AdvertisesOneHandoffToolPerWorker()
        {
            var a = FixedWorker("alpha", "a");
            var b = FixedWorker("beta", "b");
            var coordinator = ScriptedChatClient<double>.Sequence(ChatResponses.Text("done"));

            var supervisor = new SupervisorAgent<double>(coordinator, new[] { a, b });
            await supervisor.RunAsync("hi");

            var options = coordinator.ReceivedOptions[0];
            Assert.NotNull(options);
            Assert.NotNull(options.Tools);
            Assert.Contains(options.Tools, t => t.Name == "transfer_to_alpha");
            Assert.Contains(options.Tools, t => t.Name == "transfer_to_beta");
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_DefaultRoutingPrompt_ListsWorkers()
        {
            var math = FixedWorker("math", "5", "Solves arithmetic.");
            var coordinator = ScriptedChatClient<double>.Sequence(ChatResponses.Text("done"));

            var supervisor = new SupervisorAgent<double>(coordinator, new[] { math });
            await supervisor.RunAsync("hi");

            var firstRequest = coordinator.Requests[0];
            Assert.Equal(ChatRole.System, firstRequest[0].Role);
            Assert.Contains("math", firstRequest[0].Text);
            Assert.Contains("Solves arithmetic.", firstRequest[0].Text);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_NestedSupervisors_Compose()
        {
            // Leaf worker under a sub-supervisor.
            var leaf = FixedWorker("calculator", "120", "Computes factorials.");
            var subCoordinator = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("s1", "transfer_to_calculator", "{\"task\":\"5!\"}"),
                ChatResponses.Text("5! = 120"));
            var subTeam = new SupervisorAgent<double>(subCoordinator, new[] { leaf },
                new SupervisorOptions { Name = "math_team", Description = "Handles math." });

            // Top supervisor delegates to the whole sub-team.
            var topCoordinator = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("t1", "transfer_to_math_team", "{\"task\":\"compute 5!\"}"),
                ChatResponses.Text("The factorial of 5 is 120."));
            var top = new SupervisorAgent<double>(topCoordinator, new IAgent<double>[] { subTeam });

            var result = await top.RunAsync("What is 5 factorial?");

            Assert.True(result.Completed);
            Assert.Equal("The factorial of 5 is 120.", result.FinalText);
            var secondRequest = topCoordinator.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Tool
                && m.Contents.OfType<ToolResultContent>().Any(r => r.Result == "5! = 120"));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_NoWorkers_Throws()
        {
            var coordinator = ScriptedChatClient<double>.Sequence(ChatResponses.Text("x"));
            Assert.Throws<ArgumentException>(() =>
                new SupervisorAgent<double>(coordinator, Array.Empty<IAgent<double>>()));
            await Task.CompletedTask;
        }
    }
}
